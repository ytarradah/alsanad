import streamlit as st
import os
import time
import unicodedata
import re
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ----------------------
# Configuration
# ----------------------
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ"
COLLECTION_NAME = "arabic_documents_enhanced"

# API Keys
DEEPSEEK_API_KEY = "sk-14f267781a6f474a9d0ec8240383dae4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_API_KEY = "AIzaSyASlapu6AYYwOQAJJ3v-2FSKHnxIOZoPbY"

# Initialize Gemini
gemini_initial_configured = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_initial_configured = True
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Failed to configure Gemini API: {e}")

# ----------------------
# Enhanced Arabic Text Processing
# ----------------------
def normalize_arabic_text(text):
    """Enhanced Arabic text normalization for better search accuracy"""
    if not text:
        return text
    
    # Remove diacritics (تشكيل)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Normalize Arabic characters
    replacements = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',  # Alef variations
        'ى': 'ي',  # Ya variations
        'ة': 'ه',  # Ta marbuta
        'ؤ': 'و', 'ئ': 'ي',  # Hamza
        '\u200c': '',  # Zero-width non-joiner
        '\u200d': '',  # Zero-width joiner
        '\ufeff': '',  # BOM
        '\u200b': '',  # Zero-width space
        '؟': '?', '؛': ';', '،': ',',  # Punctuation
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Clean and normalize whitespace
    text = ' '.join(text.split())
    return text

def extract_arabic_keywords(text):
    """Extract meaningful Arabic keywords by removing stop words"""
    if not text:
        return []
    
    # Comprehensive Arabic stop words
    arabic_stopwords = {
        # Pronouns
        'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'أنا', 'نحن',
        'إياه', 'إياها', 'إياهم', 'إياهن', 'إياك', 'إياكم', 'إياكن', 'إياي', 'إيانا',
        
        # Demonstratives
        'هذا', 'هذه', 'ذلك', 'تلك', 'أولئك', 'هؤلاء', 'التي', 'الذي', 'اللذان', 'اللتان',
        
        # Prepositions
        'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'تحت', 'فوق', 'أمام', 'خلف',
        'بين', 'ضد', 'نحو', 'حول', 'دون', 'سوى', 'خلال', 'عبر', 'لدى', 'عند',
        
        # Conjunctions and particles
        'و', 'أو', 'أم', 'لكن', 'لكن', 'غير', 'إلا', 'بل', 'ثم', 'كذلك',
        'أن', 'إن', 'كي', 'لكي', 'حتى', 'لولا', 'لوما', 'لو', 'إذا', 'إذ', 'حيث',
        
        # Auxiliaries and modals
        'كان', 'كانت', 'كانوا', 'كن', 'يكون', 'تكون', 'يكونوا', 'تكن',
        'قد', 'لقد', 'سوف', 'لن', 'لم', 'لما', 'ليس', 'ليست', 'ليسوا', 'لسن',
        
        # Question words
        'ما', 'ماذا', 'متى', 'أين', 'لماذا', 'كم', 'أي', 'أية', 'كيف', 'أنى',
        
        # Articles and determiners
        'ال', 'كل', 'جميع', 'بعض', 'معظم',
        
        # Common short words
        'ف', 'ب', 'ك', 'ل', 'عن', 'لا', 'نعم', 'كلا'
    }
    
    # Extract Arabic words (2+ characters)
    words = re.findall(r'[\u0600-\u06FF\u0750-\u077F]{2,}', text)
    
    # Filter out stop words and normalize
    keywords = []
    for word in words:
        normalized_word = normalize_arabic_text(word)
        if len(normalized_word) > 2 and normalized_word not in arabic_stopwords:
            keywords.append(normalized_word)
    
    return list(set(keywords))  # Remove duplicates

# ----------------------
# CSS Styling
# ----------------------
def load_arabic_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; }
    .main-header { text-align: center; color: #2E8B57; font-family: 'Noto Sans Arabic', sans-serif; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; direction: rtl; }
    .sub-header { text-align: center; color: #666; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.2rem; font-weight: 400; margin-bottom: 1rem; direction: rtl; }
    .stExpander .stExpanderHeader { font-size: 1.1rem !important; font-weight: 600 !important; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; } 
    .stExpander div[data-testid="stExpanderDetails"] { direction: rtl; } 
    .stTextArea > div > div > textarea { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; min-height: 100px !important; border-radius: 10px; border: 1px solid #ccc; }
    .search-button-container { text-align: center; margin-top: 1rem; margin-bottom: 1rem; }
    div[data-testid="stButton"] > button { margin: 0 auto; display: block; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 600; border-radius: 8px; transition: background-color 0.2s ease, transform 0.2s ease; }
    div[data-testid="stButton"] > button:hover { transform: translateY(-2px); }
    .chat-container { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px 15px 5px 15px; margin: 0.5rem 0; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 500; box-shadow: 0 2px 10px rgba(0,0,0,0.1); word-wrap: break-word; }
    .bot-message { background: linear-gradient(135deg, #5DADE2 0%, #3498DB 100%); color: white; padding: 1rem; border-radius: 15px 15px 15px 5px; margin: 0.5rem 0; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 500; line-height: 1.8; box-shadow: 0 2px 10px rgba(0,0,0,0.1); word-wrap: break-word; }
    .time-taken { font-size: 0.8rem; color: #777; margin-top: 0.3rem; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif;}
    .debug-info { background: #fff3cd; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.85rem; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; border: 1px solid #ffeeba; word-wrap: break-word; }
    .debug-info-results { font-size: 0.8rem; white-space: pre-wrap; word-break: break-all; }
    .api-used { background: #e3f2fd; color: #1976d2; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.85rem; margin-top: 0.5rem; display: inline-block; font-family: 'Noto Sans Arabic', sans-serif; }
    .source-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-top: 1rem; direction: rtl; } 
    .source-info { background: #f0f2f6; padding: 0.25rem 0.4rem; border-radius: 6px; font-size: 0.75rem; color: #555; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; border: 1px solid #ddd; transition: transform 0.2s ease-in-out; display: flex; flex-direction: column; justify-content: center; overflow-wrap: break-word; word-break: break-word; min-height: 50px; }
    .source-info strong { font-size: 0.8rem; } 
    .source-info:hover { transform: translateY(-2px); box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
    .status-active { color: #28a745; font-weight: bold; }
    .status-inactive { color: #dc3545; font-weight: bold; }
    .radio-label-status-active { color: #28a745 !important; font-weight: normal !important; font-size:0.9em !important; }
    .radio-label-status-inactive { color: #dc3545 !important; font-weight: normal !important; font-size:0.9em !important; }
    .search-accuracy-boost { background: #d4edda; border: 1px solid #c3e6cb; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# Initialize Components
# ----------------------
@st.cache_resource
def init_qdrant_client():
    try:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
    except Exception as e:
        st.error(f"فشل الاتصال بـ Qdrant: {e}")
        return None

@st.cache_resource
def init_embedding_model():
    try:
        return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    except Exception as e:
        st.error(f"فشل تحميل نموذج التضمين: {e}")
        return None

# ----------------------
# IMPROVED SEARCH FUNCTION - Fixed for Better Accuracy
# ----------------------
def comprehensive_search(query, max_results=50):
    """
    البحث المحسن مع دقة أفضل للنتائج العربية
    مصمم خصيصاً لحل مشكلة النتائج غير الدقيقة
    """
    
    embedding_model = init_embedding_model()
    if not embedding_model:
        return [], "فشل تحميل نموذج التضمين.", []
    
    qdrant_client = init_qdrant_client()
    if not qdrant_client:
        return [], "فشل الاتصال بـ Qdrant.", []
    
    try:
        print(f"🔍 البحث المحسن للاستعلام: '{query}'")
        
        # إنشاء متغيرات البحث مع عتبات أقل للحصول على نتائج أكثر دقة
        search_variants = []
        
        # 1. البحث بالاستعلام الأصلي - عتبة منخفضة
        search_variants.append(('original', query, 0.1))
        
        # 2. البحث بالنص المطبع
        normalized_query = normalize_arabic_text(query)
        if normalized_query != query and normalized_query:
            search_variants.append(('normalized', normalized_query, 0.08))
        
        # 3. البحث بالكلمات المفتاحية
        keywords = extract_arabic_keywords(query)
        if keywords:
            keywords_text = ' '.join(keywords)
            search_variants.append(('keywords', keywords_text, 0.05))
            
            # 4. البحث بكلمات مفتاحية فردية مهمة
            important_keywords = [kw for kw in keywords if len(kw) > 3][:3]
            for keyword in important_keywords:
                search_variants.append(('single_keyword', keyword, 0.03))
        
        # 5. البحث بكلمات فردية من الاستعلام الأصلي
        individual_words = [w.strip() for w in query.split() if len(w.strip()) > 3][:2]
        for word in individual_words:
            search_variants.append(('individual_word', word, 0.02))
        
        # تنفيذ البحث مع كل متغير
        all_results = []
        seen_ids = set()
        search_details = []
        
        print(f"📊 سيتم اختبار {len(search_variants)} متغير بحث...")
        
        for variant_type, variant_query, threshold in search_variants:
            try:
                # إنشاء embedding للمتغير
                query_embedding = embedding_model.encode([variant_query])[0].tolist()
                
                # البحث مع هذا المتغير
                variant_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=max_results,
                    with_payload=True,
                    score_threshold=threshold
                )
                
                # إضافة النتائج الجديدة (تجنب التكرار)
                new_results_count = 0
                for result in variant_results:
                    if result.id not in seen_ids and result.payload:
                        # فحص أن النص ليس فارغاً وذو معنى
                        text = result.payload.get('text', '')
                        if len(text.strip()) > 20:  # على الأقل 20 حرف
                            seen_ids.add(result.id)
                            all_results.append(result)
                            new_results_count += 1
                
                search_details.append(f"{variant_type} (عتبة {threshold}): {new_results_count} نتيجة")
                print(f"✅ {variant_type}: {new_results_count} نتيجة جديدة")
                
                # إذا وجدنا نتائج كافية من المتغيرات الأولى، لا نحتاج للباقي
                if len(all_results) >= 15 and variant_type in ['original', 'normalized']:
                    print(f"🎯 تم العثور على {len(all_results)} نتيجة كافية، توقف البحث")
                    break
                    
            except Exception as e:
                search_details.append(f"{variant_type}: خطأ ({str(e)[:30]})")
                print(f"❌ خطأ في {variant_type}: {e}")
                continue
        
        # ترتيب النتائج حسب النقاط (الأعلى أولاً)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # اختيار أفضل النتائج
        final_results = all_results[:max_results]
        
        # إنشاء معلومات تفصيلية للتشخيص
        initial_search_details = []
        if final_results:
            initial_search_details = [
                {
                    "id": r.id,
                    "score": r.score,
                    "source": r.payload.get('source', 'N/A') if r.payload else 'N/A',
                    "text_preview": (r.payload.get('text', '')[:150] + "...") if r.payload else ''
                }
                for r in final_results[:12]  # أفضل 12 نتيجة للتشخيص
            ]
        
        # إنشاء ملخص البحث
        successful_variants = len([d for d in search_details if 'خطأ' not in d and ': 0 نتيجة' not in d])
        search_info = f"بحث محسن للدقة: {len(final_results)} نتيجة نهائية من {successful_variants}/{len(search_variants)} متغير. " + " | ".join(search_details[:5])  # أول 5 تفاصيل
        
        print(f"✅ النتائج النهائية: {len(final_results)}")
        if final_results:
            best_result = final_results[0]
            print(f"🎯 أفضل نتيجة: {best_result.payload.get('source', 'Unknown')} (نقاط: {best_result.score:.3f})")
            print(f"📄 معاينة: {best_result.payload.get('text', '')[:100]}...")
        
        return final_results, search_info, initial_search_details
        
    except Exception as e:
        error_msg = f"خطأ شامل في البحث المحسن: {str(e)}"
        print(f"❌ {error_msg}")
        
        # محاولة أخيرة بحث بسيط بعتبة منخفضة جداً
        try:
            print("🔄 محاولة بحث طوارئ...")
            emergency_embedding = embedding_model.encode([query])[0].tolist()
            emergency_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=emergency_embedding,
                limit=max_results,
                with_payload=True,
                score_threshold=0.01  # عتبة طوارئ منخفضة جداً
            )
            
            emergency_details = [
                {
                    "id": r.id,
                    "score": r.score,
                    "source": r.payload.get('source', 'N/A') if r.payload else 'N/A',
                    "text_preview": (r.payload.get('text', '')[:100] + "...") if r.payload else ''
                }
                for r in emergency_results[:10]
            ]
            
            return emergency_results, f"{error_msg} | بحث طوارئ: {len(emergency_results)} نتيجة", emergency_details
            
        except Exception as emergency_error:
            return [], f"{error_msg} | فشل بحث الطوارئ: {str(emergency_error)}", []

# ----------------------
# API Response Functions
# ----------------------
def prepare_llm_messages(user_question, context, context_info):
    system_prompt = "أنت مساعد للبحث في كتب واستفتاءات الشيخ محمد السند فقط.\nقواعد حتمية لا يمكن تجاوزها:\n1. أجب فقط من النصوص المعطاة أدناه (\"المصادر المتاحة\") - لا استثناءات.\n2. إذا لم تجد الإجابة الكاملة في النصوص، قل بوضوح: \"لم أجد إجابة كافية في المصادر المتاحة بخصوص هذا السؤال.\"\n3. ممنوع منعاً باتاً إضافة أي معلومة من خارج النصوص المعطاة. لا تستخدم معلوماتك العامة أو معرفتك السابقة.\n4. اقتبس مباشرة من النصوص عند الإجابة قدر الإمكان، مع الإشارة إلى المصدر إذا كان متاحاً في النص (مثال: [نص ١]).\n5. إذا وجدت إجابة جزئية، اذكرها وأوضح أنها غير كاملة أو تغطي جانباً من السؤال.\n6. هدفك هو تقديم إجابة دقيقة وموثوقة بناءً على ما هو متوفر في النصوص فقط.\n7. إذا كانت النصوص لا تحتوي على إجابة، لا تحاول استنتاج أو تخمين الإجابة.\nتذكر: أي معلومة ليست في النصوص أدناه = لا تذكرها أبداً. كن دقيقاً ومقتصراً على المصادر."
    user_content = f"السؤال المطروح: {user_question}\n\nالمصادر المتاحة من قاعدة البيانات فقط (أجب بناءً عليها حصراً):\n{context}\n\nمعلومات إضافية عن السياق: {context_info}\n\nالتعليمات: يرجى تقديم إجابة بناءً على النصوص أعلاه فقط. إذا لم تكن الإجابة موجودة، وضح ذلك."
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

def get_deepseek_response(messages, max_tokens=2000):
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API key مفقود."
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "deepseek-chat", "messages": messages, "temperature": 0.05, "max_tokens": max_tokens, "stream": False}
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'] if result.get('choices') and result['choices'][0].get('message') else "لم يتمكن DeepSeek من الرد."
    except requests.exceptions.Timeout:
        return "انتهت مهلة DeepSeek."
    except requests.exceptions.HTTPError as e:
        err_content = e.response.text if e.response else "No response"
        return f"خطأ DeepSeek: {e.response.status_code if e.response else 'N/A'}. تفاصيل: {err_content[:200]}"
    except Exception as e:
        return f"خطأ DeepSeek: {str(e)}"

def get_gemini_response(messages, max_tokens=2000):
    global gemini_initial_configured
    if not GEMINI_API_KEY:
        return "Gemini API key مفقود."
    try:
        if not gemini_initial_configured:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_initial_configured = True
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        proc_msgs, sys_prompt_txt = [], None
        
        if messages and messages[0]["role"] == "system":
            sys_prompt_txt = messages[0]["content"]
            for msg in messages[1:]:
                proc_msgs.append({"role": "user" if msg["role"]=="user" else "model", "parts": [msg["content"]]})
        else:
            for msg in messages:
                proc_msgs.append({"role": "user" if msg["role"]=="user" else "model", "parts": [msg["content"]]})
        
        if sys_prompt_txt:
            if proc_msgs and proc_msgs[0]["role"] == "user":
                proc_msgs[0]["parts"][0] = f"{sys_prompt_txt}\n\n---\n\n{proc_msgs[0]['parts'][0]}"
            else:
                proc_msgs.insert(0, {"role": "user", "parts": [sys_prompt_txt]})
        
        if not proc_msgs:
            return "لا توجد رسائل صالحة لـ Gemini."
        
        if len(proc_msgs) > 1 and proc_msgs[-1]["role"] == "user":
            hist, curr_msg = proc_msgs[:-1], proc_msgs[-1]["parts"][0]
            chat = model.start_chat(history=hist)
            resp = chat.send_message(curr_msg, generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens))
        elif proc_msgs and proc_msgs[0]["role"] == "user":
            resp = model.generate_content(proc_msgs[0]["parts"], generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens))
        else:
            return "بنية رسائل Gemini غير متوقعة."
        
        return resp.text
    except Exception as e:
        return f"خطأ Gemini: {str(e)}"

# ----------------------
# Status Functions
# ----------------------
@st.cache_data(ttl=300)
def get_qdrant_info():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5)
        collection_info = client.get_collection(COLLECTION_NAME)
        return {
            "status": True,
            "message": f"متصل ✓ | النقاط: {collection_info.points_count:,}",
            "details": {
                "اسم المجموعة": COLLECTION_NAME,
                "عدد النقاط": collection_info.points_count,
                "حالة الفهرسة": str(collection_info.status),
                "تهيئة Vector": str(collection_info.config.params)
            }
        }
    except Exception as e:
        if "Not found: Collection" in str(e) or "NOT_FOUND" in str(e).upper():
            return {"status": False, "message": f"غير متصل (المجموعة '{COLLECTION_NAME}' غير موجودة)", "details": {}}
        return {"status": False, "message": f"غير متصل (خطأ: {type(e).__name__})", "details": {}}

@st.cache_data(ttl=300)
def check_api_status(api_name):
    global gemini_initial_configured
    if api_name == "DeepSeek":
        if not DEEPSEEK_API_KEY:
            return False, "المفتاح مفقود"
        try:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "مرحبا"}], "max_tokens": 1, "stream": False}
            response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=7)
            return (True, "نشط ✓") if response.status_code == 200 else (False, f"خطأ ({response.status_code})")
        except Exception as e:
            return False, f"غير نشط ({type(e).__name__})"
    elif api_name == "Gemini":
        if not GEMINI_API_KEY:
            return False, "المفتاح مفقود"
        try:
            if not gemini_initial_configured:
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_initial_configured = True
            model = genai.GenerativeModel('gemini-1.5-flash')
            model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1))
            return True, "نشط ✓"
        except Exception as e:
            err = str(e).lower()
            if "api_key_invalid" in err or "permission" in err or "quota" in err or "authentication" in err:
                return False, "خطأ بالمفتاح"
            return False, f"غير نشط ({type(e).__name__})"
    return False, "API غير معروف"

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="المرجع السند - بحث", page_icon="🕌", layout="wide", initial_sidebar_state="collapsed")
    load_arabic_css()
    
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">محرك بحث الكتب والاستفتاءات المحسن - دقة عالية</p>', unsafe_allow_html=True)

    # Settings Section
    with st.expander("⚙️ الإعدادات وحالة الأنظمة", expanded=True):
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>اختر محرك الذكاء الاصطناعي:</div>", unsafe_allow_html=True)
        
        # Check API status
        deepseek_ok, deepseek_msg = check_api_status("DeepSeek")
        gemini_ok, gemini_msg = check_api_status("Gemini")
        
        llm_options = ["DeepSeek", "Gemini"]
        llm_captions = [
            f"<span class='{'radio-label-status-active' if deepseek_ok else 'radio-label-status-inactive'}'>({deepseek_msg})</span>",
            f"<span class='{'radio-label-status-active' if gemini_ok else 'radio-label-status-inactive'}'>({gemini_msg})</span>"
        ]
        
        default_index = 0 if deepseek_ok else (1 if gemini_ok else 0)
        if not deepseek_ok and not gemini_ok:
            st.warning("تنبيه: جميع محركات الذكاء الاصطناعي غير نشطة.", icon="⚠️")
        
        selected_llm = st.radio("محركات AI:", llm_options, captions=llm_captions, index=default_index, horizontal=True, key="llm_sel", label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>##### حالة قاعدة البيانات (Qdrant):</div>", unsafe_allow_html=True)
        qdrant_info = get_qdrant_info()
        status_class = "status-active" if qdrant_info["status"] else "status-inactive"
        st.markdown(f'<div style="display: flex; justify-content: center;"><div style="background: #f0f2f6; padding: 0.5rem; border-radius: 8px; text-align: center; font-family: \'Noto Sans Arabic\', sans-serif; direction: rtl; border: 1px solid #e0e0e0; font-size: 0.9rem; margin-bottom: 0.5rem; width: 90%; max-width: 450px;">Qdrant DB: <span class="{status_class}">{qdrant_info["message"]}</span></div></div>', unsafe_allow_html=True)

        st.markdown("<div style='text-align: right; font-weight: bold; margin-top:0.5rem;'>مستوى البحث:</div>", unsafe_allow_html=True)
        search_levels = ["بحث سريع (15)", "بحث متوسط (30)", "بحث شامل (50)"]
        selected_level = st.radio("مستوى البحث:", search_levels, index=1, horizontal=True, key="s_depth_radio", label_visibility="collapsed")
        max_results = {"بحث سريع (15)": 15, "بحث متوسط (30)": 30, "بحث شامل (50)": 50}[selected_level]
        
        show_debug = st.checkbox("إظهار معلومات تفصيلية", value=True, key="debug_cb")

    # Database info section
    if qdrant_info['status'] and qdrant_info.get('details'):
        with st.expander("ℹ️ معلومات قاعدة البيانات", expanded=False):
            details = qdrant_info['details']
            info_html = f"<div style='direction: rtl; padding: 1rem; background-color: #e9ecef; border-radius: 10px; margin-top:1rem; margin-bottom: 1.5rem; border: 1px solid #ced4da;'>"
            info_html += f"<h3 style='font-family: \"Noto Sans Arabic\", sans-serif; text-align:right; color: #495057;'>مجموعة: {details.get('اسم المجموعة', COLLECTION_NAME)}</h3>"
            for k, v in details.items():
                if k != "اسم المجموعة":
                    info_html += f"<p style='font-family: \"Noto Sans Arabic\", sans-serif; text-align:right; margin-bottom: 0.3rem;'><strong>{k}:</strong> {v}</p>"
            info_html += "</div>"
            st.markdown(info_html, unsafe_allow_html=True)
    elif not qdrant_info['status']:
        st.warning(f"Qdrant: {qdrant_info['message']}.", icon="⚠️")

    # Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg_item in st.session_state.messages:
            is_user = msg_item["role"] == "user"
            icon = "👤" if is_user else "🤖"
            css_class = "user-message" if is_user else "bot-message"
            st.markdown(f'<div class="{css_class}">{icon} {msg_item["content"]}</div>', unsafe_allow_html=True)
            
            if not is_user:
                # Show API used
                if "api_used" in msg_item:
                    st.markdown(f'<span class="api-used">استخدم: {msg_item["api_used"]}</span>', unsafe_allow_html=True)
                
                # Show time taken
                if "time_taken" in msg_item:
                    st.markdown(f'<div class="time-taken">⏱️ زمن: {msg_item["time_taken"]:.2f} ث</div>', unsafe_allow_html=True)
                
                # Show debug info if enabled
                if show_debug:
                    debug_parts = []
                    if "debug_info" in msg_item:
                        debug_parts.append(msg_item["debug_info"])
                    
                    if "initial_search_details" in msg_item and msg_item["initial_search_details"]:
                        details_str_parts = []
                        for d_idx, d in enumerate(msg_item["initial_search_details"]):
                            display_id = str(d.get('id', 'N/A'))
                            details_str_parts.append(f"  {d_idx+1}. ID: {display_id[:8]}... | Score: {d.get('score', 0):.3f} | Source: {d.get('source', 'N/A')} | Preview: {d.get('text_preview', 'N/A')}")
                        details_str = "\n".join(details_str_parts)
                        debug_parts.append(f"نتائج Qdrant المفصلة ({len(msg_item['initial_search_details'])}):\n{details_str}")
                    
                    if debug_parts:
                        st.markdown(f'<div class="debug-info">🔍 معلومات تفصيلية:<div class="debug-info-results">{"<hr>".join(debug_parts)}</div></div>', unsafe_allow_html=True)

                # Show sources
                if "sources" in msg_item and msg_item["sources"]:
                    st.markdown("<div style='text-align: right; margin-top:0.5rem;'><strong>المصادر المستخدمة:</strong></div><div class='source-container'>", unsafe_allow_html=True)
                    for j_idx in range(0, min(len(msg_item["sources"]), 9), 3):
                        cols = st.columns(3)
                        for k_idx, k_src_item in enumerate(msg_item["sources"][j_idx:j_idx+3]):
                            with cols[k_idx]:
                                source = k_src_item.get("source", "N/A")
                                score = k_src_item.get("score", 0)
                                st.markdown(f'<div class="source-info" title="S: {source}\nSc: {score*100:.1f}%">📄 <strong>{source}</strong><br>تطابق: {score*100:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Search input section
    st.markdown("<hr style='margin-top:1.5rem; margin-bottom:0.5rem;'>", unsafe_allow_html=True)
    
    # Accuracy improvement notice
    st.markdown('<div class="search-accuracy-boost">🎯 <strong>تحسينات دقة البحث:</strong> تم تحسين خوارزمية البحث لتقديم نتائج أكثر دقة وصلة بالاستعلام العربي</div>', unsafe_allow_html=True)
    
    _, input_main, _ = st.columns([0.2, 2.6, 0.2])
    with input_main:
        user_query = st.text_area("سؤالك...", placeholder="اكتب سؤالك هنا (مثال: صلاة ليلة الرغائب، قاعدة التسامح في أدلة السنن، حكم السيلفي في الإحرام)...", key="user_input", height=120, label_visibility="collapsed")
        
        st.markdown('<div class="search-button-container">', unsafe_allow_html=True)
        search_button = st.button("🔍 بحث وإجابة محسن", type="primary", use_container_width=False, key="send_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    # Process search when button is clicked
    if search_button and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query.strip()})
        
        start_time = time.perf_counter()
        bot_msg_data = {"api_used": selected_llm}
        
        # Enhanced search with improved accuracy
        with st.spinner(f"جاري البحث المحسن للدقة ({max_results} نتيجة)..."):
            try:
                search_results, search_info, search_details = comprehensive_search(user_query.strip(), max_results)
                bot_msg_data["initial_search_details"] = search_details
                
                # Show real-time debug info
                if show_debug:
                    st.markdown(f'<div class="search-accuracy-boost">🔍 <strong>نتائج البحث المباشرة:</strong> وجدت {len(search_results)} نتيجة | {search_info}</div>', unsafe_allow_html=True)
                
            except Exception as search_error:
                st.error(f"خطأ في البحث: {search_error}")
                search_results, search_info, search_details = [], f"خطأ في البحث: {str(search_error)}", []

        # Process search results
        if search_results:
            try:
                # Prepare context for LLM
                context_texts = []
                sources_for_llm = []
                total_chars = 0
                max_chars_context = 25000
                
                for i, result in enumerate(search_results):
                    if not result.payload:
                        continue
                    
                    source_id_str = str(result.id) if result.id is not None else f"unknown_id_{i}"
                    source_name = result.payload.get('source', f'وثيقة {source_id_str[:6]}')
                    text = result.payload.get('text', '')
                    
                    if text and len(text.strip()) > 10:  # Ensure meaningful text
                        truncated_text = text[:1500] + ("..." if len(text) > 1500 else "")
                        if total_chars + len(truncated_text) < max_chars_context:
                            context_texts.append(f"[نص {i+1} من '{source_name}']: {truncated_text}")
                            sources_for_llm.append({'source': source_name, 'score': result.score, 'id': result.id})
                            total_chars += len(truncated_text)
                        else:
                            context_texts.append(f"\n[ملاحظة: تم اقتصار النصوص. {len(search_results)-i} نص إضافي لم يرسل.]")
                            search_info += f" | اقتصار السياق، {len(search_results)-i} نصوص لم ترسل."
                            break
                
                if context_texts:
                    context_for_llm = "\n\n---\n\n".join(context_texts)
                    llm_context_info = f"أرسل {len(sources_for_llm)} نص للتحليل (~{total_chars//1000} ألف حرف)."
                    llm_messages = prepare_llm_messages(user_query.strip(), context_for_llm, llm_context_info)
                    
                    # Get LLM response
                    bot_response = ""
                    with st.spinner(f"جاري التحليل بواسطة {selected_llm}..."):
                        try:
                            if selected_llm == "DeepSeek":
                                bot_response = get_deepseek_response(llm_messages)
                            elif selected_llm == "Gemini":
                                bot_response = get_gemini_response(llm_messages)
                            else:
                                bot_response = "المحرك المحدد غير معروف."
                        except Exception as llm_error:
                            bot_response = f"خطأ في الحصول على الرد من {selected_llm}: {str(llm_error)}"
                    
                    bot_msg_data["content"] = bot_response
                    bot_msg_data["sources"] = sources_for_llm
                    bot_msg_data["debug_info"] = f"{search_info} | {llm_context_info}" if search_info else llm_context_info
                else:
                    bot_msg_data["content"] = f"تم العثور على {len(search_results)} نتيجة ولكن لا تحتوي على نصوص صالحة للمعالجة. يرجى تجربة صياغة مختلفة للسؤال."
                    bot_msg_data["debug_info"] = f"{search_info} | لا توجد نصوص صالحة في النتائج"
            
            except Exception as processing_error:
                st.error(f"خطأ في معالجة النتائج: {processing_error}")
                bot_msg_data["content"] = f"تم العثور على {len(search_results)} نتيجة ولكن حدث خطأ في المعالجة: {str(processing_error)}"
                bot_msg_data["debug_info"] = f"{search_info} | خطأ معالجة: {str(processing_error)}"
        else:
            bot_msg_data["content"] = "لم أجد أي معلومات متعلقة بسؤالك في قاعدة بيانات كتب واستفتاءات الشيخ محمد السند حالياً. يرجى محاولة صياغة السؤال بشكل مختلف أو استخدام كلمات مفتاحية أخرى."
            bot_msg_data["debug_info"] = search_info if search_info else "لا توجد نتائج من البحث المحسن."
        
        # Save response with timing
        bot_msg_data["role"] = "assistant"
        bot_msg_data["time_taken"] = time.perf_counter() - start_time
        st.session_state.messages.append(bot_msg_data)
        st.rerun()
    
    elif search_button and not user_query.strip():
        st.toast("يرجى إدخال سؤال.", icon="📝")

    # Clear chat button
    with input_main:
        if st.button("🗑️ مسح المحادثة", use_container_width=True, key="clear_btn", type="secondary"):
            st.session_state.messages = []
            st.toast("تم مسح المحادثة.", icon="🗑️")
            time.sleep(0.5)
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 1rem; font-family: "Noto Sans Arabic", sans-serif;'>
        🚀 محرك البحث المحسن للدقة العالية | يدعم البحث متعدد المتغيرات والمعالجة المتقدمة للنصوص العربية<br>
        ✨ تحسينات خاصة: عتبات نقاط محسنة، فلترة ذكية للنتائج، ترتيب متقدم حسب الصلة
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Configuration validation
    if not all([QDRANT_API_KEY, QDRANT_URL]):
        st.error("معلومات QDRANT مفقودة. تحقق من الإعدادات.")
    if not any([DEEPSEEK_API_KEY, GEMINI_API_KEY]):
        st.info("بعض مفاتيح LLM API مفقودة.", icon="ℹ️")
    
    main()
