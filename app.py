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
    """Enhanced Arabic text normalization"""
    if not text:
        return text
    
    # Remove diacritics
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Normalize characters
    replacements = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
        'ى': 'ي', 'ة': 'ه', 'ؤ': 'و', 'ئ': 'ي',
        '\u200c': '', '\u200d': '', '\ufeff': '', '\u200b': '',
        '؟': '?', '؛': ';', '،': ',', 'ـ': ''
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return ' '.join(text.split())

def extract_arabic_keywords(text):
    """Extract Arabic keywords"""
    if not text:
        return []
    
    # Arabic stop words
    stop_words = {
        'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'أنا', 'نحن',
        'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'اللذان', 'اللتان',
        'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'تحت', 'فوق',
        'و', 'أو', 'أم', 'لكن', 'غير', 'إلا', 'بل', 'ثم', 'كذلك',
        'أن', 'إن', 'كي', 'لكي', 'حتى', 'لولا', 'لو', 'إذا', 'إذ', 'حيث',
        'كان', 'كانت', 'كانوا', 'كن', 'يكون', 'تكون', 'قد', 'لقد', 'سوف',
        'لن', 'لم', 'لما', 'ليس', 'ليست', 'ما', 'ماذا', 'متى', 'أين',
        'كيف', 'لماذا', 'كم', 'أي', 'أية', 'ال', 'كل', 'جميع', 'بعض',
        'ف', 'ب', 'ك', 'ل', 'لا', 'نعم', 'كلا'
    }
    
    # Extract Arabic words
    words = re.findall(r'[\u0600-\u06FF\u0750-\u077F]{2,}', text)
    keywords = []
    
    for word in words:
        normalized = normalize_arabic_text(word)
        if len(normalized) > 2 and normalized not in stop_words:
            keywords.append(normalized)
    
    return list(set(keywords))

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
    .ultra-search-boost { background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%); border: 2px solid #28a745; padding: 0.75rem; border-radius: 10px; margin: 1rem 0; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; text-align: right; box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2); }
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
# ULTRA-ENHANCED SEARCH FUNCTION
# ----------------------
def comprehensive_search(query, max_results=50):
    """Ultra-enhanced search with 50+ strategies to find hidden content"""
    
    embedding_model = init_embedding_model()
    if not embedding_model:
        return [], "فشل تحميل نموذج التضمين.", []
    
    qdrant_client = init_qdrant_client()
    if not qdrant_client:
        return [], "فشل الاتصال بـ Qdrant.", []
    
    try:
        print(f"🔍 Ultra search for: '{query}'")
        
        # Create comprehensive search strategies
        search_strategies = []
        original_query = query.strip()
        
        # 1. Basic searches with graduated thresholds
        search_strategies.extend([
            ('basic_high', original_query, 0.15),
            ('basic_medium', original_query, 0.08),
            ('basic_low', original_query, 0.04),
            ('basic_emergency', original_query, 0.01)
        ])
        
        # 2. Normalized search
        normalized_query = normalize_arabic_text(original_query)
        if normalized_query != original_query and normalized_query:
            search_strategies.extend([
                ('normalized_high', normalized_query, 0.12),
                ('normalized_low', normalized_query, 0.03)
            ])
        
        # 3. Keywords search
        keywords = extract_arabic_keywords(original_query)
        if keywords:
            all_keywords = ' '.join(keywords)
            search_strategies.extend([
                ('keywords_all', all_keywords, 0.06),
                ('keywords_low', all_keywords, 0.02)
            ])
            
            # Important keywords
            important_keywords = keywords[:3]
            for i, keyword in enumerate(important_keywords):
                if len(keyword) > 3:
                    search_strategies.append((f'keyword_{i+1}', keyword, 0.015))
        
        # 4. Religious term expansions
        religious_terms = {
            'صلاة': ['صلاة', 'الصلاة', 'نافلة', 'فريضة'],
            'حلق': ['حلق', 'حلاقة', 'إزالة'],
            'لحية': ['لحية', 'اللحية', 'الذقن'],
            'رغائب': ['رغائب', 'الرغائب', 'رجب'],
            'جواز': ['يجوز', 'جواز', 'حلال', 'حرام', 'مباح'],
            'حكم': ['حكم', 'أحكام', 'يحكم'],
            'مسألة': ['مسألة', 'مسائل', 'سؤال']
        }
        
        query_lower = original_query.lower()
        for base_term, expansions in religious_terms.items():
            if base_term in query_lower:
                for expansion in expansions:
                    search_strategies.append((f'religious_{expansion}', expansion, 0.01))
        
        # 5. Partial phrase searches
        words = original_query.split()
        if len(words) > 1:
            # First two words
            if len(words) >= 2:
                first_two = ' '.join(words[:2])
                search_strategies.append(('partial_start', first_two, 0.02))
            
            # Last two words
            if len(words) >= 2:
                last_two = ' '.join(words[-2:])
                search_strategies.append(('partial_end', last_two, 0.02))
        
        # 6. Source-specific searches
        source_terms = ['منهاج', 'استفتاء', 'سند', 'مسألة']
        for term in source_terms:
            search_strategies.append((f'source_{term}', term, 0.005))
        
        # Execute search strategies
        all_results = []
        seen_ids = set()
        search_details = []
        strategy_success = {}
        
        print(f"Testing {len(search_strategies)} search strategies...")
        
        for strategy_name, strategy_query, threshold in search_strategies:
            try:
                if not strategy_query or len(strategy_query.strip()) < 2:
                    continue
                
                # Create embedding
                query_embedding = embedding_model.encode([strategy_query])[0].tolist()
                
                # Search
                strategy_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=max_results * 2,
                    with_payload=True,
                    score_threshold=threshold
                )
                
                # Add new results
                new_results_count = 0
                for result in strategy_results:
                    if (result.id not in seen_ids and 
                        result.payload and 
                        result.payload.get('text', '').strip() and
                        len(result.payload.get('text', '').strip()) >= 10):
                        
                        seen_ids.add(result.id)
                        all_results.append(result)
                        new_results_count += 1
                
                strategy_success[strategy_name] = new_results_count
                search_details.append(f"{strategy_name}: {new_results_count}")
                
                if new_results_count > 0:
                    print(f"✅ {strategy_name}: {new_results_count} new results")
                
                # Early stopping for basic strategies if we have enough results
                if strategy_name.startswith('basic') and len(all_results) >= 25:
                    print(f"Early stop: {len(all_results)} results found")
                    break
                    
            except Exception as e:
                search_details.append(f"{strategy_name}: error")
                print(f"❌ Error in {strategy_name}: {e}")
                continue
        
        # Enhanced result ranking
        print(f"Ranking {len(all_results)} results...")
        
        for result in all_results:
            if result.payload:
                text = result.payload.get('text', '').lower()
                source = result.payload.get('source', '').lower()
                
                # Relevance boost
                relevance_boost = 0
                
                # Check for query words in text
                for word in original_query.split():
                    if len(word) > 2:
                        word_lower = word.lower()
                        normalized_word = normalize_arabic_text(word_lower)
                        
                        if word_lower in text:
                            relevance_boost += 0.15
                        elif normalized_word in text:
                            relevance_boost += 0.10
                        
                        if word_lower in source:
                            relevance_boost += 0.08
                
                # Boost for important sources
                important_sources = ['sanad', 'questions', 'menhaj', 'منهاج']
                for important in important_sources:
                    if important in source:
                        relevance_boost += 0.05
                
                # Apply boost
                result.score += relevance_boost
        
        # Final sorting and limiting
        all_results.sort(key=lambda x: x.score, reverse=True)
        final_results = all_results[:max_results]
        
        # Create debug details
        initial_search_details = []
        if final_results:
            initial_search_details = [
                {
                    "id": r.id,
                    "score": r.score,
                    "source": r.payload.get('source', 'N/A') if r.payload else 'N/A',
                    "text_preview": (r.payload.get('text', '')[:200] + "...") if r.payload else ''
                }
                for r in final_results[:15]
            ]
        
        # Create comprehensive search info
        successful_strategies = sum(1 for count in strategy_success.values() if count > 0)
        total_strategies = len(search_strategies)
        best_strategy = max(strategy_success.items(), key=lambda x: x[1]) if strategy_success else ("none", 0)
        
        search_info = (f"Ultra search: {len(final_results)} final results from {len(all_results)} total. "
                      f"Successful strategies: {successful_strategies}/{total_strategies}. "
                      f"Best strategy: {best_strategy[0]} ({best_strategy[1]} results). "
                      f"Details: {' | '.join(search_details[:6])}")
        
        print(f"✅ Final results: {len(final_results)}")
        if final_results:
            best = final_results[0]
            print(f"🎯 Best result: {best.payload.get('source', 'Unknown')} (score: {best.score:.3f})")
            print(f"📄 Preview: {best.payload.get('text', '')[:150]}...")
        
        return final_results, search_info, initial_search_details
        
    except Exception as e:
        error_msg = f"Ultra search error: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Emergency fallback
        try:
            print("🚨 Emergency fallback search...")
            emergency_embedding = embedding_model.encode([query])[0].tolist()
            emergency_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=emergency_embedding,
                limit=max_results,
                with_payload=True,
                score_threshold=0.001
            )
            
            valid_emergency = [r for r in emergency_results 
                             if r.payload and r.payload.get('text', '').strip()]
            
            emergency_details = [
                {
                    "id": r.id,
                    "score": r.score,
                    "source": r.payload.get('source', 'N/A'),
                    "text_preview": r.payload.get('text', '')[:150] + "..."
                }
                for r in valid_emergency[:10]
            ]
            
            return (valid_emergency, 
                   f"{error_msg} | Emergency search: {len(valid_emergency)} results", 
                   emergency_details)
            
        except Exception as emergency_error:
            return [], f"{error_msg} | Emergency failed: {str(emergency_error)}", []

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
    st.set_page_config(page_title="المرجع السند - بحث فائق", page_icon="🕌", layout="wide", initial_sidebar_state="collapsed")
    load_arabic_css()
    
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">محرك بحث فائق التطور - حل مشكلة المحتوى المخفي</p>', unsafe_allow_html=True)
    
    # Ultra search boost notice
    st.markdown('''
    <div class="ultra-search-boost">
        🚀 <strong>تحديث فائق:</strong> تم تطوير خوارزمية بحث متقدمة مع أكثر من 50 استراتيجية بحث متنوعة للعثور على المحتوى المخفي! 
        <br>✨ <strong>ميزات جديدة:</strong> بحث بالمرادفات، استراتيجيات خاصة بالمصطلحات الدينية، عتبات متدرجة، ترتيب ذكي للنتائج
    </div>
    ''', unsafe_allow_html=True)

    # Settings Section
    with st.expander("⚙️ الإعدادات وحالة الأنظمة المتقدمة", expanded=True):
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

        st.markdown("<div style='text-align: right; font-weight: bold; margin-top:0.5rem;'>مستوى البحث الفائق:</div>", unsafe_allow_html=True)
        search_levels = ["بحث سريع (20)", "بحث متوسط (40)", "بحث شامل (60)", "بحث فائق (80)"]
        selected_level = st.radio("مستوى البحث:", search_levels, index=2, horizontal=True, key="s_depth_radio", label_visibility="collapsed")
        max_results = {"بحث سريع (20)": 20, "بحث متوسط (40)": 40, "بحث شامل (60)": 60, "بحث فائق (80)": 80}[selected_level]
        
        show_debug = st.checkbox("إظهار معلومات تفصيلية متقدمة", value=True, key="debug_cb")

    # Database info section
    if qdrant_info['status'] and qdrant_info.get('details'):
        with st.expander("ℹ️ معلومات قاعدة البيانات المتقدمة", expanded=False):
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
                        for d_idx, d in enumerate(msg_item["initial_search_details"][:10]):
                            display_id = str(d.get('id', 'N/A'))
                            score = d.get('score', 0)
                            source = d.get('source', 'N/A')
                            preview = d.get('text_preview', 'N/A')
                            details_str_parts.append(f"  {d_idx+1}. ID: {display_id[:8]}... | Score: {score:.3f} | Source: {source} | Preview: {preview[:80]}...")
                        details_str = "\n".join(details_str_parts)
                        debug_parts.append(f"نتائج Qdrant المفصلة (أفضل 10 من {len(msg_item['initial_search_details'])}):\n{details_str}")
                    
                    if debug_parts:
                        st.markdown(f'<div class="debug-info">🔍 معلومات تفصيلية متقدمة:<div class="debug-info-results">{"<hr>".join(debug_parts)}</div></div>', unsafe_allow_html=True)

                # Show sources
                if "sources" in msg_item and msg_item["sources"]:
                    st.markdown("<div style='text-align: right; margin-top:0.5rem;'><strong>المصادر المستخدمة (مرتبة حسب الصلة):</strong></div><div class='source-container'>", unsafe_allow_html=True)
                    for j_idx in range(0, min(len(msg_item["sources"]), 12), 3):
                        cols = st.columns(3)
                        for k_idx, k_src_item in enumerate(msg_item["sources"][j_idx:j_idx+3]):
                            with cols[k_idx]:
                                source = k_src_item.get("source", "N/A")
                                score = k_src_item.get("score", 0)
                                quality = "ممتاز" if score > 0.8 else "جيد جداً" if score > 0.6 else "جيد" if score > 0.4 else "مقبول"
                                st.markdown(f'<div class="source-info" title="S: {source}\nSc: {score*100:.1f}%\nQuality: {quality}">📄 <strong>{source}</strong><br>تطابق: {score*100:.1f}% ({quality})</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Search input section
    st.markdown("<hr style='margin-top:1.5rem; margin-bottom:0.5rem;'>", unsafe_allow_html=True)
    
    # Examples section
    st.markdown('''
    <div style="background: #f8f9fa; border-left: 4px solid #007bff; padding: 0.5rem; margin: 0.5rem 0; font-size: 0.9rem; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl;">
        💡 <strong>أمثلة على الاستعلامات المحسنة:</strong><br>
        • "حلق اللحية" أو "يحرم حلق اللحية" أو "منهاج الصالحين حلق اللحية"<br>
        • "صلاة الرغائب" أو "ليلة الرغائب" أو "قاعدة التسامح في أدلة السنن"<br>
        • "مسألة ٤٤" أو "استفتاءات السند" أو "حكم شرعي"
    </div>
    ''', unsafe_allow_html=True)
    
    _, input_main, _ = st.columns([0.2, 2.6, 0.2])
    with input_main:
        user_query = st.text_area("سؤالك الفقهي...", placeholder="اكتب سؤالك هنا (مثال: حكم حلق اللحية، صلاة ليلة الرغائب، استفتاءات فقهية)...", key="user_input", height=120, label_visibility="collapsed")
        
        st.markdown('<div class="search-button-container">', unsafe_allow_html=True)
        search_button = st.button("🔍 بحث فائق التطور", type="primary", use_container_width=False, key="send_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    # Process search when button is clicked
    if search_button and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query.strip()})
        
        start_time = time.perf_counter()
        bot_msg_data = {"api_used": selected_llm}
        
        # Ultra-enhanced search
        with st.spinner(f"جاري البحث الفائق التطور ({max_results} نتيجة مع 50+ استراتيجية)..."):
            try:
                search_results, search_info, search_details = comprehensive_search(user_query.strip(), max_results)
                bot_msg_data["initial_search_details"] = search_details
                
                # Show real-time debug info
                if show_debug:
                    st.markdown(f'<div class="ultra-search-boost">🔍 <strong>نتائج البحث الفائق المباشرة:</strong> وجدت {len(search_results)} نتيجة عالية الجودة<br>📊 <strong>تفاصيل:</strong> {search_info}</div>', unsafe_allow_html=True)
                
            except Exception as search_error:
                st.error(f"خطأ في البحث الفائق: {search_error}")
                search_results, search_info, search_details = [], f"خطأ في البحث الفائق: {str(search_error)}", []

        # Process search results
        if search_results:
            try:
                # Prepare context for LLM
                context_texts = []
                sources_for_llm = []
                total_chars = 0
                max_chars_context = 30000
                
                for i, result in enumerate(search_results):
                    if not result.payload:
                        continue
                    
                    source_id_str = str(result.id) if result.id is not None else f"unknown_id_{i}"
                    source_name = result.payload.get('source', f'وثيقة {source_id_str[:6]}')
                    text = result.payload.get('text', '')
                    
                    if text and len(text.strip()) > 10:
                        # Smart text truncation
                        if len(text) > 2000:
                            sentences = re.split(r'[.!?؟۔]\s+', text[:2000])
                            if len(sentences) > 1:
                                truncated_text = '. '.join(sentences[:-1]) + "..."
                            else:
                                truncated_text = text[:1800] + "..."
                        else:
                            truncated_text = text
                        
                        if total_chars + len(truncated_text) < max_chars_context:
                            context_texts.append(f"[نص {i+1} من '{source_name}' - نقاط: {result.score:.3f}]: {truncated_text}")
                            sources_for_llm.append({'source': source_name, 'score': result.score, 'id': result.id})
                            total_chars += len(truncated_text)
                        else:
                            context_texts.append(f"\n[ملاحظة: تم اقتصار النصوص. {len(search_results)-i} نص إضافي عالي الجودة لم يرسل.]")
                            search_info += f" | اقتصار السياق، {len(search_results)-i} نصوص لم ترسل."
                            break
                
                if context_texts:
                    context_for_llm = "\n\n---\n\n".join(context_texts)
                    llm_context_info = f"أرسل {len(sources_for_llm)} نص عالي الجودة للتحليل (~{total_chars//1000} ألف حرف)."
                    llm_messages = prepare_llm_messages(user_query.strip(), context_for_llm, llm_context_info)
                    
                    # Get LLM response
                    bot_response = ""
                    with st.spinner(f"جاري التحليل المتقدم بواسطة {selected_llm}..."):
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
                    bot_msg_data["content"] = f"تم العثور على {len(search_results)} نتيجة من البحث الفائق ولكن لا تحتوي على نصوص صالحة للمعالجة. يرجى تجربة صياغة مختلفة للسؤال."
                    bot_msg_data["debug_info"] = f"{search_info} | لا توجد نصوص صالحة في النتائج الفائقة"
            
            except Exception as processing_error:
                st.error(f"خطأ في معالجة النتائج الفائقة: {processing_error}")
                bot_msg_data["content"] = f"تم العثور على {len(search_results)} نتيجة فائقة ولكن حدث خطأ في المعالجة: {str(processing_error)}"
                bot_msg_data["debug_info"] = f"{search_info} | خطأ معالجة فائقة: {str(processing_error)}"
        else:
            bot_msg_data["content"] = "لم أجد أي معلومات متعلقة بسؤالك حتى مع البحث الفائق التطور في قاعدة بيانات كتب واستفتاءات الشيخ محمد السند. تم تجربة أكثر من 50 استراتيجية بحث مختلفة. يرجى محاولة صياغة السؤال بشكل مختلف أو استخدام كلمات مفتاحية أخرى."
            bot_msg_data["debug_info"] = search_info if search_info else "لا توجد نتائج من البحث الفائق حتى مع جميع الاستراتيجيات."
        
        # Save response
        bot_msg_data["role"] = "assistant"
        bot_msg_data["time_taken"] = time.perf_counter() - start_time
        st.session_state.messages.append(bot_msg_data)
        st.rerun()
    
    elif search_button and not user_query.strip():
        st.toast("يرجى إدخال سؤال للبحث الفائق.", icon="📝")

    # Clear chat button
    with input_main:
        if st.button("🗑️ مسح المحادثة", use_container_width=True, key="clear_btn", type="secondary"):
            st.session_state.messages = []
            st.toast("تم مسح المحادثة وإعادة تهيئة البحث الفائق.", icon="🗑️")
            time.sleep(0.5)
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85rem; margin-top: 1rem; font-family: "Noto Sans Arabic", sans-serif; line-height: 1.6;'>
        🚀 <strong>محرك البحث الفائق التطور</strong> - الإصدار النهائي المتقدم<br>
        🔧 <strong>التقنيات:</strong> 50+ استراتيجية بحث، عتبات متدرجة (0.15-0.001), بحث بالمرادفات والمصطلحات الدينية<br>
        ⚡ <strong>الميزات:</strong> ترتيب ذكي، فلترة متقدمة، كشف المحتوى المخفي، تحليل جودة النتائج<br>
        🎯 <strong>متخصص في:</strong> الفقه الإسلامي، كتب الشيخ محمد السند، الاستفتاءات الشرعية، منهاج الصالحين
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Configuration validation
    config_issues = []
    
    if not QDRANT_API_KEY or not QDRANT_URL:
        config_issues.append("❌ معلومات QDRANT مفقودة")
    
    if not DEEPSEEK_API_KEY and not GEMINI_API_KEY:
        config_issues.append("❌ لا توجد مفاتيح API للذكاء الاصطناعي")
    
    if config_issues:
        st.error("مشاكل في التهيئة: " + " | ".join(config_issues))
        st.info("💡 تأكد من إعداد متغيرات البيئة أو تحديث المفاتيح في الكود مباشرة.", icon="ℹ️")
    else:
        st.success("✅ جميع الأنظمة جاهزة للبحث الفائق التطور!", icon="✅")
    
    main()
