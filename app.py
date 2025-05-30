import streamlit as st
import os
import time
import unicodedata
import re
import requests
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ----------------------
# Configuration
# ----------------------
QDRANT_URL = os.getenv("QDRANT_URL", "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ")
COLLECTION_NAME = "arabic_documents_enhanced" 

# --- API Key Management ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-14f267781a6f474a9d0ec8240383dae4")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyASlapu6AYYwOQAJJ3v-2FSKHnxIOZoPbY")

# --- Initialize API Clients ---
gemini_initial_configured = False
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_PLACEHOLDER": 
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_initial_configured = True
        print("Gemini API configured successfully at startup.")
    except Exception as e:
        print(f"Failed to configure Gemini API at startup: {e}")
else:
    print("Gemini API key is missing or a placeholder. Gemini features will be limited.")

# ----------------------
# Enhanced Arabic Text Processing
# ----------------------
def normalize_arabic_text_enhanced(text):
    """Enhanced Arabic text normalization for better search"""
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

def extract_arabic_keywords_enhanced(text):
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
        normalized_word = normalize_arabic_text_enhanced(word)
        if len(normalized_word) > 2 and normalized_word not in arabic_stopwords:
            keywords.append(normalized_word)
    
    return list(set(keywords))  # Remove duplicates

def create_query_variants(query):
    """Create multiple variants of the query for better matching"""
    if not query:
        return [query]
    
    variants = []
    
    # 1. Original query
    variants.append(('original', query))
    
    # 2. Normalized query
    normalized_query = normalize_arabic_text_enhanced(query)
    if normalized_query != query and normalized_query:
        variants.append(('normalized', normalized_query))
    
    # 3. Keywords only
    keywords = extract_arabic_keywords_enhanced(query)
    if keywords:
        keywords_query = ' '.join(keywords)
        variants.append(('keywords', keywords_query))
    
    # 4. Individual important words (for fallback)
    important_words = []
    for word in query.split():
        if len(word) > 3:
            # Also try normalized version of the word
            normalized_word = normalize_arabic_text_enhanced(word)
            important_words.append(word)
            if normalized_word != word:
                important_words.append(normalized_word)
    
    # Add top 2 important words as separate queries
    for word in important_words[:2]:
        variants.append(('word', word))
    
    return variants

# ----------------------
# Arabic CSS Styling (Same as before)
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
    .stExpander .stRadio > label > div > p, .stExpander .stCheckbox > label > span { font-family: 'Noto Sans Arabic', sans-serif !important; } 
    .stRadio div[data-testid="stCaptionContainer"] { font-family: 'Noto Sans Arabic', sans-serif !important; margin-top: -2px; margin-right: 25px; direction: rtl; }
    .stTextArea > div > div > textarea { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; min-height: 100px !important; border-radius: 10px; border: 1px solid #ccc; }
    .search-button-container { text-align: center; margin-top: 1rem; margin-bottom: 1rem; }
    div[data-testid="stButton"] > button { margin: 0 auto; display: block; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 600; border-radius: 8px; transition: background-color 0.2s ease, transform 0.2s ease; }
    div[data-testid="stButton"] > button:hover { transform: translateY(-2px); }
    .status-box { background: #f0f2f6; padding: 0.5rem; border-radius: 8px; text-align: center; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; border: 1px solid #e0e0e0; font-size: 0.9rem; margin-bottom: 0.5rem; }
    .status-active { color: #28a745; font-weight: bold; } 
    .status-inactive { color: #dc3545; font-weight: bold; } 
    .radio-label-status-active { color: #28a745 !important; font-weight: normal !important; font-size:0.9em !important; }
    .radio-label-status-inactive { color: #dc3545 !important; font-weight: normal !important; font-size:0.9em !important; }
    .collection-info-box { direction: rtl; padding: 1rem; background-color: #e9ecef; border-radius: 10px; margin-top:1rem; margin-bottom: 1.5rem; border: 1px solid #ced4da; }
    .collection-info-box h3 { font-family: 'Noto Sans Arabic', sans-serif; text-align:right; color: #495057; }
    .collection-info-box p {font-family: 'Noto Sans Arabic', sans-serif; text-align:right; margin-bottom: 0.3rem;}
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
    div[data-testid="stSpinner"] { direction: rtl; width: 100%;}
    div[data-testid="stSpinner"] > div { display: flex; flex-direction: row-reverse; align-items: center; justify-content: flex-start; text-align: right !important; width: auto; margin: 0 auto 0 0; }
    div[data-testid="stSpinner"] p { font-size: 1.15em !important; font-family: 'Noto Sans Arabic', sans-serif !important; margin-right: 8px !important; text-align: right !important;}
    .search-debug { background: #e8f5e8; border: 1px solid #4caf50; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; font-family: 'Noto Sans Arabic', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# Status Check & Info Functions (Same as before)
# ----------------------
@st.cache_data(ttl=300)
def get_qdrant_info():
    if not QDRANT_API_KEY or QDRANT_API_KEY == "YOUR_QDRANT_API_KEY_PLACEHOLDER" or \
       not QDRANT_URL or QDRANT_URL == "YOUR_QDRANT_URL_PLACEHOLDER": 
        return {"status": False, "message": "بيانات Qdrant مفقودة أو هي قيم نائبة.", "details": {}}
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5) 
        collection_info = client.get_collection(COLLECTION_NAME)
        return {"status": True, "message": f"متصل ✓ | النقاط: {collection_info.points_count:,}",
                "details": {"اسم المجموعة": COLLECTION_NAME, "عدد النقاط": collection_info.points_count, 
                            "حالة الفهرسة": str(collection_info.status), "تهيئة Vector": str(collection_info.config.params),
                            "تهيئة Quantization": str(collection_info.config.quantization_config) if collection_info.config.quantization_config else "لا"}}
    except Exception as e:
        print(f"Qdrant status error: {e}")
        if "Not found: Collection" in str(e) or "NOT_FOUND" in str(e).upper():
             return {"status": False, "message": f"غير متصل (المجموعة '{COLLECTION_NAME}' غير موجودة)", "details": {}}
        return {"status": False, "message": f"غير متصل (خطأ: {type(e).__name__})", "details": {}}

@st.cache_data(ttl=300)
def check_api_status(api_name):
    global gemini_initial_configured 
    if api_name == "DeepSeek":
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_PLACEHOLDER": 
            return False, "المفتاح مفقود أو نائب"
        try:
            h = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            d = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "مرحبا"}], "max_tokens": 1, "stream": False}
            r = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=h, json=d, timeout=7)
            return (True, "نشط ✓") if r.status_code == 200 else (False, f"خطأ ({r.status_code})") 
        except Exception as e: print(f"DeepSeek status error: {e}"); return False, f"غير نشط ({type(e).__name__})"
    elif api_name == "Gemini":
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_PLACEHOLDER": 
            return False, "المفتاح مفقود أو نائب"
        try:
            if not gemini_initial_configured: genai.configure(api_key=GEMINI_API_KEY); gemini_initial_configured = True
            m = genai.GenerativeModel('gemini-1.5-flash') 
            m.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1,max_output_tokens=1))
            return True, "نشط ✓"
        except Exception as e:
            print(f"Gemini status error: {e}")
            err = str(e).lower()
            if "api_key_invalid" in err or "permission" in err or "quota" in err or "authentication" in err: return False, "خطأ بالمفتاح" 
            return False, f"غير نشط ({type(e).__name__})"
    return False, "API غير معروف"

# ----------------------
# Initialize Components Resource (Same as before)
# ----------------------
@st.cache_resource
def init_qdrant_client_resource():
    if not QDRANT_API_KEY or QDRANT_API_KEY == "YOUR_QDRANT_API_KEY_PLACEHOLDER" or \
       not QDRANT_URL or QDRANT_URL == "YOUR_QDRANT_URL_PLACEHOLDER": 
        st.error("بيانات اتصال Qdrant مفقودة أو هي قيم نائبة."); return None
    try: return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10) 
    except Exception as e: st.error(f"فشل الاتصال بـ Qdrant: {e}"); return None

@st.cache_resource
def init_embedding_model_resource():
    try:
        model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' # 768 dimensions
        print(f"Initializing embedding model: {model_name}")
        return SentenceTransformer(model_name)
    except Exception as e: st.error(f"فشل تحميل نموذج التضمين '{model_name}': {e}"); return None

# ----------------------
# ENHANCED SEARCH FUNCTION - This is the new improved version
# ----------------------
def comprehensive_search(query, max_results=50):
    """
    Enhanced comprehensive search with Arabic text processing and multiple strategies
    """
    
    embedding_model = init_embedding_model_resource()
    if not embedding_model: 
        return [], "فشل تحميل نموذج التضمين.", []
    
    qdrant_c = init_qdrant_client_resource()
    if not qdrant_c: 
        return [], "فشل الاتصال بـ Qdrant.", []
    
    try:
        print(f"Enhanced search for: '{query}'")
        
        # Create multiple query variants for better matching
        query_variants = create_query_variants(query)
        
        # Search with multiple strategies
        all_results = []
        seen_ids = set()
        search_info_parts = []
        
        print(f"Testing {len(query_variants)} query variants...")
        
        # Use different thresholds for different query types
        variant_thresholds = {
            'original': 0.2,      # Your diagnostic showed 0.2 works well
            'normalized': 0.18,   # Slightly lower for normalized
            'keywords': 0.15,     # Lower for keyword-only search
            'word': 0.12          # Lowest for individual words
        }
        
        for variant_type, variant_query in query_variants:
            try:
                # Choose threshold based on variant type
                threshold = variant_thresholds.get(variant_type, 0.15)
                
                # Create embedding for this variant
                query_embedding = embedding_model.encode([variant_query])[0].tolist()
                print(f"Query embedding dim: {len(query_embedding)}")
                
                # Search with this variant
                search_results_qdrant = qdrant_c.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=max_results // 2,  # Get more results per variant
                    with_payload=True,
                    score_threshold=threshold
                )
                
                # Add new results (avoid duplicates)
                new_results_count = 0
                for result in search_results_qdrant:
                    if result.id not in seen_ids:
                        seen_ids.add(result.id)
                        all_results.append(result)
                        new_results_count += 1
                
                if new_results_count > 0:
                    search_info_parts.append(f"{variant_type} ({threshold}): {new_results_count} نتيجة")
                    print(f"✅ {variant_type} found {new_results_count} new results")
                else:
                    search_info_parts.append(f"{variant_type}: 0 نتائج")
                
                # If we have enough good results, don't need all variants
                if len(all_results) >= 25:
                    break
                    
            except Exception as e:
                print(f"❌ Error with variant {variant_type}: {e}")
                search_info_parts.append(f"{variant_type}: خطأ")
                continue
        
        # Sort all results by score (highest first)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to requested number of results
        final_results = all_results[:max_results]
        
        # Create detailed search info for debugging
        initial_search_details = []
        if final_results:
            initial_search_details = [
                {
                    "id": sr.id,
                    "score": sr.score,
                    "source": sr.payload.get('source', 'N/A') if sr.payload else 'N/A',
                    "text_preview": (sr.payload.get('text', '')[:100] + "...") if sr.payload else ''
                } 
                for sr in final_results[:15]  # Show top 15 in debug
            ]
        
        # Create comprehensive search info
        total_variants_tried = len([p for p in search_info_parts if 'خطأ' not in p])
        search_info = f"بحث محسن: {len(final_results)} نتيجة من {total_variants_tried} متغير. " + " | ".join(search_info_parts)
        
        print(f"✅ Final results: {len(final_results)}")
        
        # Additional fallback if still no results
        if len(final_results) == 0:
            print("🔄 Trying emergency fallback with very low threshold...")
            try:
                emergency_embedding = embedding_model.encode([query])[0].tolist()
                emergency_results = qdrant_c.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=emergency_embedding,
                    limit=max_results,
                    with_payload=True,
                    score_threshold=0.05  # Very low threshold
                )
                
                if emergency_results:
                    emergency_details = [
                        {
                            "id": sr.id,
                            "score": sr.score,
                            "source": sr.payload.get('source', 'N/A') if sr.payload else 'N/A',
                            "text_preview": (sr.payload.get('text', '')[:100] + "...") if sr.payload else ''
                        } 
                        for sr in emergency_results
                    ]
                    
                    return emergency_results, f"{search_info} | بحث طوارئ: {len(emergency_results)} نتيجة", emergency_details
            except Exception as emergency_error:
                print(f"❌ Emergency fallback failed: {emergency_error}")
        
        return final_results, search_info, initial_search_details
        
    except Exception as e:
        print(f"Enhanced search error: {e}")
        error_msg = f"خطأ بحث محسن: {str(e)[:100]}"
        
        # Final fallback to basic search
        try:
            print("🔄 Trying basic fallback search...")
            simple_embedding = embedding_model.encode([query])[0].tolist()
            fallback_results = qdrant_c.search(
                collection_name=COLLECTION_NAME,
                query_vector=simple_embedding,
                limit=max_results,
                with_payload=True,
                score_threshold=0.1  # Lower threshold for fallback
            )
            
            fallback_details = [
                {
                    "id": sr.id,
                    "score": sr.score,
                    "source": sr.payload.get('source', 'N/A') if sr.payload else 'N/A',
                    "text_preview": (sr.payload.get('text', '')[:100] + "...") if sr.payload else ''
                } 
                for sr in fallback_results
            ]
            
            print(f"✅ Basic fallback found: {len(fallback_results)} results")
            return fallback_results, f"{error_msg} | بحث أساسي: {len(fallback_results)} نتيجة", fallback_details
            
        except Exception as fallback_error:
            print(f"❌ All search methods failed: {fallback_error}")
            return [], f"{error_msg} | فشل كل طرق البحث: {str(fallback_error)[:50]}", []

# ----------------------
# API Response Functions (Same as before but with better error handling)
# ----------------------
def prepare_llm_messages(user_question, context, context_info):
    system_prompt = "أنت مساعد للبحث في كتب واستفتاءات الشيخ محمد السند فقط.\nقواعد حتمية لا يمكن تجاوزها:\n1. أجب فقط من النصوص المعطاة أدناه (\"المصادر المتاحة\") - لا استثناءات.\n2. إذا لم تجد الإجابة الكاملة في النصوص، قل بوضوح: \"لم أجد إجابة كافية في المصادر المتاحة بخصوص هذا السؤال.\"\n3. ممنوع منعاً باتاً إضافة أي معلومة من خارج النصوص المعطاة. لا تستخدم معلوماتك العامة أو معرفتك السابقة.\n4. اقتبس مباشرة من النصوص عند الإجابة قدر الإمكان، مع الإشارة إلى المصدر إذا كان متاحاً في النص (مثال: [نص ١]).\n5. إذا وجدت إجابة جزئية، اذكرها وأوضح أنها غير كاملة أو تغطي جانباً من السؤال.\n6. هدفك هو تقديم إجابة دقيقة وموثوقة بناءً على ما هو متوفر في النصوص فقط.\n7. إذا كانت النصوص لا تحتوي على إجابة، لا تحاول استنتاج أو تخمين الإجابة.\nتذكر: أي معلومة ليست في النصوص أدناه = لا تذكرها أبداً. كن دقيقاً ومقتصراً على المصادر."
    user_content = (f"السؤال المطروح: {user_question}\n\nالمصادر المتاحة من قاعدة البيانات فقط (أجب بناءً عليها حصراً):\n{context}\n\nمعلومات إضافية عن السياق: {context_info}\n\nالتعليمات: يرجى تقديم إجابة بناءً على النصوص أعلاه فقط. إذا لم تكن الإجابة موجودة، وضح ذلك.")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

def get_deepseek_response(messages, max_tokens=2000):
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_PLACEHOLDER": 
        return "DeepSeek API key مفقود أو نائب."
    try:
        h = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        d = {"model": "deepseek-chat", "messages": messages, "temperature": 0.05, "max_tokens": max_tokens, "stream": False}
        r = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=h, json=d, timeout=90)
        r.raise_for_status()
        res = r.json()
        return res['choices'][0]['message']['content'] if res.get('choices') and res['choices'][0].get('message') else "لم يتمكن DeepSeek من الرد."
    except requests.exceptions.Timeout: return "انتهت مهلة DeepSeek."
    except requests.exceptions.HTTPError as e:
        err_c = e.response.text if e.response else "No response"
        print(f"DeepSeek HTTP error: {e.response.status_code if e.response else 'N/A'} - {err_c}")
        return f"خطأ DeepSeek: {e.response.status_code if e.response else 'N/A'}. تفاصيل: {err_c[:200]}"
    except Exception as e: print(f"DeepSeek API error: {e}"); return f"خطأ DeepSeek: {str(e)}"

def get_gemini_response(messages, max_tokens=2000):
    global gemini_initial_configured
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_PLACEHOLDER": 
        return "Gemini API key مفقود أو نائب."
    try:
        if not gemini_initial_configured: genai.configure(api_key=GEMINI_API_KEY); gemini_initial_configured = True
        model = genai.GenerativeModel('gemini-1.5-flash')
        proc_msgs, sys_prompt_txt = [], None
        if messages and messages[0]["role"] == "system":
            sys_prompt_txt = messages[0]["content"]
            for msg in messages[1:]: proc_msgs.append({"role": "user" if msg["role"]=="user" else "model", "parts": [msg["content"]]})
        else:
            for msg in messages: proc_msgs.append({"role": "user" if msg["role"]=="user" else "model", "parts": [msg["content"]]})
        if sys_prompt_txt:
             if proc_msgs and proc_msgs[0]["role"] == "user": proc_msgs[0]["parts"][0] = f"{sys_prompt_txt}\n\n---\n\n{proc_msgs[0]['parts'][0]}"
             else: proc_msgs.insert(0, {"role": "user", "parts": [sys_prompt_txt]})
        if not proc_msgs: return "لا توجد رسائل صالحة لـ Gemini."
        if len(proc_msgs) > 1 and proc_msgs[-1]["role"] == "user":
            hist, curr_msg = proc_msgs[:-1], proc_msgs[-1]["parts"][0]
            chat = model.start_chat(history=hist)
            resp = chat.send_message(curr_msg, generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens))
        elif proc_msgs and proc_msgs[0]["role"] == "user":
            resp = model.generate_content(proc_msgs[0]["parts"], generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens))
        else: return "بنية رسائل Gemini غير متوقعة."
        return resp.text
    except Exception as e: print(f"Gemini API error: {e}"); return f"خطأ Gemini: {str(e)}"

# ----------------------
# Enhanced Debug Function
# ----------------------
def show_search_debug(search_res, db_dbg_info, initial_details, user_q, show_debug=True):
    """Enhanced debugging display for search results"""
    if not show_debug:
        return
    
    # Enhanced debug box
    debug_content = f"""
🔍 **نتائج البحث المحسن:**
- الاستعلام: "{user_q}"
- النتائج الموجودة: {len(search_res)}
- أفضل نتيجة: {max([r.score for r in search_res]):.3f} إذا وجدت
- معلومات البحث: {db_dbg_info}
    """
    
    if search_res:
        debug_content += f"\n\n**أفضل 3 نتائج:**\n"
        for i, result in enumerate(search_res[:3]):
            source = result.payload.get('source', 'Unknown') if result.payload else 'Unknown'
            score = result.score
            preview = result.payload.get('text', '')[:150] + '...' if result.payload else 'No text'
            debug_content += f"{i+1}. {source} (Score: {score:.3f})\n   {preview}\n\n"
    else:
        debug_content += "\n❌ **لم يتم العثور على نتائج**"
    
    st.markdown(f'<div class="search-debug">{debug_content}</div>', unsafe_allow_html=True)

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="المرجع السند - بحث", page_icon="🕌", layout="wide", initial_sidebar_state="collapsed")
    load_arabic_css()
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند</h1><p class="sub-header">محرك بحث الكتب والاستفتاءات المحسن</p>', unsafe_allow_html=True)

    with st.expander("⚙️ الإعدادات وحالة الأنظمة", expanded=True):
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>اختر محرك الذكاء الاصطناعي:</div>", unsafe_allow_html=True)
        llm_opts, llm_caps, active_llms_idx = [], [], []
        llm_apis = [("DeepSeek", "DeepSeek"), ("Gemini", "Gemini")]
        def_idx = 0
        for i, (disp, internal) in enumerate(llm_apis):
            ok, msg = check_api_status(internal)
            llm_opts.append(disp); llm_caps.append(f"<span class='{'radio-label-status-active' if ok else 'radio-label-status-inactive'}'>({msg})</span>")
            if ok: active_llms_idx.append(i)
        if active_llms_idx and def_idx not in active_llms_idx: def_idx = active_llms_idx[0]
        elif not active_llms_idx: st.warning("تنبيه: جميع محركات الذكاء الاصطناعي غير نشطة.", icon="⚠️")
        sel_llm = st.radio("محركات AI:", llm_opts, captions=llm_caps, index=def_idx, horizontal=True, key="llm_sel", label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>##### حالة قاعدة البيانات (Qdrant):</div>", unsafe_allow_html=True)
        q_info_data = get_qdrant_info()
        q_stat_txt = f'<span class="{"status-active" if q_info_data["status"] else "status-inactive"}">{q_info_data["message"]}</span>'
        st.markdown(f'<div style="display: flex; justify-content: center;"><div class="status-box" style="width: 90%; max-width: 450px;">Qdrant DB: {q_stat_txt}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='text-align: right; font-weight: bold; margin-top:0.5rem;'>مستوى البحث:</div>", unsafe_allow_html=True)
        s_depth_opts = ["بحث سريع (15)", "بحث متوسط (30)", "بحث شامل (50)"]
        s_depth = st.radio("مستوى البحث:", s_depth_opts, index=1, horizontal=True, key="s_depth_radio", label_visibility="collapsed")
        max_db_res = {"بحث سريع (15)": 15, "بحث متوسط (30)": 30, "بحث شامل (50)": 50}[s_depth]
        show_dbg = st.checkbox("إظهار معلومات تفصيلية", value=True, key="debug_cb")

    if q_info_data['status'] and q_info_data['details']:
        with st.expander("ℹ️ معلومات قاعدة البيانات", expanded=False):
            st.markdown(f"<div class='collection-info-box'><h3>مجموعة: {q_info_data['details'].get('اسم المجموعة', COLLECTION_NAME)}</h3>" +
                        "".join([f"<p><strong>{k}:</strong> {v}</p>" for k,v in q_info_data['details'].items() if k!="اسم المجموعة"]) +
                        "</div>", unsafe_allow_html=True)
    elif not q_info_data['status']: st.warning(f"Qdrant: {q_info_data['message']}.", icon="⚠️")

    if "messages" not in st.session_state: st.session_state.messages = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg_item in st.session_state.messages:
            is_user = msg_item["role"] == "user"
            st.markdown(f'<div class="{"user-message" if is_user else "bot-message"}">{"👤" if is_user else "🤖"} {msg_item["content"]}</div>', unsafe_allow_html=True)
            if not is_user:
                if "api_used" in msg_item: st.markdown(f'<span class="api-used">استخدم: {msg_item["api_used"]}</span>', unsafe_allow_html=True)
                if "time_taken" in msg_item: st.markdown(f'<div class="time-taken">⏱️ زمن: {msg_item["time_taken"]:.2f} ث</div>', unsafe_allow_html=True)
                
                full_debug_info_parts = []
                if "debug_info" in msg_item: full_debug_info_parts.append(msg_item["debug_info"])
                
                if show_dbg and "initial_search_details" in msg_item and msg_item["initial_search_details"]:
                    details_str_parts = []
                    for d_idx, d in enumerate(msg_item["initial_search_details"]):
                        display_id = str(d.get('id', 'N/A')) 
                        details_str_parts.append(f"  {d_idx+1}. ID: {display_id[:8]}... | Score: {d.get('score', 0):.3f} | Source: {d.get('source', 'N/A')} | Preview: {d.get('text_preview', 'N/A')}")
                    details_str = "\n".join(details_str_parts)
                    full_debug_info_parts.append(f"نتائج Qdrant المفصلة ({len(msg_item['initial_search_details'])}):\n{details_str}")
                
                if show_dbg and full_debug_info_parts: 
                    st.markdown(f'<div class="debug-info">🔍 معلومات تفصيلية:<div class="debug-info-results">{"<hr>".join(full_debug_info_parts)}</div></div>', unsafe_allow_html=True)

                if "sources" in msg_item and msg_item["sources"]:
                    st.markdown("<div style='text-align: right; margin-top:0.5rem;'><strong>المصادر المستخدمة:</strong></div><div class='source-container'>", unsafe_allow_html=True)
                    for j_idx in range(0, min(len(msg_item["sources"]), 9), 3):
                        cols = st.columns(3)
                        for k_idx, k_src_item in enumerate(msg_item["sources"][j_idx:j_idx+3]):
                            with cols[k_idx]: st.markdown(f'<div class="source-info" title="S: {k_src_item.get("source", "N/A")}\nSc: {k_src_item.get("score",0)*100:.1f}%">📄 <strong>{k_src_item.get("source","N/A")}</strong><br>تطابق: {k_src_item.get("score",0)*100:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Search input section
    st.markdown("<hr style='margin-top:1.5rem; margin-bottom:0.5rem;'>", unsafe_allow_html=True)
    _, input_main, _ = st.columns([0.2, 2.6, 0.2])
    with input_main:
        user_q = st.text_area("سؤالك...", placeholder="اكتب سؤالك هنا (مثال: صلاة ليلة الرغائب، قاعدة التسامح في أدلة السنن)...", key="user_input", height=120, label_visibility="collapsed")
        st.markdown('<div class="search-button-container">', unsafe_allow_html=True)
        send_btn = st.button("🔍 بحث وإجابة محسن", type="primary", use_container_width=False, key="send_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    # Process search when button is clicked
    if send_btn and user_q.strip():
        st.session_state.messages.append({"role": "user", "content": user_q.strip()})
        s_time = time.perf_counter()
        bot_msg_data = {"api_used": sel_llm}
        
        # Enhanced search with better error handling
        with st.spinner(f"جاري البحث المحسن ({max_db_res} نتيجة)..."):
            try:
                search_res, db_dbg_info, initial_details = comprehensive_search(user_q.strip(), max_results=max_db_res)
                bot_msg_data["initial_search_details"] = initial_details
                
                # Show debug information if enabled
                if show_dbg:
                    show_search_debug(search_res, db_dbg_info, initial_details, user_q.strip(), show_debug=True)
                
            except Exception as search_error:
                st.error(f"خطأ في البحث: {search_error}")
                search_res, db_dbg_info, initial_details = [], f"خطأ في البحث: {str(search_error)}", []

        # Process search results
        if search_res:
            try:
                ctx_texts, srcs_for_llm = [], []
                total_ch, max_ch_ctx = 0, 25000
                
                for i, res_item in enumerate(search_res):
                    # Enhanced error handling for result processing
                    try:
                        src_id_str = str(res_item.id) if res_item.id is not None else f"unknown_id_{i}"
                        src_name = res_item.payload.get('source', f'وثيقة {src_id_str[:6]}') if res_item.payload else f'وثيقة {src_id_str[:6]}'
                        txt = res_item.payload.get('text', '') if res_item.payload else ''
                        
                        if txt and len(txt.strip()) > 0:  # Ensure text is not empty
                            trunc_txt = txt[:1500] + ("..." if len(txt) > 1500 else "")
                            if total_ch + len(trunc_txt) < max_ch_ctx:
                                ctx_texts.append(f"[نص {i+1} من '{src_name}']: {trunc_txt}")
                                srcs_for_llm.append({'source': src_name, 'score': res_item.score, 'id': res_item.id})
                                total_ch += len(trunc_txt)
                            else:
                                ctx_texts.append(f"\n[ملاحظة: تم اقتصار النصوص. {len(search_res)-i} نص إضافي لم يرسل.]")
                                db_dbg_info += f" | اقتصار السياق، {len(search_res)-i} نصوص لم ترسل."
                                break
                    except Exception as result_error:
                        print(f"Error processing result {i}: {result_error}")
                        continue
                
                if ctx_texts:  # Only proceed if we have valid context
                    ctx_for_llm = "\n\n---\n\n".join(ctx_texts)
                    llm_ctx_info = f"أرسل {len(srcs_for_llm)} نص للتحليل (~{total_ch//1000} ألف حرف)."
                    llm_msgs = prepare_llm_messages(user_q.strip(), ctx_for_llm, llm_ctx_info)
                    
                    bot_response = ""
                    with st.spinner(f"جاري التحليل بواسطة {sel_llm}..."):
                        try:
                            if sel_llm == "DeepSeek": 
                                bot_response = get_deepseek_response(llm_msgs)
                            elif sel_llm == "Gemini": 
                                bot_response = get_gemini_response(llm_msgs)
                            else: 
                                bot_response = "المحرك المحدد غير معروف."
                        except Exception as llm_error:
                            bot_response = f"خطأ في الحصول على الرد من {sel_llm}: {str(llm_error)}"
                    
                    bot_msg_data["content"] = bot_response
                    bot_msg_data["sources"] = srcs_for_llm
                    bot_msg_data["debug_info"] = f"{db_dbg_info} | {llm_ctx_info}" if db_dbg_info else llm_ctx_info
                else:
                    bot_msg_data["content"] = f"تم العثور على {len(search_res)} نتيجة ولكن لا يحتوون على نصوص صالحة للمعالجة. يرجى تجربة استعلام مختلف."
                    bot_msg_data["debug_info"] = f"{db_dbg_info} | لا توجد نصوص صالحة في النتائج"
            
            except Exception as processing_error:
                st.error(f"خطأ في معالجة النتائج: {processing_error}")
                bot_msg_data["content"] = f"تم العثور على {len(search_res)} نتيجة ولكن حدث خطأ في المعالجة: {str(processing_error)}"
                bot_msg_data["debug_info"] = f"{db_dbg_info} | خطأ معالجة: {str(processing_error)}"
        else:
            bot_msg_data["content"] = "لم أجد أي معلومات متعلقة بسؤالك في قاعدة بيانات كتب واستفتاءات الشيخ محمد السند حالياً. يرجى محاولة صياغة السؤال بشكل مختلف أو استخدام كلمات مفتاحية أخرى."
            bot_msg_data["debug_info"] = db_dbg_info if db_dbg_info else "لا توجد نتائج من البحث المحسن."
        
        bot_msg_data["role"] = "assistant"
        bot_msg_data["time_taken"] = time.perf_counter() - s_time
        st.session_state.messages.append(bot_msg_data)
        st.rerun()
    elif send_btn and not user_q.strip(): 
        st.toast("يرجى إدخال سؤال.", icon="📝")

    # Clear chat button
    with input_main:
        if st.button("🗑️ مسح المحادثة", use_container_width=True, key="clear_btn", type="secondary"):
            st.session_state.messages = []
            st.toast("تم مسح المحادثة.", icon="🗑️")
            time.sleep(0.5)
            st.rerun()

    # Footer with enhancement info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 1rem;'>
        🚀 محرك البحث المحسن | يدعم البحث متعدد المتغيرات والمعالجة المتقدمة للنصوص العربية
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Configuration checks
    if not all([QDRANT_API_KEY and QDRANT_API_KEY != "YOUR_QDRANT_API_KEY_PLACEHOLDER", 
                QDRANT_URL and QDRANT_URL != "YOUR_QDRANT_URL_PLACEHOLDER"]):
        st.error("معلومات QDRANT مفقودة أو هي قيم افتراضية. تحقق من الإعدادات.")
    if not any([DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "YOUR_DEEPSEEK_API_KEY_PLACEHOLDER", 
                GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_PLACEHOLDER"]): 
        st.info("بعض مفاتيح LLM API مفقودة أو هي قيم افتراضية.", icon="ℹ️")
    
    main()
