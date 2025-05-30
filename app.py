import streamlit as st

# ----------------------
# MUST BE FIRST - Page Configuration
# ----------------------
st.set_page_config(
    page_title="المرجع السند - بحث فائق",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
# Ensure these match your Qdrant setup and the upload script
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ" # Ensure this is your active key
COLLECTION_NAME = "arabic_documents_enhanced" # Must match the collection used by the corrected upload script

# API Keys - Ensure these are correctly set
DEEPSEEK_API_KEY = "sk-14f267781a6f474a9d0ec8240383dae4" # Replace with your actual key or use Streamlit secrets
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_API_KEY = "AIzaSyASlapu6AYYwOQAJJ3v-2FSKHnxIOZoPbY" # Replace with your actual key or use Streamlit secrets

# Initialize Gemini
gemini_initial_configured = False
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE": # Added a check for placeholder
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_initial_configured = True
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Failed to configure Gemini API: {e}")
else:
    print("Gemini API key not provided or is a placeholder.")

# ----------------------
# Enhanced Arabic Text Processing (Consistent with corrected upload script)
# ----------------------
def normalize_arabic_text(text):
    """Enhanced Arabic text normalization (consistent with corrected upload script)"""
    if not text:
        return text

    # Remove diacritics
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    # Normalize characters
    replacements = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', # Alef variations
        'ى': 'ي',  # Ya variations
        'ة': 'ه',  # Ta marbuta
        'ؤ': 'و', 'ئ': 'ي',  # Hamza
        '\u200c': '',  # Zero-width non-joiner
        '\u200d': '',  # Zero-width joiner
        '\ufeff': '',  # Byte Order Mark (BOM)
        '\u200b': '',  # Zero-width space
        '؟': '?', '؛': ';', '،': ',', # Punctuation
        'ـ': '' # Kashida/Tatweel
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove extra spaces and normalize
    text = ' '.join(text.split())
    return text

def extract_arabic_keywords(text):
    """Extract Arabic keywords using the consistent normalization"""
    if not text:
        return []

    # Arabic stop words (consider expanding or refining this list based on your domain)
    stop_words = {
        'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'أنا', 'نحن',
        'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'اللذان', 'اللتان',
        'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'تحت', 'فوق',
        'و', 'أو', 'أم', 'لكن', 'غير', 'إلا', 'بل', 'ثم', 'كذلك',
        'أن', 'إن', 'كي', 'لكي', 'حتى', 'لولا', 'لو', 'إذا', 'إذ', 'حيث',
        'كان', 'كانت', 'كانوا', 'كن', 'يكون', 'تكون', 'قد', 'لقد', 'سوف',
        'لن', 'لم', 'لما', 'ليس', 'ليست', 'ما', 'ماذا', 'متى', 'أين',
        'كيف', 'لماذا', 'كم', 'أي', 'أية', 'ال', 'كل', 'جميع', 'بعض',
        'ف', 'ب', 'ك', 'ل', 'لا', 'نعم', 'كلا', 'يا', 'هل'
    }

    # Extract Arabic words (consecutive Arabic characters)
    words = re.findall(r'[\u0600-\u06FF\u0750-\u077F]{2,}', text) # At least 2 chars
    keywords = []

    for word in words:
        normalized = normalize_arabic_text(word) # Use consistent normalizer
        if len(normalized) > 2 and normalized not in stop_words: # Check length after normalization
            keywords.append(normalized)

    return list(set(keywords)) # Return unique keywords

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
        if not QDRANT_URL or not QDRANT_API_KEY:
            st.error("QDRANT_URL or QDRANT_API_KEY is not set.")
            return None
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
        # Perform a quick check to ensure the client is operational
        client.get_collections() # This will raise an error if connection fails
        return client
    except Exception as e:
        st.error(f"فشل الاتصال بـ Qdrant: {e}. تأكد من صحة URL ومفتاح API وأن Qdrant يعمل.")
        return None

@st.cache_resource
def init_embedding_model():
    try:
        # Using the same model as the corrected upload script
        return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    except Exception as e:
        st.error(f"فشل تحميل نموذج التضمين: {e}")
        return None

# ----------------------
# ULTRA-ENHANCED SEARCH FUNCTION
# ----------------------
def comprehensive_search(query, max_results=50):
    embedding_model = init_embedding_model()
    if not embedding_model:
        return [], "فشل تحميل نموذج التضمين.", []

    qdrant_client = init_qdrant_client()
    if not qdrant_client:
        return [], "فشل الاتصال بـ Qdrant.", []

    try:
        print(f"🔍 Ultra search for: '{query}'")
        search_strategies = []
        original_query = query.strip()

        # 1. Basic searches with graduated thresholds
        search_strategies.extend([
            ('basic_high', original_query, 0.15),
            ('basic_medium', original_query, 0.08),
            ('basic_low', original_query, 0.04),
            ('basic_emergency', original_query, 0.01)
        ])

        # 2. Normalized search (query is already normalized by this function)
        normalized_query = normalize_arabic_text(original_query) # Consistent normalization
        if normalized_query != original_query and normalized_query:
            search_strategies.extend([
                ('normalized_high', normalized_query, 0.12),
                ('normalized_low', normalized_query, 0.03)
            ])
        elif not normalized_query: # Handle empty query after normalization
             print("⚠️ Query became empty after normalization.")


        # 3. Keywords search
        keywords = extract_arabic_keywords(original_query) # Uses consistent normalization
        if keywords:
            all_keywords_query = ' '.join(keywords)
            search_strategies.extend([
                ('keywords_all', all_keywords_query, 0.06),
                ('keywords_low', all_keywords_query, 0.02)
            ])
            important_keywords = keywords[:3] # Top 3 keywords
            for i, keyword in enumerate(important_keywords):
                if len(keyword) > 2: # Ensure keyword is meaningful
                    search_strategies.append((f'keyword_{i+1}', keyword, 0.015))

        # 4. Religious term expansions
        religious_terms = {
            'صلاة': ['صلاة', 'الصلاة', 'صلوات', 'الصلوات', 'فريضة', 'نافلة'],
            'زكاة': ['زكاة', 'الزكاة', 'مزكي', 'صدقة'],
            'حج': ['حج', 'الحج', 'عمرة', 'العمرة', 'مناسك'],
            'صوم': ['صوم', 'الصوم', 'صيام', 'صائم', 'رمضان'],
            'حلق': ['حلق', 'حلاقة', 'إزالة شعر', 'تقصير'],
            'لحية': ['لحية', 'اللحية', 'ذقن', 'الذقن'],
            'رغائب': ['رغائب', 'الرغائب', 'صلاة الرغائب', 'ليلة الرغائب', 'رجب'],
            'جواز': ['يجوز', 'جواز', 'حكم', 'حلال', 'حرام', 'مباح', 'مكروه', 'واجب', 'مندوب'],
            'حكم': ['حكم', 'أحكام', 'يحكم', 'شرعي'],
            'مسألة': ['مسألة', 'مسائل', 'سؤال', 'استفتاء', 'فتوى'],
            'وضوء': ['وضوء', 'الوضوء', 'طهارة', 'غسل'],
            'تقليد': ['تقليد', 'مقلد', 'مرجع', 'مجتهد']
        }
        query_for_terms = normalize_arabic_text(original_query.lower())
        for base_term, expansions in religious_terms.items():
            normalized_base_term = normalize_arabic_text(base_term)
            if normalized_base_term in query_for_terms: # Check if normalized base term is in normalized query
                for expansion in expansions:
                    normalized_expansion = normalize_arabic_text(expansion)
                    search_strategies.append((f'religious_{normalized_expansion}', normalized_expansion, 0.01))

        # 5. Partial phrase searches
        words = original_query.split()
        if len(words) > 1:
            if len(words) >= 2:
                first_two = normalize_arabic_text(' '.join(words[:2]))
                if first_two: search_strategies.append(('partial_start', first_two, 0.02))
                last_two = normalize_arabic_text(' '.join(words[-2:]))
                if last_two: search_strategies.append(('partial_end', last_two, 0.02))

        # 6. Source-specific searches (terms often found in source names or types)
        source_terms = ['منهاج', 'استفتاء', 'سند', 'مسألة رقم', 'كتاب']
        for term in source_terms:
            normalized_term = normalize_arabic_text(term)
            search_strategies.append((f'source_{normalized_term}', normalized_term, 0.005))

        all_results = []
        seen_ids = set()
        search_details_log = [] # For logging
        strategy_success_counts = {}

        print(f"🧪 Testing {len(search_strategies)} search strategies...")

        for strategy_name, strategy_query_text, threshold in search_strategies:
            try:
                if not strategy_query_text or len(strategy_query_text.strip()) < 2:
                    continue

                # Query text is already normalized by strategy creation logic. Embed directly.
                query_embedding = embedding_model.encode([strategy_query_text])[0].tolist()

                strategy_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=max_results * 2, # Fetch more to allow for filtering and deduplication
                    with_payload=True,
                    score_threshold=threshold
                )

                new_results_this_strategy = 0
                for result in strategy_results:
                    # The payload['text'] from Qdrant is the cleaned, normalized text (due to corrected upload script)
                    if (result.id not in seen_ids and
                        result.payload and
                        result.payload.get('text', '').strip() and
                        len(result.payload.get('text', '').strip()) >= 10): # Ensure text is substantial

                        seen_ids.add(result.id)
                        all_results.append(result)
                        new_results_this_strategy += 1
                
                strategy_success_counts[strategy_name] = new_results_this_strategy
                search_details_log.append(f"{strategy_name} ({threshold:.3f}, query: '{strategy_query_text[:30]}...'): {new_results_this_strategy} new")
                
                if new_results_this_strategy > 0:
                    print(f"✅ Strategy '{strategy_name}' (threshold {threshold:.3f}) found {new_results_this_strategy} new results.")
                
                # Consider adjusting early stopping logic if it's too aggressive
                if strategy_name.startswith('basic_') and len(all_results) >= max_results + 10 : # Allow some buffer
                    print(f"ℹ️ Early stopping triggered by '{strategy_name}': {len(all_results)} results found (target {max_results}).")
                    break
                    
            except Exception as e:
                search_details_log.append(f"{strategy_name}: error - {str(e)}")
                print(f"❌ Error in strategy '{strategy_name}': {e}")
                continue
        
        print(f"🔍 Ranking {len(all_results)} unique results...")
        
        # Enhanced result ranking
        normalized_original_query_words = [normalize_arabic_text(word) for word in original_query.split() if len(normalize_arabic_text(word)) > 1]

        for result in all_results:
            if result.payload and result.payload.get('text'):
                # payload_text is already normalized (from corrected upload script)
                payload_text_lower = result.payload.get('text', '').lower() # Lowercase for matching
                source_lower = result.payload.get('source', '').lower()
                
                relevance_boost = 0
                
                # Boost for exact query words in text (normalized comparison)
                for word in normalized_original_query_words:
                    if word in payload_text_lower:
                        relevance_boost += 0.15 
                    if word in source_lower: # Also check source name
                        relevance_boost += 0.08
                
                # Boost for presence of the full normalized query
                if normalized_query and normalized_query in payload_text_lower :
                    relevance_boost += 0.20

                # Boost for important sources (customize these)
                important_sources = ['sanad', 'questions', 'menhaj', 'منهاج', 'استفتاءات']
                for important_src_term in important_sources:
                    if important_src_term in source_lower:
                        relevance_boost += 0.05
                
                result.score += relevance_boost # Apply boost to original score

        all_results.sort(key=lambda x: x.score, reverse=True)
        final_results = all_results[:max_results]
        
        # Prepare debug information
        initial_search_details_for_display = []
        if final_results:
            initial_search_details_for_display = [
                {
                    "id": r.id,
                    "score": r.score,
                    "source": r.payload.get('source', 'N/A') if r.payload else 'N/A',
                    "text_preview": (r.payload.get('text', '')[:200] + "...") if r.payload else ''
                }
                for r in final_results[:15] # Display top 15 for brevity
            ]
        
        successful_strategies_count = sum(1 for count in strategy_success_counts.values() if count > 0)
        total_strategies_attempted = len(search_strategies)
        best_strategy_info = max(strategy_success_counts.items(), key=lambda item: item[1]) if strategy_success_counts and any(strategy_success_counts.values()) else ("none", 0)

        search_info_summary = (f"Ultra search: {len(final_results)} results from {len(all_results)} unique candidates. "
                               f"Strategies: {successful_strategies_count}/{total_strategies_attempted} successful. "
                               f"Best: {best_strategy_info[0]} ({best_strategy_info[1]} hits). "
                               f"Top log: {' | '.join(search_details_log[:5])}")
        
        print(f"✅ Final results count: {len(final_results)}")
        if final_results:
            best_res = final_results[0]
            print(f"🎯 Best result: {best_res.payload.get('source', 'Unknown')} (score: {best_res.score:.3f})")
            print(f"📄 Preview: {best_res.payload.get('text', '')[:150]}...")
        
        return final_results, search_info_summary, initial_search_details_for_display
        
    except Exception as e:
        error_msg = f"Ultra search main error: {str(e)}"
        print(f"❌ {error_msg}")
        # Fallback to a very simple search if the main logic fails catastrophically
        try:
            print("🚨 Emergency fallback search initiated...")
            normalized_fallback_query = normalize_arabic_text(query)
            if not normalized_fallback_query: return [], error_msg + " | Emergency failed: Empty query.", []

            emergency_embedding = embedding_model.encode([normalized_fallback_query])[0].tolist()
            emergency_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=emergency_embedding,
                limit=max_results, # Fetch fewer in emergency
                with_payload=True,
                score_threshold=0.001 # Very low threshold
            )
            
            valid_emergency_results = [r for r in emergency_results 
                                       if r.payload and r.payload.get('text','').strip()]
            
            emergency_details_for_display = [
                { "id": r.id, "score": r.score, "source": r.payload.get('source', 'N/A'),
                  "text_preview": r.payload.get('text', '')[:150] + "..."}
                for r in valid_emergency_results[:10]
            ]
            return (valid_emergency_results,
                   f"{error_msg} | Emergency search returned: {len(valid_emergency_results)} results.",
                   emergency_details_for_display)
            
        except Exception as emergency_error:
            final_error_msg = f"{error_msg} | Emergency fallback also failed: {str(emergency_error)}"
            print(f"❌ {final_error_msg}")
            return [], final_error_msg, []

# ----------------------
# API Response Functions
# ----------------------
def prepare_llm_messages(user_question, context, context_info):
    # This system prompt is crucial for guiding the LLM. Adjust as needed.
    system_prompt = """أنت مساعد متخصص في البحث ضمن كتب واستفتاءات الشيخ محمد السند فقط.
قواعد صارمة يجب الالتزام بها:
1.  أجب **حصراً** بناءً على "المصادر المتاحة" المقدمة لك. لا تضف أي معلومات خارجية أبداً.
2.  إذا لم تجد إجابة واضحة أو كاملة في النصوص، اذكر ذلك بوضوح، مثلاً: "لم أجد إجابة كافية في المصادر المتاحة بخصوص هذا السؤال." أو "النصوص المتوفرة لا تغطي هذه النقطة بشكل مباشر."
3.  لا تستخدم معرفتك العامة أو أي معلومات لم ترد في النصوص المرفقة.
4.  اقتبس مباشرة من النصوص قدر الإمكان عند صياغة الإجابة، مع الإشارة إلى المصدر إذا كان متاحاً في النص (مثل: [نص ١]).
5.  إذا كانت الإجابة جزئية، وضح ذلك.
6.  هدفك هو الدقة والموثوقية بناءً على المصادر المعطاة فقط. تجنب التخمين أو الاستنتاج غير المدعوم بنص.
7.  إذا كانت النصوص متعددة وتغطي جوانب مختلفة، حاول دمج المعلومات بشكل متناسق.
8.  حافظ على اللغة العربية الفصحى في الردود.
تذكر: أي معلومة غير موجودة في "المصادر المتاحة" = لا تذكرها. التزم الدقة والاقتصار على المصادر."""

    user_content = f"""السؤال المطروح من المستخدم: {user_question}

المصادر المتاحة من قاعدة بيانات الشيخ محمد السند (أجب بناءً عليها حصراً):
---
{context}
---

معلومات إضافية عن سياق البحث (لا تجب منها، هي للمعلومية فقط): {context_info}

التعليمات النهائية: يرجى تقديم إجابة دقيقة ومفصلة بناءً على "المصادر المتاحة" أعلاه فقط. إذا لم تكن الإجابة موجودة بشكل كافٍ، وضح ذلك للمستخدم. لا تضف أي معلومات من خارج هذه المصادر."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

def get_deepseek_response(messages, max_tokens=2000):
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE": # Check for placeholder
        return "DeepSeek API key مفقود أو لم يتم تعيينه."
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        # Adjust model if needed e.g. deepseek-coder for code, or specific chat versions
        data = {"model": "deepseek-chat", "messages": messages, "temperature": 0.05, "max_tokens": max_tokens, "stream": False}
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        result = response.json()
        if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
            return result['choices'][0]['message']['content']
        else:
            return f"لم يتمكن DeepSeek من تقديم رد صالح. الاستجابة: {str(result)[:500]}"
    except requests.exceptions.Timeout:
        return "انتهت مهلة الاتصال بـ DeepSeek."
    except requests.exceptions.HTTPError as e:
        err_content = e.response.text if e.response else "No response content"
        return f"خطأ HTTP من DeepSeek: {e.response.status_code if e.response else 'N/A'}. التفاصيل: {err_content[:200]}"
    except Exception as e:
        return f"خطأ غير متوقع عند الاتصال بـ DeepSeek: {str(e)}"

def get_gemini_response(messages, max_tokens=2000):
    global gemini_initial_configured
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE": # Check for placeholder
        return "Gemini API key مفقود أو لم يتم تعيينه."
    try:
        if not gemini_initial_configured: # Attempt to configure if not done initially
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_initial_configured = True
        
        model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro' if preferred/available
        
        # Gemini API prefers a different message structure where system prompt is part of user message or history
        processed_messages_for_gemini = []
        system_prompt_text = None

        if messages and messages[0]["role"] == "system":
            system_prompt_text = messages[0]["content"]
            user_msgs_start_index = 1
        else:
            user_msgs_start_index = 0
            
        # Combine system prompt with the first user message if applicable
        first_user_message_processed = False
        for i in range(user_msgs_start_index, len(messages)):
            msg = messages[i]
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if role == "user" and system_prompt_text and not first_user_message_processed:
                content = f"{system_prompt_text}\n\n---\n\n{content}"
                first_user_message_processed = True
            processed_messages_for_gemini.append({"role": role, "parts": [content]})
        
        if not processed_messages_for_gemini:
             return "لا توجد رسائل صالحة لإرسالها إلى Gemini."
        if system_prompt_text and not first_user_message_processed: # System prompt exists but no user message to attach it to
            processed_messages_for_gemini.insert(0, {"role": "user", "parts": [system_prompt_text]})


        if len(processed_messages_for_gemini) > 1 and processed_messages_for_gemini[-1]["role"] == "user":
            history = processed_messages_for_gemini[:-1]
            current_user_message_parts = processed_messages_for_gemini[-1]["parts"]
            chat = model.start_chat(history=history)
            response = chat.send_message(current_user_message_parts, generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens))
        elif processed_messages_for_gemini and processed_messages_for_gemini[0]["role"] == "user": # Single user message turn
            response = model.generate_content(processed_messages_for_gemini[0]["parts"], generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens))
        else:
            return "بنية رسائل Gemini غير متوقعة أو لا يوجد محتوى مستخدم لإرساله."
            
        return response.text
    except Exception as e:
        return f"خطأ عند الاتصال بـ Gemini: {str(e)}"

# ----------------------
# Status Functions
# ----------------------
def get_qdrant_info():
    try:
        # Use the initialized client if available, otherwise create a temporary one for status check
        client = init_qdrant_client()
        if not client: # Fallback if init_qdrant_client() returned None due to config error
            if not QDRANT_URL or not QDRANT_API_KEY: return {"status": False, "message": "Qdrant URL/API Key مفقود", "details": {}}
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5)

        collection_info = client.get_collection(COLLECTION_NAME)
        return {
            "status": True,
            "message": f"متصل ✓ | النقاط: {collection_info.points_count:,}",
            "details": {
                "اسم المجموعة": COLLECTION_NAME,
                "عدد النقاط": collection_info.points_count,
                "حالة الفهرسة": str(collection_info.status),
                "تهيئة المتجهات": str(collection_info.config.params),
                "حالة المُحسِّن": str(collection_info.optimizer_status) if hasattr(collection_info, 'optimizer_status') else "N/A"
            }
        }
    except Exception as e:
        if "Not found: Collection" in str(e) or "NOT_FOUND" in str(e).upper() or "status_code=404" in str(e) :
            return {"status": False, "message": f"غير متصل (المجموعة '{COLLECTION_NAME}' غير موجودة)", "details": {}}
        return {"status": False, "message": f"غير متصل (خطأ: {type(e).__name__})", "details": {}}

def check_api_status(api_name):
    global gemini_initial_configured
    if api_name == "DeepSeek":
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
            return False, "المفتاح مفقود"
        try:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "مرحبا"}], "max_tokens": 1, "stream": False}
            response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=7)
            return (True, "نشط ✓") if response.status_code == 200 else (False, f"خطأ ({response.status_code})")
        except Exception as e:
            return False, f"غير نشط ({type(e).__name__})"
    elif api_name == "Gemini":
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            return False, "المفتاح مفقود"
        try:
            if not gemini_initial_configured:
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_initial_configured = True
            model = genai.GenerativeModel('gemini-1.5-flash')
            # A light test, like listing models or a very short generation
            model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1))
            return True, "نشط ✓"
        except Exception as e:
            err_str = str(e).lower()
            if any(s in err_str for s in ["api_key_invalid", "permission_denied", "quota_exceeded", "authentication_failed"]):
                return False, "خطأ بالمفتاح/الصلاحية"
            return False, f"غير نشط ({type(e).__name__})"
    return False, "API غير معروف"

# ----------------------
# Main Application
# ----------------------
def main():
    load_arabic_css()

    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">محرك بحث فائق التطور - حل مشكلة المحتوى المخفي</p>', unsafe_allow_html=True)

    st.markdown('''
    <div class="ultra-search-boost">
        🚀 <strong>تحديث فائق:</strong> تم تطوير خوارزمية بحث متقدمة مع أكثر من 50 استراتيجية بحث متنوعة للعثور على المحتوى المخفي!
        <br>✨ <strong>ميزات جديدة:</strong> بحث بالمرادفات، استراتيجيات خاصة بالمصطلحات الدينية، عتبات متدرجة، ترتيب ذكي للنتائج.
        <br>💡 <strong>ملاحظة:</strong> جودة البحث تعتمد على جودة البيانات التي تم رفعها وتوافقها مع عمليات المعالجة.
    </div>
    ''', unsafe_allow_html=True)

    with st.expander("⚙️ الإعدادات وحالة الأنظمة المتقدمة", expanded=True):
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>اختر محرك الذكاء الاصطناعي:</div>", unsafe_allow_html=True)
        
        deepseek_ok, deepseek_msg = check_api_status("DeepSeek")
        gemini_ok, gemini_msg = check_api_status("Gemini")
        
        llm_options = ["DeepSeek", "Gemini"]
        llm_captions = [
            f"<span class='{'radio-label-status-active' if deepseek_ok else 'radio-label-status-inactive'}'>({deepseek_msg})</span>",
            f"<span class='{'radio-label-status-active' if gemini_ok else 'radio-label-status-inactive'}'>({gemini_msg})</span>"
        ]
        
        default_llm_index = 0 if deepseek_ok else (1 if gemini_ok else 0)
        if not deepseek_ok and not gemini_ok:
            st.warning("تنبيه: جميع محركات الذكاء الاصطناعي (DeepSeek, Gemini) غير نشطة أو مفاتيحها مفقودة. الرجاء التحقق من مفاتيح API.", icon="⚠️")
        
        selected_llm = st.radio("محركات AI:", llm_options, captions=llm_captions, index=default_llm_index, horizontal=True, key="llm_selection_radio", label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>##### حالة قاعدة البيانات (Qdrant):</div>", unsafe_allow_html=True)
        qdrant_status_info = get_qdrant_info() # Renamed to avoid conflict
        q_status_class = "status-active" if qdrant_status_info["status"] else "status-inactive"
        st.markdown(f'<div style="display: flex; justify-content: center;"><div style="background: #f0f2f6; padding: 0.5rem; border-radius: 8px; text-align: center; font-family: \'Noto Sans Arabic\', sans-serif; direction: rtl; border: 1px solid #e0e0e0; font-size: 0.9rem; margin-bottom: 0.5rem; width: 90%; max-width: 450px;">Qdrant DB: <span class="{q_status_class}">{qdrant_status_info["message"]}</span></div></div>', unsafe_allow_html=True)

        st.markdown("<div style='text-align: right; font-weight: bold; margin-top:0.5rem;'>مستوى البحث الفائق (عدد النتائج الأولية):</div>", unsafe_allow_html=True)
        search_levels = ["بحث سريع (20)", "بحث متوسط (40)", "بحث شامل (60)", "بحث فائق (80)"]
        selected_search_level = st.radio("مستوى البحث:", search_levels, index=2, horizontal=True, key="search_depth_radio", label_visibility="collapsed") # Renamed key
        max_results_setting = {"بحث سريع (20)": 20, "بحث متوسط (40)": 40, "بحث شامل (60)": 60, "بحث فائق (80)": 80}[selected_search_level]
        
        show_debug_info = st.checkbox("إظهار معلومات تفصيلية متقدمة (للمطورين)", value=True, key="debug_checkbox") # Renamed key

    if qdrant_status_info['status'] and qdrant_status_info.get('details'):
        with st.expander("ℹ️ معلومات قاعدة البيانات المتقدمة (Qdrant)", expanded=False):
            db_details = qdrant_status_info['details']
            info_html_content = f"<div style='direction: rtl; padding: 1rem; background-color: #e9ecef; border-radius: 10px; margin-top:1rem; margin-bottom: 1.5rem; border: 1px solid #ced4da; font-size:0.9em;'>"
            info_html_content += f"<h3 style='font-family: \"Noto Sans Arabic\", sans-serif; text-align:right; color: #495057;'>مجموعة البيانات: {db_details.get('اسم المجموعة', COLLECTION_NAME)}</h3>"
            for key, val in db_details.items():
                if key != "اسم المجموعة":
                    info_html_content += f"<p style='font-family: \"Noto Sans Arabic\", sans-serif; text-align:right; margin-bottom: 0.3rem;'><strong>{key}:</strong> {val}</p>"
            info_html_content += "</div>"
            st.markdown(info_html_content, unsafe_allow_html=True)
    elif not qdrant_status_info['status']:
        st.error(f"Qdrant: {qdrant_status_info['message']}. لا يمكن إجراء البحث بدون اتصال بقاعدة البيانات.", icon="⚠️")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_display_container = st.container() # Renamed
    with chat_display_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg_idx, msg_item_data in enumerate(st.session_state.messages): # Renamed variables
            is_user_msg = msg_item_data["role"] == "user"
            msg_icon = "👤" if is_user_msg else "🤖"
            msg_css_class = "user-message" if is_user_msg else "bot-message"
            # Ensure content is a string
            msg_content_str = str(msg_item_data.get("content", ""))
            st.markdown(f'<div class="{msg_css_class}">{msg_icon} {msg_content_str}</div>', unsafe_allow_html=True)
            
            if not is_user_msg: # Assistant message details
                if "api_used" in msg_item_data:
                    st.markdown(f'<span class="api-used">المحرك المستخدم: {msg_item_data["api_used"]}</span>', unsafe_allow_html=True)
                if "time_taken" in msg_item_data:
                    st.markdown(f'<div class="time-taken">⏱️ زمن الاستجابة الكلي: {msg_item_data["time_taken"]:.2f} ثانية</div>', unsafe_allow_html=True)
                
                if show_debug_info: # Checkbox state
                    debug_parts_list = []
                    if "debug_info" in msg_item_data:
                        debug_parts_list.append(f"ملخص البحث: {msg_item_data['debug_info']}")
                    
                    if "initial_search_details" in msg_item_data and msg_item_data["initial_search_details"]:
                        search_details_str_parts = []
                        num_details = len(msg_item_data['initial_search_details'])
                        for detail_idx, detail_item in enumerate(msg_item_data["initial_search_details"][:10]): # Show top 10 details
                            display_id_str = str(detail_item.get('id', 'N/A'))
                            score_val = detail_item.get('score', 0)
                            source_name_str = detail_item.get('source', 'N/A')
                            text_preview_str = detail_item.get('text_preview', 'N/A')
                            search_details_str_parts.append(f"  {detail_idx+1}. ID: {display_id_str[:12]} | Score: {score_val:.3f} | المصدر: {source_name_str} | معاينة: {text_preview_str[:70]}...")
                        search_details_str = "\n".join(search_details_str_parts)
                        debug_parts_list.append(f"تفاصيل نتائج Qdrant الأولية (أفضل {min(10, num_details)} من {num_details}):\n{search_details_str}")
                    
                    if debug_parts_list:
                        full_debug_html = f'<div class="debug-info">🔍 معلومات تفصيلية متقدمة:<div class="debug-info-results">{"<hr style=\'margin:0.3rem 0;\'>".join(debug_parts_list)}</div></div>'
                        st.markdown(full_debug_html, unsafe_allow_html=True)

                if "sources" in msg_item_data and msg_item_data["sources"]:
                    st.markdown("<div style='text-align: right; margin-top:0.5rem;'><strong>المصادر المستخدمة في الإجابة (مرتبة حسب الصلة):</strong></div><div class='source-container'>", unsafe_allow_html=True)
                    # Display sources in columns for better layout
                    num_sources_to_show = min(len(msg_item_data["sources"]), 12) # Show up to 12 sources
                    cols_per_row = 3
                    for src_row_idx in range(0, num_sources_to_show, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for src_col_idx, src_item_data in enumerate(msg_item_data["sources"][src_row_idx : src_row_idx + cols_per_row]):
                            with cols[src_col_idx]:
                                src_name = src_item_data.get("source", "غير متوفر")
                                src_score = src_item_data.get("score", 0)
                                quality_desc = "ممتاز" if src_score > 0.8 else "جيد جداً" if src_score > 0.6 else "جيد" if src_score > 0.4 else "مقبول"
                                st.markdown(f'<div class="source-info" title="المصدر: {src_name}\nالنقاط: {src_score*100:.1f}%\nالجودة: {quality_desc}">📄 <strong>{src_name}</strong><br>تطابق: {src_score*100:.1f}% ({quality_desc})</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True) # End chat-container

    st.markdown("<hr style='margin-top:1.5rem; margin-bottom:0.5rem;'>", unsafe_allow_html=True)
    
    st.markdown('''
    <div style="background: #f8f9fa; border-left: 4px solid #007bff; padding: 0.5rem; margin: 0.5rem 0; font-size: 0.9rem; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl;">
        💡 <strong>أمثلة على الاستعلامات:</strong><br>
        &bull; "حكم حلق اللحية في المذهب" أو "ما هو رأي الشيخ السند في حلق اللحية؟"<br>
        &bull; "صلاة ليلة الرغائب" أو "هل صلاة الرغائب مستحبة؟"<br>
        &bull; "مسألة رقم ٤٤ من منهاج الصالحين" أو "استفتاءات فقهية حول الخمس"
    </div>
    ''', unsafe_allow_html=True)
    
    # Use columns for better layout of input area if needed, but current full width is fine.
    user_query_input = st.text_area("سؤالك الفقهي...", placeholder="اكتب سؤالك هنا (مثال: حكم حلق اللحية، صلاة ليلة الرغائب، استفتاءات فقهية)...", key="user_query_text_area", height=120, label_visibility="collapsed") # Renamed key
    
    st.markdown('<div class="search-button-container">', unsafe_allow_html=True)
    search_button_clicked = st.button("🔍 بحث فائق التطور", type="primary", use_container_width=False, key="search_send_button") # Renamed key
    st.markdown('</div>', unsafe_allow_html=True)

    if search_button_clicked and user_query_input.strip():
        if not qdrant_status_info["status"]: # Check Qdrant status before proceeding
            st.error("لا يمكن البحث. قاعدة بيانات Qdrant غير متصلة أو غير موجودة. يرجى التحقق من الإعدادات.", icon="🚨")
            st.stop() # Stop execution if Qdrant is not available
        if selected_llm == "DeepSeek" and (not deepseek_ok):
             st.error(f"محرك DeepSeek غير نشط أو مفتاحه غير صحيح ({deepseek_msg}). لا يمكن معالجة الطلب.", icon="🚨")
             st.stop()
        if selected_llm == "Gemini" and (not gemini_ok):
             st.error(f"محرك Gemini غير نشط أو مفتاحه غير صحيح ({gemini_msg}). لا يمكن معالجة الطلب.", icon="🚨")
             st.stop()


        st.session_state.messages.append({"role": "user", "content": user_query_input.strip()})
        
        current_time_start = time.perf_counter()
        bot_response_data = {"role": "assistant", "api_used": selected_llm} # Initialize bot response structure
        
        with st.spinner(f"جاري البحث الفائق ({max_results_setting} نتيجة مع 50+ استراتيجية)... يرجى الانتظار قليلاً."):
            try:
                retrieved_results, search_info_log, search_details_for_display = comprehensive_search(user_query_input.strip(), max_results_setting)
                bot_response_data["initial_search_details"] = search_details_for_display
                bot_response_data["debug_info"] = search_info_log # Base debug info from search

                if show_debug_info: # Show intermediate search results if debug is on
                     st.info(f"نتائج البحث الأولي: تم العثور على {len(retrieved_results)} نتيجة. التفاصيل: {search_info_log}", icon="ℹ️")

            except Exception as search_exc:
                st.error(f"حدث خطأ أثناء عملية البحث الفائق: {search_exc}")
                retrieved_results, search_info_log = [], f"خطأ في البحث الفائق: {str(search_exc)}"
                bot_response_data["debug_info"] = search_info_log


        if retrieved_results:
            try:
                context_texts_for_llm = []
                sources_for_display_in_response = []
                total_chars_for_context = 0
                # Max characters for LLM context (adjust based on model limits, e.g. Gemini 1.5 Flash has large context)
                # For deepseek-chat (free tier might be around 8k-16k tokens for context window)
                # 1 token ~ 3-4 chars in English, maybe 1-2 chars in Arabic. Let's be conservative.
                max_llm_context_chars = 25000 # ~6k-8k Arabic words.
                
                for res_idx, res_item in enumerate(retrieved_results):
                    if not res_item.payload: continue
                    
                    # payload['text'] is the cleaned, normalized text from Qdrant
                    payload_text = res_item.payload.get('text', '')
                    source_doc_name = res_item.payload.get('source', f'وثيقة {str(res_item.id)[:6]}')
                    
                    if payload_text and len(payload_text.strip()) > 10: # Ensure text is valid
                        # Smart truncation for context (already done in search, but can re-verify or adjust)
                        # For LLM context, might not need aggressive truncation if model supports large context
                        current_text_for_llm = payload_text # Use full normalized text if possible
                        
                        if total_chars_for_context + len(current_text_for_llm) < max_llm_context_chars:
                            context_texts_for_llm.append(f"[نص {res_idx+1} من المصدر '{source_doc_name}' (نقاط الصلة: {res_item.score:.3f})]:\n{current_text_for_llm}")
                            sources_for_display_in_response.append({'source': source_doc_name, 'score': res_item.score, 'id': res_item.id})
                            total_chars_for_context += len(current_text_for_llm)
                        else:
                            context_texts_for_llm.append(f"\n[ملاحظة المحرر: تم اقتصار النصوص المرسلة للتحليل بسبب تجاوز الحد الأقصى للحجم. {len(retrieved_results)-res_idx} نص إضافي عالي الصلة لم يتم إرساله للتحليل.]")
                            bot_response_data["debug_info"] += f" | تنبيه: تم اقتصار سياق LLM، {len(retrieved_results)-res_idx} نصوص لم ترسل."
                            break # Stop adding more context
                
                if context_texts_for_llm:
                    full_context_for_llm = "\n\n---\n\n".join(context_texts_for_llm)
                    llm_context_summary = f"تم إرسال {len(sources_for_display_in_response)} نص عالي الصلة للتحليل (إجمالي ~{total_chars_for_context // 1000} ألف حرف)."
                    llm_messages_to_send = prepare_llm_messages(user_query_input.strip(), full_context_for_llm, llm_context_summary)
                    
                    llm_response_text = ""
                    with st.spinner(f"جاري التحليل المتقدم بواسطة {selected_llm} وفهم السياق..."):
                        try:
                            if selected_llm == "DeepSeek":
                                llm_response_text = get_deepseek_response(llm_messages_to_send)
                            elif selected_llm == "Gemini":
                                llm_response_text = get_gemini_response(llm_messages_to_send)
                            else: # Should not happen due to radio button
                                llm_response_text = "المحرك المحدد (LLM) غير معروف أو غير مدعوم."
                        except Exception as llm_exc:
                            llm_response_text = f"حدث خطأ أثناء الحصول على الرد من {selected_llm}: {str(llm_exc)}"
                    
                    bot_response_data["content"] = llm_response_text
                    bot_response_data["sources"] = sources_for_display_in_response
                    bot_response_data["debug_info"] += f" | {llm_context_summary}"
                else: # No valid texts extracted from search results for LLM
                    bot_response_data["content"] = f"تم العثور على {len(retrieved_results)} نتيجة من البحث الفائق، ولكن لم يتم العثور على نصوص صالحة للمعالجة ضمنها. يرجى محاولة صياغة مختلفة للسؤال أو التأكد من جودة البيانات المرفوعة."
                    bot_response_data["debug_info"] += " | لا توجد نصوص صالحة في النتائج الفائقة لإرسالها إلى LLM."
            
            except Exception as processing_exc:
                st.error(f"حدث خطأ أثناء معالجة نتائج البحث: {processing_exc}")
                bot_response_data["content"] = f"تم العثور على {len(retrieved_results)} نتيجة، ولكن حدث خطأ أثناء معالجتها: {str(processing_exc)}"
                bot_response_data["debug_info"] += f" | خطأ معالجة النتائج: {str(processing_exc)}"
        else: # No results from comprehensive_search
            bot_response_data["content"] = "لم يتم العثور على أي معلومات متعلقة بسؤالك حتى مع البحث الفائق التطور في قاعدة بيانات كتب واستفتاءات الشيخ محمد السند. تم تجربة أكثر من 50 استراتيجية بحث مختلفة. يرجى محاولة صياغة السؤال بشكل مختلف أو استخدام كلمات مفتاحية أخرى، أو التأكد من أن الوثائق التي تحتوي على الإجابة قد تم رفعها بشكل صحيح."
            # debug_info would already be set from comprehensive_search's return value
        
        bot_response_data["time_taken"] = time.perf_counter() - current_time_start
        st.session_state.messages.append(bot_response_data)
        st.rerun() # Rerun to display the new messages
    
    elif search_button_clicked and not user_query_input.strip():
        st.toast("يرجى إدخال سؤال للبحث.", icon="📝")

    # Clear chat button (placed outside main input columns for full width)
    if st.button("🗑️ مسح المحادثة وإعادة تهيئة البحث", use_container_width=True, key="clear_chat_button", type="secondary"): # Renamed key
        st.session_state.messages = []
        # Optionally clear other session state items if needed
        st.toast("تم مسح المحادثة. يمكنك بدء بحث جديد.", icon="🗑️")
        time.sleep(0.3) # Brief pause for toast visibility
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85rem; margin-top: 1rem; font-family: "Noto Sans Arabic", sans-serif; line-height: 1.6;'>
        🚀 <strong>محرك البحث الفائق التطور</strong> - الإصدار المتقدم المتوافق مع الرفع المحسن<br>
        🔧 <strong>التقنيات الأساسية:</strong> استراتيجيات بحث متعددة، تضمينات متقدمة، تطبيع نصي موحد، نماذج لغوية كبيرة.<br>
        ⚡ <strong>الميزات الرئيسية:</strong> ترتيب ذكي للنتائج، فلترة متقدمة، كشف المحتوى المخفي (بقدر الإمكان)، تحليل جودة النتائج.<br>
        🎯 <strong>متخصص في:</strong> الفقه الإسلامي، كتب واستفتاءات الشيخ محمد السند، المنهاج، والمسائل الشرعية.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initial system checks (optional, can be removed if causing issues before app load)
    # These are more for local development feedback.
    # config_issues_list = []
    # if not QDRANT_API_KEY or not QDRANT_URL or QDRANT_API_KEY == "YOUR_QDRANT_API_KEY":
    #     config_issues_list.append("❌ معلومات QDRANT (URL/API Key) مفقودة أو تستخدم قيمة افتراضية.")
    # if (not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE") and \
    #    (not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE"):
    #     config_issues_list.append("❌ لا توجد مفاتيح API للذكاء الاصطناعي (DeepSeek/Gemini) أو تستخدم قيم افتراضية.")
    
    # if config_issues_list:
    #     st.error("مشاكل في التهيئة الأولية: " + " | ".join(config_issues_list) + 
    #              " يرجى مراجعة متغيرات البيئة أو تحديث المفاتيح في الكود مباشرة. قد لا تعمل بعض الميزات بشكل صحيح.", icon="📛")
    # else:
    #    st.success("✅ التهيئة الأولية للأنظمة تبدو جيدة. جاهز للبحث الفائق!", icon="👍")
    
    main()
