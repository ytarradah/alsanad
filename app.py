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
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
        'ى': 'ي', 'ة': 'ه', 'ؤ': 'و', 'ئ': 'ي',
        '\u200c': '', '\u200d': '', '\ufeff': '', '\u200b': ''
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return ' '.join(text.split())

def extract_arabic_keywords(text):
    """Extract Arabic keywords"""
    if not text:
        return []
    
    stop_words = {
        'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل',
        'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي',
        'هو', 'هي', 'هم', 'هن', 'أنت', 'أنا', 'نحن',
        'كان', 'كانت', 'يكون', 'تكون', 'قد', 'لقد',
        'ما', 'ماذا', 'متى', 'أين', 'كيف', 'لماذا',
        'ال', 'و', 'ف', 'ب', 'ك', 'ل', 'أن', 'إن', 'لا'
    }
    
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
    .stTextArea > div > div > textarea { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; }
    .chat-container { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; direction: rtl; text-align: right; }
    .bot-message { background: linear-gradient(135deg, #5DADE2 0%, #3498DB 100%); color: white; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; direction: rtl; text-align: right; }
    .debug-info { background: #fff3cd; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.85rem; direction: rtl; text-align: right; }
    .status-active { color: #28a745; font-weight: bold; }
    .status-inactive { color: #dc3545; font-weight: bold; }
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
# ENHANCED SEARCH FUNCTION
# ----------------------
def comprehensive_search(query, max_results=50):
    """Enhanced search with multiple strategies"""
    
    embedding_model = init_embedding_model()
    if not embedding_model:
        return [], "فشل تحميل نموذج التضمين.", []
    
    qdrant_client = init_qdrant_client()
    if not qdrant_client:
        return [], "فشل الاتصال بـ Qdrant.", []
    
    try:
        print(f"Enhanced search for: '{query}'")
        
        # Create query variants
        variants = []
        
        # 1. Original query
        variants.append(('original', query, 0.2))
        
        # 2. Normalized query
        normalized = normalize_arabic_text(query)
        if normalized != query:
            variants.append(('normalized', normalized, 0.18))
        
        # 3. Keywords
        keywords = extract_arabic_keywords(query)
        if keywords:
            keywords_query = ' '.join(keywords)
            variants.append(('keywords', keywords_query, 0.15))
        
        # 4. Individual words
        words = [w for w in query.split() if len(w) > 3]
        for word in words[:2]:
            variants.append(('word', word, 0.12))
        
        # Search with each variant
        all_results = []
        seen_ids = set()
        search_info = []
        
        for variant_type, variant_query, threshold in variants:
            try:
                # Create embedding
                embedding = embedding_model.encode([variant_query])[0].tolist()
                
                # Search
                results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=embedding,
                    limit=max_results // 2,
                    with_payload=True,
                    score_threshold=threshold
                )
                
                # Add new results
                new_count = 0
                for result in results:
                    if result.id not in seen_ids:
                        seen_ids.add(result.id)
                        all_results.append(result)
                        new_count += 1
                
                search_info.append(f"{variant_type}: {new_count}")
                print(f"✅ {variant_type} found {new_count} results")
                
                if len(all_results) >= 20:
                    break
                    
            except Exception as e:
                print(f"❌ Error with {variant_type}: {e}")
                search_info.append(f"{variant_type}: خطأ")
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        final_results = all_results[:max_results]
        
        # Create debug info
        initial_details = []
        if final_results:
            initial_details = [
                {
                    "id": r.id,
                    "score": r.score,
                    "source": r.payload.get('source', 'N/A') if r.payload else 'N/A',
                    "text_preview": (r.payload.get('text', '')[:100] + "...") if r.payload else ''
                }
                for r in final_results[:10]
            ]
        
        search_summary = f"بحث محسن: {len(final_results)} نتيجة. " + " | ".join(search_info)
        
        return final_results, search_summary, initial_details
        
    except Exception as e:
        print(f"Search error: {e}")
        return [], f"خطأ في البحث: {str(e)}", []

# ----------------------
# API Functions
# ----------------------
def get_deepseek_response(messages, max_tokens=2000):
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API key مفقود."
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "deepseek-chat", "messages": messages, "temperature": 0.05, "max_tokens": max_tokens}
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'] if result.get('choices') else "لم يتمكن DeepSeek من الرد."
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
        
        # Convert messages to Gemini format
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"خطأ Gemini: {str(e)}"

def prepare_llm_messages(user_question, context, context_info):
    system_prompt = "أنت مساعد للبحث في كتب واستفتاءات الشيخ محمد السند فقط. أجب فقط من النصوص المعطاة أدناه."
    user_content = f"السؤال: {user_question}\n\nالمصادر المتاحة:\n{context}\n\nمعلومات إضافية: {context_info}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

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
            "details": {"اسم المجموعة": COLLECTION_NAME, "عدد النقاط": collection_info.points_count}
        }
    except Exception as e:
        return {"status": False, "message": f"غير متصل: {str(e)}", "details": {}}

@st.cache_data(ttl=300)
def check_api_status(api_name):
    if api_name == "DeepSeek":
        if not DEEPSEEK_API_KEY:
            return False, "المفتاح مفقود"
        try:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "test"}], "max_tokens": 1}
            response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=5)
            return (True, "نشط ✓") if response.status_code == 200 else (False, f"خطأ ({response.status_code})")
        except Exception:
            return False, "غير نشط"
    
    elif api_name == "Gemini":
        if not GEMINI_API_KEY:
            return False, "المفتاح مفقود"
        try:
            if not gemini_initial_configured:
                genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            model.generate_content("test")
            return True, "نشط ✓"
        except Exception:
            return False, "غير نشط"
    
    return False, "غير معروف"

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="المرجع السند - بحث", page_icon="🕌", layout="wide")
    load_arabic_css()
    
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">محرك بحث الكتب والاستفتاءات المحسن</p>', unsafe_allow_html=True)

    # Settings
    with st.expander("⚙️ الإعدادات وحالة الأنظمة", expanded=True):
        st.markdown("### اختر محرك الذكاء الاصطناعي:")
        
        # Check API status
        deepseek_ok, deepseek_msg = check_api_status("DeepSeek")
        gemini_ok, gemini_msg = check_api_status("Gemini")
        
        llm_options = ["DeepSeek", "Gemini"]
        llm_captions = [f"DeepSeek ({deepseek_msg})", f"Gemini ({gemini_msg})"]
        
        default_index = 0 if deepseek_ok else (1 if gemini_ok else 0)
        selected_llm = st.radio("محركات AI:", llm_options, captions=llm_captions, index=default_index, horizontal=True)
        
        st.markdown("---")
        st.markdown("### حالة قاعدة البيانات:")
        qdrant_info = get_qdrant_info()
        status_class = "status-active" if qdrant_info["status"] else "status-inactive"
        st.markdown(f'<div class="{status_class}">Qdrant: {qdrant_info["message"]}</div>', unsafe_allow_html=True)
        
        st.markdown("### مستوى البحث:")
        search_levels = ["بحث سريع (15)", "بحث متوسط (30)", "بحث شامل (50)"]
        selected_level = st.radio("مستوى البحث:", search_levels, index=1, horizontal=True)
        max_results = {"بحث سريع (15)": 15, "بحث متوسط (30)": 30, "بحث شامل (50)": 50}[selected_level]
        
        show_debug = st.checkbox("إظهار معلومات تفصيلية", value=True)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        css_class = "user-message" if is_user else "bot-message"
        icon = "👤" if is_user else "🤖"
        st.markdown(f'<div class="{css_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
        
        if not is_user:
            if "time_taken" in msg:
                st.markdown(f'<div style="font-size:0.8rem;color:#777;margin-top:0.3rem;">⏱️ زمن: {msg["time_taken"]:.2f} ث</div>', unsafe_allow_html=True)
            
            if show_debug and "debug_info" in msg:
                st.markdown(f'<div class="debug-info">🔍 {msg["debug_info"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input section
    st.markdown("---")
    user_query = st.text_area("سؤالك...", placeholder="اكتب سؤالك هنا (مثال: صلاة ليلة الرغائب)...", height=100)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🔍 بحث وإجابة محسن", type="primary", use_container_width=True)

    # Process search
    if search_button and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query.strip()})
        
        start_time = time.perf_counter()
        
        # Search
        with st.spinner(f"جاري البحث المحسن ({max_results} نتيجة)..."):
            search_results, search_info, search_details = comprehensive_search(user_query.strip(), max_results)
        
        # Debug display
        if show_debug:
            st.markdown(f'<div class="debug-info">🔍 وجدت {len(search_results)} نتيجة | {search_info}</div>', unsafe_allow_html=True)
        
        # Process results
        if search_results:
            try:
                # Prepare context
                context_texts = []
                sources = []
                total_chars = 0
                max_chars = 25000
                
                for i, result in enumerate(search_results):
                    if not result.payload:
                        continue
                    
                    text = result.payload.get('text', '')
                    source = result.payload.get('source', f'وثيقة {i+1}')
                    
                    if text and total_chars + len(text) < max_chars:
                        truncated_text = text[:1500] + ("..." if len(text) > 1500 else "")
                        context_texts.append(f"[نص {i+1} من '{source}']: {truncated_text}")
                        sources.append({'source': source, 'score': result.score})
                        total_chars += len(truncated_text)
                    
                    if len(context_texts) >= 10:  # Limit contexts
                        break
                
                if context_texts:
                    context = "\n\n---\n\n".join(context_texts)
                    context_info = f"أرسل {len(sources)} نص للتحليل"
                    messages = prepare_llm_messages(user_query.strip(), context, context_info)
                    
                    # Get LLM response
                    with st.spinner(f"جاري التحليل بواسطة {selected_llm}..."):
                        if selected_llm == "DeepSeek":
                            bot_response = get_deepseek_response(messages)
                        else:
                            bot_response = get_gemini_response(messages)
                    
                    # Save response
                    time_taken = time.perf_counter() - start_time
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_response,
                        "time_taken": time_taken,
                        "debug_info": f"{search_info} | {context_info}"
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "تم العثور على نتائج لكن لا تحتوي على نصوص صالحة للمعالجة.",
                        "debug_info": search_info
                    })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"حدث خطأ في معالجة النتائج: {str(e)}",
                    "debug_info": search_info
                })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "لم أجد أي معلومات متعلقة بسؤالك في قاعدة البيانات. يرجى تجربة صياغة مختلفة.",
                "debug_info": search_info
            })
        
        st.rerun()
    
    elif search_button and not user_query.strip():
        st.toast("يرجى إدخال سؤال.", icon="📝")

    # Clear chat
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
