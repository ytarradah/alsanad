import streamlit as st
import time
import json
import requests

# ----------------------
# Configuration
# ----------------------
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Dw82gEuSqeeloMVxaGp48Q2oU-W3NjLSibtM-pqRHzk"
COLLECTION_NAME = "arabic_documents_384"
DEEPSEEK_API_KEY = "sk-14f267781a6f474a9d0ec8240383dae4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ----------------------
# Fast Arabic CSS Styling
# ----------------------
def load_arabic_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 2rem 0;
        direction: rtl;
        line-height: 1.4;
    }
    
    .main-subtitle {
        text-align: center;
        color: #6c757d;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        direction: rtl;
        font-style: italic;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .source-info {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 12px;
        margin-top: 0.8rem;
        font-size: 1rem;
        color: #495057;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        border-left: 4px solid #2E8B57;
    }
    
    .source-percentage {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        margin-right: 0.5rem;
        font-size: 0.9rem;
    }
    
    .connection-status {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        direction: rtl;
        text-align: center;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 0.9rem;
    }
    
    .connection-status.disconnected {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    }
    
    .stTextArea > div > div > textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 18px !important;
        min-height: 120px !important;
        padding: 20px !important;
        line-height: 1.6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# Fast Connection Check
# ----------------------
@st.cache_data(ttl=60)  # Cache for 1 minute
def check_connection_status():
    status = {"qdrant": False, "deepseek": False}
    
    # Quick Qdrant test
    try:
        headers = {"api-key": QDRANT_API_KEY}
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}", 
                               headers=headers, timeout=3)
        status["qdrant"] = response.status_code == 200
    except:
        pass
    
    # Quick DeepSeek test
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions",
                               headers=headers, 
                               json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                               timeout=3)
        status["deepseek"] = response.status_code == 200
    except:
        pass
    
    return status

# ----------------------
# Fast Document Search - STRICT DATABASE ONLY
# ----------------------
def search_in_database_only(query, top_k=3):
    """STRICT: Search ONLY in uploaded Qdrant database"""
    try:
        # Try embedding with fallback
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            query_embedding = model.encode([query])[0].tolist()
        except:
            return None  # No embedding = no search
        
        # Search in Qdrant database ONLY
        headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
        search_data = {"vector": query_embedding, "limit": top_k, "with_payload": True}
        
        response = requests.post(f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
                               headers=headers, json=search_data, timeout=10)
        
        if response.status_code == 200:
            return response.json().get("result", [])
        return None
        
    except Exception as e:
        st.error(f"خطأ في البحث: {e}")
        return None

# ----------------------
# STRICT DeepSeek Response - DATABASE ONLY
# ----------------------
def get_strict_database_response(context, query):
    """Get response STRICTLY from database context only"""
    
    # VERY STRICT system prompt
    system_prompt = f"""أنت مساعد ذكي للمرجع الديني الشيخ محمد السند. 

    قواعد صارمة:
    1. أجب فقط من النصوص المرفقة من قاعدة البيانات
    2. لا تستخدم أي معلومات من خارج النصوص المرفقة
    3. إذا لم تجد الإجابة في النصوص المرفقة، قل "لا توجد معلومات في قاعدة البيانات"
    4. لا تضيف معلومات من معرفتك العامة
    5. أجب باللغة العربية فقط
    6. اذكر النص الأصلي عند الإمكان

    النصوص من قاعدة البيانات:
    {context}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"السؤال: {query}\n\nأجب فقط من النصوص المرفقة أعلاه."}
    ]
    
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.1,  # Very low temperature for strict responses
            "max_tokens": 500,   # Shorter responses
            "stream": False
        }
        
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", 
                               headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "خطأ في الحصول على الإجابة"
            
    except Exception as e:
        return f"خطأ: {e}"

# ----------------------
# Main Fast Application
# ----------------------
def main():
    st.set_page_config(
        page_title="موقع المرجع الديني الشيخ محمد السند",
        page_icon="🕌",
        layout="wide"
    )
    
    load_arabic_css()
    
    # Header 
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند - دام ظله</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">محرك بحث الكتب والاستفتاءات</p>', unsafe_allow_html=True)
    
    # Fast connection check
    status = check_connection_status()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''<div class="connection-status {'disconnected' if not status['qdrant'] else ''}">
        {'🟢 قاعدة البيانات متصلة' if status['qdrant'] else '🔴 قاعدة البيانات غير متصلة'}</div>''', 
        unsafe_allow_html=True)
    with col2:
        st.markdown(f'''<div class="connection-status {'disconnected' if not status['deepseek'] else ''}">
        {'🤖 الذكاء الاصطناعي متصل' if status['deepseek'] else '⚠️ الذكاء الاصطناعي غير متاح'}</div>''', 
        unsafe_allow_html=True)
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
            if "sources" in message:
                for source in message["sources"]:
                    percentage = source.get('percentage', 0)
                    source_name = source["source"].replace('.txt', '').replace('.pdf', '').replace('.docx', '')
                    st.markdown(f'''
                    <div class="source-info">
                        <span class="source-percentage">{percentage}%</span>
                        📚 المرجع: {source_name}
                    </div>''', unsafe_allow_html=True)
    
    # Input form
    st.markdown('<h3 style="text-align: center; direction: rtl; font-family: \'Noto Sans Arabic\', sans-serif;">💭 اكتب سؤالك هنا:</h3>', unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_area("", placeholder="اكتب سؤالك هنا واضغط Ctrl+Enter للإرسال...", height=120)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button("إرسال السؤال 📤", type="primary", use_container_width=True)
    
    # Process question
    if submit and user_question and user_question.strip():
        if not status["qdrant"] or not status["deepseek"]:
            st.error("⚠️ النظام غير متصل بالكامل")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_question.strip()})
        
        # Search ONLY in database
        with st.spinner("🔍 جاري البحث في قاعدة البيانات..."):
            search_results = search_in_database_only(user_question.strip())
        
        if search_results:
            # Prepare context from database ONLY
            context_texts = []
            sources = []
            
            for result in search_results:
                payload = result.get("payload", {})
                text = payload.get('text', '')
                if text:  # Only add non-empty texts
                    context_texts.append(text)
                    sources.append({
                        'source': payload.get('source', 'مجهول'),
                        'score': result.get('score', 0.0),
                        'percentage': min(100, int(result.get('score', 0.0) * 100))
                    })
            
            if context_texts:
                context = "\n---\n".join(context_texts)
                
                # Get STRICT response from database only
                with st.spinner("🤖 جاري تحضير الإجابة من قاعدة البيانات..."):
                    response = get_strict_database_response(context, user_question.strip())
                
                # Add response with sources
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "لا توجد معلومات ذات صلة في قاعدة البيانات المحملة للإجابة على سؤالك."
                })
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "لم أتمكن من البحث في قاعدة البيانات. تأكد من الاتصال."
            })
        
        st.rerun()
    
    # Management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("مسح المحادثة 🗑️"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("إحصائيات 📊"):
            total = len(st.session_state.messages)
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.info(f"إجمالي: {total} | أسئلة: {user_msgs}")

if __name__ == "__main__":
    main()
