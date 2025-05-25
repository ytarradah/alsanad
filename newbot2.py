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
GEMINI_API_KEY = "AIzaSyAHlTww82Qw5PdFDPjKQIwA7f0FRUO-nFQ"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

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
    
    .source-group {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 12px;
        margin-top: 0.8rem;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        border-left: 4px solid #2E8B57;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    
    .source-item {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        white-space: nowrap;
    }
    
    .ai-indicator {
        background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: inline-block;
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
# Fast Connection Check with Gemini
# ----------------------
@st.cache_data(ttl=60)  # Cache for 1 minute
def check_connection_status():
    status = {"qdrant": False, "deepseek": False, "gemini": False}
    
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
    
    # Quick Gemini test
    try:
        url = f"{GEMINI_BASE_URL}?key={GEMINI_API_KEY}"
        data = {"contents": [{"parts": [{"text": "hi"}]}]}
        response = requests.post(url, json=data, timeout=3)
        status["gemini"] = response.status_code == 200
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
# AI Response with DeepSeek + Gemini Fallback
# ----------------------
def get_ai_response_with_fallback(context, query):
    """Try DeepSeek first, fallback to Gemini if it fails"""
    
    strict_prompt = f"""أنت مساعد ذكي للمرجع الديني الشيخ محمد السند. 

    قواعد صارمة:
    1. أجب فقط من النصوص المرفقة من قاعدة البيانات
    2. لا تستخدم أي معلومات من خارج النصوص المرفقة
    3. إذا لم تجد الإجابة في النصوص المرفقة، قل "لا توجد معلومات في قاعدة البيانات"
    4. لا تضيف معلومات من معرفتك العامة
    5. أجب باللغة العربية فقط

    النصوص من قاعدة البيانات:
    {context}

    السؤال: {query}

    أجب فقط من النصوص المرفقة أعلاه."""
    
    # Try DeepSeek first
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": strict_prompt}],
            "temperature": 0.1,
            "max_tokens": 500,
            "stream": False
        }
        
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", 
                               headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return content, "DeepSeek"
    except Exception as e:
        print(f"DeepSeek failed: {e}")
    
    # Fallback to Gemini
    try:
        url = f"{GEMINI_BASE_URL}?key={GEMINI_API_KEY}"
        data = {
            "contents": [{"parts": [{"text": strict_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 500
            }
        }
        
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            content = response.json()['candidates'][0]['content']['parts'][0]['text']
            return content, "Gemini"
    except Exception as e:
        print(f"Gemini failed: {e}")
    
    return "خطأ في الحصول على الإجابة من كلا النظامين", "Error"

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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'''<div class="connection-status {'disconnected' if not status['qdrant'] else ''}">
        {'🟢 قاعدة البيانات متصلة' if status['qdrant'] else '🔴 قاعدة البيانات غير متصلة'}</div>''', 
        unsafe_allow_html=True)
    with col2:
        st.markdown(f'''<div class="connection-status {'disconnected' if not status['deepseek'] else ''}">
        {'🚀 DeepSeek متصل' if status['deepseek'] else '⚠️ DeepSeek غير متاح'}</div>''', 
        unsafe_allow_html=True)
    with col3:
        st.markdown(f'''<div class="connection-status {'disconnected' if not status['gemini'] else ''}">
        {'🤖 Gemini متصل' if status['gemini'] else '⚠️ Gemini غير متاح'}</div>''', 
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
            
            # Show AI indicator
            if "ai_used" in message:
                ai_name = message["ai_used"]
                ai_color = "🚀" if ai_name == "DeepSeek" else "🤖" if ai_name == "Gemini" else "❌"
                st.markdown(f'<div class="ai-indicator">{ai_color} تم الرد بواسطة: {ai_name}</div>', unsafe_allow_html=True)
            
            # Show sources in groups (3-4 per line)
            if "sources" in message and message["sources"]:
                sources = message["sources"]
                # Group sources in chunks of 3-4
                for i in range(0, len(sources), 3):
                    group = sources[i:i+3]
                    source_items = []
                    for source in group:
                        percentage = source.get('percentage', 0)
                        source_name = source["source"].replace('.txt', '').replace('.pdf', '').replace('.docx', '')
                        source_items.append(f'<span class="source-item">{percentage}% {source_name}</span>')
                    
                    st.markdown(f'''
                    <div class="source-group">
                        📚 المراجع: {' '.join(source_items)}
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
        if not status["qdrant"]:
            st.error("⚠️ قاعدة البيانات غير متصلة")
            return
        
        if not (status["deepseek"] or status["gemini"]):
            st.error("⚠️ لا يوجد نظام ذكاء اصطناعي متاح")
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
                
                # Get AI response with fallback
                with st.spinner("🤖 جاري تحضير الإجابة من قاعدة البيانات..."):
                    response, ai_used = get_ai_response_with_fallback(context, user_question.strip())
                
                # Add response with sources and AI indicator
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources,
                    "ai_used": ai_used
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "لا توجد معلومات ذات صلة في قاعدة البيانات المحملة للإجابة على سؤالك.",
                    "ai_used": "System"
                })
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "لم أتمكن من البحث في قاعدة البيانات. تأكد من الاتصال.",
                "ai_used": "System"
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
