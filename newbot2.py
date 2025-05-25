import streamlit as st
import time
from datetime import datetime
import json
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd

# ----------------------
# Configuration
# ----------------------
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Dw82gEuSqeeloMVxaGp48Q2oU-W3NjLSibtM-pqRHzk"
COLLECTION_NAME = "arabic_documents_384"
DEEPSEEK_API_KEY = "sk-14f267781a6f474a9d0ec8240383dae4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ----------------------
# Arabic CSS Styling
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
    
    .loading-spinner {
        text-align: center;
        color: #2E8B57;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 1rem 0;
        direction: rtl;
        padding: 0.8rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border: 1px solid #2E8B57;
        box-shadow: 0 2px 8px rgba(46, 139, 87, 0.1);
    }
    
    .chat-container {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
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
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
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
    
    .stSelectbox > div > div > select {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# Initialize Components
# ----------------------
@st.cache_resource
def init_qdrant_client():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return client
    except Exception as e:
        st.error(f"فشل في الاتصال بقاعدة البيانات: {e}")
        return None

@st.cache_resource
def init_embedding_model():
    try:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as e:
        st.error(f"فشل في تحميل نموذج التضمين: {e}")
        return None

# ----------------------
# Connection Status Check
# ----------------------
def check_connection_status():
    """Check connection status for both Qdrant and DeepSeek"""
    status = {
        "qdrant": False,
        "deepseek": False
    }
    
    # Test Qdrant connection
    try:
        qdrant_client = init_qdrant_client()
        if qdrant_client:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            status["qdrant"] = True
    except Exception:
        status["qdrant"] = False
    
    # Test DeepSeek connection
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        test_data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            status["deepseek"] = True
    except Exception:
        status["deepseek"] = False
    
    return status

def show_connection_status():
    """Display connection status"""
    status = check_connection_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if status["qdrant"]:
            st.markdown('''
            <div class="connection-status">
                🟢 قاعدة البيانات متصلة
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="connection-status disconnected">
                🔴 قاعدة البيانات غير متصلة
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        if status["deepseek"]:
            st.markdown('''
            <div class="connection-status">
                🤖 الذكاء الاصطناعي متصل
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="connection-status disconnected">
                ⚠️ الذكاء الاصطناعي غير متاح
            </div>
            ''', unsafe_allow_html=True)
    
    return status

# ----------------------
# DeepSeek Chat Function
# ----------------------
def get_deepseek_response(messages, max_retries=3):
    """Get response from DeepSeek API"""
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            response = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                st.error(f"خطأ في API: {response.status_code}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                st.error(f"فشل في الحصول على الاستجابة: {e}")
    
    return "عذراً، لم أتمكن من الحصول على استجابة في الوقت الحالي."

# ----------------------
# Document Search Function
# ----------------------
def search_documents(query, top_k=5):
    """Search for relevant documents using vector similarity"""
    try:
        # Generate embedding for the query
        embedding_model = init_embedding_model()
        if not embedding_model:
            return []
        
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Search in Qdrant
        qdrant_client = init_qdrant_client()
        if not qdrant_client:
            return []
        
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        return search_results
        
    except Exception as e:
        st.error(f"فشل في البحث: {e}")
        return []

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(
        page_title="المساعد الذكي للوثائق العربية",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load Arabic CSS
    load_arabic_css()
    
    # Header - centered text only
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند - دام ظله</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">محرك بحث الكتب والاستفتاءات</p>', unsafe_allow_html=True)
    
    # Show connection status
    connection_status = show_connection_status()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
                if "sources" in message:
                    for source in message["sources"]:
                        percentage = source.get('percentage', int(source['score'] * 100))
                        source_name = source["source"].replace('.txt', '').replace('.pdf', '').replace('.docx', '')
                        st.markdown(f'''
                        <div class="source-info">
                            <span class="source-percentage">{percentage}%</span>
                            📚 المرجع: {source_name}
                        </div>
                        ''', unsafe_allow_html=True)
    
    # Chat input with bigger text area and Enter key support
    st.markdown('<h3 style="text-align: center; direction: rtl; font-family: \'Noto Sans Arabic\', sans-serif;">💭 اكتب سؤالك هنا:</h3>', unsafe_allow_html=True)
    
    # Use form for Enter key functionality and bigger text area
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_area(
            "",
            placeholder="اكتب سؤالك هنا واضغط Ctrl+Enter للإرسال...\n\nمثال: ما هو موضوع الوثيقة الأولى؟\nأو: لخص محتوى الملف الثاني",
            height=120,
            key="user_input"
        )
        
        # Submit button (also works with Enter in form)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button("إرسال السؤال 📤", type="primary", use_container_width=True)
    
    # Process user input
    if submit_button and user_question and user_question.strip():
        # Check connections before processing
        if not connection_status["qdrant"]:
            st.warning("⚠️ قاعدة البيانات غير متصلة. لا يمكن البحث في الوثائق.")
            return
            
        if not connection_status["deepseek"]:
            st.warning("⚠️ الذكاء الاصطناعي غير متاح. لا يمكن تقديم إجابات ذكية.")
            return
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question.strip()})
        
        # Search for relevant documents
        with st.spinner(""):
            st.markdown('<div class="loading-spinner">🔍 جاري البحث في المراجع والكتب...</div>', unsafe_allow_html=True)
            search_results = search_documents(user_question.strip())
        
        if search_results:
            # Prepare context for DeepSeek
            context_texts = []
            sources = []
            
            for result in search_results:
                context_texts.append(result.payload.get('text', ''))
                sources.append({
                    'source': result.payload.get('source', 'مجهول'),
                    'score': result.score,
                    'percentage': min(100, int(result.score * 100))  # Convert to percentage
                })
            
            context = "\n\n".join(context_texts)
            
            # Prepare messages for DeepSeek
            system_prompt = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة باللغة العربية بناءً على الوثائق المقدمة.
            
            تعليمات مهمة:
            1. أجب باللغة العربية فقط
            2. استخدم المعلومات من الوثائق المقدمة فقط
            3. كن دقيقاً ومفيداً
            4. إذا لم تجد الإجابة في الوثائق، قل ذلك بوضوح
            5. اذكر المصادر عند الإمكان
            6. اجعل إجاباتك مفصلة ومفيدة
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"السياق من الوثائق:\n{context}\n\nالسؤال: {user_question.strip()}"}
            ]
            
            # Get response from DeepSeek
            with st.spinner(""):
                st.markdown('<div class="loading-spinner">🤖 جاري تحضير الإجابة الشرعية...</div>', unsafe_allow_html=True)
                bot_response = get_deepseek_response(messages)
            
            # Add bot response to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "sources": sources
            })
        else:
            # No relevant documents found
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "عذراً، لم أجد معلومات ذات صلة في الوثائق المحملة للإجابة على سؤالك. تأكد من أن سؤالك متعلق بمحتوى الوثائق المرفوعة."
            })
        
        # Rerun to update chat
        st.rerun()
    
    # Chat management buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("مسح المحادثة 🗑️"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("إحصائيات المحادثة 📊"):
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            st.info(f"إجمالي الرسائل: {total_messages} | أسئلتك: {user_messages} | إجابات المساعد: {bot_messages}")

# ----------------------
# Run the Application
# ----------------------
if __name__ == "__main__":
    main()