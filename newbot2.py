import streamlit as st
import time
from datetime import datetime
import json
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
from openai import OpenAI
import google.generativeai as genai

# ----------------------
# Configuration
# ----------------------
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ"
COLLECTION_NAME = "arabic_documents_384"

# API Keys
OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-14f267781a6f474a9d0ec8240383dae4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_API_KEY = "AIzaSyASlapu6AYYwOQAJJ3v-2FSKHnxIOZoPbY"  # New API key

# Initialize APIs
client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

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
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        direction: rtl;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 1rem;
        direction: rtl;
    }
    
    .status-box {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 1rem auto;
        text-align: center;
        font-family: 'Noto Sans Arabic', sans-serif;
        direction: rtl;
        max-width: 400px;
    }
    
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #dc3545;
        font-weight: bold;
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
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #5DADE2 0%, #3498DB 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1.1rem;
        font-weight: 500;
        line-height: 1.8;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .source-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-top: 1rem;
        direction: rtl;
    }
    
    .source-info {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 10px;
        font-size: 0.85rem;
        color: #666;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        border: 1px solid #ddd;
    }
    
    .api-used {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        display: inline-block;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    
    .stTextArea > div > div > textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1.1rem;
        min-height: 100px !important;
    }
    
    .stSelectbox > div > div > select {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    
    .search-button {
        text-align: center;
        margin-top: 1rem;
    }
    
    div[data-testid="stButton"] > button {
        width: 200px;
        margin: 0 auto;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# Database Status Check
# ----------------------
@st.cache_data(ttl=60)  # Cache for 1 minute
def check_database_status():
    """Check if Qdrant database is connected and active"""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collection_info = client.get_collection(COLLECTION_NAME)
        point_count = collection_info.points_count
        return True, point_count
    except Exception as e:
        print(f"Database connection error: {e}")
        return False, 0

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
# API Response Functions
# ----------------------
def get_openai_response(messages):
    """Get response from OpenAI API"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"خطأ في OpenAI API: {e}")
        return "عذراً، لم أتمكن من الحصول على استجابة من OpenAI."

def get_deepseek_response(messages):
    """Get response from DeepSeek API"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500,
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
            st.error(f"خطأ في DeepSeek API: {response.status_code}")
            return "عذراً، لم أتمكن من الحصول على استجابة من DeepSeek."
            
    except Exception as e:
        st.error(f"فشل في الحصول على الاستجابة: {e}")
        return "عذراً، لم أتمكن من الحصول على استجابة في الوقت الحالي."

def get_gemini_response(messages):
    """Get response from Gemini API"""
    try:
        # Try different model names
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                
                # Format messages for Gemini
                prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        prompt += f"النظام: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        prompt += f"المستخدم: {msg['content']}\n\n"
                
                response = model.generate_content(prompt)
                return response.text
            except:
                continue
                
        # If all models fail
        st.error("فشل في استخدام جميع نماذج Gemini المتاحة")
        return "عذراً، لم أتمكن من الحصول على استجابة من Gemini."
        
    except Exception as e:
        st.error(f"خطأ في Gemini API: {e}")
        return "عذراً، لم أتمكن من الحصول على استجابة من Gemini."

# ----------------------
# Document Search Function
# ----------------------
def search_documents(query, top_k=10):  # Increased to search more documents
    """Search for relevant documents using vector similarity"""
    try:
        embedding_model = init_embedding_model()
        if not embedding_model:
            return []
        
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        qdrant_client = init_qdrant_client()
        if not qdrant_client:
            return []
        
        # Search with higher limit to get more comprehensive results
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=0.3  # Lower threshold to include more results
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
        page_title="موقع المرجع الديني الشيخ محمد السند - دام ظله",
        page_icon="🕌",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load Arabic CSS
    load_arabic_css()
    
    # Header
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند - دام ظله</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">محرك بحث الكتب والاستفتاءات</p>', unsafe_allow_html=True)
    
    # Database Status
    db_status, vector_count = check_database_status()
    if db_status:
        st.markdown(f'<div class="status-box">حالة قاعدة البيانات: <span class="status-active">متصل ✓</span> | عدد المتجهات: {vector_count:,}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box">حالة قاعدة البيانات: <span class="status-inactive">غير متصل ✗</span></div>', unsafe_allow_html=True)
    
    # Search Engine Selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_engine = st.selectbox(
            "اختر محرك البحث الذكي",
            ["OpenAI API", "DeepSeek", "Gemini"],
            index=0,
            key="search_engine"
        )
    
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
                
                # Display API used
                if "api_used" in message:
                    st.markdown(f'<span class="api-used">تم استخدام: {message["api_used"]}</span>', unsafe_allow_html=True)
                
                # Display sources in grid
                if "sources" in message and message["sources"]:
                    st.markdown('<div class="source-container">', unsafe_allow_html=True)
                    cols = st.columns(3)
                    for idx, source in enumerate(message["sources"][:6]):  # Show up to 6 sources
                        with cols[idx % 3]:
                            percentage = source["score"] * 100
                            st.markdown(
                                f'<div class="source-info">📄 المصدر: {source["source"]}<br>التطابق: {percentage:.1f}%</div>', 
                                unsafe_allow_html=True
                            )
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input with larger text area
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create centered columns for input
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        user_question = st.text_area(
            "اسأل سؤالك هنا...",
            placeholder="اكتب سؤالك هنا... مثال: ما هو حكم الصلاة في السفر؟",
            key="user_input",
            height=100
        )
        
        # Centered search button
        st.markdown('<div class="search-button">', unsafe_allow_html=True)
        send_button = st.button("🔍 بحث", type="primary", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process user input
    if send_button and user_question:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Search for relevant documents
        with st.spinner("جاري البحث في جميع الكتب والاستفتاءات..."):
            search_results = search_documents(user_question, top_k=10)
        
        if search_results:
            # Prepare context for AI
            context_texts = []
            sources = []
            
            # Use more results for context
            for result in search_results:
                context_texts.append(result.payload.get('text', ''))
                sources.append({
                    'source': result.payload.get('source', 'مجهول'),
                    'score': result.score
                })
            
            context = "\n\n".join(context_texts)
            
            # Prepare messages for AI with stricter system prompt
            system_prompt = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة الدينية والشرعية باللغة العربية بناءً على كتب واستفتاءات المرجع الديني الشيخ محمد السند فقط.
            
            قواعد صارمة جداً:
            1. استخدم فقط المعلومات الموجودة في النصوص المقدمة - لا تضف أي معلومات من خارج هذه النصوص
            2. إذا لم تجد الإجابة الكاملة في النصوص المقدمة، قل "لم أجد إجابة كافية في المصادر المتاحة"
            3. لا تستنتج أو تخمن أو تضيف معلومات غير موجودة في النصوص
            4. اقتبس من النصوص المقدمة مباشرة عند الإمكان
            5. أجب باللغة العربية الفصحى فقط
            6. كن دقيقاً جداً في نقل الأحكام الشرعية كما هي في المصادر
            7. إذا كانت الإجابة موجودة جزئياً، اذكر ما وجدته واذكر أن هناك تفاصيل أخرى قد تكون مطلوبة
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"المصادر المتاحة من كتب واستفتاءات الشيخ محمد السند:\n{context}\n\nالسؤال: {user_question}\n\nأجب فقط بناءً على المصادر المقدمة أعلاه."}
            ]
            
            # Get response based on selected engine
            with st.spinner(f"جاري تحضير الإجابة باستخدام {search_engine}..."):
                if search_engine == "OpenAI API":
                    bot_response = get_openai_response(messages)
                elif search_engine == "DeepSeek":
                    bot_response = get_deepseek_response(messages)
                else:  # Gemini
                    bot_response = get_gemini_response(messages)
            
            # Add bot response to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "sources": sources[:6],  # Show top 6 sources
                "api_used": search_engine
            })
        else:
            # No relevant documents found
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "عذراً، لم أجد معلومات ذات صلة في كتب واستفتاءات الشيخ محمد السند للإجابة على سؤالك. يرجى إعادة صياغة السؤال أو السؤال عن موضوع آخر.",
                "api_used": search_engine
            })
        
        # Rerun to update chat
        st.rerun()
    
    # Clear chat button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ----------------------
# Run the Application
# ----------------------
if __name__ == "__main__":
    main()
