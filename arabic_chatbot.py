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
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ"
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
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        direction: rtl;
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
        padding: 0.5rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #666;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    
    .file-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        direction: rtl;
        text-align: center;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
    
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
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
# Get File Statistics
# ----------------------
def get_file_statistics():
    """Get statistics about uploaded files"""
    try:
        qdrant_client = init_qdrant_client()
        if not qdrant_client:
            return {}
        
        # Get collection info
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        total_vectors = collection_info.vectors_count or 0
        
        # Get sample of points to analyze files
        if total_vectors > 0:
            sample_points = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=True
            )[0]
            
            # Extract file statistics
            files = {}
            for point in sample_points:
                source = point.payload.get('source', 'Unknown')
                if source not in files:
                    files[source] = {
                        'chunks': 0,
                        'file_type': point.payload.get('file_type', ''),
                        'upload_time': point.payload.get('upload_time', '')
                    }
                files[source]['chunks'] += 1
            
            return {
                'total_vectors': total_vectors,
                'total_files': len(files),
                'files': files
            }
        else:
            return {'total_vectors': 0, 'total_files': 0, 'files': {}}
            
    except Exception as e:
        st.error(f"فشل في الحصول على الإحصائيات: {e}")
        return {}

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
    
    # Header
    st.markdown('<h1 class="main-header">🤖 المساعد الذكي للوثائق العربية</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.markdown("### 🔧 الأدوات")
    page = st.sidebar.selectbox(
        "اختر الصفحة",
        ["💬 المحادثة", "📁 إدارة الملفات", "📊 الإحصائيات"]
    )
    
    if page == "💬 المحادثة":
        chat_page()
    elif page == "📁 إدارة الملفات":
        files_page()
    elif page == "📊 الإحصائيات":
        statistics_page()

# ----------------------
# Chat Page
# ----------------------
def chat_page():
    st.markdown("### 💬 تحدث مع وثائقك")
    
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
                        st.markdown(f'<div class="source-info">📄 المصدر: {source["source"]} | النقاط: {source["score"]:.2f}</div>', unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "اسأل سؤالك هنا...",
            placeholder="مثال: ما هو موضوع الوثيقة الأولى؟",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("إرسال 📤", type="primary")
    
    # Process user input
    if send_button and user_question:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Search for relevant documents
        with st.spinner("جاري البحث في الوثائق..."):
            search_results = search_documents(user_question)
        
        if search_results:
            # Prepare context for DeepSeek
            context_texts = []
            sources = []
            
            for result in search_results:
                context_texts.append(result.payload.get('text', ''))
                sources.append({
                    'source': result.payload.get('source', 'مجهول'),
                    'score': result.score
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
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"السياق من الوثائق:\n{context}\n\nالسؤال: {user_question}"}
            ]
            
            # Get response from DeepSeek
            with st.spinner("جاري تحضير الإجابة..."):
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
                "content": "عذراً، لم أجد معلومات ذات صلة في الوثائق المحملة للإجابة على سؤالك."
            })
        
        # Rerun to update chat
        st.rerun()
    
    # Clear chat button
    if st.button("مسح المحادثة 🗑️"):
        st.session_state.messages = []
        st.rerun()

# ----------------------
# Files Management Page
# ----------------------
def files_page():
    st.markdown("### 📁 إدارة الملفات المحملة")
    
    # Get file statistics
    stats = get_file_statistics()
    
    if stats.get('total_files', 0) > 0:
        st.markdown(f'<div class="file-stats">📊 إجمالي الملفات: {stats["total_files"]} | إجمالي الأجزاء: {stats["total_vectors"]}</div>', unsafe_allow_html=True)
        
        # Display files table
        files_data = []
        for filename, info in stats['files'].items():
            files_data.append({
                'اسم الملف': filename,
                'نوع الملف': info['file_type'],
                'عدد الأجزاء': info['chunks'],
                'تاريخ الرفع': info['upload_time']
            })
        
        df = pd.DataFrame(files_data)
        st.dataframe(df, use_container_width=True)
        
        # File operations
        st.markdown("### 🔧 عمليات الملفات")
        
        selected_file = st.selectbox("اختر ملف للعمليات", list(stats['files'].keys()))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("عرض معاينة 👁️"):
                preview_file(selected_file)
        
        with col2:
            if st.button("البحث في الملف 🔍"):
                search_in_file(selected_file)
        
        with col3:
            if st.button("حذف الملف ❌", type="secondary"):
                delete_file(selected_file)
    else:
        st.info("لا توجد ملفات محملة بعد. استخدم أداة الرفع لإضافة وثائقك.")

# ----------------------
# Statistics Page
# ----------------------
def statistics_page():
    st.markdown("### 📊 إحصائيات النظام")
    
    stats = get_file_statistics()
    
    if stats.get('total_files', 0) > 0:
        # Overall statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("إجمالي الملفات", stats['total_files'])
        
        with col2:
            st.metric("إجمالي الأجزاء", stats['total_vectors'])
        
        with col3:
            avg_chunks = stats['total_vectors'] / stats['total_files'] if stats['total_files'] > 0 else 0
            st.metric("متوسط الأجزاء لكل ملف", f"{avg_chunks:.1f}")
        
        # File type distribution
        file_types = {}
        for info in stats['files'].values():
            file_type = info['file_type'] or 'غير محدد'
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        st.markdown("### 📈 توزيع أنواع الملفات")
        
        # Create a simple bar chart using Streamlit
        if file_types:
            st.bar_chart(file_types)
        
        # Database health check
        st.markdown("### 🏥 صحة قاعدة البيانات")
        
        try:
            qdrant_client = init_qdrant_client()
            if qdrant_client:
                collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                st.success("✅ قاعدة البيانات تعمل بشكل طبيعي")
                st.info(f"حالة المجموعة: {collection_info.status}")
            else:
                st.warning("⚠️ مشكلة في الاتصال بقاعدة البيانات")
        except Exception as e:
            st.error(f"❌ خطأ في قاعدة البيانات: {e}")
    else:
        st.info("لا توجد بيانات لعرض الإحصائيات.")

# ----------------------
# Helper Functions
# ----------------------
def preview_file(filename):
    """Preview file content"""
    try:
        qdrant_client = init_qdrant_client()
        if not qdrant_client:
            return
        
        # Get some chunks from this file
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={"must": [{"key": "source", "match": {"value": filename}}]},
            limit=3,
            with_payload=True
        )[0]
        
        if results:
            st.markdown(f"### 👁️ معاينة الملف: {filename}")
            
            for i, point in enumerate(results):
                preview_text = point.payload.get('text_preview', point.payload.get('text', ''))[:300]
                st.markdown(f'<div class="arabic-text"><strong>الجزء {i+1}:</strong><br>{preview_text}...</div>', unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning("لم يتم العثور على محتوى للملف")
            
    except Exception as e:
        st.error(f"فشل في عرض الملف: {e}")

def search_in_file(filename):
    """Search within a specific file"""
    search_query = st.text_input(f"البحث في {filename}")
    
    if search_query:
        try:
            # This would implement file-specific search
            st.info("ميزة البحث في الملف المحدد قيد التطوير")
        except Exception as e:
            st.error(f"فشل في البحث: {e}")

def delete_file(filename):
    """Delete a file from the database"""
    if st.button(f"تأكيد حذف {filename}", type="secondary"):
        try:
            qdrant_client = init_qdrant_client()
            if qdrant_client:
                # Delete all points with this source
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector={"filter": {"must": [{"key": "source", "match": {"value": filename}}]}}
                )
                st.success(f"تم حذف الملف {filename} بنجاح")
                st.rerun()
            else:
                st.error("فشل في الاتصال بقاعدة البيانات")
        except Exception as e:
            st.error(f"فشل في حذف الملف: {e}")

# ----------------------
# Run the Application
# ----------------------
if __name__ == "__main__":
    main()