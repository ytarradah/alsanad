import streamlit as st
import time
import json
import requests
import asyncio
import concurrent.futures

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
# Ultra Fast CSS
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
    
    .source-compact {
        background: #f8f9fa;
        padding: 0.6rem;
        border-radius: 10px;
        margin-top: 0.6rem;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        border-left: 3px solid #28a745;
        font-size: 0.9rem;
    }
    
    .source-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.3rem;
        display: inline-block;
    }
    
    .ai-badge {
        background: #6f42c1;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    
    .status-fast {
        background: #28a745;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        margin: 0.3rem;
        direction: rtl;
        text-align: center;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        display: inline-block;
        min-width: 140px;
    }
    
    .status-fast.offline {
        background: #dc3545;
    }
    
    .status-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    
    .loading-message {
        text-align: center;
        color: #2E8B57;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0;
        direction: rtl;
        padding: 1rem;
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        border-radius: 12px;
        border: 2px solid #28a745;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
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
# SUPER FAST Status Check (no caching)
# ----------------------
def quick_status_check():
    """Ultra quick status check - 1 second max each"""
    status = {"qdrant": False, "deepseek": False, "gemini": False}
    
    # Test Qdrant (1 sec timeout)
    try:
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}", 
                               headers={"api-key": QDRANT_API_KEY}, timeout=1)
        status["qdrant"] = response.status_code == 200
    except:
        status["qdrant"] = False
    
    # Test DeepSeek (1 sec timeout)
    try:
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions",
                               headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                               json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                               timeout=1)
        status["deepseek"] = response.status_code == 200
    except:
        status["deepseek"] = False
    
    # Test Gemini (1 sec timeout)
    try:
        response = requests.post(f"{GEMINI_BASE_URL}?key={GEMINI_API_KEY}",
                               json={"contents": [{"parts": [{"text": "hi"}]}]}, timeout=1)
        status["gemini"] = response.status_code == 200
    except:
        status["gemini"] = False
    
    return status

# ----------------------
# ENHANCED Database Search with better accuracy
# ----------------------
def enhanced_search(query):
    """More accurate database search with better matching"""
    try:
        # Quick embedding
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            query_embedding = model.encode([query])[0].tolist()
        except:
            return None
        
        # Enhanced search with more results for better accuracy
        search_data = {
            "vector": query_embedding, 
            "limit": 5,  # More results for better accuracy
            "with_payload": True,
            "score_threshold": 0.3  # Only return results above 30% similarity
        }
        
        response = requests.post(f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
                               headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
                               json=search_data, timeout=8)
        
        if response.status_code == 200:
            results = response.json().get("result", [])
            # Filter results with decent scores
            filtered_results = [r for r in results if r.get('score', 0) > 0.3]
            return filtered_results if filtered_results else results[:3]
        return None
        
    except Exception as e:
        print(f"Search error: {e}")
        return None

# ----------------------
# IMPROVED AI Response with better context handling
# ----------------------
def get_enhanced_ai_response(context, query):
    """Get better AI response with improved context handling"""
    
    # Better prompt for more accurate responses
    prompt = f"""أنت مساعد ذكي للمرجع الديني الشيخ محمد السند. اقرأ النصوص التالية بعناية:

النصوص من قاعدة البيانات:
{context}

السؤال: {query}

تعليمات:
1. ابحث في النصوص أعلاه عن إجابة للسؤال
2. إذا وجدت معلومات متعلقة بالسؤال، أجب بناءً عليها
3. إذا لم تجد إجابة واضحة، قل "لم أجد معلومات كافية في المراجع المتاحة"
4. أجب باللغة العربية واستشهد من النصوص
5. كن دقيقاً ومفصلاً في الإجابة"""
    
    def try_deepseek():
        try:
            response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions",
                                   headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                                   json={"model": "deepseek-chat", 
                                        "messages": [{"role": "user", "content": prompt}], 
                                        "temperature": 0.1,  # Lower temperature for more accuracy
                                        "max_tokens": 600},  # More tokens for detailed answers
                                   timeout=10)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'], "DeepSeek"
        except Exception as e:
            print(f"DeepSeek error: {e}")
        return None, None
    
    def try_gemini():
        try:
            response = requests.post(f"{GEMINI_BASE_URL}?key={GEMINI_API_KEY}",
                                   json={"contents": [{"parts": [{"text": prompt}]}],
                                        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600}},
                                   timeout=10)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text'], "Gemini"
        except Exception as e:
            print(f"Gemini error: {e}")
        return None, None
    
    # Try both with longer timeout for better accuracy
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_deepseek = executor.submit(try_deepseek)
        future_gemini = executor.submit(try_gemini)
        
        try:
            for future in concurrent.futures.as_completed([future_deepseek, future_gemini], timeout=12):
                result, ai_name = future.result()
                if result:
                    return result, ai_name
        except:
            pass
    
    return "لم أتمكن من الحصول على إجابة من النظم المتاحة", "Error"

# ----------------------
# MAIN ULTRA FAST APP
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
    
    # Super fast status check
    if "last_status_check" not in st.session_state or time.time() - st.session_state.last_status_check > 30:
        st.session_state.status = quick_status_check()
        st.session_state.last_status_check = time.time()
    
    status = st.session_state.status
    
    # Centered and bigger status display
    st.markdown('<div class="status-container">', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="status-fast {'offline' if not status['qdrant'] else ''}">🟢 قاعدة البيانات</div>
    <div class="status-fast {'offline' if not status['deepseek'] else ''}">🚀 DeepSeek</div>
    <div class="status-fast {'offline' if not status['gemini'] else ''}">🤖 Gemini</div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
            
            # AI indicator
            if "ai_used" in message and message["ai_used"] != "Error":
                st.markdown(f'<div class="ai-badge">{"🚀" if message["ai_used"] == "DeepSeek" else "🤖"} {message["ai_used"]}</div>', unsafe_allow_html=True)
            
            # Compact sources (remove duplicates)
            if "sources" in message and message["sources"]:
                unique_sources = {}
                for source in message["sources"]:
                    src_name = source["source"].replace('.txt', '').replace('.pdf', '').replace('.docx', '')
                    if src_name not in unique_sources or source["percentage"] > unique_sources[src_name]:
                        unique_sources[src_name] = source["percentage"]
                
                source_text = " ".join([f'<span class="source-badge">{perc}% {name}</span>' 
                                       for name, perc in unique_sources.items()])
                st.markdown(f'<div class="source-compact">📚 {source_text}</div>', unsafe_allow_html=True)
    
    # Input
    st.markdown('<h3 style="text-align: center; direction: rtl; font-family: \'Noto Sans Arabic\', sans-serif;">💭 اكتب سؤالك هنا:</h3>', unsafe_allow_html=True)
    
    with st.form(key="fast_chat", clear_on_submit=True):
        user_question = st.text_area("", placeholder="اكتب سؤالك واضغط Ctrl+Enter...", height=120)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button("إرسال سريع ⚡", type="primary", use_container_width=True)
    
    # Process
    if submit and user_question and user_question.strip():
        if not status["qdrant"]:
            st.error("⚠️ قاعدة البيانات غير متاحة")
            return
        
        if not (status["deepseek"] or status["gemini"]):
            st.error("⚠️ لا يوجد ذكاء اصطناعي متاح")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_question.strip()})
        
        # Show loading message for search
        search_placeholder = st.empty()
        search_placeholder.markdown('<div class="loading-message">🔍 جاري البحث في قاعدة البيانات...</div>', unsafe_allow_html=True)
        
        # Enhanced search with better accuracy
        search_results = enhanced_search(user_question.strip())
        search_placeholder.empty()  # Remove search message
        
        if search_results:
            # Show loading message for AI response
            ai_placeholder = st.empty()
            ai_placeholder.markdown('<div class="loading-message">🤖 جاري تحضير الإجابة من المراجع...</div>', unsafe_allow_html=True)
            
            # Better context preparation
            context_texts = []
            sources = []
            
            for result in search_results:
                payload = result.get("payload", {})
                text = payload.get('text', '')
                if text:
                    context_texts.append(text)  # Keep full text for better accuracy
                    sources.append({
                        'source': payload.get('source', 'مجهول'),
                        'percentage': min(100, int(result.get('score', 0.0) * 100))
                    })
            
            if context_texts:
                context = "\n\n---\n\n".join(context_texts)
                
                # Get enhanced AI response
                response, ai_used = get_enhanced_ai_response(context, user_question.strip())
                ai_placeholder.empty()  # Remove AI loading message
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources,
                    "ai_used": ai_used
                })
            else:
                ai_placeholder.empty()
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "لا توجد معلومات مناسبة في قاعدة البيانات",
                    "ai_used": "System"
                })
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "لم أتمكن من البحث في قاعدة البيانات أو لا توجد نتائج مطابقة",
                "ai_used": "System"
            })
        
        st.rerun()
    
    # Quick management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ مسح"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📊 عدد"):
            st.info(f"الرسائل: {len(st.session_state.messages)}")

if __name__ == "__main__":
    main()
