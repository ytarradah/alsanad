import streamlit as st
import time
from datetime import datetime
import json
import requests
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai

# ----------------------
# Configuration
# ----------------------
# Ensure this matches your Qdrant setup and the collection used by your upload script
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ"
COLLECTION_NAME = "arabic_documents_enhanced" # CONFIRMED: This is the collection name you are using

# --- API Key Management ---
OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-14f267781a6f474a9d0ec8240383dae4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_API_KEY = "AIzaSyASlapu6AYYwOQAJJ3v-2FSKHnxIOZoPbY"


# --- Initialize API Clients ---
openai_client = None
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_PLACEHOLDER": # Basic check
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
else:
    print("OpenAI API key is missing or a placeholder. OpenAI features will be limited.")


gemini_initial_configured = False
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_PLACEHOLDER": # Basic check
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_initial_configured = True
        print("Gemini API configured successfully at startup.")
    except Exception as e:
        print(f"Failed to configure Gemini API at startup: {e}")
else:
    print("Gemini API key is missing or a placeholder. Gemini features will be limited.")

# ----------------------
# Arabic CSS Styling
# ----------------------
def load_arabic_css():
    """Loads custom CSS for Arabic text, RTL layout, and overall styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700&display=swap');
    body { 
        font-family: 'Noto Sans Arabic', sans-serif; 
        direction: rtl; 
    }
    .main-header { text-align: center; color: #2E8B57; font-family: 'Noto Sans Arabic', sans-serif; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; direction: rtl; }
    .sub-header { text-align: center; color: #666; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.2rem; font-weight: 400; margin-bottom: 1rem; direction: rtl; }
    .stExpander .stExpanderHeader { font-size: 1.1rem !important; font-weight: 600 !important; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; } 
    .stExpander div[data-testid="stExpanderDetails"] { direction: rtl; } 
    .stExpander .stRadio > label > div > p, .stExpander .stCheckbox > label > span { font-family: 'Noto Sans Arabic', sans-serif !important; } 
    .stRadio div[data-testid="stCaptionContainer"] { 
        font-family: 'Noto Sans Arabic', sans-serif !important;
        margin-top: -2px; 
        margin-right: 25px; 
        direction: rtl;
    }
    .stTextArea > div > div > textarea { 
        direction: rtl; 
        text-align: right; 
        font-family: 'Noto Sans Arabic', sans-serif; 
        font-size: 1.1rem; 
        min-height: 100px !important; 
        border-radius: 10px; 
        border: 1px solid #ccc; 
    }
    .search-button-container { text-align: center; margin-top: 1rem; margin-bottom: 1rem; }
    div[data-testid="stButton"] > button { 
        margin: 0 auto; 
        display: block; 
        font-family: 'Noto Sans Arabic', sans-serif; 
        font-size: 1.1rem; 
        font-weight: 600; 
        border-radius: 8px; 
        transition: background-color 0.2s ease, transform 0.2s ease; 
    }
    div[data-testid="stButton"] > button:hover { transform: translateY(-2px); }
    .status-box { 
        background: #f0f2f6; 
        padding: 0.5rem; 
        border-radius: 8px; 
        text-align: center; 
        font-family: 'Noto Sans Arabic', sans-serif; 
        direction: rtl; 
        border: 1px solid #e0e0e0; 
        font-size: 0.9rem; 
        margin-bottom: 0.5rem;
    }
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
    .api-used { background: #e3f2fd; color: #1976d2; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.85rem; margin-top: 0.5rem; display: inline-block; font-family: 'Noto Sans Arabic', sans-serif; }
    .source-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-top: 1rem; direction: rtl; } 
    .source-info { 
        background: #f0f2f6; 
        padding: 0.25rem 0.4rem; 
        border-radius: 6px; 
        font-size: 0.75rem; 
        color: #555; 
        direction: rtl; 
        text-align: right; 
        font-family: 'Noto Sans Arabic', sans-serif; 
        border: 1px solid #ddd; 
        transition: transform 0.2s ease-in-out; 
        display: flex; 
        flex-direction: column; 
        justify-content: center;
        overflow-wrap: break-word; 
        word-break: break-word;     
        min-height: 50px; 
    }
    .source-info strong { font-size: 0.8rem; } 
    .source-info:hover { transform: translateY(-2px); box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
    div[data-testid="stSpinner"] { 
        direction: rtl; 
        width: 100%;
    }
    div[data-testid="stSpinner"] > div { 
        display: flex;
        flex-direction: row-reverse; 
        align-items: center;
        justify-content: flex-start; 
        text-align: right !important; 
        width: auto; 
        margin: 0 auto 0 0; 
    }
    div[data-testid="stSpinner"] p { 
        font-size: 1.15em !important; 
        font-family: 'Noto Sans Arabic', sans-serif !important;
        margin-right: 8px !important; 
        text-align: right !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# Status Check & Info Functions
# ----------------------
@st.cache_data(ttl=300)
def get_qdrant_info():
    if not QDRANT_API_KEY or not QDRANT_URL: 
        return {"status": False, "message": "بيانات اتصال Qdrant (URL أو المفتاح) مفقودة.", "details": {}}
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5) 
        collection_info = client.get_collection(COLLECTION_NAME) # Uses the global COLLECTION_NAME
        return {
            "status": True,
            "message": f"متصل ✓ | النقاط: {collection_info.points_count:,}",
            "details": {
                "اسم المجموعة": COLLECTION_NAME,
                "عدد النقاط (Vectors)": collection_info.points_count,
                "حالة الفهرسة": str(collection_info.status),
                "تهيئة Vector Params": str(collection_info.config.params),
                "تهيئة Quantization": str(collection_info.config.quantization_config) if collection_info.config.quantization_config else "لا يوجد",
            }
        }
    except Exception as e:
        print(f"Qdrant connection error during status check: {e}")
        if "Not found: Collection" in str(e) or "NOT_FOUND" in str(e).upper():
             return {"status": False, "message": f"غير متصل (المجموعة '{COLLECTION_NAME}' غير موجودة أو خطأ: {type(e).__name__})", "details": {}}
        return {"status": False, "message": f"غير متصل (خطأ: {type(e).__name__})", "details": {}}

@st.cache_data(ttl=300)
def check_api_status(api_name):
    global gemini_initial_configured 
    if api_name == "OpenAI":
        if not openai_client: return False, "التهيئة فشلت أو المفتاح مفقود/غير صالح"
        try: openai_client.models.list(limit=1); return True, "نشط ✓" 
        except Exception as e: print(f"OpenAI API error for status check: {e}"); return False, f"غير نشط (خطأ: {type(e).__name__})"
    elif api_name == "DeepSeek":
        if not DEEPSEEK_API_KEY: return False, "المفتاح مفقود"
        try:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "مرحبا"}], "max_tokens": 1, "stream": False}
            response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=7)
            return (True, "نشط ✓") if response.status_code == 200 else (False, f"خطأ ({response.status_code})") 
        except Exception as e: print(f"DeepSeek API error for status check: {e}"); return False, f"غير نشط (خطأ: {type(e).__name__})"
    elif api_name == "Gemini":
        if not GEMINI_API_KEY: return False, "المفتاح مفقود"
        try:
            if not gemini_initial_configured: 
                 genai.configure(api_key=GEMINI_API_KEY); gemini_initial_configured = True
            model = genai.GenerativeModel('gemini-1.5-flash') 
            model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1))
            return True, "نشط ✓"
        except Exception as e:
            print(f"Gemini API error during status check: {e}")
            err_msg = str(e).lower()
            if "api_key_invalid" in err_msg or "permission" in err_msg or "quota" in err_msg or "authentication" in err_msg: return False, "خطأ بالمفتاح/الإذن/الحصة" 
            return False, f"غير نشط ({type(e).__name__})"
    return False, "API غير معروف"

# ----------------------
# Initialize Components Resource
# ----------------------
@st.cache_resource
def init_qdrant_client_resource():
    if not QDRANT_API_KEY or not QDRANT_URL:
        st.error("بيانات اتصال Qdrant (URL أو المفتاح) مفقودة أو غير مهيأة بشكل صحيح.")
        return None
    try: return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10) 
    except Exception as e: st.error(f"فشل الاتصال بـ Qdrant: {e}"); return None

# CRITICAL CHANGE HERE: Ensure the correct embedding model is used
@st.cache_resource
def init_embedding_model_resource():
    """Initializes and returns the sentence embedding model. Cached for performance."""
    try:
        # This model produces 768 dimensions, matching your upload script and Qdrant collection expectation
        model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        print(f"Initializing embedding model: {model_name}") # Log model initialization
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"فشل تحميل نموذج التضمين (Embeddings) '{model_name}': {e}")
        return None

# ----------------------
# Document Search Functions
# ----------------------
def comprehensive_search(query, max_results=50):
    embedding_model = init_embedding_model_resource()
    if not embedding_model:
        return [], "فشل تحميل نموذج التضمين للبحث. لا يمكن المتابعة."

    qdrant_c = init_qdrant_client_resource()
    if not qdrant_c:
        return [], "فشل الاتصال بقاعدة بيانات Qdrant. لا يمكن المتابعة."

    try:
        print(f"Generating embedding for query: '{query}' using model: {type(embedding_model)}") # Log query embedding
        query_embedding = embedding_model.encode([query])[0].tolist()
        print(f"Query embedding dimension: {len(query_embedding)}") # Log dimension of query vector

        current_score_threshold = 0.20 

        search_results = qdrant_c.search(
            collection_name=COLLECTION_NAME, # Uses the global COLLECTION_NAME
            query_vector=query_embedding,
            limit=max_results,
            with_payload=True, 
            score_threshold=current_score_threshold
        )
        
        search_info = f"البحث الأساسي: وجدت {len(search_results)} نتيجة بعتبة {current_score_threshold} لـ '{COLLECTION_NAME}'."

        if len(search_results) < 5 and ' ' in query: 
            search_info += " | نتائج قليلة، محاولة البحث بالكلمات المفتاحية..."
            all_keyword_results = []
            keywords = [kw for kw in query.split() if len(kw) > 2] 
            for keyword in keywords[:3]: 
                keyword_embedding = embedding_model.encode([keyword])[0].tolist()
                keyword_search_results = qdrant_c.search(
                    collection_name=COLLECTION_NAME, query_vector=keyword_embedding, 
                    limit=5, with_payload=True, score_threshold=0.15 
                )
                all_keyword_results.extend(keyword_search_results)
            
            seen_ids = {res.id for res in search_results}
            combined_results = list(search_results) 
            for res in all_keyword_results:
                if res.id not in seen_ids: 
                    seen_ids.add(res.id); combined_results.append(res)
            search_results = sorted(combined_results, key=lambda x: x.score, reverse=True)[:max_results]
            search_info += f" | بعد الكلمات المفتاحية: {len(search_results)} نتيجة إجمالية."
        
        return search_results, search_info
    except Exception as e:
        print(f"Error in comprehensive_search: {e}")
        error_content = str(e)
        if "vector dimension error" in error_content.lower() or "expected dim" in error_content.lower():
            st.error(f"خطأ في أبعاد المتجه! تأكد من تطابق نموذج التضمين مع المجموعة '{COLLECTION_NAME}'. التفاصيل: {error_content}")
        elif "Not found: Collection" in error_content or "NOT_FOUND" in error_content.upper():
            st.error(f"خطأ فادح: المجموعة '{COLLECTION_NAME}' غير موجودة في Qdrant. يرجى التحقق من اسم المجموعة وعملية الرفع.")
        return [], f"خطأ أثناء البحث في قاعدة البيانات: {error_content}"

# ----------------------
# API Response Functions with Strict Instructions
# ----------------------
def prepare_llm_messages(user_question, context, context_info):
    system_prompt = """أنت مساعد للبحث في كتب واستفتاءات الشيخ محمد السند فقط.
قواعد حتمية لا يمكن تجاوزها:
1. أجب فقط من النصوص المعطاة أدناه ("المصادر المتاحة") - لا استثناءات.
2. إذا لم تجد الإجابة الكاملة في النصوص، قل بوضوح: "لم أجد إجابة كافية في المصادر المتاحة بخصوص هذا السؤال."
3. ممنوع منعاً باتاً إضافة أي معلومة من خارج النصوص المعطاة. لا تستخدم معلوماتك العامة أو معرفتك السابقة.
4. اقتبس مباشرة من النصوص عند الإجابة قدر الإمكان، مع الإشارة إلى المصدر إذا كان متاحاً في النص (مثال: [نص ١]).
5. إذا وجدت إجابة جزئية، اذكرها وأوضح أنها غير كاملة أو تغطي جانباً من السؤال.
6. هدفك هو تقديم إجابة دقيقة وموثوقة بناءً على ما هو متوفر في النصوص فقط.
7. إذا كانت النصوص لا تحتوي على إجابة، لا تحاول استنتاج أو تخمين الإجابة.
تذكر: أي معلومة ليست في النصوص أدناه = لا تذكرها أبداً. كن دقيقاً ومقتصراً على المصادر."""
    user_content = (f"السؤال المطروح: {user_question}\n\nالمصادر المتاحة من قاعدة البيانات فقط (أجب بناءً عليها حصراً):\n{context}\n\nمعلومات إضافية عن السياق: {context_info}\n\nالتعليمات: يرجى تقديم إجابة بناءً على النصوص أعلاه فقط. إذا لم تكن الإجابة موجودة، وضح ذلك.")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

def get_openai_response(messages, max_tokens=2000):
    if not openai_client: return "OpenAI client غير مهيأ."
    try:
        response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=0.05, max_tokens=max_tokens)
        return response.choices[0].message.content
    except Exception as e: 
        print(f"OpenAI API error: {e}")
        return f"خطأ OpenAI: {str(e)}"

def get_deepseek_response(messages, max_tokens=2000):
    if not DEEPSEEK_API_KEY: return "DeepSeek API key مفقود."
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "deepseek-chat", "messages": messages, "temperature": 0.05, "max_tokens": max_tokens, "stream": False}
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'] if result.get('choices') and result['choices'][0].get('message') else "لم يتمكن DeepSeek من إنشاء رد."
    except requests.exceptions.Timeout:
        return "انتهت مهلة الانتظار مع DeepSeek."
    except requests.exceptions.HTTPError as e:
        err_content = e.response.text if e.response else "No response content"
        print(f"DeepSeek HTTP error: {e.response.status_code if e.response else 'N/A'} - {err_content}")
        return f"خطأ في DeepSeek: {e.response.status_code if e.response else 'N/A'}. التفاصيل: {err_content[:200]}"
    except Exception as e: 
        print(f"DeepSeek API error: {e}")
        return f"خطأ DeepSeek: {str(e)}"

def get_gemini_response(messages, max_tokens=2000):
    global gemini_initial_configured
    if not GEMINI_API_KEY: return "Gemini API key مفقود."
    try:
        if not gemini_initial_configured: 
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_initial_configured = True
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        processed_messages_for_gemini = []
        system_instruction_text = None

        if messages and messages[0]["role"] == "system":
            system_instruction_text = messages[0]["content"]
            for msg in messages[1:]: 
                role = "user" if msg["role"] == "user" else "model" 
                processed_messages_for_gemini.append({"role": role, "parts": [msg["content"]]})
        else: 
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                processed_messages_for_gemini.append({"role": role, "parts": [msg["content"]]})

        if system_instruction_text:
             if processed_messages_for_gemini and processed_messages_for_gemini[0]["role"] == "user":
                 processed_messages_for_gemini[0]["parts"][0] = f"{system_instruction_text}\n\n---\n\n{processed_messages_for_gemini[0]['parts'][0]}"
             else: 
                 processed_messages_for_gemini.insert(0, {"role": "user", "parts": [system_instruction_text]})
        
        if not processed_messages_for_gemini:
            return "لا توجد رسائل صالحة لإرسالها إلى Gemini بعد معالجة التعليمات النظامية."

        if len(processed_messages_for_gemini) > 1 and processed_messages_for_gemini[-1]["role"] == "user":
            chat_history = processed_messages_for_gemini[:-1]
            current_user_message_content = processed_messages_for_gemini[-1]["parts"][0]
            for entry in chat_history: 
                if entry["role"] not in ["user", "model"]:
                    print(f"Invalid role in Gemini chat history: {entry['role']}") 
                    return "خطأ داخلي: دور غير صالح في سجل محادثة Gemini."
            
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(current_user_message_content, generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens))
        elif processed_messages_for_gemini and processed_messages_for_gemini[0]["role"] == "user": 
            response = model.generate_content(
                processed_messages_for_gemini[0]["parts"], 
                generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens)
            )
        else: 
            return "بنية رسائل Gemini غير متوقعة أو لا يوجد محتوى لإرساله."

        return response.text
    except Exception as e: 
        print(f"Gemini API error: {e}")
        return f"خطأ Gemini: {str(e)}"

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="المرجع السند - محرك بحث", page_icon="🕌", layout="wide", initial_sidebar_state="collapsed")
    load_arabic_css()
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند - دام ظله</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">محرك بحث الكتب والاستفتاءات الذكي</p>', unsafe_allow_html=True)

    with st.expander("⚙️ إعدادات البحث وحالة الأنظمة", expanded=True):
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>اختر محرك الذكاء الاصطناعي للمعالجة:</div>", unsafe_allow_html=True)
        llm_option_names, llm_captions_with_status, active_llms = [], [], []
        llm_apis_to_check = [("DeepSeek", "DeepSeek"), ("OpenAI", "OpenAI"), ("Gemini", "Gemini")]
        default_llm_index = 0
        for i, (display_name, internal_name) in enumerate(llm_apis_to_check):
            status_ok, status_msg = check_api_status(internal_name)
            llm_option_names.append(display_name)
            llm_captions_with_status.append(f"<span class='{'radio-label-status-active' if status_ok else 'radio-label-status-inactive'}'>({status_msg})</span>")
            if status_ok: active_llms.append(i)
        if active_llms and default_llm_index not in active_llms: default_llm_index = active_llms[0]
        elif not active_llms: st.warning("تنبيه: جميع محركات الذكاء الاصطناعي غير نشطة.", icon="⚠️")
        selected_llm = st.radio("محركات الذكاء الاصطناعي:", llm_option_names, captions=llm_captions_with_status, index=default_llm_index, horizontal=True, key="llm_select", label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>##### حالة قاعدة البيانات (Qdrant):</div>", unsafe_allow_html=True)
        q_info = get_qdrant_info()
        q_status_text = f'<span class="{"status-active" if q_info["status"] else "status-inactive"}">{q_info["message"]}</span>'
        st.markdown(f'<div style="display: flex; justify-content: center;"><div class="status-box" style="width: 90%; max-width: 450px;">Qdrant DB: {q_status_text}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem; margin-top: 0.5rem;'>مستوى البحث في قاعدة البيانات:</div>", unsafe_allow_html=True)
        search_depth_options = ["بحث سريع (10 نتائج)", "بحث متوسط (25 نتيجة)", "بحث شامل (50 نتيجة)"]
        search_depth = st.radio("مستوى البحث:", search_depth_options, index=1, horizontal=True, key="search_depth_radio", label_visibility="collapsed")
        max_db_results = {"بحث سريع (10 نتائج)": 10, "بحث متوسط (25 نتيجة)": 25, "بحث شامل (50 نتيجة)": 50}[search_depth]
        show_debug = st.checkbox("إظهار معلومات البحث التفصيلية (للمطورين)", value=True, key="debug_checkbox")

    if q_info['status'] and q_info['details']:
        with st.expander("ℹ️ معلومات عن قاعدة البيانات المتصل بها", expanded=False):
            st.markdown(f"<div class='collection-info-box'><h3>تفاصيل مجموعة: {q_info['details'].get('اسم المجموعة', COLLECTION_NAME)}</h3>" +
                        "".join([f"<p><strong>{k}:</strong> {v}</p>" for k, v in q_info['details'].items() if k != "اسم المجموعة"]) +
                        "<p><small>ملاحظة: معلومات عامة عن المجموعة.</small></p></div>", unsafe_allow_html=True)
    elif not q_info['status']: st.warning(f"تعذر الاتصال بـ Qdrant: {q_info['message']}.", icon="⚠️")

    if "messages" not in st.session_state: st.session_state.messages = []
    chat_display_container = st.container()
    with chat_display_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            msg_class = "user-message" if message["role"] == "user" else "bot-message"
            st.markdown(f'<div class="{msg_class}">{"👤" if message["role"] == "user" else "🤖"} {message["content"]}</div>', unsafe_allow_html=True)
            if message["role"] == "assistant":
                if "api_used" in message: st.markdown(f'<span class="api-used">تم استخدام: {message["api_used"]}</span>', unsafe_allow_html=True)
                if "time_taken" in message: st.markdown(f'<div class="time-taken">⏱️ زمن الاستجابة: {message["time_taken"]:.2f} ثانية</div>', unsafe_allow_html=True)
                if show_debug and "debug_info" in message: st.markdown(f'<div class="debug-info">🔍 معلومات تفصيلية: {message["debug_info"]}</div>', unsafe_allow_html=True)
                if "sources" in message and message["sources"]:
                    st.markdown("<div style='text-align: right; width: 100%; margin-top: 0.5rem;'><strong>المصادر المرجعية:</strong></div><div class='source-container'>", unsafe_allow_html=True)
                    for j in range(0, min(len(message["sources"]), 9), 3): # Display up to 9 sources
                        cols = st.columns(3)
                        for k_idx, k_src in enumerate(message["sources"][j:j+3]):
                            with cols[k_idx]: 
                                st.markdown(f'<div class="source-info" title="المصدر: {k_src.get("source", "N/A")}\nالتطابق: {k_src.get("score", 0)*100:.1f}%">📄 <strong>{k_src.get("source", "N/A")}</strong><br>تطابق: {k_src.get("score",0)*100:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin-top: 1.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
    _, input_area_main, _ = st.columns([0.2, 2.6, 0.2])
    with input_area_main:
        user_question = st.text_area("اسأل سؤالك هنا...", placeholder="اكتب سؤالك هنا...", key="user_input_main", height=120, label_visibility="collapsed")
        st.markdown('<div class="search-button-container">', unsafe_allow_html=True)
        send_button = st.button("🔍 بحث وإجابة", type="primary", use_container_width=False, key="send_button_main")
        st.markdown('</div>', unsafe_allow_html=True)

    if send_button and user_question.strip():
        st.session_state.messages.append({"role": "user", "content": user_question.strip()})
        start_time = time.perf_counter()
        with st.spinner(f"جاري البحث في قاعدة البيانات ({max_db_results} نتيجة)..."):
            search_results, db_debug_info = comprehensive_search(user_question.strip(), max_results=max_db_results)
        
        if search_results:
            context_texts, sources_for_llm = [], []
            total_chars, max_chars_context = 0, 25000 # Max characters for LLM context
            for i, result in enumerate(search_results):
                # CORRECTED: Convert result.id to string before slicing
                source_name = result.payload.get('source', f'وثيقة {str(result.id)[:6]}') if result.id else result.payload.get('source', f'وثيقة غير محددة {i}')
                text = result.payload.get('text', '')
                
                if text:
                    truncated_text = text[:1500] + ("..." if len(text) > 1500 else "") # Truncate individual texts
                    if total_chars + len(truncated_text) < max_chars_context:
                        context_texts.append(f"[نص {i+1} من '{source_name}']: {truncated_text}")
                        sources_for_llm.append({'source': source_name, 'score': result.score, 'id': result.id})
                        total_chars += len(truncated_text)
                    else:
                        # Context limit reached
                        context_texts.append(f"\n[ملاحظة: تم اقتصار النصوص المرسلة للتحليل. تم العثور على {len(search_results)-i} نص إضافي متعلق بالموضوع ولكن لم يتم إرساله.]")
                        db_debug_info += f" | تم اقتصار السياق، {len(search_results)-i} نصوص إضافية لم ترسل."
                        break
            
            context_for_llm = "\n\n---\n\n".join(context_texts)
            llm_context_info = f"تم إرسال {len(sources_for_llm)} نص للتحليل (إجمالي ~{total_chars//1000} ألف حرف)."
            llm_messages = prepare_llm_messages(user_question.strip(), context_for_llm, llm_context_info)
            bot_response_content = ""
            with st.spinner(f"جاري تحليل النتائج وتوليد الإجابة باستخدام {selected_llm}..."):
                if selected_llm == "OpenAI": bot_response_content = get_openai_response(llm_messages)
                elif selected_llm == "DeepSeek": bot_response_content = get_deepseek_response(llm_messages)
                elif selected_llm == "Gemini": bot_response_content = get_gemini_response(llm_messages)
                else: bot_response_content = "المحرك المحدد غير معروف. يرجى الاختيار من القائمة."
            
            time_taken = time.perf_counter() - start_time
            st.session_state.messages.append({"role": "assistant", "content": bot_response_content, "sources": sources_for_llm, "api_used": selected_llm, "time_taken": time_taken, "debug_info": f"{db_debug_info} | {llm_context_info}"})
        else:
            time_taken = time.perf_counter() - start_time
            no_results_msg = "لم أجد أي معلومات متعلقة بسؤالك في قاعدة بيانات كتب واستفتاءات الشيخ محمد السند حالياً. يرجى محاولة صياغة السؤال بشكل مختلف أو استخدام كلمات مفتاحية أخرى."
            st.session_state.messages.append({"role": "assistant", "content": no_results_msg, "api_used": selected_llm, "time_taken": time_taken, "debug_info": db_debug_info})
        
        st.rerun()
    elif send_button and not user_question.strip(): 
        st.toast("يرجى إدخال سؤال في مربع النص.", icon="📝")

    with input_area_main:
        if st.button("🗑️ مسح المحادثة بالكامل", use_container_width=True, key="clear_chat_button", type="secondary"):
            st.session_state.messages = []
            st.toast("تم مسح المحادثة.", icon="🗑️")
            time.sleep(0.5) 
            st.rerun()

# ----------------------
# Run the Application
# ----------------------
if __name__ == "__main__":
    # Basic check for essential configurations
    if not all([QDRANT_API_KEY, QDRANT_URL]):
        st.error("معلومات اتصال QDRANT (URL أو API_KEY) مفقودة. يرجى التحقق من الإعدادات في بداية الملف.")
        st.warning("وظائف البحث في قاعدة البيانات لن تعمل بدون هذه المعلومات.", icon="⚠️")
    
    # Check for at least one LLM API key
    if not any([OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_PLACEHOLDER", 
                DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "YOUR_DEEPSEEK_API_KEY_PLACEHOLDER", 
                GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_PLACEHOLDER"]):
        st.info("لم يتم العثور على مفاتيح LLM API صالحة. يرجى تحديثها لتفعيل جميع محركات الذكاء الاصطناعي.", icon="ℹ️")

    main()
