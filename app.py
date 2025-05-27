import streamlit as st
import time
from datetime import datetime # Not explicitly used, but good to have for potential future features
import json # Not explicitly used, but good to have for potential future features
import requests
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
# import pandas as pd # Pandas was imported but not used. Removed for cleaner imports.
from openai import OpenAI
import google.generativeai as genai

# ----------------------
# Configuration
# ----------------------
# Actual Qdrant URL and API Key from your initial script
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ"
COLLECTION_NAME = "arabic_documents_enhanced" # Or your specific collection name

# --- API Key Management ---
# WARNING: Storing API keys directly in code is a security risk.
# Consider Streamlit Secrets (st.secrets) or environment variables for production.

# Actual API keys from your initial script
OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-14f267781a6f474a9d0ec8240383dae4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_API_KEY = "AIzaSyASlapu6AYYwOQAJJ3v-2FSKHnxIOZoPbY"


# --- Initialize API Clients ---
openai_client = None
# Placeholder to check if the key is still a default placeholder (though we are using actual keys now)
generic_openai_placeholder = "YOUR_OPENAI_API_KEY_PLACEHOLDER"
if OPENAI_API_KEY and OPENAI_API_KEY != generic_openai_placeholder:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
else:
    if OPENAI_API_KEY == generic_openai_placeholder: # Should not happen if keys are correctly inserted
        print("OpenAI API key is a placeholder. OpenAI features will be limited.")
    elif not OPENAI_API_KEY:
        print("OpenAI API key is missing. OpenAI features will be limited.")


gemini_initial_configured = False
generic_gemini_placeholder = "YOUR_GEMINI_API_KEY_PLACEHOLDER"
if GEMINI_API_KEY and GEMINI_API_KEY != generic_gemini_placeholder:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_initial_configured = True
        print("Gemini API configured successfully at startup.")
    except Exception as e:
        print(f"Failed to configure Gemini API at startup: {e}")
else:
    if GEMINI_API_KEY == generic_gemini_placeholder: # Should not happen
        print("Gemini API key is a placeholder at startup. Gemini features will be limited.")
    elif not GEMINI_API_KEY:
        print("Gemini API key is missing at startup. Gemini features will be limited.")

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
        direction: rtl; /* Ensure body direction is RTL */
    }
    /* General Page Elements */
    .main-header { text-align: center; color: #2E8B57; font-family: 'Noto Sans Arabic', sans-serif; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; direction: rtl; }
    .sub-header { text-align: center; color: #666; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.2rem; font-weight: 400; margin-bottom: 1rem; direction: rtl; }
    
    /* Streamlit Widgets Styling */
    .stExpander .stExpanderHeader { font-size: 1.1rem !important; font-weight: 600 !important; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; } 
    .stExpander div[data-testid="stExpanderDetails"] { direction: rtl; } 
    .stExpander .stRadio > label > div > p, .stExpander .stCheckbox > label > span { font-family: 'Noto Sans Arabic', sans-serif !important; } 
    
    .stRadio div[data-testid="stCaptionContainer"] { 
        font-family: 'Noto Sans Arabic', sans-serif !important;
        margin-top: -2px; 
        margin-right: 25px; /* Adjust based on radio button icon */
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
        /* width: 200px; */ /* Let Streamlit handle width with use_container_width or default */
        margin: 0 auto; 
        display: block; 
        font-family: 'Noto Sans Arabic', sans-serif; 
        font-size: 1.1rem; 
        font-weight: 600; 
        border-radius: 8px; 
        transition: background-color 0.2s ease, transform 0.2s ease; 
    }
    div[data-testid="stButton"] > button:hover { transform: translateY(-2px); }

    /* Status and Info Boxes */
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

    /* Chat Area Styling */
    .chat-container { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px 15px 5px 15px; margin: 0.5rem 0; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 500; box-shadow: 0 2px 10px rgba(0,0,0,0.1); word-wrap: break-word; }
    .bot-message { background: linear-gradient(135deg, #5DADE2 0%, #3498DB 100%); color: white; padding: 1rem; border-radius: 15px 15px 15px 5px; margin: 0.5rem 0; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 500; line-height: 1.8; box-shadow: 0 2px 10px rgba(0,0,0,0.1); word-wrap: break-word; }
    
    /* Metadata and Debug Styling */
    .time-taken { font-size: 0.8rem; color: #777; margin-top: 0.3rem; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif;}
    .debug-info { background: #fff3cd; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.85rem; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; border: 1px solid #ffeeba; word-wrap: break-word; }
    .api-used { background: #e3f2fd; color: #1976d2; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.85rem; margin-top: 0.5rem; display: inline-block; font-family: 'Noto Sans Arabic', sans-serif; }
    
    /* Source Display Styling */
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
        overflow-wrap: break-word; /* Ensure long source names wrap */
        word-break: break-word;     /* Ensure long source names wrap */
        min-height: 50px; /* Adjust as needed for consistent height */
    }
    .source-info strong { font-size: 0.8rem; } 
    .source-info:hover { transform: translateY(-2px); box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# Status Check & Info Functions
# ----------------------
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_qdrant_info():
    """Checks Qdrant connection status and retrieves collection information."""
    # generic_qdrant_placeholder = "YOUR_QDRANT_API_KEY_PLACEHOLDER" # Not needed as keys are directly used
    if not QDRANT_API_KEY or not QDRANT_URL: # Basic check if keys are somehow empty
        return {"status": False, "message": "بيانات اتصال Qdrant (URL أو المفتاح) مفقودة.", "details": {}}
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5) # Short timeout for status check
        collection_info = client.get_collection(COLLECTION_NAME)
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
        return {"status": False, "message": f"غير متصل (خطأ: {type(e).__name__})", "details": {}}

@st.cache_data(ttl=300) # Cache for 5 minutes
def check_api_status(api_name):
    """Checks the status of the given LLM API."""
    global gemini_initial_configured # To manage Gemini's one-time configuration

    if api_name == "OpenAI":
        if not openai_client: return False, "التهيئة فشلت أو المفتاح مفقود/غير صالح"
        try:
            openai_client.models.list(limit=1) # Light call to check connectivity
            return True, "نشط ✓"
        except Exception as e:
            print(f"OpenAI API error for status check: {e}")
            return False, f"غير نشط (خطأ: {type(e).__name__})"

    elif api_name == "DeepSeek":
        # generic_deepseek_placeholder = "YOUR_DEEPSEEK_API_KEY_PLACEHOLDER" # Not needed
        if not DEEPSEEK_API_KEY: # Basic check
            return False, "المفتاح مفقود"
        try:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "مرحبا"}], "max_tokens": 1, "stream": False}
            response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=7)
            return (True, "نشط ✓") if response.status_code == 200 else (False, f"خطأ ({response.status_code})")
        except Exception as e:
            print(f"DeepSeek API error for status check: {e}")
            return False, f"غير نشط (خطأ: {type(e).__name__})"

    elif api_name == "Gemini":
        # generic_gemini_placeholder_local = "YOUR_GEMINI_API_KEY_PLACEHOLDER" # Not needed
        if not GEMINI_API_KEY: # Basic check
            return False, "المفتاح مفقود"
        try:
            if not gemini_initial_configured: # Configure if not already done
                 genai.configure(api_key=GEMINI_API_KEY)
                 gemini_initial_configured = True

            model = genai.GenerativeModel('gemini-1.5-flash')
            model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1))
            return True, "نشط ✓"
        except Exception as e:
            print(f"Gemini API error during status check: {e}")
            err_msg = str(e).lower()
            if "api_key_invalid" in err_msg or "permission" in err_msg or "quota" in err_msg or "authentication" in err_msg:
                return False, "خطأ بالمفتاح/الإذن/الحصة"
            return False, f"غير نشط (خطأ: {type(e).__name__})"
    return False, "API غير معروف"

# ----------------------
# Initialize Components Resource
# ----------------------
@st.cache_resource
def init_qdrant_client_resource():
    """Initializes and returns a Qdrant client. Cached for performance."""
    # generic_qdrant_placeholder = "YOUR_QDRANT_API_KEY_PLACEHOLDER" # Not needed
    if not QDRANT_API_KEY or not QDRANT_URL:
        st.error("بيانات اتصال Qdrant (URL أو المفتاح) مفقودة أو غير مهيأة بشكل صحيح.")
        return None
    try:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10) # Longer timeout for operations
    except Exception as e:
        st.error(f"فشل الاتصال بـ Qdrant: {e}")
        return None

@st.cache_resource
def init_embedding_model_resource():
    """Initializes and returns the sentence embedding model. Cached for performance."""
    try:
        return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        st.error(f"فشل تحميل نموذج التضمين (Embeddings): {e}")
        return None

# ----------------------
# Document Search Functions
# ----------------------
def comprehensive_search(query, max_results=50):
    """
    Searches the Qdrant database for documents matching the query.
    Includes a fallback to search by keywords if initial results are sparse.
    """
    embedding_model = init_embedding_model_resource()
    if not embedding_model:
        return [], "فشل تحميل نموذج التضمين للبحث. لا يمكن المتابعة."

    qdrant_c = init_qdrant_client_resource()
    if not qdrant_c:
        return [], "فشل الاتصال بقاعدة بيانات Qdrant. لا يمكن المتابعة."

    try:
        query_embedding = embedding_model.encode([query])[0].tolist()
        search_results = qdrant_c.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=max_results,
            with_payload=True, 
            score_threshold=0.20 
        )
        
        if len(search_results) < 5 and ' ' in query: 
            all_keyword_results = []
            keywords = [kw for kw in query.split() if len(kw) > 2] 

            for keyword in keywords[:3]: 
                keyword_embedding = embedding_model.encode([keyword])[0].tolist()
                keyword_search_results = qdrant_c.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=keyword_embedding,
                    limit=5, 
                    with_payload=True,
                    score_threshold=0.15 
                )
                all_keyword_results.extend(keyword_search_results)
            
            seen_ids = {res.id for res in search_results}
            combined_results = list(search_results) 
            for res in all_keyword_results:
                if res.id not in seen_ids:
                    seen_ids.add(res.id)
                    combined_results.append(res)
            
            search_results = sorted(combined_results, key=lambda x: x.score, reverse=True)[:max_results]
        
        return search_results, f"تم البحث في القاعدة: وجدت {len(search_results)} نتيجة."
    except Exception as e:
        print(f"Error in comprehensive_search: {e}")
        return [], f"خطأ أثناء البحث في قاعدة البيانات: {str(e)}"

# ----------------------
# API Response Functions with Strict Instructions
# ----------------------
def prepare_llm_messages(user_question, context, context_info):
    """
    Prepares the message list for the LLM, including the system prompt and user question with context.
    """
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
    
    user_content = (
        f"السؤال المطروح: {user_question}\n\n"
        f"المصادر المتاحة من قاعدة البيانات فقط (أجب بناءً عليها حصراً):\n{context}\n\n"
        f"معلومات إضافية عن السياق: {context_info}\n\n"
        "التعليمات: يرجى تقديم إجابة بناءً على النصوص أعلاه فقط. إذا لم تكن الإجابة موجودة، وضح ذلك."
    )
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

def get_openai_response(messages, max_tokens=2000):
    """Gets a response from the OpenAI API."""
    if not openai_client:
        return "OpenAI client غير مهيأ. يرجى التحقق من مفتاح API في الإعدادات."
    try:
        model_to_use = "gpt-3.5-turbo"
        response = openai_client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.05, 
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error in get_openai_response: {e}")
        if "context_length_exceeded" in str(e).lower():
            return "النص المرسل طويل جداً لـ OpenAI. يرجى تقليل عمق البحث أو تبسيط السؤال."
        return f"خطأ في الاتصال بـ OpenAI: {str(e)}"

def get_deepseek_response(messages, max_tokens=2000):
    """Gets a response from the DeepSeek API."""
    # generic_deepseek_placeholder = "YOUR_DEEPSEEK_API_KEY_PLACEHOLDER" # Not needed
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API key مفقود في الإعدادات."
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat", 
            "messages": messages,
            "temperature": 0.05,
            "max_tokens": max_tokens,
            "stream": False 
        }
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=90) 
        response.raise_for_status() 
        
        result = response.json()
        if result.get('choices') and result['choices'][0].get('message'):
            return result['choices'][0]['message']['content']
        else:
            print(f"DeepSeek unexpected response structure: {result}")
            return "لم يتمكن DeepSeek من إنشاء رد (استجابة غير متوقعة)."
    except requests.exceptions.Timeout:
        return "انتهت مهلة الانتظار مع DeepSeek. قد يكون النص طويلاً جداً أو الشبكة بطيئة."
    except requests.exceptions.HTTPError as e:
        err_content = e.response.text
        print(f"DeepSeek HTTP error: {e.response.status_code} - {err_content}")
        return f"خطأ في DeepSeek: {e.response.status_code}. التفاصيل: {err_content[:200]}" 
    except Exception as e:
        print(f"Error in get_deepseek_response: {e}")
        return f"خطأ غير متوقع في الاتصال بـ DeepSeek: {str(e)}"

def get_gemini_response(messages, max_tokens=2000):
    """
    Gets a response from the Google Gemini API.
    """
    global gemini_initial_configured
    # generic_gemini_placeholder_local = "YOUR_GEMINI_API_KEY_PLACEHOLDER" # Not needed
    if not GEMINI_API_KEY:
         return "Gemini API key مفقود في الإعدادات."

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
                 current_first_message_content = processed_messages_for_gemini[0]["parts"][0]
                 processed_messages_for_gemini[0]["parts"][0] = f"{system_instruction_text}\n\n---\n\n{current_first_message_content}"
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
            response = chat.send_message(
                current_user_message_content,
                generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens)
            )
        elif processed_messages_for_gemini and processed_messages_for_gemini[0]["role"] == "user":
            response = model.generate_content(
                processed_messages_for_gemini[0]["parts"], 
                generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens)
            )
        else:
            print(f"Unexpected message structure for Gemini: {processed_messages_for_gemini}")
            return "لا توجد رسائل مناسبة لإرسالها إلى Gemini أو أن بنية الرسائل غير متوقعة."

        return response.text
    except Exception as e:
        print(f"Gemini API error in get_gemini_response: {e}")
        err_msg = str(e).lower()
        if "api_key_invalid" in err_msg or "permission" in err_msg or "quota" in err_msg or "authentication" in err_msg:
            return f"خطأ في مفتاح Gemini أو الأذونات أو الحصة: تأكد من صحة المفتاح وصلاحيته."
        if "context length" in err_msg or "token limit" in err_msg or "request payload size" in err_msg:
            return "النص المرسل طويل جداً لـ Gemini. يرجى تقليل عمق البحث أو تبسيط السؤال."
        return f"خطأ في الاتصال بـ Gemini: {str(e)}"

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

        llm_option_names = []
        llm_captions_with_status = []
        llm_apis_to_check = [("DeepSeek", "DeepSeek"), ("OpenAI", "OpenAI"), ("Gemini", "Gemini")]
        
        default_llm_index = 0 
        active_llms = []

        for i, (display_name, internal_name) in enumerate(llm_apis_to_check):
            status_ok, status_msg = check_api_status(internal_name)
            status_indicator_text = status_msg 
            status_class = "radio-label-status-active" if status_ok else "radio-label-status-inactive"
            
            llm_option_names.append(display_name)
            llm_captions_with_status.append(f"<span class='{status_class}'>({status_indicator_text})</span>")
            if status_ok:
                active_llms.append(i)
        
        if active_llms and default_llm_index not in active_llms:
            default_llm_index = active_llms[0]
        elif not active_llms: 
            st.warning("تنبيه: جميع محركات الذكاء الاصطناعي غير نشطة حالياً. قد لا تعمل وظيفة الإجابة.", icon="⚠️")


        if llm_option_names:
            selected_llm = st.radio(
                label="محركات الذكاء الاصطناعي:",
                options=llm_option_names,
                captions=llm_captions_with_status,
                index=default_llm_index,
                horizontal=True,
                key="llm_select_with_captions",
                label_visibility="collapsed" 
            )
        else: 
            selected_llm = st.radio(
                "محركات الذكاء الاصطناعي:",
                ["DeepSeek", "OpenAI", "Gemini"], index=0, horizontal=True, key="llm_select_plain_fb"
            )
            st.caption("لم يتمكن من جلب حالة محركات الذكاء الاصطناعي للخيارات.")
        
        st.markdown("---")
        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem;'>##### حالة قاعدة البيانات (Qdrant):</div>", unsafe_allow_html=True)
        q_info = get_qdrant_info()
        q_status_html_class = "status-active" if q_info['status'] else "status-inactive"
        q_status_text = f'<span class="{q_status_html_class}">{q_info["message"]}</span>'
        st.markdown(f'<div style="display: flex; justify-content: center;"><div class="status-box" style="width: 90%; max-width: 450px;">Qdrant DB: {q_status_text}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 0.5rem; margin-top: 0.5rem;'>مستوى البحث في قاعدة البيانات:</div>", unsafe_allow_html=True)
        search_depth = st.radio(
            "مستوى البحث في قاعدة البيانات:",
            ["بحث سريع (10 نتائج)", "بحث متوسط (25 نتيجة)", "بحث شامل (50 نتيجة)"],
            index=1, horizontal=True, key="search_depth_radio",
            help="يؤثر على عدد النتائج الأولية من قاعدة البيانات قبل إرسالها للتحليل. البحث الشامل قد يستغرق وقتاً أطول.",
            label_visibility="collapsed"
        )
        depth_map = {"بحث سريع (10 نتائج)": 10, "بحث متوسط (25 نتيجة)": 25, "بحث شامل (50 نتيجة)": 50}
        max_db_results = depth_map[search_depth]
        
        show_debug = st.checkbox("إظهار معلومات البحث التفصيلية (للمطورين)", value=False, key="debug_checkbox")

    if q_info['status'] and q_info['details']:
        with st.expander("ℹ️ معلومات عن قاعدة البيانات المتصل بها", expanded=False):
            st.markdown('<div class="collection-info-box">', unsafe_allow_html=True)
            st.markdown(f"<h3>تفاصيل مجموعة: {q_info['details'].get('اسم المجموعة', COLLECTION_NAME)}</h3>", unsafe_allow_html=True)
            for key, value in q_info['details'].items():
                if key != "اسم المجموعة": 
                    st.markdown(f"<p><strong>{key}:</strong> {str(value)}</p>", unsafe_allow_html=True)
            st.markdown("<p><small>ملاحظة: هذه معلومات عامة عن المجموعة. لا يعرض هذا التطبيق حاليًا الملفات المصدر الفردية التي تم استخدامها لإنشاء هذه المجموعة.</small></p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    elif not q_info['status']:
         st.warning(f"تعذر الاتصال بقاعدة بيانات Qdrant: {q_info['message']}. وظيفة البحث لن تعمل.", icon="⚠️")


    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_display_container = st.container()
    with chat_display_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else: 
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
                if "api_used" in message:
                    st.markdown(f'<span class="api-used">تم استخدام: {message["api_used"]}</span>', unsafe_allow_html=True)
                if "time_taken" in message:
                    st.markdown(f'<div class="time-taken">⏱️ زمن الاستجابة: {message["time_taken"]:.2f} ثانية</div>', unsafe_allow_html=True)
                if show_debug and "debug_info" in message:
                    st.markdown(f'<div class="debug-info">🔍 معلومات تفصيلية: {message["debug_info"]}</div>', unsafe_allow_html=True)
                
                if "sources" in message and message["sources"]:
                    st.markdown("<div style='text-align: right; width: 100%; margin-top: 0.5rem;'><strong>المصادر المرجعية من قاعدة البيانات:</strong></div>", unsafe_allow_html=True)
                    st.markdown('<div class="source-container">', unsafe_allow_html=True)
                    sources_to_show = message["sources"][:9]
                    num_sources_to_display = len(sources_to_show)
                    cols_per_row = 3
                    
                    for j in range(0, num_sources_to_display, cols_per_row):
                        row_sources = sources_to_show[j : j + cols_per_row]
                        cols = st.columns(len(row_sources)) 

                        for k, source_item in enumerate(row_sources):
                            with cols[k]:
                                percentage = source_item["score"] * 100
                                source_name = source_item["source"] if source_item.get("source") else f'مصدر {j+k+1}'
                                st.markdown(f'<div class="source-info" title="المصدر: {source_name}\nالتطابق: {percentage:.1f}%">📄 <strong>{source_name}</strong><br>تطابق: {percentage:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin-top: 1.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)

    input_area_spacer1, input_area_main, input_area_spacer2 = st.columns([0.2, 2.6, 0.2]) 
    with input_area_main:
        user_question = st.text_area(
            "اسأل سؤالك هنا...",
            placeholder="اكتب سؤالك هنا المتعلق بكتب واستفتاءات الشيخ السند...",
            key="user_input_main",
            height=120,
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="search-button-container">', unsafe_allow_html=True)
        send_button = st.button("🔍 بحث وإجابة", type="primary", use_container_width=False, key="send_button_main") 
        st.markdown('</div>', unsafe_allow_html=True)

    if send_button and user_question.strip():
        st.session_state.messages.append({"role": "user", "content": user_question.strip()})
        start_time = time.perf_counter()

        search_spinner_msg = f"جاري البحث في قاعدة البيانات ({max_db_results} نتيجة كحد أقصى)..."
        with st.spinner(search_spinner_msg):
            search_results, db_debug_info = comprehensive_search(user_question.strip(), max_results=max_db_results)

        if search_results:
            context_texts, sources_for_llm = [], []
            total_chars_for_llm, max_chars_llm_context = 0, 25000 
            
            for i, result in enumerate(search_results):
                text = result.payload.get('text', '')
                source_payload_name = result.payload.get('source', f'وثيقة رقم {result.id[:8]}') 
                
                if text:
                    max_individual_text_len = 1500 
                    truncated_text = text[:max_individual_text_len] + ("..." if len(text) > max_individual_text_len else "")
                    
                    if total_chars_for_llm + len(truncated_text) < max_chars_llm_context:
                        context_texts.append(f"[نص {i+1} من '{source_payload_name}']: {truncated_text}")
                        sources_for_llm.append({'source': source_payload_name, 'score': result.score, 'id': result.id})
                        total_chars_for_llm += len(truncated_text)
                    else:
                        num_omitted = len(search_results) - i
                        context_texts.append(f"\n[ملاحظة: تم اقتصار النصوص المرسلة للتحليل بسبب طول السياق. تم العثور على {num_omitted} نص إضافي متعلق بالموضوع ولكن لم يتم إرساله.]")
                        db_debug_info += f" | تم اقتصار السياق، {num_omitted} نصوص إضافية لم ترسل."
                        break
            
            context_for_llm = "\n\n---\n\n".join(context_texts)
            llm_context_info = f"تم البحث في قاعدة البيانات والعثور على {len(search_results)} نص. تم إرسال أكثر {len(sources_for_llm)} نص صلة للتحليل (إجمالي ~{total_chars_for_llm // 1000} ألف حرف)."
            
            llm_messages = prepare_llm_messages(user_question.strip(), context_for_llm, llm_context_info)
            
            bot_response_content = ""
            llm_spinner_msg = f"جاري تحليل النتائج وتوليد الإجابة باستخدام {selected_llm}..."
            with st.spinner(llm_spinner_msg):
                if selected_llm == "OpenAI":
                    bot_response_content = get_openai_response(llm_messages)
                elif selected_llm == "DeepSeek":
                    bot_response_content = get_deepseek_response(llm_messages)
                elif selected_llm == "Gemini":
                    bot_response_content = get_gemini_response(llm_messages)
                else:
                    bot_response_content = "المحرك المحدد غير معروف. يرجى الاختيار من القائمة."
            
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            
            st.session_state.messages.append({
                "role": "assistant", "content": bot_response_content,
                "sources": sources_for_llm, "api_used": selected_llm,
                "time_taken": time_taken,
                "debug_info": f"{db_debug_info} | {llm_context_info}"
            })
        else: 
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            no_results_message = "لم أجد أي معلومات متعلقة بسؤالك في قاعدة بيانات كتب واستفتاءات الشيخ محمد السند حالياً. يرجى محاولة صياغة السؤال بشكل مختلف أو استخدام كلمات مفتاحية أخرى."
            st.session_state.messages.append({
                "role": "assistant", "content": no_results_message, "api_used": selected_llm, 
                "time_taken": time_taken,
                "debug_info": db_debug_info if 'db_debug_info' in locals() and db_debug_info else "لم يتم العثور على نتائج من قاعدة البيانات."
            })
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
    # Initial check for critical configurations (placeholders are now actual keys)
    # These checks are less critical now but can catch if a key is accidentally cleared.
    if not QDRANT_API_KEY or not QDRANT_URL:
        st.error("يرجى التأكد من تعيين QDRANT_URL و QDRANT_API_KEY في الشيفرة المصدرية.")
        st.warning("لن تعمل وظائف البحث في قاعدة البيانات بدون بيانات اتصال Qdrant صحيحة.", icon="⚠️")
    
    if not OPENAI_API_KEY and not DEEPSEEK_API_KEY and not GEMINI_API_KEY:
        st.info("لم يتم العثور على مفاتيح LLM API. يرجى تحديثها لتفعيل جميع محركات الذكاء الاصطناعي.", icon="ℹ️")

    main()
