import streamlit as st
import time
from datetime import datetime
import json
import requests
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import re
from typing import List, Dict, Any

# ----------------------
# Configuration
# ----------------------
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ"
COLLECTION_NAME = "arabic_documents_enhanced"

# API Keys
OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-14f267781a6f474a9d0ec8240383dae4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_API_KEY = "AIzaSyASlapu6AYYwOQAJJ3v-2FSKHnxIOZoPbY"

# Initialize API Clients
openai_client = None
generic_openai_placeholder = "YOUR_OPENAI_KEY_PLACEHOLDER_EXAMPLE" 
if OPENAI_API_KEY and OPENAI_API_KEY != generic_openai_placeholder: 
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"OpenAI API error in get_openai_response: {e}")
        if "context_length_exceeded" in str(e).lower(): 
            return "النص طويل جداً لـ OpenAI. يرجى تقليل عمق البحث أو محاولة سؤال أبسط."
        return f"خطأ في OpenAI: {str(e)}"

def get_deepseek_response(messages, max_tokens=2000):
    generic_deepseek_placeholder = "YOUR_DEEPSEEK_KEY_PLACEHOLDER_EXAMPLE"
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == generic_deepseek_placeholder: 
        return "DeepSeek API key غير مهيأ أو غير صالح."
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "deepseek-chat", "messages": messages, "temperature": 0.05, "max_tokens": max_tokens}
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=90) 
        response.raise_for_status() 
        result = response.json()
        return result['choices'][0]['message']['content'] if result.get('choices') else "لم يتمكن DeepSeek من إنشاء رد."
    except requests.exceptions.Timeout:
        return "انتهت مهلة الانتظار مع DeepSeek. قد يكون النص طويلاً جداً أو الشبكة بطيئة."
    except requests.exceptions.HTTPError as e:
        err_content = e.response.text 
        print(f"DeepSeek HTTP error: {e.response.status_code} - {err_content}")
        return f"خطأ في DeepSeek: {e.response.status_code}. التفاصيل: {err_content[:200]}" 
    except Exception as e:
        print(f"Error in get_deepseek_response: {e}")
        return f"خطأ غير متوقع في DeepSeek: {str(e)}"

def get_gemini_response(messages, max_tokens=2000):
    global gemini_initial_configured
    generic_gemini_placeholder = "YOUR_GEMINI_KEY_PLACEHOLDER_EXAMPLE"
    if not GEMINI_API_KEY or GEMINI_API_KEY == generic_gemini_placeholder:
         return "Gemini API key غير مهيأ أو غير صالح."
    try:
        if not gemini_initial_configured:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_initial_configured = True 
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        processed_messages_for_gemini = []
        system_instruction_text = None

        if messages[0]["role"] == "system":
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
                 processed_messages_for_gemini[0]["parts"][0] = system_instruction_text + "\n\n---\n\n" + processed_messages_for_gemini[0]["parts"][0]
             else: 
                 processed_messages_for_gemini.insert(0, {"role": "user", "parts": [system_instruction_text]})
        
        if not processed_messages_for_gemini:
            return "لا توجد رسائل صالحة لإرسالها إلى Gemini بعد معالجة التعليمات النظامية."

        if len(processed_messages_for_gemini) > 1:
            chat_history = processed_messages_for_gemini[:-1]
            current_user_message_content = processed_messages_for_gemini[-1]["parts"][0]
            for entry in chat_history: 
                if entry["role"] not in ["user", "model"]:
                    print(f"Invalid role in Gemini chat history: {entry['role']}") 
                    return "خطأ داخلي: دور غير صالح في سجل محادثة Gemini."
            
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(current_user_message_content, generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens))
        elif processed_messages_for_gemini: 
            response = model.generate_content(
                processed_messages_for_gemini[0]["parts"][0], 
                generation_config=genai.types.GenerationConfig(temperature=0.05, max_output_tokens=max_tokens)
            )
        else: 
            return "لا توجد رسائل لإرسالها إلى Gemini."

        return response.text
    except Exception as e:
        print(f"Gemini API error in get_gemini_response: {e}")
        err_msg = str(e).lower()
        if "api_key_invalid" in err_msg or "permission" in err_msg or "quota" in err_msg or "authentication" in err_msg: 
            return f"خطأ في مفتاح Gemini أو الأذونات أو الحصة: {err_msg}"
        if "context length" in err_msg or "token limit" in err_msg: 
            return "النص طويل جداً لـ Gemini. يرجى تقليل عمق البحث."
        return f"خطأ في Gemini: {str(e)}"

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="المرجع السند - محرك بحث", page_icon="🕌", layout="wide", initial_sidebar_state="collapsed")
    load_arabic_css() 
    
    st.markdown('<h1 class="main-header">موقع المرجع الديني الشيخ محمد السند - دام ظله</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">محرك بحث الكتب والاستفتاءات الذكي مع ذاكرة المحادثة</p>', unsafe_allow_html=True)

    with st.expander("⚙️ إعدادات البحث وحالة الأنظمة", expanded=True):
        st.markdown("<div style='text-align: right; margin-bottom: 1rem;'><strong>اختر محرك الذكاء الاصطناعي للمعالجة:</strong></div>", unsafe_allow_html=True)

        # Enhanced API status display
        col1, col2, col3 = st.columns(3)
        llm_apis_to_check = [("DeepSeek", "DeepSeek"), ("OpenAI", "OpenAI"), ("Gemini", "Gemini")]
        api_statuses = {}
        
        for i, (display_name, internal_name) in enumerate(llm_apis_to_check):
            status_ok, status_msg = check_api_status(internal_name)
            api_statuses[display_name] = status_ok
            
            with [col1, col2, col3][i]:
                status_class = "status-active" if status_ok else "status-inactive"
                status_text = "متاح ✓" if status_ok else "غير متاح ✗"
                st.markdown(f'<div class="status-box {status_class}">{display_name}<br>{status_text}</div>', unsafe_allow_html=True)

        # Radio selection with better styling
        available_apis = [name for name, status in api_statuses.items() if status]
        if available_apis:
            selected_llm = st.radio(
                "اختر المحرك:", 
                available_apis, 
                index=0, 
                horizontal=True,
                key="llm_select_enhanced"
            )
        else:
            st.error("لا توجد محركات ذكاء اصطناعي متاحة حالياً!")
            selected_llm = "DeepSeek"  # fallback
        
        st.markdown("---") 
        
        # Database status
        st.markdown("<div style='text-align: right;'><strong>حالة قاعدة البيانات:</strong></div>", unsafe_allow_html=True)
        q_info = get_qdrant_info() 
        q_status_class = "status-active" if q_info['status'] else "status-inactive"
        q_status_text = q_info["message"]
        st.markdown(f'<div class="status-box {q_status_class}">Qdrant DB: {q_status_text}</div>', unsafe_allow_html=True)

        # Search depth options
        st.markdown("<div style='text-align: right; margin-top: 1rem;'><strong>مستوى البحث في قاعدة البيانات:</strong></div>", unsafe_allow_html=True)
        search_depth = st.radio(
            "مستوى البحث:",
            ["بحث سريع (15 نتائج)", "بحث متوسط (30 نتيجة)", "بحث شامل (50 نتيجة)", "بحث عميق (75 نتيجة)"],
            index=1, 
            horizontal=True, 
            key="search_depth_radio_enhanced"
        )
        depth_map = {
            "بحث سريع (15 نتائج)": 15, 
            "بحث متوسط (30 نتيجة)": 30, 
            "بحث شامل (50 نتيجة)": 50,
            "بحث عميق (75 نتيجة)": 75
        }
        max_db_results = depth_map[search_depth]
        
        # Advanced options
        col_a, col_b = st.columns(2)
        with col_a:
            show_debug = st.checkbox("إظهار معلومات البحث التفصيلية", value=False)
        with col_b:
            use_memory = st.checkbox("استخدام ذاكرة المحادثة", value=True, help="يساعد في الأسئلة المتتابعة والمرجعية")

    # Database info section
    if q_info['status'] and q_info['details']:
        with st.expander("ℹ️ معلومات عن قاعدة البيانات المتصل بها", expanded=False):
            st.markdown('<div class="collection-info-box">', unsafe_allow_html=True)
            for key, value in q_info['details'].items():
                st.markdown(f"<p><strong>{key}:</strong> {str(value)}</p>", unsafe_allow_html=True)
            st.markdown("<p><small>ملاحظة: هذه معلومات عامة عن المجموعة. تم تحسين خوارزمية البحث للحصول على نتائج أكثر دقة.</small></p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state: 
        st.session_state.messages = []
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    # Chat display
    chat_display_container = st.container() 
    with chat_display_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.messages): 
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else: 
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
                
                # Memory info display
                if use_memory and "used_memory" in message and message["used_memory"]:
                    st.markdown('<div class="memory-info">🧠 تم استخدام سياق المحادثة السابق</div>', unsafe_allow_html=True)
                
                if "api_used" in message:
                    st.markdown(f'<span class="api-used">تم استخدام: {message["api_used"]}</span>', unsafe_allow_html=True)
                if "time_taken" in message:
                    st.markdown(f'<div class="time-taken">⏱️ زمن الاستجابة: {message["time_taken"]:.2f} ثانية</div>', unsafe_allow_html=True)
                if show_debug and "debug_info" in message:
                    st.markdown(f'<div class="debug-info">🔍 {message["debug_info"]}</div>', unsafe_allow_html=True)
                
                if "sources" in message and message["sources"]:
                    st.markdown("<div style='text-align: right; width: 100%; margin-top: 1rem;'><strong>المصادر المرجعية من قاعدة البيانات:</strong></div>", unsafe_allow_html=True)
                    st.markdown('<div class="source-container">', unsafe_allow_html=True)
                    sources_to_show = message["sources"][:9] 
                    for j in range(0, len(sources_to_show), 3): 
                        cols = st.columns(3) 
                        for k in range(3): 
                            if j + k < len(sources_to_show): 
                                source = sources_to_show[j+k]
                                percentage = source["score"] * 100
                                source_name = source["source"] if source["source"] else f'مصدر {j+k+1}'
                                with cols[k]: 
                                    st.markdown(f'<div class="source-info">📄 <strong>{source_name}</strong><br>تطابق: {percentage:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True) 
        st.markdown('</div>', unsafe_allow_html=True) 

    st.markdown("<hr>", unsafe_allow_html=True) 
    
    # Input area
    input_area_spacer1, input_area_main, input_area_spacer2 = st.columns([0.5, 3, 0.5])
    with input_area_main:
        user_question = st.text_area(
            "اسأل سؤالك هنا...", 
            placeholder="اكتب سؤالك هنا... يمكنك الإشارة إلى الأسئلة السابقة باستخدام كلمات مثل 'هذا' أو 'المذكور سابقاً'", 
            key="user_input_enhanced", 
            height=120
        )
        
        col_send, col_clear = st.columns([2, 1])
        with col_send:
            send_button = st.button("🔍 بحث وإجابة", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ مسح المحادثة", use_container_width=True):
                st.session_state.messages = []
                st.session_state.search_history = []
                st.rerun()

    # Main search logic
    if send_button and user_question:
        st.session_state.messages.append({"role": "user", "content": user_question}) 
        start_time = time.perf_counter() 
        
        # Get conversation context for memory
        conversation_context = ""
        used_memory = False
        if use_memory and len(st.session_state.messages) > 1:
            conversation_context = get_conversation_context(st.session_state.messages[:-1])
            if conversation_context:
                used_memory = True
                user_question = enhance_query_with_context(user_question, conversation_context)
        
        # Enhanced search
        search_spinner_msg = f"جاري البحث المُحسَّن في قاعدة البيانات (حتى {max_db_results} نتيجة)..."
        with st.spinner(search_spinner_msg):
            search_results, db_debug_info = enhanced_hybrid_search(user_question, max_results=max_db_results)
        
        if search_results: 
            context_texts, sources_for_llm = [], []
            total_chars_for_llm, max_chars_llm_context = 0, 35000  # Increased limit
            
            for i, result in enumerate(search_results):
                text = result.payload.get('text', '')
                source_payload_name = result.payload.get('source', f'مصدر رقم {i+1}')
                if text:
                    max_ind_text_len = 2500  # Increased individual text length
                    text = text[:max_ind_text_len] + "..." if len(text) > max_ind_text_len else text
                    if total_chars_for_llm + len(text) < max_chars_llm_context: 
                        context_texts.append(f"[نص {i+1} من '{source_payload_name}']: {text}")
                        sources_for_llm.append({'source': source_payload_name, 'score': result.score})
                        total_chars_for_llm += len(text)
                    else: 
                        if (rem := len(search_results) - i) > 0: 
                            context_texts.append(f"\n[ملاحظة: تم اقتصار النصوص المرسلة للتحليل. تم العثور على {rem} نص إضافي متعلق بالموضوع.]")
                        break 
            
            context_for_llm = "\n\n---\n\n".join(context_texts)
            llm_context_info = f"تم البحث المُحسَّن والعثور على {len(search_results)} نص. تم إرسال أكثر {len(sources_for_llm)} نص صلة للتحليل."
            
            # Prepare messages with memory
            llm_messages = prepare_llm_messages_with_memory(
                st.session_state.messages[-1]["content"],  # Original question without context enhancement
                context_for_llm, 
                llm_context_info,
                conversation_context if used_memory else ""
            )
            
            bot_response_content = "" 
            llm_spinner_msg = f"جاري تحليل النتائج وتوليد الإجابة باستخدام {selected_llm}..."
            with st.spinner(llm_spinner_msg): 
                if selected_llm == "OpenAI":
                    bot_response_content = get_openai_response(llm_messages)
                elif selected_llm == "DeepSeek":
                    bot_response_content = get_deepseek_response(llm_messages)
                elif selected_llm == "Gemini":
                    bot_response_content = get_gemini_response(llm_messages)
            
            end_time = time.perf_counter() 
            time_taken = end_time - start_time 
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response_content,
                "sources": sources_for_llm, 
                "api_used": selected_llm,
                "time_taken": time_taken,
                "debug_info": f"{db_debug_info} | {llm_context_info}",
                "used_memory": used_memory
            })
        else: 
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            no_results_message = f"لم أجد أي معلومات متعلقة بسؤالك في قاعدة بيانات كتب واستفتاءات الشيخ محمد السند. تم استخدام البحث المُحسَّن ولكن لم تُعثر على نتائج مطابقة.\n\nاقتراحات:\n• جرب صياغة السؤال بشكل مختلف\n• استخدم كلمات مفتاحية أكثر عمومية\n• تأكد من كتابة السؤال باللغة العربية"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": no_results_message, 
                "api_used": selected_llm,
                "time_taken": time_taken,
                "debug_info": db_debug_info if 'db_debug_info' in locals() else "لم يتم العثور على نتائج من البحث المُحسَّن.",
                "used_memory": used_memory
            })
        st.rerun() 

# ----------------------
# Run the Application
# ----------------------
if __name__ == "__main__":
    generic_qdrant_placeholder = "YOUR_QDRANT_API_KEY_PLACEHOLDER"
    if QDRANT_API_KEY == generic_qdrant_placeholder: 
        st.error("يرجى تعيين مفتاح QDRANT_API_KEY الخاص بك في الشيفرة المصدرية.")
        st.warning("لن تعمل وظائف البحث في قاعدة البيانات بدون مفتاح Qdrant صحيح.")
    main()"OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")

gemini_initial_configured = False
generic_gemini_placeholder = "YOUR_GEMINI_KEY_PLACEHOLDER_EXAMPLE"
if GEMINI_API_KEY and GEMINI_API_KEY != generic_gemini_placeholder: 
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_initial_configured = True 
        print("Gemini API configured successfully at startup.")
    except Exception as e:
        print(f"Failed to configure Gemini API at startup: {e}")

# ----------------------
# Enhanced Arabic CSS Styling
# ----------------------
def load_arabic_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Noto Sans Arabic', sans-serif; }
    .main-header { text-align: center; color: #2E8B57; font-family: 'Noto Sans Arabic', sans-serif; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; direction: rtl; }
    .sub-header { text-align: center; color: #666; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.2rem; font-weight: 400; margin-bottom: 1rem; direction: rtl; }
    .stExpander .stExpanderHeader { font-size: 1.1rem !important; font-weight: 600 !important; font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; } 
    .stExpander div[data-testid="stExpanderDetails"] { direction: rtl; } 
    .stExpander .stRadio > label > div > p, .stExpander .stCheckbox > label > span { font-family: 'Noto Sans Arabic', sans-serif !important; } 
    
    .status-box { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0.6rem; 
        border-radius: 10px; 
        text-align: center; 
        font-family: 'Noto Sans Arabic', sans-serif; 
        direction: rtl; 
        border: 1px solid #dee2e6; 
        font-size: 0.95rem; 
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-active { color: #28a745; font-weight: bold; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-color: #28a745; } 
    .status-inactive { color: #dc3545; font-weight: bold; background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border-color: #dc3545; } 
    
    .stRadio div[data-testid="stCaptionContainer"] { 
        font-family: 'Noto Sans Arabic', sans-serif !important;
        margin-top: -2px; 
        margin-right: 25px; 
    }
    .radio-label-status-active { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724 !important; 
        font-weight: 600 !important; 
        font-size: 0.85em !important;
        padding: 2px 8px;
        border-radius: 15px;
        border: 1px solid #28a745;
    }
    .radio-label-status-inactive { 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24 !important; 
        font-weight: 600 !important; 
        font-size: 0.85em !important;
        padding: 2px 8px;
        border-radius: 15px;
        border: 1px solid #dc3545;
    }

    .time-taken { font-size: 0.8rem; color: #777; margin-top: 0.3rem; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif;}
    .debug-info { background: #fff3cd; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.85rem; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; border: 1px solid #ffeeba; }
    
    .chat-container { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 15px 15px 5px 15px; margin: 0.5rem 0; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 500; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .bot-message { background: linear-gradient(135deg, #5DADE2 0%, #3498DB 100%); color: white; padding: 1rem; border-radius: 15px 15px 15px 5px; margin: 0.5rem 0; direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 500; line-height: 1.8; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    
    .source-container { 
        display: grid; 
        grid-template-columns: repeat(3, 1fr); 
        gap: 10px; 
        margin-top: 1rem; 
        direction: rtl; 
    } 
    .source-info { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0.4rem 0.6rem; 
        border-radius: 8px; 
        font-size: 0.8rem; 
        color: #495057; 
        direction: rtl; 
        text-align: right; 
        font-family: 'Noto Sans Arabic', sans-serif; 
        border: 1px solid #dee2e6; 
        transition: all 0.3s ease-in-out; 
        display: flex; 
        flex-direction: column; 
        justify-content: center;
        overflow-wrap: break-word;
        word-break: break-word; 
        min-height: 60px;
        max-height: 80px;
    }
    .source-info strong { font-size: 0.85rem; color: #212529; } 
    .source-info:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-color: #2196f3;
    }
    
    .api-used { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); color: #0d47a1; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.85rem; margin-top: 0.5rem; display: inline-block; font-family: 'Noto Sans Arabic', sans-serif; border: 1px solid #2196f3; }
    .stTextArea > div > div > textarea { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; min-height: 100px !important; border-radius: 10px; border: 2px solid #ccc; transition: border-color 0.3s ease; }
    .stTextArea > div > div > textarea:focus { border-color: #2E8B57 !important; box-shadow: 0 0 0 2px rgba(46, 139, 87, 0.2); }
    
    .search-button-container { text-align: center; margin-top: 1rem; margin-bottom: 1rem; }
    div[data-testid="stButton"] > button { width: 200px; margin: 0 auto; display: block; font-family: 'Noto Sans Arabic', sans-serif; font-size: 1.1rem; font-weight: 600; border-radius: 25px; transition: all 0.3s ease; background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%); }
    div[data-testid="stButton"] > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3); }
    
    .collection-info-box { direction: rtl; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top:1rem; margin-bottom: 1.5rem; border: 1px solid #ced4da; }
    .collection-info-box h3 { font-family: 'Noto Sans Arabic', sans-serif; text-align:right; color: #495057; }
    .collection-info-box p {font-family: 'Noto Sans Arabic', sans-serif; text-align:right; margin-bottom: 0.3rem;}
    
    .memory-info {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
        border: 1px solid #ffb74d;
        color: #e65100;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# Status Check & Info Functions
# ----------------------
@st.cache_data(ttl=300) 
def get_qdrant_info():
    generic_qdrant_placeholder = "YOUR_QDRANT_API_KEY_PLACEHOLDER"
    if not QDRANT_API_KEY or QDRANT_API_KEY == generic_qdrant_placeholder: 
        return {"status": False, "message": "مفتاح Qdrant غير مهيأ", "details": {}}
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5)
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
        print(f"Qdrant connection error: {e}")
        return {"status": False, "message": f"غير متصل ({type(e).__name__})", "details": {}}

@st.cache_data(ttl=300) 
def check_api_status(api_name):
    global gemini_initial_configured 
    if api_name == "OpenAI":
        if not openai_client: return False, "التهيئة فشلت أو المفتاح مفقود"
        try: openai_client.models.list(); return True, "نشط" 
        except Exception as e: print(f"OpenAI API error for status check: {e}"); return False, f"غير نشط"
    
    elif api_name == "DeepSeek":
        generic_deepseek_placeholder = "YOUR_DEEPSEEK_KEY_PLACEHOLDER_EXAMPLE"
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == generic_deepseek_placeholder: 
            return False, "المفتاح غير صالح أو مفقود"
        try:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "مرحبا"}], "max_tokens": 5}
            r = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data, timeout=7)
            return (True, "نشط") if r.status_code == 200 else (False, f"غير نشط") 
        except Exception as e: print(f"DeepSeek API error for status check: {e}"); return False, f"غير نشط"
    
    elif api_name == "Gemini":
        generic_gemini_placeholder = "YOUR_GEMINI_KEY_PLACEHOLDER_EXAMPLE"
        if not GEMINI_API_KEY or GEMINI_API_KEY == generic_gemini_placeholder:
            return False, "المفتاح غير صالح أو مفقود"
        try:
            if not gemini_initial_configured: 
                 genai.configure(api_key=GEMINI_API_KEY)
                 gemini_initial_configured = True 

            model = genai.GenerativeModel('gemini-1.5-flash') 
            model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1))
            return True, "نشط"
        except Exception as e:
            print(f"Gemini API error during status check: {e}")
            return False, f"غير نشط"
    return False, "API غير معروف"

# ----------------------
# Initialize Components Resource 
# ----------------------
@st.cache_resource 
def init_qdrant_client_resource():
    generic_qdrant_placeholder = "YOUR_QDRANT_API_KEY_PLACEHOLDER"
    if not QDRANT_API_KEY or QDRANT_API_KEY == generic_qdrant_placeholder: 
        return None
    try: return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
    except Exception as e: st.error(f"فشل الاتصال بـ Qdrant: {e}"); return None

@st.cache_resource 
def init_embedding_model_resource():
    try: return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    except Exception as e: st.error(f"فشل تحميل نموذج التضمين: {e}"); return None

# ----------------------
# Enhanced Document Search Functions
# ----------------------
def preprocess_arabic_query(query: str) -> str:
    """Enhanced Arabic query preprocessing"""
    # Remove diacritics (tashkeel)
    arabic_diacritics = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065A\u065B\u065C\u065D\u065E\u065F\u0670'
    query = ''.join(char for char in query if char not in arabic_diacritics)
    
    # Normalize Arabic characters
    query = query.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    query = query.replace('ة', 'ه')  # Ta marbuta to ha
    query = query.replace('ي', 'ى').replace('ى', 'ي')  # Alif maksura normalization
    
    return query.strip()

def enhanced_hybrid_search(query: str, max_results: int = 50) -> tuple:
    """Enhanced hybrid search with multiple strategies"""
    embedding_model = init_embedding_model_resource()
    if not embedding_model: 
        return [], "فشل تحميل نموذج البحث"
    
    qdrant_c = init_qdrant_client_resource()
    if not qdrant_c: 
        return [], "فشل الاتصال بقاعدة البيانات"
    
    try:
        # Preprocess query
        processed_query = preprocess_arabic_query(query)
        original_query = query
        
        all_results = []
        search_strategies = []
        
        # Strategy 1: Exact semantic search with original query
        try:
            query_embedding = embedding_model.encode([original_query])[0].tolist()
            exact_results = qdrant_c.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=max_results,
                with_payload=True,
                score_threshold=0.15  # Lowered threshold
            )
            all_results.extend(exact_results)
            search_strategies.append(f"البحث الدلالي المباشر: {len(exact_results)} نتيجة")
        except Exception as e:
            print(f"Error in exact search: {e}")
        
        # Strategy 2: Processed query search
        if processed_query != original_query:
            try:
                processed_embedding = embedding_model.encode([processed_query])[0].tolist()
                processed_results = qdrant_c.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=processed_embedding,
                    limit=max_results,
                    with_payload=True,
                    score_threshold=0.15
                )
                # Add unique results
                existing_ids = {r.id for r in all_results}
                new_results = [r for r in processed_results if r.id not in existing_ids]
                all_results.extend(new_results)
                search_strategies.append(f"البحث المُحسَّن: {len(new_results)} نتيجة إضافية")
            except Exception as e:
                print(f"Error in processed search: {e}")
        
        # Strategy 3: Keyword-based search with lower threshold
        keywords = [word.strip() for word in original_query.split() if len(word.strip()) > 2]
        if len(keywords) > 1:
            try:
                keyword_results = []
                for keyword in keywords:
                    kw_embedding = embedding_model.encode([keyword])[0].tolist()
                    kw_results = qdrant_c.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=kw_embedding,
                        limit=15,
                        with_payload=True,
                        score_threshold=0.1  # Very low threshold for keywords
                    )
                    keyword_results.extend(kw_results)
                
                # Add unique keyword results
                existing_ids = {r.id for r in all_results}
                new_keyword_results = [r for r in keyword_results if r.id not in existing_ids]
                all_results.extend(new_keyword_results)
                search_strategies.append(f"البحث بالكلمات المفتاحية: {len(new_keyword_results)} نتيجة إضافية")
            except Exception as e:
                print(f"Error in keyword search: {e}")
        
        # Strategy 4: Fuzzy search with character variations
        if len(all_results) < 10:
            try:
                fuzzy_queries = []
                # Add variations
                fuzzy_queries.append(query.replace('ي', 'ى'))
                fuzzy_queries.append(query.replace('ى', 'ي'))
                fuzzy_queries.append(query.replace('ة', 'ه'))
                fuzzy_queries.append(query.replace('أ', 'ا'))
                
                fuzzy_results = []
                for fq in fuzzy_queries:
                    if fq != query:
                        try:
                            fq_embedding = embedding_model.encode([fq])[0].tolist()
                            fq_results = qdrant_c.search(
                                collection_name=COLLECTION_NAME,
                                query_vector=fq_embedding,
                                limit=10,
                                with_payload=True,
                                score_threshold=0.12
                            )
                            fuzzy_results.extend(fq_results)
                        except:
                            continue
                
                # Add unique fuzzy results
                existing_ids = {r.id for r in all_results}
                new_fuzzy_results = [r for r in fuzzy_results if r.id not in existing_ids]
                all_results.extend(new_fuzzy_results)
                if new_fuzzy_results:
                    search_strategies.append(f"البحث المرن: {len(new_fuzzy_results)} نتيجة إضافية")
            except Exception as e:
                print(f"Error in fuzzy search: {e}")
        
        # Remove duplicates and sort by score
        unique_results = {}
        for result in all_results:
            if result.id not in unique_results or result.score > unique_results[result.id].score:
                unique_results[result.id] = result
        
        final_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)[:max_results]
        
        debug_info = f"استراتيجيات البحث المستخدمة: {' | '.join(search_strategies)}. المجموع النهائي: {len(final_results)} نتيجة"
        
        return final_results, debug_info
        
    except Exception as e:
        print(f"Error in enhanced_hybrid_search: {e}")
        return [], f"خطأ في البحث: {str(e)}"

# ----------------------
# Memory Management
# ----------------------
def get_conversation_context(messages: List[Dict], max_context_length: int = 1000) -> str:
    """Get recent conversation context for follow-up questions"""
    if len(messages) < 2:
        return ""
    
    # Get last few exchanges
    recent_messages = messages[-4:] if len(messages) >= 4 else messages
    context_parts = []
    
    for msg in recent_messages:
        role = "المستخدم" if msg["role"] == "user" else "المساعد"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        context_parts.append(f"{role}: {content}")
    
    context = "\n".join(context_parts)
    return context[:max_context_length] + "..." if len(context) > max_context_length else context

def enhance_query_with_context(current_query: str, conversation_context: str) -> str:
    """Enhance current query with conversation context"""
    if not conversation_context:
        return current_query
    
    # Simple context enhancement
    enhanced_query = f"{current_query}"
    
    # Add context if query seems like a follow-up
    follow_up_indicators = ["هذا", "ذلك", "نفس", "أيضا", "كذلك", "المذكور", "السابق"]
    if any(indicator in current_query for indicator in follow_up_indicators):
        enhanced_query = f"السياق السابق: {conversation_context}\n\nالسؤال الحالي: {current_query}"
    
    return enhanced_query

# ----------------------
# API Response Functions with Memory
# ----------------------
def prepare_llm_messages_with_memory(user_question: str, context: str, context_info: str, conversation_memory: str = ""):
    system_prompt = """أنت مساعد للبحث في كتب واستفتاءات الشيخ محمد السند فقط.
قواعد حتمية لا يمكن تجاوزها:
1. أجب فقط من النصوص المعطاة أدناه ("المصادر المتاحة") - لا استثناءات.
2. إذا لم تجد الإجابة الكاملة في النصوص، قل بوضوح: "لم أجد إجابة كافية في المصادر المتاحة بخصوص هذا السؤال."
3. ممنوع منعاً باتاً إضافة أي معلومة من خارج النصوص المعطاة. لا تستخدم معلوماتك العامة أو معرفتك السابقة.
4. اقتبس مباشرة من النصوص عند الإجابة قدر الإمكان، مع الإشارة إلى المصدر إذا كان متاحاً في النص (مثال: [نص ١]).
5. إذا وجدت إجابة جزئية، اذكرها وأوضح أنها غير كاملة أو تغطي جانباً من السؤال.
6. يمكنك الرجوع إلى السياق السابق للمحادثة إذا كان متوفراً لفهم السؤال بشكل أفضل.
7. إذا كانت النصوص لا تحتوي على إجابة، لا تحاول استنتاج أو تخمين الإجابة.
تذكر: أي معلومة ليست في النصوص أدناه = لا تذكرها أبداً. كن دقيقاً ومقتصراً على المصادر."""
    
    memory_section = f"\n\nسياق المحادثة السابق:\n{conversation_memory}\n---\n" if conversation_memory else ""
    
    user_content = f"{memory_section}السؤال المطروح: {user_question}\n\nالمصادر المتاحة من قاعدة البيانات فقط (أجب بناءً عليها حصراً):\n{context}\n\nمعلومات إضافية عن السياق: {context_info}\n\nالتعليمات: يرجى تقديم إجابة بناءً على النصوص أعلاه فقط. إذا لم تكن الإجابة موجودة، وضح ذلك."
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

# ----------------------
# API Response Functions (keeping existing ones but using new prepare function)
# ----------------------
def get_openai_response(messages, max_tokens=2000):
    if not openai_client: return "OpenAI client غير مهيأ. يرجى التحقق من مفتاح API."
    try:
        model_to_use = "gpt-3.5-turbo"
        response = openai_client.chat.completions.create(
            model=model_to_use, messages=messages, temperature=0.05, max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(
