import streamlit as st
import requests
import json

st.set_page_config(
    page_title="المرجع الديني الشيخ محمد السند حفظه الله",
    page_icon="🕌"
)

st.markdown("# 🕌 المرجع الديني الشيخ محمد السند حفظه الله")
st.markdown("### مساعد ذكي للاستفتاءات والأحكام الشرعية")

# Simple chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"👤 **السائل:** {message['content']}")
    else:
        st.write(f"🕌 **المرجع:** {message['content']}")

# Input form
with st.form("religious_form", clear_on_submit=True):
    question = st.text_area("اكتب استفتاءك الشرعي:", height=120)
    submit = st.form_submit_button("🕌 استفتاء شرعي", type="primary")

if submit and question:
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Simple response (you can connect to DeepSeek API here)
    response = "بسم الله الرحمن الرحيم\n\nشكراً لاستفتائكم. يرجى المراجعة مع المراجع الدينية المعتمدة.\n\nوالله أعلم"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()