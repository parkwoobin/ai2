import streamlit as st

def render():
    st.title("챗봇")
    
    user_input = st.text_input("메시지를 입력하세요", "")
    
    if st.button("Send"):
        if user_input:
            st.write(f"챗봇 응답: {chatbot_response(user_input)}")
        else:
            st.warning("메시지를 입력하세요.")
    
def chatbot_response(message):
    # 여기에 OpenAI API 호출 등을 넣을 수 있음.
    return f"'{message}'에 대한 응답입니다."