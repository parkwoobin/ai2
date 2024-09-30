import streamlit as st
from db import update_api_key, get_api_key
import os
import time
#
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)
        
# 대화 기록 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 챗봇 페이지
def show_chatbot_page():
    st.title("패션 도우미")

    # API 키가 세션에 이미 있는 경우 자동 입력
    with st.sidebar:
        api_key_input = st.text_input("API 키 입력", value=st.session_state['api_key'])
        if st.button("API 키 저장"):
            st.session_state['api_key'] = api_key_input
            # 현재 로그인된 사용자가 있는지 확인
            if 'username' in st.session_state:
                update_api_key(st.session_state['username'], api_key_input)  # DB에 API 키 저장
                st.success("API 키가 저장되었습니다.")
            else:
                st.error("로그인되어 있지 않습니다.")

        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"],label_visibility="collapsed")
    
    

   

    # 이전 대화 불러오기
    
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    prompt = st.chat_input("질문을 입력하세요.")

    # 이미지 업로드
    
    if uploaded_file is not None:
        # 이미지 저장
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"이미지가 업로드되었습니다: {uploaded_file.name}")

        
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        querry = f"{prompt}"
        response = querry
        
        with st.chat_message("ai"):
            st.write_stream(stream_data(message["message"]))
        
        st.session_state.chat_history.append({"role": "user", "message": prompt})
        st.session_state.chat_history.append({"role": "ai", "message": response})
        
show_chatbot_page()