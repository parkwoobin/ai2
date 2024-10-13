import streamlit as st
from db import update_api_key, get_api_key, get_personal_info
import os
import time
from chatbot_class import Chatbot
from weather_func import get_weather
# 스트리밍 데이터를 생성하는 함수
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

# 대화 기록 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# API 키 초기화
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
else:
    st.session_state.api_key = get_api_key(st.session_state['username'])


# 챗봇 페이지를 보여주는 함수
def show_chatbot_page():
    user_id = st.session_state['id']
    existing_info = get_personal_info(user_id)
    weather = get_weather(st.session_state['location'])
    chatbot = Chatbot(
        api_key=st.session_state.api_key,
        user_info=existing_info,
        weather=weather,
        model_name="gpt-4o-mini-2024-07-18"
    )
    st.title("패션 도우미")  # 페이지 제목 설정

    # 사이드바에 API 키 입력 및 저장 기능 추가
    with st.sidebar:
        # 세션에 저장된 API 키가 있으면 자동으로 입력
        api_key_input = st.text_input("API 키 입력", value=st.session_state.api_key, type="password")
        if st.button("API 키 저장"):
            st.session_state.api_key = api_key_input
            # 현재 로그인된 사용자가 있는지 확인
            if 'username' in st.session_state:
                update_api_key(st.session_state['username'], api_key_input)  # DB에 API 키 저장
                st.success("API 키가 저장되었습니다.")
            else:
                st.error("로그인되어 있지 않습니다.")

        # 이미지 업로드 기능 추가
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # 이전 대화 불러오기
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    # 사용자 입력을 받는 입력창
    prompt = st.chat_input("질문을 입력하세요.")

    # 이미지 업로드 처리
    if uploaded_file is not None:
        # 이미지 저장 경로 설정 및 파일 저장
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"이미지가 업로드되었습니다: {uploaded_file.name}")

    # 사용자가 입력한 질문 처리
    if prompt:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.write(prompt)
        querry = f"{prompt}"
        response = chatbot.generate(querry)

        # AI 응답 메시지 표시
        with st.chat_message("ai"):
            st.write_stream(stream_data(response))

        # 대화 기록에 사용자와 AI 메시지 추가
        st.session_state.chat_history.append({"role": "user", "message": prompt})
        st.session_state.chat_history.append({"role": "ai", "message": response})

# 챗봇 페이지 표시 함수 호출
show_chatbot_page()