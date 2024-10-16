import streamlit as st
from db import update_api_key, get_api_key, get_personal_info
import os
import time
import base64
import uuid  # uuid 라이브러리 추가
from chatbot_class import Chatbot, GPT
from weather_func import get_weather, get_location

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

# 이미지 인코딩 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 챗봇 페이지를 보여주는 함수
def show_chatbot_page():
    user_id = st.session_state['id']
    existing_info = get_personal_info(user_id)
    weather = get_weather(get_location())
    chatbot = GPT(
        api_key=st.session_state.api_key,
        user_info=existing_info,
        weather=weather,
        model="gpt-4o-mini"
    )
    st.title("패션 도우미")  # 페이지 제목 설정

    # 사이드바에 API 키 입력 및 저장 기능 추가
    with st.sidebar:
        api_key_input = st.text_input("API 키 입력", value=st.session_state.api_key, type="password")
        if st.button("API 키 저장"):
            st.session_state.api_key = api_key_input
            if 'username' in st.session_state:
                update_api_key(st.session_state['username'], api_key_input)
                st.success("API 키가 저장되었습니다.")
            else:
                st.error("로그인되어 있지 않습니다.")

        # 이미지 업로드 기능 추가
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # 이전 대화 불러오기
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.markdown(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"], caption="업로드된 이미지")

    # 사용자 입력을 받는 입력창
    prompt = st.chat_input("질문을 입력하세요.")

    # 이미지 업로드 처리
    img_prompt = None
    if uploaded_file is not None:
        # 고유한 파일 이름 생성
        unique_filename = f"{uuid.uuid4()}.png"
        # 이미지 저장 경로 설정
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)  # 디렉토리 생성

        save_path = os.path.join(upload_dir, unique_filename)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"이미지가 업로드되었습니다: {unique_filename}")

        # 이미지를 base64로 인코딩하여 저장
        img_prompt = encode_image(save_path)
        image_url = f"data:image/{uploaded_file.type.split('/')[-1]};base64,{img_prompt}"

    # 사용자가 입력한 질문 처리
    if prompt or img_prompt:
        with st.chat_message("user"):
            user_content = []
            if prompt:
                st.write(prompt)
                user_content.append({"type": "text", "text": prompt})
            if img_prompt:
                st.image(uploaded_file, caption="업로드된 이미지")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })

            st.session_state.chat_history.append({
                "role": "user",
                "content": user_content
            })

        # 이미지가 있을 경우 리스트에 추가하여 전달
        img_list = [img_prompt] if img_prompt else None
        response = chatbot.generate(st.session_state.chat_history, text_prompt=prompt, img_prompt=img_list)

        # AI 응답 메시지 표시
        with st.chat_message("ai"):
            st.write_stream(stream_data(response))

        # AI 응답 메시지를 chat_history에 추가
        st.session_state.chat_history.append({
            "role": "ai",
            "content": [{"type": "text", "text": response}]
        })

# 챗봇 페이지 표시 함수 호출
show_chatbot_page()
