import streamlit as st

def render():
    st.title("로그인")
    
    username = st.text_input("아이디")
    password = st.text_input("비밀번호", type="password")
    
    if st.button("로그인"):
        if username == "admin" and password == "admin":  # 간단한 로그인 예시
            st.success("로그인 성공!")
        else:
            st.error("로그인 실패!")