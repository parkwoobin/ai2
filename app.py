import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="AI 서비스", page_icon="🤖", layout="wide")

# 공통 헤더: 상단 고정 CSS 스타일
def render_header():
    st.markdown("""
        <style>
            .header {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                background-color: #4CAF50;
                color: white;
                text-align: center;
                padding: 10px;
                font-size: 24px;
                z-index: 1000;
            }
            body {
                padding-top: 60px;  /* 헤더 높이만큼 여백 추가 */
            }
        </style>
        <div class="header">
            🤖 AI 서비스 플랫폼 - 상단 고정 헤더
        </div>
    """, unsafe_allow_html=True)

# 공통 푸터
def render_footer():
    st.markdown("""
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #f1f1f1;
                color: black;
                text-align: center;
                padding: 10px;
            }
        </style>
        <div class="footer">
            &copy; 2024 AI 서비스 플랫폼 - All Rights Reserved
        </div>
    """, unsafe_allow_html=True)

# 페이지 네비게이션 바
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("챗봇", "로그인", "회원가입", "이미지 올리기", "이미지 탐색"))

# 페이지에 공통 헤더 추가
render_header()

# 페이지 내용 예시
if page == "챗봇":
    st.title("챗봇 페이지")
elif page == "로그인":
    st.title("로그인 페이지")
elif page == "회원가입":
    st.title("회원가입 페이지")
elif page == "이미지 올리기":
    st.title("이미지 올리기 페이지")
elif page == "이미지 탐색":
    st.title("이미지 탐색 페이지")

# 페이지에 공통 푸터 추가
render_footer()