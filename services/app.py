import streamlit as st
import sys
import os

# 상위 폴더의 경로를 Python 모듈 검색 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_option_menu import option_menu
from weather import get_weather
from user import get_user_info
from login import show_login_page, show_signup_page

# 사이드바에 로그인 및 회원가입 페이지 표시
st.sidebar.title("로그인 / 회원가입")

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    # 로그인되지 않은 경우, 로그인 또는 회원가입 화면을 사이드바에 표시
    if st.session_state.get('signup'):
        show_signup_page()
    else:
        show_login_page()
else:
    # 로그인된 경우, 사용자 정보와 로그아웃 버튼을 사이드바에 표시
    st.sidebar.write(f"환영합니다, {st.session_state['username']}님!")
    if st.sidebar.button("로그아웃"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['api_key'] = None
        st.experimental_rerun()

# 상단 네비게이션 메뉴 생성
selected = option_menu(
    menu_title=None,  # 메뉴바 제목 없음
    options=["Home", "Weather", "User Info"],  # 메뉴 옵션
    icons=["house", "cloud-sun", "person"],  # 아이콘 추가
    menu_icon="cast",  # 메뉴바 왼쪽 상단 아이콘
    default_index=0,  # 기본 선택된 메뉴
    orientation="horizontal",  # 메뉴를 상단에 가로로 표시
)

# 각 메뉴에 따른 화면 구성
if selected == "Home":
    st.title("Welcome to the Home Page")
    st.write("This is the home page of the Streamlit app.")
    
elif selected == "Weather":
    st.title("Weather Information")
    try:
        st.write("Fetching weather information...")
        weather_info = get_weather()  # weather.py의 get_weather() 호출
        if weather_info:
            st.write(f"Location: {weather_info['location']}")
            st.write(f"Temperature: {weather_info['temperature']}°C")
            st.write(f"Condition: {weather_info['condition']}")
        else:
            st.warning("No weather information available.")
    except Exception as e:
        st.error(f"Error fetching weather information: {e}")
        
elif selected == "User Info":
    st.title("User Information")
    try:
        st.write("Fetching user information...")
        user_info = get_user_info()  # user.py의 get_user_info() 호출
        if user_info:
            st.write(user_info)
        else:
            st.warning("No user information available.")
    except Exception as e:
        st.error(f"Error fetching user information: {e}")
