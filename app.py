import streamlit as st
from db import initialize_database
from streamlit_option_menu import option_menu
from services.weather import get_weather
from services.user import get_user_info
from services.login import show_login_page  # 로그인 페이지를 가져옵니다.

# 데이터베이스 테이블 생성
initialize_database()

# 로그인 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['id'] = ""
    st.session_state['username'] = ""
    st.session_state['api_key'] = ""

def logout():
    st.session_state.logged_in = False
    st.session_state['id'] = ""
    st.session_state['username'] = ""
    st.session_state['api_key'] = ""
    st.rerun()

# 페이지 불러오기
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

if st.session_state.logged_in:
    # 상단 네비게이션 바 (로그인한 경우에만 표시)
    selected = option_menu(
        menu_title=None,  # 메뉴바 제목 없음
        options=["홈", "날씨", "개인정보"],  # 메뉴 옵션
        icons=["house", "cloud-sun", "person"],  # 아이콘 추가
        menu_icon="cast",  # 메뉴바 왼쪽 상단 아이콘
        default_index=0,  # 기본 선택된 메뉴
        orientation="horizontal",  # 메뉴를 상단에 가로로 표시
    )   
    
    if selected == "홈":
        st.title("Welcome to the Home Page")
        st.write("This is the home page of the Streamlit app.")
        
    elif selected == "날씨":
        try:
            weather_info = get_weather()  # weather.py의 get_weather() 호출
            if weather_info:
                st.write(f"Location: {weather_info['location']}")
                st.write(f"Temperature: {weather_info['temperature']}°C")
                st.write(f"Condition: {weather_info['condition']}")
            else:
                st.warning("No weather information available.")
        except Exception as e:
            st.error(f"Error fetching weather information: {e}")
            
    elif selected == "개인정보":
        try:
            user_info = get_user_info()  # user.py의 get_user_info() 호출
            if user_info:
                st.write(user_info)
            else:
                st.warning("No user information available.")
        except Exception as e:
            st.error(f"Error fetching user information: {e}")

    # 로그아웃 버튼을 사이드바에 추가
    if st.sidebar.button("로그아웃"):
        logout()

else:
    # 로그인 페이지 표시
    show_login_page()  # 로그인 페이지를 호출하여 표시
