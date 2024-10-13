import streamlit as st
import hydralit_components as hc
from weather import get_weather
from user import get_user_info

# 네비게이션 바의 메뉴 정의
menu_data = [
    {'label': "Home", 'id': "Home"},
    {'label': "Weather", 'id': "Weather"},
    {'label': "User Info", 'id': "User Info"},
]

# 네비게이션 바의 스타일 설정
over_theme = {
    'txc_inactive': 'purple',
    'menu_background': 'red',
    'txc_active': 'yellow',
    'option_active': 'blue'
}

# Hydralit 네비게이션 바 생성
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=True,  # Streamlit 마커 숨기기
)

# 페이지별 콘텐츠 표시
st.title("My Streamlit App")  # 애플리케이션 제목

st.write("---")  # 섹션 구분선

# Home 섹션
if menu_id == "Home":
    st.header("Welcome to the Home Page")
    st.write("This is the home page of the Streamlit app.")

# Weather 섹션
if menu_id == "Weather":
    st.header("Weather Information")
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

# User Info 섹션
if menu_id == "User Info":
    st.header("User Information")
    try:
        st.write("Fetching user information...")
        user_info = get_user_info()  # user.py의 get_user_info() 호출
        if user_info:
            st.write(user_info)
        else:
            st.warning("No user information available.")
    except Exception as e:
        st.error(f"Error fetching user information: {e}")

st.write("---")  # 섹션 구분선
