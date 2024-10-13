
import streamlit as st
from weather_func import NaverWeatherCrawler

def get_weather_streamlit():
    # Streamlit에서 사용자로부터 위치 입력 받기
    st.title("날씨 정보")
    location = st.text_input("위치를 입력하세요", "서울")
    
    
    # 날씨 정보 가져오기
    if location:
        crawler = NaverWeatherCrawler(location)
        weather_info = crawler.get_weather()

        if weather_info:
            st.write(f"위치: {weather_info['location']}")
            st.write(f"온도: {weather_info['temperature']}°C")
            st.write(f"날씨 상태: {weather_info['condition']}")
        else:
            st.error("날씨 정보를 가져오지 못했습니다.")

get_weather_streamlit()