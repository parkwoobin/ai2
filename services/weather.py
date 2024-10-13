import requests
from bs4 import BeautifulSoup
import streamlit as st

def get_weather():
    class NaverWeatherCrawler:
        def __init__(self, location):
            self.base_url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query="
            self.location = location

        def get_weather(self):
            url = self.base_url + self.location + " 날씨"
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                weather_info = self.parse_weather(soup)
                return weather_info
            else:
                return None

        def parse_weather(self, soup):
            weather = {}
            temp_element = soup.find('div', {'class': 'temperature_text'})
            condition_element = soup.find('span', {'class': 'weather before_slash'})
            location_element = soup.find('h2', {'class': 'title'})

            weather['temperature'] = temp_element.text if temp_element else 'N/A'
            weather['temperature'] = weather['temperature'].replace('현재 온도', '').replace('°', '').strip()
            weather['condition'] = condition_element.text if condition_element else 'N/A'
            weather['location'] = location_element.text if location_element else 'N/A'
            return weather

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
