import requests
from bs4 import BeautifulSoup
import streamlit as st

mapping = {
    'Seoul': '서울',
    'Busan': '부산',
    'Daegu': '대구',
    'Incheon': '인천',
    'Gwangju': '광주',
    'Daejeon': '대전',
    'Ulsan': '울산',
    'Sejong': '세종',
    'Yongin': '용인',
    'Suwon': '수원',
    'Changwon': '창원',
    'Jeonju': '전주',
    'Cheongju': '청주',
    'Jeju': '제주',
    'Gangneung': '강릉',
    'Jecheon': '제천',
    'Chuncheon': '춘천',
    'Gyeongju': '경주',
}

# 사용자의 위치 정보 가져오기
def get_location():
    try:
        # IP 주소 확인 (ipinfo.io 무료 API 사용 예제)
        ip_response = requests.get("https://ipinfo.io/json")
        ip_data = ip_response.json()

        # 위치 정보 추출
        city = ip_data.get("city")
        if city:
            city_kor = mapping.get(city, city)  # 한글로 변환된 도시명
            st.session_state['location'] = city_kor
            return city_kor
        else:
            st.error("도시 정보를 가져올 수 없습니다.")
            return None

    except requests.RequestException:
        st.error("위치 정보를 가져오는 데 실패했습니다.")
        return None

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
            st.error("날씨 정보를 가져오는 데 실패했습니다.")
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

def get_weather():
    # 사용자의 위치 정보 가져오기
    if 'location' not in st.session_state:
        location = get_location()
    else:
        location = st.session_state['location']
    
    if location:
        crawler = NaverWeatherCrawler(location)
        return crawler.get_weather()
    else:
        return None
    
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