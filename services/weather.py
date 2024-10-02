import requests
from bs4 import BeautifulSoup

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

if __name__ == "__main__":
    location = "서울"
    crawler = NaverWeatherCrawler(location)
    weather = crawler.get_weather()
    if weather:
        print(f"Location: {weather['location']}")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Condition: {weather['condition']}")
    else:
        print("Failed to retrieve weather information.")