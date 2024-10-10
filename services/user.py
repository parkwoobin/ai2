import streamlit as st
import sqlite3
from datetime import datetime

class PersonalInfoService:
    def __init__(self, db_name='personal_info.db'):
        self.conn = sqlite3.connect(db_name)
        self.c = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.c.execute('''CREATE TABLE IF NOT EXISTS personal_info
                          (name TEXT, birthdate DATE, gender TEXT, height REAL, weight REAL, personal_color TEXT, mbti TEXT)''')

    def add_data(self, name, birthdate, gender, height, weight, personal_color, mbti):
        self.c.execute('INSERT INTO personal_info (name, birthdate, gender, height, weight, personal_color, mbti) VALUES (?, ?, ?, ?, ?, ?, ?)', 
                       (name, birthdate, gender, height, weight, personal_color, mbti))
        self.conn.commit()

    def get_all_data(self):
        self.c.execute("SELECT * FROM personal_info")
        return self.c.fetchall()

# Streamlit 페이지 구성
st.title("개인정보 입력")

service = PersonalInfoService()

name = st.text_input("이름", key="name_input")
birthdate = st.date_input("생년월일", datetime(2000, 1, 1), key="birthdate_input")
gender = st.selectbox("성별", ["남성", "여성", "기타"], key="gender_select")
height = st.number_input("키 (cm)", min_value=0.0, key="height_input")
weight = st.number_input("체중 (kg)", min_value=0.0, key="weight_input")

# 퍼스널 컬러 입력
personal_color = st.text_input("퍼스널컬러", placeholder="모를 시 빈칸으로 두시오", key="personal_color_input")

# MBTI 입력
mbti = st.text_input("MBTI", placeholder="모를 시 빈칸으로 두시오", key="mbti_input")

# 데이터 저장 버튼
if st.button("데이터 저장"):
    service.add_data(name, birthdate, gender, height, weight, personal_color, mbti)
    st.success("데이터가 성공적으로 저장되었습니다.")

# 저장된 데이터 보기
if st.checkbox("저장된 데이터 보기", key="view_data"):
    data = service.get_all_data()
    st.write(data)
