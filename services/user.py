import streamlit as st
import sqlite3
from datetime import datetime
from db import get_personal_info, add_personal_info, update_personal_info


def get_user_info():
    # 세션 상태에서 사용자 ID 가져오기
    user_id = st.session_state.get('id')
    username = st.session_state.get('username')

    if user_id is None:
        st.error("로그인이 필요합니다.")
        return

    st.title(f"{username}님의 개인정보 입력")

    # 데이터베이스에서 기존 개인 정보 가져오기
    existing_info = get_personal_info(user_id)

    if existing_info:
        # 기존 정보가 있으면 필드에 미리 채워넣기
        name = st.text_input("이름", value=existing_info[1], key="name_input")
        birthdate = st.date_input(
            "생년월일",
            datetime.strptime(existing_info[2], '%Y-%m-%d'),
            key="birthdate_input"
        )
        gender_options = ["남성", "여성", "기타"]
        gender_index = gender_options.index(existing_info[3]) if existing_info[3] in gender_options else 0
        gender = st.selectbox("성별", gender_options, index=gender_index, key="gender_select")
        height = st.number_input("키 (cm)", min_value=0.0, value=existing_info[4], key="height_input")
        weight = st.number_input("체중 (kg)", min_value=0.0, value=existing_info[5], key="weight_input")
        personal_color = st.text_input(
            "퍼스널컬러", value=existing_info[6],
            placeholder="모를 시 빈칸으로 두시오",
            key="personal_color_input"
        )
        mbti = st.text_input(
            "MBTI", value=existing_info[7],
            placeholder="모를 시 빈칸으로 두시오",
            key="mbti_input"
        )
    else:
        # 기존 정보가 없으면 빈 필드로 표시
        name = st.text_input("이름", key="name_input")
        birthdate = st.date_input("생년월일", datetime(2000, 1, 1), key="birthdate_input")
        gender = st.selectbox("성별", ["남성", "여성", "기타"], key="gender_select")
        height = st.number_input("키 (cm)", min_value=0.0, key="height_input")
        weight = st.number_input("체중 (kg)", min_value=0.0, key="weight_input")
        personal_color = st.text_input("퍼스널컬러", placeholder="모를 시 빈칸으로 두시오", key="personal_color_input")
        mbti = st.text_input("MBTI", placeholder="모를 시 빈칸으로 두시오", key="mbti_input")

    # 데이터 저장 버튼
    if st.button("데이터 저장"):
        # 날짜를 문자열로 변환
        birthdate_str = birthdate.strftime('%Y-%m-%d')

        if existing_info:
            # 기존 정보가 있으면 업데이트
            update_personal_info(user_id, name, birthdate_str, gender, height, weight, personal_color, mbti)
            st.success("데이터가 성공적으로 업데이트되었습니다.")
        else:
            # 기존 정보가 없으면 새로 추가
            add_personal_info(user_id, name, birthdate_str, gender, height, weight, personal_color, mbti)
            st.success("데이터가 성공적으로 저장되었습니다.")

    # 저장된 데이터 보기
    if st.checkbox("저장된 데이터 보기", key="view_data"):
        data = get_personal_info(user_id)
        if data:
            st.write(f"**이름:** {data[1]}")
            st.write(f"**생년월일:** {data[2]}")
            st.write(f"**성별:** {data[3]}")
            st.write(f"**키:** {data[4]} cm")
            st.write(f"**체중:** {data[5]} kg")
            st.write(f"**퍼스널컬러:** {data[6]}")
            st.write(f"**MBTI:** {data[7]}")
        else:
            st.write("저장된 데이터가 없습니다.")

get_user_info() 