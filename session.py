import streamlit as st
from db import update_api_key, get_api_key

# 로그인 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.session_state['api_key'] = ""

# API 키 자동 로딩
if 'api_key' not in st.session_state:
    if st.session_state['logged_in']:
        try:
            # 로그인되어 있으면 DB에서 API 키 가져오기
            api_key = get_api_key(st.session_state['username'])
            if api_key:
                st.session_state['api_key'] = api_key
        except Exception as e:
            st.error("API 키를 불러오는 중 오류 발생: " + str(e))
    else:
        st.session_state['api_key'] = ""