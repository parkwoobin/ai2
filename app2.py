import streamlit as st
from db import initialize_database, get_api_key
from services.weather import get_location
# 데이터베이스 테이블 생성
initialize_database()

# 로그인 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['id'] = ""
    st.session_state['username'] = ""
    st.session_state['api_key'] = ""
    st.session_state['location'] = get_location()

def logout():
    st.session_state.logged_in = False
    st.session_state['id'] = ""
    st.session_state['username'] = ""
    st.session_state['api_key'] = ""
    st.rerun()
    
if st.session_state.logged_in:
    col1, col2, col3 = st.columns(3,gap='small')
    with col1:
        if st.button("Home"):
            st.switch_page("services/chatbot.py")
    with col2:
        if st.button("Page 1"):
            st.switch_page("services/weather.py")
    with col3: 
        if st.button("Page 2"):
            st.switch_page("services/user.py")

login_page = st.Page("services/login.py", title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
chatbot = st.Page(
    "services/chatbot.py", title="챗봇", icon=":material/chat:", default=True
    )
weahter = st.Page(
    "services/weather.py", title="날씨", icon=":material/wb_sunny:"
    )
user_info = st.Page(
    "services/user.py", title="개인 정보", icon=":material/notification_important:"
    )
if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Reports": [chatbot, weahter, user_info],
            
        },
        position="hidden"
    )
else:
    pg = st.navigation([login_page])
    
pg.run()