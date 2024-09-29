import streamlit as st
from db import create_table, get_api_key

# 데이터베이스 테이블 생성
create_table()

# 로그인 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['id'] = ""
    st.session_state['api_key'] = ""

def login():
    if st.button("Log in"):
        
        st.session_state.logged_in = True
        st.rerun()

def logout():

    st.session_state.logged_in = False
    st.session_state['id'] = ""
    st.session_state['api_key'] = ""
    
    st.rerun()

login_page = st.Page("services/login.py", title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

chatbot = st.Page(
    "services/chatbot.py", title="Dashboard", icon=":material/dashboard:", default=True
)
browse = st.Page("services/browse.py", title="Bug reports", icon=":material/bug_report:")
register = st.Page(
    "services/register.py", title="System alerts", icon=":material/notification_important:"
)

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Reports": [chatbot, browse, register],
            
        }
    )
else:
    pg = st.navigation([login_page])
    
pg.run()