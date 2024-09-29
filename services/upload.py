import streamlit as st
import os

def render():
    st.title("이미지 올리기")
    
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        save_folder = "assets/uploads/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # 파일 저장
        file_path = os.path.join(save_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(file_path, caption="업로드한 이미지", use_column_width=True)
        st.success("이미지 업로드 성공!")