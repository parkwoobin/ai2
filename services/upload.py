import streamlit as st
import os
from db import add_user_image, get_user_images, delete_user_image
from PIL import Image
import uuid

# 이미지 리사이즈 및 저장 함수
def save_and_resize_image(uploaded_file, output_size=(640, 640)):
    unique_filename = f"{uuid.uuid4()}.jpeg"
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    save_path = os.path.join(upload_dir, unique_filename)
    image = Image.open(uploaded_file)
    
    # 이미지가 RGBA 모드일 경우 RGB로 변환
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    resized_image = image.resize(output_size)
    resized_image.save(save_path, format="JPEG")  # JPEG 포맷으로 저장
    
    return unique_filename, save_path  # 이미지 파일 이름과 경로 반환

# 메인 렌더링 함수
def render():
    st.title("이미지 올리기")
    
    user_id = st.session_state.get('id', '')
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=["jpg", "png", "jpeg"])

    # 이미지 저장 및 업로드 기록 추가
    if uploaded_file is not None:
        img_name, img_url = save_and_resize_image(uploaded_file)
        st.success(f"이미지가 업로드되었습니다: {img_url}")
        add_user_image(user_id, img_name, img_url)

    # 특정 사용자가 업로드한 이미지 표시
    user_images = get_user_images(user_id)
    st.subheader("업로드한 이미지 갤러리")

    # 이미지를 3열 갤러리 형태로 나열
    num_columns = 3
    for i in range(0, len(user_images), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            if i + j < len(user_images):
                filename, filepath, upload_date = user_images[i + j]
                with col:
                    st.image(filepath, caption=f"{filename[:10]}... ({upload_date})", use_column_width=True)
                    if st.button("삭제", key=f"delete_{filename}"):
                        if delete_user_image(user_id, filename):
                            st.success("이미지가 삭제되었습니다.")
                            st.rerun()  # 페이지 새로고침
                        else:
                            st.error("이미지 삭제에 실패했습니다.")
                            
render()
