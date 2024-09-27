from fastapi import FastAPI, Request, Depends, Form, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from models import SessionLocal, Image, engine
import os
from pathlib import Path
from PIL import Image as PILImage

app = FastAPI()

# 정적 파일 제공 (CSS, 이미지)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 이미지 업로드 페이지
@app.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# 이미지 업로드 처리
@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    
    # 파일 저장
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    # DB에 이미지 정보 저장
    new_image = Image(filename=file.filename)
    db.add(new_image)
    db.commit()

    return RedirectResponse("/gallery", status_code=302)

# 이미지 갤러리 페이지 (읽기 권한이 있는 이미지만 표시)
@app.get("/gallery")
async def gallery_page(request: Request, db: Session = Depends(get_db)):
    images = db.query(Image).filter(Image.readable == 1).all()
    return templates.TemplateResponse("index.html", {"request": request, "images": images})

# 이미지 처리 함수
def process_image(image_path: str):
    # PyTorch 모델 가중치 불러오기 및 처리
    import torch
    model = torch.load("model_weights.pth")
    model.eval()
    
    # 이미지 불러오기 및 처리
    img = PILImage.open(image_path)
    img_tensor = torch.tensor(img).unsqueeze(0)  # 텐서로 변환 및 배치 차원 추가
    output = model(img_tensor)
    
    # 처리된 결과를 PIL 이미지로 변환
    processed_image = output.squeeze(0).detach().numpy()
    return PILImage.fromarray(processed_image)

# 이미지 수정 처리 (여기서는 단순히 읽기 가능 여부를 토글)
@app.post("/update/{image_id}")
async def update_image(image_id: int, db: Session = Depends(get_db)):
    image = db.query(Image).filter(Image.id == image_id).first()
    if image:
        image.readable = 1 - image.readable  # 읽기 가능/불가능 토글
        db.commit()
    return RedirectResponse("/gallery", status_code=302)