from fastapi import FastAPI, Request, Response, Depends, Form, Cookie, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer
from sqlalchemy.orm import Session
from models import User, Image, ChatHistory, Base, engine, SessionLocal  # 모델 및 데이터베이스 설정 불러오기
from passlib.context import CryptContext
from pathlib import Path
import uuid
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# DB 테이블 초기화 (앱 시작 시 한 번만 실행)
Base.metadata.create_all(bind=engine)

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 비밀번호 암호화 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 시크릿 키로 세션 관리
SECRET_KEY = "my-secret-key"  # 배포 시 환경 변수로 관리하는 것이 좋습니다.
serializer = URLSafeSerializer(SECRET_KEY)

# 비밀번호 해시
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# 비밀번호 검증 함수
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)



# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 비밀번호 검증 함수
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# 세션 생성 함수
def create_session_cookie(username: str) -> str:
    session_id = serializer.dumps(username)
    return session_id

# 쿠키에서 세션 복원 함수
def get_session_from_cookie(session_cookie: str) -> str:
    try:
        username = serializer.loads(session_cookie)
        return username
    except Exception:
        return None

# 루트 경로를 index.html로 연결
@app.get("/")
async def root(request: Request, session_id: str = Cookie(None)):
    username = get_session_from_cookie(session_id)
    if not username:
        return RedirectResponse("/login")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_user": username  # 사용자 정보 전달
    })

# 로그인 페이지
@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# 로그인 처리 함수에서 쿠키 설정
@app.post("/login")
async def login_user(
    response: Response, 
    request: Request, 
    username: str = Form(...),  
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    print(f"Attempting login for: {username}")
    
    # 사용자 정보 조회
    user = db.query(User).filter(User.username == username).first()

    # 사용자 정보 및 비밀번호 확인
    if not user or not verify_password(password, user.hashed_password):
        print("Login failed: Invalid credentials")
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    # 세션 ID 생성
    session_id = create_session_cookie(username)

    # 쿠키에 session_id 설정
    response.set_cookie(
        key="session_id", 
        value=session_id, 
        httponly=True, 
        secure=False,  # 개발 환경에서는 False, 배포 시에는 True로 설정
        max_age=60*60*24,  # 쿠키 유효 기간 1일
        path="/"  # 전체 경로에서 세션 유지
    )
    
    print(f"Login successful, session ID: {session_id}")
    return RedirectResponse(url="/", status_code=302)

# 로그아웃 처리
@app.post("/logout")
async def logout_user(response: Response):
    response.delete_cookie("session_id")
    return RedirectResponse("/login", status_code=302)

# 사용자가 업로드한 이미지 보기
@app.get("/my_images")
async def my_images(request: Request, db: Session = Depends(get_db), session_id: str = Cookie(None)):
    username = get_session_from_cookie(session_id)
    if not username:
        return RedirectResponse("/login")

    # 로그인한 사용자가 업로드한 이미지 가져오기
    user = db.query(User).filter(User.username == username).first()
    images = db.query(Image).filter(Image.user_id == user.id).all()
    return templates.TemplateResponse("my_images.html", {"request": request, "images": images, "current_user": username})

# 업로드 페이지 GET 요청
@app.get("/upload")
async def upload_page(request: Request, session_id: str = Cookie(None)):
    username = get_session_from_cookie(session_id)
    if not username:
        return RedirectResponse("/login")

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "current_user": username
    })

# 이미지 업로드 처리 (POST 요청)
@app.post("/upload")
async def upload_image(
    request: Request, 
    file: UploadFile = File(...),  
    db: Session = Depends(get_db), 
    session_id: str = Cookie(None)
):
    # 세션에서 사용자 정보 가져오기
    username = get_session_from_cookie(session_id)
    if not username:
        # status_code를 303으로 설정하여 클라이언트가 GET 요청을 보내도록 함
        return RedirectResponse("/login", status_code=303)

    # 사용자 정보 가져오기
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return RedirectResponse("/login", status_code=303)

    try:
        # 고유 파일 이름 생성
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = Path("uploads") / unique_filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # DB에 파일 정보 저장
        new_image = Image(filename=unique_filename, user_id=user.id)
        db.add(new_image)
        db.commit()

        # 업로드 후 index 페이지로 리다이렉션
        return RedirectResponse("/", status_code=302)

    except Exception as e:
        print(f"File upload error: {e}")
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "error": "File upload failed.",
            "current_user": username
        })