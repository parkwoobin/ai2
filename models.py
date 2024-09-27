from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import bcrypt

# 데이터베이스 URL 설정 (SQLite 예시)
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"  # 다른 DB를 사용할 경우 URL을 변경 (PostgreSQL, MySQL 등)

# 데이터베이스 엔진 생성
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# 세션 로컬 객체 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 베이스 클래스 생성 (모든 모델이 상속)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    images = relationship("Image", back_populates="user")
    chats = relationship("ChatHistory", back_populates="user")

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))  # user_id 외래키
    user = relationship("User", back_populates="images")

class ChatHistory(Base):
    __tablename__ = "chat_histories"
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="chats")
    


