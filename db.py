import sqlite3
import bcrypt
from cryptography.fernet import Fernet, InvalidToken
import os

# secret.key 파일에서 키를 로드하는 함수
def load_fernet_key():
    # secret.key 파일에서 키 로드
    with open("secret.key", "rb") as key_file:
        key = key_file.read()
    return Fernet(key)

# Fernet 키 로드 및 암호화 객체 초기화
try:
    cipher_suite = load_fernet_key()
except FileNotFoundError:
    print("secret.key 파일을 찾을 수 없습니다. 키를 생성하거나 올바른 위치에 파일이 있는지 확인하세요.")
    cipher_suite = None

# 비밀번호 해시화
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

# 비밀번호 검증
def check_password(password, hashed):
    if isinstance(hashed, str): 
        hashed = hashed.encode('utf-8')
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# API 키 암호화
def encrypt_api_key(api_key):
    if cipher_suite is None:
        print("Cipher suite is not initialized. Cannot encrypt the API key.")
        return None
    return cipher_suite.encrypt(api_key.encode('utf-8'))

# API 키 복호화
def decrypt_api_key(encrypted_key):
    if cipher_suite is None:
        print("Cipher suite is not initialized. Cannot decrypt the API key.")
        return None
    try:
        return cipher_suite.decrypt(encrypted_key).decode('utf-8')
    except InvalidToken:
        print("Invalid Token: Unable to decrypt the API key.")
        return None

# 데이터베이스 연결
def create_connection():
    try:
        conn = sqlite3.connect('users.db')
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

# 테이블 생성
def create_table():
    conn = create_connection()
    if conn:
        with conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL,
                api_key BLOB
            );
            ''')
        conn.close()

# 회원가입
def register_user(username, password):
    conn = create_connection()
    if conn:
        hashed_password = hash_password(password)
        with conn:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.close()

# 로그인
def login_user(username, password):
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password(password, user[2]):
            return user
    return None

# API 키 업데이트
def update_api_key(username, api_key):
    conn = create_connection()
    if conn:
        encrypted_key = encrypt_api_key(api_key)
        with conn:
            conn.execute('UPDATE users SET api_key = ? WHERE username = ?', (encrypted_key, username))
        conn.close()

# API 키 불러오기
def get_api_key(username):
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('SELECT api_key FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return decrypt_api_key(result[0])
    return None
