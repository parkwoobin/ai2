import sqlite3
import bcrypt
from cryptography.fernet import Fernet

# 비밀번호 해시화
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

# 비밀번호 검증 (데이터베이스에서 불러온 해시된 비밀번호는 문자열이므로 바이트로 변환)
def check_password(password, hashed):
    if isinstance(hashed, str):
        hashed = hashed.encode('utf-8')  # 문자열을 바이트로 변환
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# 암호화를 위한 대칭키 생성 (앱을 시작할 때 또는 환경 변수로 관리)
FERNET_KEY = Fernet.generate_key()  # 안전하게 저장 필요, 재생성 시 복호화 불가
cipher_suite = Fernet(FERNET_KEY)

# API 키 암호화
def encrypt_api_key(api_key):
    encrypted_key = cipher_suite.encrypt(api_key.encode('utf-8'))
    return encrypted_key

# API 키 복호화
def decrypt_api_key(encrypted_key):
    decrypted_key = cipher_suite.decrypt(encrypted_key).decode('utf-8')
    return decrypted_key

# 데이터베이스 연결
def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

# 테이블 생성 함수 (유저 테이블에 api_key 필드 추가)
def create_table():
    conn = create_connection()
    with conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            api_key BLOB  -- 암호화된 API 키 저장
        );
        ''')
    conn.close()

# 회원가입 시 사용자 정보 저장 (비밀번호 해시화, API 키는 없을 수 있음)
def register_user(username, password):
    conn = create_connection()
    hashed_password = hash_password(password)
    with conn:
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.close()

# 로그인 시 비밀번호 검증
def login_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password(password, user[2]):  # 비밀번호 검증
        return user
    else:
        return None

# 유저의 API 키 업데이트 (API 키 암호화 후 저장)
def update_api_key(username, api_key):
    conn = create_connection()
    encrypted_key = encrypt_api_key(api_key)
    with conn:
        conn.execute('UPDATE users SET api_key = ? WHERE username = ?', (encrypted_key, username))
    conn.close()

# 유저의 API 키 불러오기 (복호화)
def get_api_key(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT api_key FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        return decrypt_api_key(result[0])  # 복호화하여 반환
    return None