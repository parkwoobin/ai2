import sqlite3
import bcrypt
from cryptography.fernet import Fernet, InvalidToken
from cryptography.fernet import Fernet

# secret.key 파일에서 키를 불러오는 함수
def load_key():
    with open('secret.key', 'rb') as key_file:
        return key_file.read()

# 키를 로드하여 Fernet 객체를 생성
FERNET_KEY = load_key()
cipher_suite = Fernet(FERNET_KEY)

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
def load_fernet_key():
    # 기존에 생성된 키를 안전하게 로드하도록 환경 변수나 파일로 관리하는 것을 추천
    # FERNET_KEY = Fernet.generate_key()  # 앱 재시작 시 키가 바뀌면 복호화가 불가능하므로, 환경변수나 파일로 키를 관리하세요.
    # 예: 환경 변수에서 로드 (os.getenv('FERNET_KEY')) 또는 파일에서 로드
    return Fernet(FERNET_KEY)  # FERNET_KEY는 적절히 관리되어야 함

cipher_suite = load_fernet_key()

# API 키 암호화
def encrypt_api_key(api_key):
    return cipher_suite.encrypt(api_key.encode('utf-8'))

# API 키 복호화
def decrypt_api_key(encrypted_key):
    try:
        return cipher_suite.decrypt(encrypted_key).decode('utf-8')
    except InvalidToken:
        print(f"Invalid Token: Unable to decrypt the API key.")
        return None

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
        print(f"Encrypted API key: {result[0]}")  # 암호화된 값 출력
        return decrypt_api_key(result[0])  # 복호화하여 반환
    return None