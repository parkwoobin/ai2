from cryptography.fernet import Fernet

# Fernet 키 생성
key = Fernet.generate_key()

# 생성된 키를 secret.key 파일에 저장
with open("secret.key", "wb") as key_file:
    key_file.write(key)

print("Fernet 키가 secret.key 파일에 저장되었습니다.")
