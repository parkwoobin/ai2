import chardet

# 파일의 인코딩 감지
with open('requirements.txt', 'rb') as f:
    result = chardet.detect(f.read())

print(f"Detected encoding: {result['encoding']}")
