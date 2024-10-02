def convert_to_utf8(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

if __name__ == "__main__":
    convert_to_utf8('/Users/seungwoo/Workspace/AIService2/requirements.txt')