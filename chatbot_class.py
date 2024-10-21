from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from openai import OpenAI
import base64

# 대화 기록 예시
'''
st.session_state['chat_history'] = [
  {
    "role": "system", 
    "content": [{"type": "text", "text": "You are a helpful assistant."}]
  },
  {
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/3/36/Danbo_Cheese.jpg"}},
      {"type": "text", "text": "What is this?"}
    ]
  }
]
'''
import requests
import base64

class GPT:
    def __init__(self, api_key, weather, user_info, model="gpt-4o-mini"):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.model = model
        SYS_PROMPT = f"""
            [시스템 정보]
            사용자 이름: {user_info[1]}
            사용자 생년월일: {user_info[2]}
            사용자 성별: {user_info[3]}
            사용자 키: {user_info[4]}
            사용자 체중: {user_info[5]}
            사용자 퍼스널컬러: {user_info[6]}
            사용자 MBTI: {user_info[7]}

            위치: {weather['location']}
            온도: {weather['temperature']}°C
            날씨: {weather['condition']}

            사용자의 요구에 따라 패션에 대한 정보를 제공하는 챗봇입니다.
            편안한 말투로 사용자의 질문에 답변하세요.
            사용자와 현재 날씨 정보를 기반으로 추천해야합니다.
            의류 트렌드와 사용자에게 어울릴 스타일을 추천해드립니다.
            사용자가 읽어야할 텍스트가 너무 많지 않게 해주세요. 가독성을 고려해주세요.
            """
        self.messages = [{'role': 'system', 'content': SYS_PROMPT}]
        
    def _encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def generate(self, text_prompt=None, img_prompt=None):
        if not text_prompt and not img_prompt:
            return "텍스트 또는 이미지를 입력해주세요."
        
        contents = []
        if text_prompt:
            contents.append({'type': 'text', 'text': text_prompt})
        
        if img_prompt:
            img = self._encode_image(img_prompt)
            if img:
                contents.append({'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{img}"}})
        
        self.messages.append({'role': 'user', 'content': contents})
        
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": 300
        }
        
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload, timeout=10)
            response = response.json()
            
            if 'choices' in response:
                assistant_text = response['choices'][0]['message']['content']
                self.messages.append({'role': 'assistant', 'content': assistant_text})
                return assistant_text
            else:
                error_message = response.get('error', {}).get('message', 'Unknown error occurred')
                print("Error:", error_message)
                return f"Error: {error_message}"
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return "API 요청 중 문제가 발생했습니다."


class Chatbot:
    def __init__(self,
                 api_key,
                 weather,
                 user_info,
                 model = "gpt-4o-mini"
                 ):
        self.model = ChatOpenAI(api_key=api_key, model=model)
        # 프롬프트 템플릿 정의
        SYS_PROMPT = f"""
[시스템 정보]
사용자 이름: {user_info[1]}
사용자 생년월일: {user_info[2]}
사용자 성별: {user_info[3]}
사용자 키: {user_info[4]}
사용자 체중: {user_info[5]}
사용자 퍼스널컬러: {user_info[6]}
사용자 MBTI: {user_info[7]}

위치: {weather['location']}
온도: {weather['temperature']}°C
날씨: {weather['condition']}

사용자의 요구에 따라 패션에 대한 정보를 제공하는 챗봇입니다.
편안한 말투로 사용자의 질문에 답변하세요.
사용자와 현재 날씨 정보를 기반으로 추천해야합니다.
의류 트렌드와 사용자에게 어울릴 스타일을 추천해드립니다.
사용자가 읽어야할 텍스트가 너무 많지 않게 해주세요. 가독성을 고려해주세요.

[대화 내역]
{{history}}

[현재 대화]
사용자: {{input}}
어시스턴트:
"""
        self.prompt = ChatPromptTemplate.from_template(SYS_PROMPT)
        self.history = ChatMessageHistory()
        
    def generate(self, input_message):
        # 사용자의 메시지를 히스토리에 추가
        self.history.add_user_message(input_message)
        
        # 히스토리를 문자열로 변환
        history_str = ""
        for msg in self.history.messages[:-1]:  # 현재 메시지를 제외한 히스토리
            if msg.type == "human":
                history_str += f"사용자: {msg.content}\n"
            elif msg.type == "ai":
                history_str += f"어시스턴트: {msg.content}\n"
        
        # 프롬프트 생성
        prompt_input = self.prompt.format_prompt(history=history_str, input=input_message).to_messages()
        
        # 모델로부터 응답 생성
        response = self.model(prompt_input)
        
        # 어시스턴트의 응답을 히스토리에 추가
        self.history.add_ai_message(response.content)
        
        return response.content
