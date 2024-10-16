from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory
from openai import OpenAI
import base64

class GPT:
    def __init__(self, 
                 api_key, 
                 weather,
                 user_info,
                 model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
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
        self.messages = [
            {'role': 'system', 'content': f'{SYS_PROMPT}'},
        ]
    
    def _encode_image(self, image_path):
        # 이미지를 base64로 인코딩하는 함수입니다. gpt에 이미지를 넘기기 위해서는 인터넷url이 아닌경우 base64로 인코딩하여 넘겨야합니다.
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate(self, chat_history, text_prompt, img_prompt=None):
        # 대화 기록에서 메시지를 재구성하여 추가
        messages = []
        for message in chat_history:
            content_list = []
            for content in message["content"]:
                if content["type"] == "text":
                    content_list.append({"type": "text", "text": content["text"]})
                elif content["type"] == "image_url":
                    # 이미 인코딩된 이미지 URL을 그대로 사용
                    content_list.append({"type": "image_url", "image_url": content["image_url"]})
            messages.append({"role": message["role"], "content": content_list})
        
        # 새로운 사용자 입력 추가
        user_content = [{"type": "text", "text": text_prompt}]
        if img_prompt:
            for img in img_prompt:
                img_type = img.split(';')[0].split('/')[-1]
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
        
        # 메시지에 새로운 사용자 요청 추가
        messages.append({"role": "user", "content": user_content})
        
        # 모델 예측 요청
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        # AI 응답 추가
        ai_content = [{"type": "text", "text": completion.choices[0].message.content}]
        self.messages.append({"role": "assistant", "content": ai_content})
        
        return completion.choices[0].message.content

    
class Chatbot_openai:
    def __init__(self,
                 api_key,
                 weather,
                 user_info,
                 model_name
                 ):
        self.model = OpenAI(api_key=api_key)
        self.model_name = model_name
        
    def generate(self, chat_history, input_message,input_image=None):
        # 이전 대화 불러오기
        messages = [
            {
                "role": "assistant",
                "content": self.SYS_PROMPT 
            }
        ]
        for message in chat_history:
            role = "assistant" if message["role"] == "user" else "user"
            history = {"role": role, "message": message["message"]}
            messages.append(history)    
        message.append(
            {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": input_image,
                            }
                        },
                    ],
                }
        )
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=300,
        )

        return response.choices[0]

class Chatbot:
    def __init__(self,
                 api_key,
                 weather,
                 user_info,
                 model_name
                 ):
        self.model = ChatOpenAI(api_key=api_key, model=model_name)
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
