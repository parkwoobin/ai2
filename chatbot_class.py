from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory

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
