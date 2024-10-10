from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class User:
    def __init__(self):
        self.name = "name"
        self.birthday = "1008/09/01"
        self.gender = None
        self.height = None
        self.weight = None
        self.personal_color = None
        self.mbti = None

def _format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
class Chatbot:
    def __init__(self,
                 api_key,
                 weather,
                 user_info,
                 model_name
                 ):
        self.model = ChatOpenAI(api_key=api_key, model=model_name)
        # 프롬프트 템플릿을 정의합니다.
        # 앞, 중간, 뒤로 나누어서 프롬프트를 정의합니다.
        # 중간은 사용자의 입력을 그대로 전달합니다.
        SYS_PROMPT = f"""
        사용자 이름 : {name}
        사용자 생년월일 : {birthday}
        사용자 성별 : {gender}
        사용자 키 : {height}
        사용자 체중 : {weight}
        사용자 퍼스널컬러 : {personal_color}
        사용자 MBTI : {mbti}
        """
        self.prompt = ChatPromptTemplate.from_template(SYS_PROMPT)
        
        self.chain = (
            {'question': RunnablePassthrough()}  # 'context'는 retriever와 format_docs를 통해 설정되고, 'question'은 그대로 전달됩니다.
            | self.prompt  # 프롬프트 템플릿을 적용합니다.
            | self.model  # 모델을 호출합니다.
            | StrOutputParser()  # 출력 파서를 통해 모델의 출력을 문자열로 변환합니다.
            ) 

    def generate(self, input_message):
        return self.chain.invoke(input_message)
    
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    api = os.getenv("OPENAI_API_KEY")
    
    chatbot = Chatbot(
        api,
        weather=
        user_info=
        model_name="gpt-4o",
    )
        