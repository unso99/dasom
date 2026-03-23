import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities import SerpAPIWrapper
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# SerpAPI
params = {"engine": "google", "gl": "kr", "hl": "ko", "num": "3"}

search = SerpAPIWrapper(params=params)

class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    company: str = Field(description="메일을 보낸 사람의 회사 정보")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")

def create_email_chain():
    parser = PydanticOutputParser(pydantic_object=EmailSummary)
    prompt = load_prompt("email_summary.yaml", encoding="utf-8")
    prompt = prompt.partial(format=parser.get_format_instructions())
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return prompt | llm | parser

def create_chain(prompt_type):
    # ✅ 이메일 요약은 별도 체인 반환
    if prompt_type == "이메일 요약":
        return create_email_chain()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신의 이름은 'Dasom'이며, 사용자를 돕는 AI 어시스턴트입니다.

## 응답 원칙
- 한국어로 자연스럽고 명확하게 답변합니다.
- 질문의 의도를 정확히 파악하고 핵심부터 답변합니다.
- 모르는 내용은 솔직히 모른다고 말합니다.
- 코드 작성 시 간결하고 실행 가능한 예제를 제공합니다.

## 답변 형식
- 불필요한 서론 없이 바로 본론으로 들어갑니다.
- 복잡한 내용은 단계별로 설명합니다."""),
        ("user", "{question}"),
    ])
    if prompt_type == "SNS 게시글":
        prompt = load_prompt("sns.yaml", encoding="utf-8")
    if prompt_type == "요약":
        prompt = load_prompt("summary.yaml", encoding="utf-8")

    return prompt | ChatOpenAI(model="gpt-4o", temperature=0.7) | StrOutputParser()

def render_email_summary(result: EmailSummary):
    st.markdown(f"""
**발신자:** {result.person}  
**회사:** {result.company}  
**이메일:** {result.email}  
**제목:** {result.subject}  
**미팅 일정:** {result.date}  

---

**요약**  
{result.summary}
""")

st.title("Dasom")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_prompt = st.selectbox(
        "프롬프트를 선택해 주세요.", ("기본모드", "SNS 게시글", "요약", "이메일 요약"), index=0
    )
    if selected_prompt == "이메일 요약":
        st.info("📧 이메일 원문을 채팅창에 붙여넣으세요.")

if clear_btn:
    st.session_state["messages"] = []

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

print_messages()

placeholder = "이메일 내용을 붙여넣으세요!" if selected_prompt == "이메일 요약" else "궁금한 내용을 물어보세요!"
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.chat_message("user").write(user_input)
    chain = create_chain(selected_prompt)

    with st.chat_message("assistant"):
        if selected_prompt == "이메일 요약":
            try:
                result = chain.invoke({"email_conversation": user_input})
                render_email_summary(result)

                # ✅ 테스트: 발신자 정보로 검색
                query = f"{result.person} {result.company} {result.email}"
                search_result = search.run(query)
                
                st.markdown("---")
                st.markdown("**🔍 발신자 검색 결과**")
                st.write(search_result)

                ai_answer = f"발신자: {result.person} / 회사: {result.company} / 요약: {result.summary}"
            except Exception as e:
                ai_answer = "⚠️ 이메일 형식의 내용을 입력해 주세요.\n\nFrom, Subject, 본문이 포함된 이메일을 붙여넣으면 요약해드립니다."
                st.markdown(ai_answer)
        else:
            container = st.empty()
            ai_answer = ""
            for token in chain.stream({"question": user_input}):
                ai_answer += token
                container.markdown(ai_answer)

    add_message("user", user_input)
    add_message("assistant", ai_answer)