import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

st.title("Dasom")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_prompt = st.selectbox(
        "프롬프트를 선택해 주세요.", ("기본모드","SNS 게시글","요약"), index=0
    )

if clear_btn:
    st.session_state["messages"] = []

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role,message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def create_chain(prompt_type):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """당신의 이름은 'Dasom'이며, 사용자를 돕는 AI 어시스턴트입니다.

## 응답 원칙
- 한국어로 자연스럽고 명확하게 답변합니다.
- 질문의 의도를 정확히 파악하고 핵심부터 답변합니다.
- 모르는 내용은 솔직히 모른다고 말합니다.
- 코드 작성 시 간결하고 실행 가능한 예제를 제공합니다.

## 답변 형식
- 불필요한 서론 없이 바로 본론으로 들어갑니다.
- 복잡한 내용은 단계별로 설명합니다."""),
            ("user", "{question}"),
        ]
    )
    if prompt_type == "SNS 게시글":
        prompt = load_prompt("sns.yaml", encoding="utf-8")
    if prompt_type == "요약":
        prompt = load_prompt("summary.yaml", encoding="utf-8")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    return chain

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.chat_message("user").write(user_input)
    chain = create_chain(selected_prompt)
    response = chain.stream({"question": user_input})

    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""

        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    add_message("user",user_input)
    add_message("assistant",ai_answer)