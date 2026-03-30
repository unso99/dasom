import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import tempfile
import os

from retriever import build_retriever
from multimodal import stream_multimodal_answer

load_dotenv()


# ── Pydantic 모델 ──────────────────────────────────────────────────────────────

class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    company: str = Field(description="메일을 보낸 사람의 회사 정보")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")


# ── 체인 팩토리 ────────────────────────────────────────────────────────────────

def create_email_chain():
    parser = PydanticOutputParser(pydantic_object=EmailSummary)
    prompt = load_prompt("email_summary.yaml", encoding="utf-8")
    prompt = prompt.partial(format=parser.get_format_instructions())
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return prompt | llm | parser


def create_rag_chain(retriever):
    """업로드된 PDF를 기반으로 답변하는 RAG 체인을 생성합니다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신의 이름은 'Dasom'이며, 주어진 문서를 기반으로 사용자의 질문에 답변하는 AI 어시스턴트입니다.

## 응답 원칙
- 반드시 아래 제공된 문서(context) 내용을 근거로 답변합니다.
- 문서에 없는 내용은 "해당 내용은 문서에서 찾을 수 없습니다."라고 솔직히 말합니다.
- 한국어로 자연스럽고 명확하게 답변합니다.
- 출처가 되는 내용은 요약해서 함께 제시합니다.

## 문서 내용
{context}"""),
        ("user", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model="gpt-4o", temperature=0)
        | StrOutputParser()
    )


def create_chain(prompt_type):
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


# ── 렌더링 헬퍼 ───────────────────────────────────────────────────────────────

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


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# ── 앱 레이아웃 ────────────────────────────────────────────────────────────────

st.title("Dasom")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "image_path" not in st.session_state:
    st.session_state["image_path"] = None

with st.sidebar:
    clear_btn = st.button("대화 초기화")

    selected_prompt = st.selectbox(
        "프롬프트를 선택해 주세요.",
        ("기본모드", "SNS 게시글", "요약", "이메일 요약", "PDF 문서 QA", "이미지 QA"),
        index=0,
    )

    # ── PDF 업로드 UI (PDF 문서 QA 선택 시에만 표시) ──
    if selected_prompt == "PDF 문서 QA":
        st.divider()
        uploaded_file = st.file_uploader(
            "📄 PDF 파일을 업로드하세요",
            type=["pdf"],
            help="업로드한 PDF를 기반으로 질문에 답변합니다.",
        )

        if uploaded_file:
            # 이미 같은 파일이 처리된 경우 재처리 방지
            if st.session_state.get("uploaded_filename") != uploaded_file.name:
                with st.spinner("📚 문서를 분석하는 중입니다..."):
                    # 임시 파일로 저장 후 retriever 생성
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        st.session_state["retriever"] = build_retriever(tmp_path)
                        st.session_state["uploaded_filename"] = uploaded_file.name
                        st.success(f"✅ '{uploaded_file.name}' 분석 완료!")
                    except Exception as e:
                        st.error(f"❌ 문서 처리 중 오류가 발생했습니다: {e}")
                    finally:
                        os.unlink(tmp_path)
            else:
                st.success(f"✅ '{uploaded_file.name}' 분석 완료!")
        else:
            # 파일이 없으면 retriever 초기화
            st.session_state["retriever"] = None
            st.session_state.pop("uploaded_filename", None)
            st.info("📎 PDF를 업로드하면 문서 기반 QA를 시작합니다.")

    elif selected_prompt == "이메일 요약":
        st.info("📧 이메일 원문을 채팅창에 붙여넣으세요.")

    # ── 이미지 업로드 UI (이미지 QA 선택 시에만 표시) ──
    elif selected_prompt == "이미지 QA":
        st.divider()
        selected_model = st.selectbox(
            "🤖 모델 선택",
            ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o"],
            index=0,
        )
        system_prompt = st.text_area(
            "시스템 프롬프트",
            "당신은 이미지를 분석하는 AI 어시스턴트입니다.\n이미지에 대해 한국어로 친절하고 상세하게 답변해 주세요.",
            height=150,
        )
        uploaded_image = st.file_uploader(
            "🖼️ 이미지를 업로드하세요",
            type=["jpg", "jpeg", "png"],
            help="업로드한 이미지를 기반으로 질문에 답변합니다.",
        )

        if uploaded_image:
            # 캐시 디렉토리 생성
            os.makedirs(".cache/files", exist_ok=True)
            image_path = f".cache/files/{uploaded_image.name}"

            if st.session_state.get("uploaded_image_name") != uploaded_image.name:
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.read())
                st.session_state["image_path"] = image_path
                st.session_state["uploaded_image_name"] = uploaded_image.name

            st.image(image_path, caption=uploaded_image.name, use_container_width=True)
        else:
            st.session_state["image_path"] = None
            st.session_state.pop("uploaded_image_name", None)
            st.info("🖼️ 이미지를 업로드하면 이미지 기반 QA를 시작합니다.")

if clear_btn:
    st.session_state["messages"] = []

print_messages()

# 입력 placeholder 동적 설정
placeholder_map = {
    "이메일 요약": "이메일 내용을 붙여넣으세요!",
    "PDF 문서 QA": "문서에 대해 궁금한 내용을 물어보세요!",
    "이미지 QA": "이미지에 대해 궁금한 내용을 물어보세요!",
}
placeholder = placeholder_map.get(selected_prompt, "궁금한 내용을 물어보세요!")

user_input = st.chat_input(placeholder)

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):

        # ── PDF 문서 QA 분기 ──────────────────────────────────────────────────
        if selected_prompt == "PDF 문서 QA":
            retriever = st.session_state.get("retriever")
            if retriever is None:
                ai_answer = "⚠️ 먼저 사이드바에서 PDF 파일을 업로드해 주세요."
                st.markdown(ai_answer)
            else:
                container = st.empty()
                ai_answer = ""
                rag_chain = create_rag_chain(retriever)
                for token in rag_chain.stream(user_input):
                    ai_answer += token
                    container.markdown(ai_answer)

        # ── 이미지 QA 분기 ────────────────────────────────────────────────────
        elif selected_prompt == "이미지 QA":
            image_path = st.session_state.get("image_path")
            if image_path is None:
                ai_answer = "⚠️ 먼저 사이드바에서 이미지를 업로드해 주세요."
                st.markdown(ai_answer)
            else:
                container = st.empty()
                ai_answer = ""
                for token in stream_multimodal_answer(
                    image_path=image_path,
                    user_prompt=user_input,
                    system_prompt=system_prompt,
                    model_name=selected_model,
                ):
                    ai_answer += token
                    container.markdown(ai_answer)

        # ── 이메일 요약 분기 ──────────────────────────────────────────────────
        elif selected_prompt == "이메일 요약":
            chain = create_email_chain()
            try:
                result = chain.invoke({"email_conversation": user_input})
                render_email_summary(result)
                ai_answer = (
                    f"발신자: {result.person} / 회사: {result.company} / 요약: {result.summary}"
                )
            except Exception:
                ai_answer = (
                    "⚠️ 이메일 형식의 내용을 입력해 주세요.\n\n"
                    "From, Subject, 본문이 포함된 이메일을 붙여넣으면 요약해드립니다."
                )
                st.markdown(ai_answer)

        # ── 일반 모드 분기 ────────────────────────────────────────────────────
        else:
            chain = create_chain(selected_prompt)
            container = st.empty()
            ai_answer = ""
            for token in chain.stream({"question": user_input}):
                ai_answer += token
                container.markdown(ai_answer)

    add_message("user", user_input)
    add_message("assistant", ai_answer)