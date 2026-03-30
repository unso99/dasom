from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def build_retriever(file_path: str):
    """
    PDF 파일을 로드하고 FAISS 벡터스토어 기반 retriever를 반환합니다.

    Args:
        file_path (str): PDF 파일 경로

    Returns:
        retriever: FAISS vectorstore retriever
    """
    # 단계 1: 문서 로드
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: 벡터스토어 생성
    vectorstore = FAISS.from_documents(
        documents=split_documents,
        embedding=embeddings,
    )

    # 단계 5: retriever 반환
    retriever = vectorstore.as_retriever()
    return retriever