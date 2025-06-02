## PDF 질의응답 서비스
# 2. Streamlit 기반 웹 서비스 구현
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "apikey"

# Streamlit 제목 설정
st.title("PDF 기반 GPT-4 질의응답")

# PDF 파일 업로드
pdf = st.file_uploader('PDF 파일을 업로드 하세요.', type='pdf')

if pdf:
    # PDF 텍스트 추출
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # 텍스트를 작은 단위(chunk)로 분할
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # 임베딩을 이용한 벡터 저장소 생성
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    # 사용자 질의 입력
    query = st.text_input("PDF 내용에 대해 질문하세요:")

    if query:
        # 유사도 겁색으로 관련 chunk 추출
        docs = vector_store.similarity_search(query, k=3)

        # GPT-4 모델로 질의응답 체인 설정
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        qa_chain = load_qa_chain(llm, chain_type='stuff')

        # 질의에 대한 답변 생성
        response = qa_chain.run(input_documents=docs, question=query)

        # 결과 표시
        st.subheader("답변 결과")
        st.write(response)

# 실행 'streamlit run qna.py'