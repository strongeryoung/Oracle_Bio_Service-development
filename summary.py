# 검색 증강 생성
# 실습 1. PDF 요약 서비스
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback  # 최신 버전 호환

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "apikey"

# 텍스트를 분할하여 임베딩 후 FAISS 벡터 DB 생성
def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    documents = FAISS.from_texts(chunks, embeddings)
    return documents


# Streamlit 메인 애플리케이션
def main():
    st.title("GPT-4 기반 PDF 요약기")
    st.divider()

    # PDF 파일 업로드
    pdf = st.file_uploader('PDF 파일을 업로드 하세요.', type='pdf')

    if pdf:
        # PDF 텍스트 추출
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # 텍스트 처리 후 벡터 DB 생성
        documents = process_text(text)

        # 요약 쿼리
        query = "업로드 된 PDF 파일의 내용을 약 3~5 문장으로 요약해 주세요."

        if query:
            # 벡터 유사도 기반 문서 검색
            docs = documents.similarity_search(query, k=3)

            # GPT-4 모델로 질의응답 체인 설정
            llm = ChatOpenAI(model="gpt-4", temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            # 요약 생성 및 API 사용 비용 출력
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader("요약 결과:")
            st.write(response)


# 애플리케이션 수행
if __name__ == '__main__':
    main()

# 터미널에 'streamlit run summary.py'