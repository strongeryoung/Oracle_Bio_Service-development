import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType

os.environ["OPENAI_API_KEY"] = "apikey"

# Streamlit 웹 제목 설정
st.title("CSV 데이터 분석 웹 서비스")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type="csv")

if uploaded_file:
    # CSV 파일을 데이터프레임으로 읽기
    df = pd.read_csv(uploaded_file)
    st.write("### 업로드된 데이터 미리보기:")
    st.dataframe(df.head())

    # LangChain 데이터프레임용 분석 에이전트 생성
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model='gpt-4o'),
        df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    # 사용자 데이터 분석 질문 입력
    st.write("### 데이터 분석 질문 입력")
    user_query = st.text_input("질문 입력", placeholder="질문을 입력하세요.")

    # 분석 수행 및 결과 출력
    if st.button("분석 시작"):
        if user_query:
            with st.spinner("분석 중..."):
                result = agent.run(user_query)
                st.write("### 분석 결과:")
                st.write(result)

# 실행: 'streamlit run csv_reader.py'