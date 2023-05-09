from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat PDF', page_icon=':speech_balloon:')
    st.title('Local Chat PDF')
    st.header('Ask Chat PDF')

    pdf = st.file_uploader('Upload PDF', type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=850,
            length_function=len,
        )

        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()

        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_input = st.text_input('Ask Questions about the PDF')
        if user_input:
            docs = knowledge_base.similarity_search(user_input)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_input)
            st.write(response)

if __name__ == '__main__':
    main()