from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain


# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def generate_response(message, docs):
    llm = ChatOpenAI(model_name="gpt-4")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    ai_response = chain.run(input_documents=docs, question=message)
    return ai_response

def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text    

def main():
    load_dotenv()
    st.header("Chat with your PDF")

    pdf = st.file_uploader('Upload PDF', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=450,
            length_function=len,
        )

        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()

        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_input = get_text()

        print(user_input)

        if user_input:
            docs = knowledge_base.similarity_search(user_input)
            print(docs)
            output = generate_response(user_input, docs)
            # store the output 
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


if __name__ == '__main__':
    main()