from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain import SQLDatabase, SQLDatabaseChain


# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'summary_memory' not in st.session_state:
    st.session_state['summary_memory'] = ""

def generate_response(message):
    dburi = "sqlite:///accounts.db"
    db = SQLDatabase.from_uri(dburi)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)    
    memory.save_context({"input": message}, {"output": st.session_state.summary_memory})
    
    db_chain = SQLDatabaseChain.from_llm(
        llm=llm,
        memory= memory,
        db=db, 
        verbose=True,
        top_k=20
    )
    
    ai_response = db_chain.run(st.session_state.summary_memory + message)
    memory=memory.load_memory_variables({})
    st.session_state.summary_memory = memory["history"]

    return ai_response

def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text    

def main():
    load_dotenv()
    st.header('Chat with Database')
    # st.sidebar.title('Choose models').subheader('Choose the model you want to use')
    user_input = get_text()
    
    if user_input:
        output = generate_response(user_input)

        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

if __name__ == '__main__':
    main()