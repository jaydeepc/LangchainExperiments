from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE


# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def generate_response(message):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm)
    
    conversation = ConversationChain(
        llm=llm, 
        memory=st.session_state.entity_memory,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        verbose=True
    )
    # print(conversation)
    ai_response = conversation.predict(input=message)
    return ai_response

def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text    

def main():
    load_dotenv()
    st.header('Chat with the AI')
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