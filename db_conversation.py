import os
import streamlit as st
from dotenv import load_dotenv
from langchain import SQLDatabase, SQLDatabaseChain
from langchain.chat_models import ChatOpenAI


load_dotenv()
dburi = "sqlite:///chinook.db"
db = SQLDatabase.from_uri(dburi)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

db_chain.run("Give a financial report from the dataset. Write a detailed email to the director of Thoughtworks, with this report")
