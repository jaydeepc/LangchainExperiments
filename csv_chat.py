import os
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI


load_dotenv()
filepath = "flipkart.csv"
llm = ChatOpenAI(model_name="gpt-3.5-turbo") #OpenAI(temperature=0)
agent = create_csv_agent(llm=llm, path=filepath, verbose=True)
agent.run("How many total reviews are there from India? Just give me the number")