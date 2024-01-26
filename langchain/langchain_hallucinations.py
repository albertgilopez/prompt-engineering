""" This example shows how to use the LLMCheckerChain to check for hallucinations. """

import os

import langchain

from langchain_openai import ChatOpenAI
from langchain.chains import LLMCheckerChain

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

langchain.debug = True

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model="gpt-3.5-turbo"
    )

text = "What type of mamal lays the biggest eggs?"

checker_chain = LLMCheckerChain.from_llm(llm=llm, verbose=True)
response = checker_chain.invoke(text)

print(response)

langchain.debug = False