# https://python.langchain.com/docs/modules/chains
# !pip install langchain

import os 
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llmOpenAI = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Simple Sequential Chain
prompt1 = ChatPromptTemplate.from_template(
    "Sugiereme un nombre para una empresa \
     que produce {product}?")

chain1 = LLMChain(llm=llmOpenAI, prompt=prompt1, output_key="company_name")

# Sequential Chain
prompt2 = ChatPromptTemplate.from_template(
    "Escribe un tagline para una empresa llamada {company_name}")
chain2 = LLMChain(llm=llmOpenAI, prompt=prompt2)

simple_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

result = simple_chain.run("aceite de oliva")
print(result)

# LangChainDeprecationWarning: The function `run` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead. warn_deprecated(

result = simple_chain.invoke("aceite de oliva")
print(result)