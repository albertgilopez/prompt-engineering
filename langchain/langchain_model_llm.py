# https://python.langchain.com/docs/get_started/introduction
# !pip install langchain

import os 
from langchain_openai import OpenAI

# Importar la librería dotenv para cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=OPENAI_API_KEY)

result = llm.predict("La mejor forma de empezar el día es ")
print(result)

# LangChainDeprecationWarning: The function `predict` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead. warn_deprecated(

result = llm.invoke("La mejor forma de empezar el día es ")
print(result)
