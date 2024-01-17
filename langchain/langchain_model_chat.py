# https://python.langchain.com/docs/get_started/introduction
# !pip install langchain

import os 
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Importar la librería dotenv para cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# user | system | assistant | function
chatModel = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

messages = [

    SystemMessage(content="Tu eres un traductor de español a ingles"),
    HumanMessage(
        content='''Hola, como estas? ayudame a traducir esto: "Quiero aprender inteligencia artificial"'''),
    AIMessage(content="La traducción es: I want to learn artificial intelligence"),
    HumanMessage(
        content='''Gracias, y como se dice "no tengo mucho tiempo para aprender"?'''),

]

response = chatModel(messages)
print(response)

# LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:

# !pip install -U langchain-community
from langchain_community.chat_models import ChatOpenAI

chatModel = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

messages = [

    SystemMessage(content="Tu eres un traductor de español a ingles"),
    HumanMessage(
        content='''Hola, como estas? ayudame a traducir esto: "Quiero aprender inteligencia artificial"'''),
    AIMessage(content="La traducción es: I want to learn artificial intelligence"),
    HumanMessage(
        content='''Gracias, y como se dice "no tengo mucho tiempo para aprender"?'''),

]

response = chatModel(messages)
print(response.content)
