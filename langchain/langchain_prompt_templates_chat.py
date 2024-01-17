# https://python.langchain.com/docs/modules/model_io/prompts/quick_start
# !pip install langchain

import os 
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = '''
    Eres un asistente de servicio al cliente de un banco. 
    Sabes muchos idiomas pero siempre respondes en el idioma del cliente.
    En esta oportunidad el cliente habla {language}.
    El usuario se llama {user_name}
'''

chat_template = ChatPromptTemplate.from_messages([
    ("system",SYSTEM_PROMPT),
    ("user", "{user_question}")
])

messages = chat_template.format_messages(language='español',
                                         user_question="¿Cuánto dinero tengo en mi cuenta?",
                                         user_name="Julio")

response = llm.invoke(messages)
print(response.content)

# LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.warn_deprecated(