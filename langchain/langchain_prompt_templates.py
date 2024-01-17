# https://python.langchain.com/docs/modules/model_io/prompts/quick_start
# !pip install langchain

import os 
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = '''
    Eres un asistente de servicio al cliente de un banco. 
    Sabes muchos idiomas pero siempre respondes en el idioma del cliente.
    En esta oportunidad el cliente habla {language}.
    La consulta del cliente es:
    "{user_question}"
    '''

prompt_template = PromptTemplate.from_template(SYSTEM_PROMPT)
prompt = prompt_template.format(language='español', user_question='¿Cuánto dinero tengo en mi cuenta?')

result = llm.invoke(prompt)
print(result)
