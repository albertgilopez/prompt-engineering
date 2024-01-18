# https://python.langchain.com/docs/modules/agents/quick_start
# !pip install langchain

import os 

import langchain
from langchain_openai import ChatOpenAI
# from langchain.agents import tools, initialize_agent
from langchain.agents import tools, tool, load_tools
from langchain.agents import AgentType

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from datetime import date

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model="gpt-3.5-turbo"
    )

@tool
def get_date(dummy: str) -> str:
    """
    Returns the current date. The dummy argument is ignored.
    """
    return str(date.today())

tools = load_tools(["wikipedia"])

# langchain.debug = True
# agent = initialize_agent(
#     tools=tools + [get_date],
#     llm=llm,
#     agent= AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # ReAct, ejecutan una acción, lo observan y actuan
#     verbose = True,
#     handle_parsing_errors=True # Si se encuentra un error en el proceso, se devuelve el input inicial
# )

# LangChainDeprecationWarning: The function `initialize_agent` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use Use new agent constructor methods like create_react_agent, create_json_agent, create_structured_chat_agent, etc. instead.

langchain.debug = True
prompt = hub.pull("hwchase17/react")
# https://blog.langchain.dev/langchainhub/

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input" : "Cuál fue el primer presidente de Colombia?"})

langchain.debug = False

# ZERO_SHOT_REACT_DESCRIPTION cuando se quiere que el agente reaccione sin ningun ejemplo
# ONE_SHOT_REACT_DESCRIPTION cuando se quiere que el agente reaccione con un ejemplo
# FEW_SHOT_REACT_DESCRIPTION cuando se quiere que el agente reaccione con algunos ejemplos
