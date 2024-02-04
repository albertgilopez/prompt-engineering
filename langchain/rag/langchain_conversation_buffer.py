""" Example of using a conversation buffer memory with langchain """

# https://python.langchain.com/docs/modules/memory/
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# from langchain.memory import ConversationBufferWindowMemory

from langchain_openai import OpenAI
from langchain.prompts.prompt import PromptTemplate

from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv
load_dotenv()

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.invoke(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
        return result["response"]

llm = OpenAI(temperature=0)

# https://milvus.io/blog/conversational-memory-in-langchain.md
# memory = ConversationBufferMemory()
# memory = ConversationBufferWindowMemory(k=1)

# Creating a conversation chain with memory

template = """
The following is a conversation between a human and an AI assistant.
The AI is talkative and provides lots of specific detals from its context.
If the AI does not know the answer, it will ask the human for clarification or says it does not know.

Current conversation:
{history}

Human: {input}
AI Assistant: """

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
    # memory=ConversationBufferWindowMemory(k=1),
    verbose=True,
)
response = count_tokens(
    conversation, 
    "Hello, how are you?"
)

response = count_tokens(
    conversation, 
    "What is your name?"
)

print(response)
