""" Example of using the ConversationChain with multiple memory types."""

# https://python.langchain.com/docs/modules/memory/multiple_memory
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory

from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

from langchain_openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(temperature=0)

# Define ConversationBufferMemory (for retaining all past messages)
buffer_memory = ConversationBufferMemory(memory_key="chat_history_lines", input_key="input")

# Define ConversationSummaryMemory (for summarizing the conversation)
summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")

# Combine both memory types
memory = CombinedMemory(memories=[buffer_memory, summary_memory])

# CHAIN

_DEFAULT_TEMPLATE = """

The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context
If the AI does not know the answer to a question, it truthfully says it does not know.

Summary of conversation:
{history}

Current conversation:
{chat_history_lines}

Human: {input}

AI:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input", "chat_history_lines"],
    template=_DEFAULT_TEMPLATE,
)

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    memory=memory,
)

response = conversation.invoke("Hi")
response = conversation.invoke("Yes, I need help with something. Can you help me?")

print(response)