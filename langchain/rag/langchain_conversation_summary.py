""" Script to test the conversation summary memory module. """

# https://python.langchain.com/docs/modules/memory/types/summarys
from langchain.memory import ConversationSummaryMemory

from langchain_openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(temperature=0)

memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "Hello"}, {"output": "Hey"})

print(memory.load_memory_variables({}))