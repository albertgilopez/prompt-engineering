""" Example of using the langchain memory. """

# https://python.langchain.com/docs/modules/memory/
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# Creating a conversation chain with memory
memory = ConversationBufferMemory()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    streaming=True,
)

chain = ConversationChain(llm=llm, memory=memory)

# User inputs a message
user_input = "Hi, how are you?"
# Processing the user input in the conversation chain
response = chain.predict(input=user_input)

print(response)

# User inputs another message
user_input = "I'm doing great, thank you for asking."
# Processing the user input in the conversation chain
response = chain.predict(input=user_input)

print(response)
print(memory.chat_memory.messages)