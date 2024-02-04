""" Simple example of using the langchain KG memory and conversation chain. """

# https://python.langchain.com/docs/modules/memory/types/kg
from langchain.memory import ConversationKGMemory

from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

from langchain_openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(temperature=0)

memory = ConversationKGMemory(llm=llm, return_messages=True)

memory.save_context({"input": "say hi to sam"}, {"output": "who is sam"})
memory.save_context({"input": "sam is a friend"}, {"output": "okay"})

print(memory.load_memory_variables({"input": "who is sam"}))
print(memory.get_current_entities("what's Sams favorite color?"))
print(memory.get_knowledge_triplets("her favorite color is red"))

# CHAIN

template = """
The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
conversation_with_kg = ConversationChain(
    llm=llm, verbose=True, prompt=prompt, memory=ConversationKGMemory(llm=llm)
)

response = conversation_with_kg.predict(input="Hi, what's up?")

print(response)

response = conversation_with_kg.predict(
    input="My name is James and I'm helping Will. He's an engineer."
)

print(response)

response = conversation_with_kg.predict(input="What do you know about Will?")

print(response)