""" Example of using the retrievers module to get relevant documents from a list of words. """

from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import KNNRetriever

from dotenv import load_dotenv
load_dotenv()

# RETRIEVERS: https://python.langchain.com/docs/modules/data_connection/retrievers/

# Example of using the KNNRetriever
words = ["cat", "dog", "computer", "animal"]
retriever = KNNRetriever.from_texts(words, OpenAIEmbeddings())

result = retriever.get_relevant_documents("dog")
print(result)


# Example of using the PubMedRetriever  
from langchain_community.retrievers import PubMedRetriever

retriever = PubMedRetriever()
documents = retriever.get_relevant_documents("covid-19")

for document in documents:
    print(document.metadata["Title"]+"\n")


# Example of using Custom Retriever
from langchain.schema import Document, BaseRetriever

class MyRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> list[Document]:
        # Implement your own logic here
        # Retrieve and process documents based on the query
        # Return a list of relevant documents

        relevant_documents = []

        # Youur retrieval logic goes here

        return relevant_documents