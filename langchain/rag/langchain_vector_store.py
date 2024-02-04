""" Sample code to use the vector store to search for similar documents. """

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import CharacterTextSplitter

import json

from dotenv import load_dotenv
load_dotenv()

# https://arxiv.org/abs/2310.06825
loader = ArxivLoader(query="2310.06825")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
query = "What is this document about?"

similar_vectors = vectorstore.similarity_search(query)
similar_vectors_scoring = vectorstore.similarity_search_with_score(query)

print(similar_vectors[0].page_content)
print(similar_vectors_scoring[0][1])

# print('Similarity search:')
# print(vectorstore.similarity_search(query))

# print('Similarity search with score:')
# print(vectorstore.similarity_search_with_score(query))

# TUTORIAL https://www.gettingstarted.ai/tutorial-chroma-db-best-vector-database-for-langchain-store-embeddings/