""" This example shows how to use the map-reduce chain to summarize a document. """

import os

import langchain
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

langchain.debug = True

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model="gpt-3.5-turbo"
    )

pdf_file_path = "path/to/pdf/file"
pdf_loader = PyPDFLoader(pdf_file_path)

docs = pdf_loader.load_and_split()

chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.invoke(docs)

langchain.debug = False
