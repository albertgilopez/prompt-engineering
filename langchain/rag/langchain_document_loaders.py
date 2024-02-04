""" Example of using document loaders to load data from different sources. """


# https://python.langchain.com/docs/integrations/document_loaders/web_base
from langchain_community.document_loaders import WebBaseLoader

# from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders import WikipediaLoader

from dotenv import load_dotenv
load_dotenv()

# DOCUMENT LOADERS: https://python.langchain.com/docs/modules/data_connection/document_loaders/

# loader = TextLoader(file_path="path/to/text/file")
# documents = loader.load()

# loader = WikipediaLoader(title="LangChain")
# documents = loader.load()

loader = WebBaseLoader("https://www.uab.cat/enginyeria/")
documents = loader.load() # we can also use lazy_load() method for loading data into memory as needed

print(documents)