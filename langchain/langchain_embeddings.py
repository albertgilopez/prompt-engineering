from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
text = "This is a simple query"
query_result = embeddings.embed_query(text)

# print(query_result)
# print(len(query_result))

words = ["cat", "dog", "computer", "animal"]
embeddings = OpenAIEmbeddings()
doc_vectors = embeddings.embed_documents(words)

# print(doc_vectors)
# print(len(doc_vectors))

from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd 

x = np.array(doc_vectors)
dist = squareform(pdist(x))

df = pd.DataFrame(
    data=dist,
    index=words,
    columns=words
)

df.style.background_gradient(cmap='coolwarm')