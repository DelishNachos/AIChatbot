import chromadb
from chromadb.utils import embedding_functions
import numpy as np

# Set up embedding function and ChromaDB
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient("chroma_data/")

# Specify the collection name
collection_name = "memory"

# Retrieve or create the collection
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)

# Example query using an embedding
#query_embedding = [embedding_function("your_query_text")]
query_embedding_list = embedding_function("query_text")
#print("Type of Query Embedding:", type(query_embedding))
#query_embedding = np.array(embedding_list)
# results = collection.query(query_embeddings=query_embedding_list, n_results=1)
results = collection.query(query_texts="pirate", n_results=2)

list_things = results['documents'][0]
print(list_things)
print(type(list_things))
print(''.join(list_things))
print(collection.count())
# Access the documents returned by the query
#queried_documents = results.documents

#Print or process the queried documents
# for document in results:
#     print(document)
