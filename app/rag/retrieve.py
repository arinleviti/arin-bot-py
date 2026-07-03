import chromadb
from sentence_transformers import SentenceTransformer

# Open the existing database — note: get_collection, not get_or_create_collection,
# since we expect it to already exist from running ingest.py
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection(name="arin_knowledge")

model = SentenceTransformer("all-MiniLM-L6-v2")  # Load the pre-trained model for generating embeddings

query = "what does arin do in his spare time?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    # creates a list with only one element, the query embedding, since we only have one query
    query_embeddings=[query_embedding],
    n_results=5
)  # Generate an embedding for the query

documents  =results['documents'][0]  # Get the documents from the results
metadata = results['metadatas'][0]  # Get the metadata from the results

for doc, metadata in zip(documents, metadata):
    print(f"--- from {metadata['source']} ({metadata['heading']}) ---")
    print(doc)
    print()
