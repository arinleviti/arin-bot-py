import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Retrieval setup (same as retrieve.py) ---
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection(name="arin_knowledge")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Generation setup ---
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def retrieve_context(question, n_results=3):
    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results["documents"][0]


def answer_question(question):
    context_chunks = retrieve_context(question)
    context_text = "\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful assistant on Arin's portfolio website. "
        "Answer the user's question using ONLY the context provided below. "
        "If the context doesn't contain the answer, say you don't have that information.\n\n"
        f"Context:\n{context_text}"
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    question = "what AI agent work has arin done?"
    print(f"Question: {question}\n")
    print(f"Answer: {answer_question(question)}")