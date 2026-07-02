from app.schemas import Message
import os
from typing import List
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# --- Retrieval setup ---
# PersistentClient connects to the existing database built by ingest.py
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection(name="arin_knowledge")

# --- Generation setup ---
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def retrieve_context(question, n_results=3):
    # Pass query_texts instead of query_embeddings — Chroma handles the embedding
    # internally using its built-in model, so we don't need sentence-transformers at all
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )
    return results["documents"][0]


# None makes the parameter optional. in TS we use ? to make a parameter optional, but in Python we use None as the default value. If the caller doesn't provide a value for history, it will be None.
def answer_question(question: str, history: List[Message] = None) -> str:

    if history is None:
        history = []

    context_chunks = retrieve_context(question)
    context_text = "\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful assistant on Arin's portfolio website. "
        "Answer the user's question using ONLY the context provided below. "
        "If the context doesn't contain the answer, say you don't have that information.\n\n"
        f"Context:\n{context_text}"
    )

    # Create the system message as a Message instance — Pydantic validates it on creation
    system_message = Message(role="system", content=system_prompt)

    # Start the messages list with the system prompt.
    # .model_dump() converts the Message instance into a plain object that Groq understands
    messages = [system_message.model_dump()]

    # Add all previous conversation turns to the messages list.
    # .model_dump() converts each Message instance into a plain object for Groq.
    # extend() adds each item individually, oldest message first
    messages.extend([m.model_dump() for m in history])

    # Finally, append the brand new question from the user as the last item.
    # This is a plain object directly — no Message instance needed here since
    # it's the current turn, not something we're storing or validating elsewhere.
    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    question = "what AI agent work has arin done?"
    print(f"Question: {question}\n")
    print(f"Answer: {answer_question(question)}")