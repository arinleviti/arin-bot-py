

from app.schemas import Message
#os is a built-in Python module that provides a way to interact with the operating system, including reading environment variables. In this case, it's used to access the GROQ_API_KEY from the environment.
import os
import re #python's built-in regular expression library, used for pattern matching and string manipulation. In this case, it's used to split the reformulated questions based on a specific pattern (the '---' separator).
from typing import List
# chromadb is a library for working with ChromaDB, which is a database purpose-built for storing these text-to-vector conversions and answering "find me the closest matches" queries efficiently. It's an open-source project
import chromadb
from dotenv import load_dotenv
from groq import Groq
from collections import defaultdict

load_dotenv()

# --- Retrieval setup ---
# PersistentClient connects to the existing database built by ingest.py, like PrismaClient connects to a Postgres database. It doesn't create a new database; it just connects to the one that already exists on disk.
# PersistentClient — data is saved to actual files on disk, at whatever path you give it ("data/chroma_db" in your case), so it survives your program restarting.
# So "persistent" describes durability across restarts
# PersistentClient is the equivalent of PrismaClient in TypeScript, which connects to a database and allows you to query it. In this case, it's connecting to a ChromaDB database that was created by ingest.py.
client = chromadb.PersistentClient(path="data/chroma_db")
# inside the database I just connected to, give me the specific collection named arin_knowledge
collection = client.get_collection(name="arin_knowledge")

# --- Generation setup ---
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
ANSWER_NOT_FOUND_MARKER = "ANSWER NOT FOUND"

def retrieve_context(question, n_results=5):
    # Pass query_texts instead of query_embeddings — Chroma handles the embedding
    # internally using its built-in model, so we don't need sentence-transformers at all
    # collection.query(...) = you're asking it a question, and what comes back is a dictionary, not just a plain list of matching texts
    # That dictionary bundles together several parallel pieces of information about the matches, keyed by name — "documents" (the actual matched text), and others like "metadatas" 
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )
    return results["documents"][0]


def question_main_logic (question: str, history: List[Message] = None) -> str:
    answer = answer_question(question, history)
    if ANSWER_NOT_FOUND_MARKER in answer:
        print(f"[retry] no answer for '{question}' — reformulating")
        reformulated_questions = reformulate_question(question)
        if reformulated_questions:
            last_index = len(reformulated_questions) - 1
                # Only the last reformulated attempt switches the prompt back to
                # the current/friendly fallback instruction. Every attempt before
                # that still uses the marker, so the loop can detect failure and
                # move on to the next reformulation.
                # enumerate lets you loop through a list while getting both the index and the item.
            for i, rq in enumerate(reformulated_questions):
                is_last_attempt = (i == last_index)
                answer =answer_question(rq, history, use_current_fallback=is_last_attempt,)
            
                if ANSWER_NOT_FOUND_MARKER not in answer:
                    print(f"[retry] reformulation #{i + 1} succeeded: '{rq}'")
                    break
    return answer

# None makes the parameter optional. in TS we use ? to make a parameter optional, but in Python we use None as the default value. If the caller doesn't provide a value for history, it will be None.
def answer_question(question: str, history: List[Message] = None, use_current_fallback: bool = False,) -> str:

    if history is None:
        history = []

    context_chunks = retrieve_context(question)
    context_text = "\n\n".join(context_chunks)

    if use_current_fallback:
         # Friendly/final mode: the graceful-fallback instruction and the positive
        # skill-framing instruction are the same "voice" (how to handle a gap
        # nicely for the visitor), so they belong together here.
        fallback_instruction = (
            "If the topic is genuinely absent from the context, suggest the visitor reach out to Arin directly.\n\n"
            "IMPORTANT: If someone asks about a skill, info or technology Arin hasn't listed, do not simply say he doesn't know it. "
            "Frame it honestly but compellingly: Arin is entirely self-taught — no CS degree, no bootcamp. "
            "He built his way into AI engineering from film production and game development through sheer determination. "
            "In a recent example, he went from no Python experience to building and deploying a full RAG pipeline "
            "— chunking, embeddings, vector database, LLM integration, Docker, Cloud Run — in a matter of days. "
            "That is his learning velocity. A technology he hasn't used yet is not a red flag; "
            "it is simply the next thing on a very short list, and his track record shows exactly how fast that list shrinks. "
            "The recruiters who have hired him have consistently been technical people who recognised this immediately.\n\n"
        )
    else:
        fallback_instruction =  (
            f"If the topic is genuinely absent from the context, respond with exactly the text "
            f"{ANSWER_NOT_FOUND_MARKER} and nothing else — no punctuation, no explanation, no extra words.\n\n"
        )

    system_prompt = (
        "You are a helpful, friendly assistant on Arin Leviti's portfolio website. "
        "Your job is to represent Arin honestly and positively to visitors — recruiters, collaborators, and anyone curious about his work. "
        "Answer questions using the context provided below as your primary source. "
        + fallback_instruction +
        "Keep answers focused and skimmable — aim for 4 sentences maximum."
        "Only go longer if the visitor explicitly asks for more detail.\n\n"
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
        model="openai/gpt-oss-120b",
        messages=messages,
    )
    return response.choices[0].message.content

def reformulate_question(question: str) -> str:

    reformulation_prompt = (
    "The following question could not be answered from the available "
    "reference documents. Generate exactly 3 alternative phrasings of the same "
    "underlying question"
    "that might match how the source documents describe this topic."
    "Separate the 3 phrasings with a line containing only ---.\n\n"
    f"Original question: {question}"
    ) 
    response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": reformulation_prompt}]
    )
    raw = response.choices[0].message.content
    raw_split = re.split(r"\s*-{3,}\s*", raw)
    if len(raw_split) < 3:
        return []
    return [q.strip() for q in raw_split[:3]]


if __name__ == "__main__":
    question = "what AI agent work has arin done?"
    print(f"Question: {question}\n")
    print(f"Answer: {answer_question(question)}")