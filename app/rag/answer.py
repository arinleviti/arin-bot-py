from app.schemas import Message
#os is a built-in Python module that provides a way to interact with the operating system, including reading environment variables. In this case, it's used to access the GROQ_API_KEY from the environment.
import os
# json lets us parse the arguments the model sends back for a tool call — they
# arrive as a JSON string, not a ready-to-use Python dict, so we need to decode them.
import json
from typing import List
# chromadb is a library for working with ChromaDB, which is a database purpose-built for storing these text-to-vector conversions and answering "find me the closest matches" queries efficiently. It's an open-source project
import chromadb
from dotenv import load_dotenv
from groq import Groq
from collections import defaultdict
import re
from app.tools.send_email import notify_arin

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

# The literal marker the model is told to return when it can't answer from context.
# Kept as a constant so the prompt text and the detection checks below can never drift apart.
ANSWER_NOT_FOUND_MARKER = "ANSWER NOT FOUND"

# --- Tool definitions ---
# This is the "menu" of functions we offer the model. It's pure description —
# metadata the model reads to decide WHETHER and WHEN to call this function.
# It does NOT execute anything by itself; the actual Python function
# (notify_arin, imported above from app.tools.send_email) is what runs when
# we detect the model asked for it.
#
# TS comparison: this is like handing the model a typed interface signature
# plus a docstring, without giving it the function body — it only knows the
# shape and purpose, not the implementation.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "notify_arin",
            "description": (
                "Sends Arin a real email notification about a visitor who wants to hire "
                "or contact him. Only call this once the visitor has actually provided "
                "contact info (email or similar) — either in their current message or "
                "an earlier one in this conversation. If they have expressed interest "
                "but have NOT given contact info yet, do not call this tool — ask them "
                "for their email first, and call this tool only after they provide it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "visitor_message": {
                        "type": "string",
                        "description": "A short summary of what the visitor said or wants.",
                    },
                    "visitor_contact": {
                        "type": "string",
                        "description": "The visitor's email or other contact info. Required — do not call this tool without it.",
                    },
                },
                "required": ["visitor_message", "visitor_contact"],
            },
        },
    }
]


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


def question_main_logic(question: str, history: List[Message] = None) -> str:
    # Tools are enabled on this first call — this is where we expect a
    # hiring-interest message to be caught, most of the time.
    answer = answer_question(question, history, enable_tools=True)  # noqa: keep enable_tools explicit
    if ANSWER_NOT_FOUND_MARKER in answer:
        print(f"[retry] no answer for '{question}' — reformulating")
        reformulated_questions = reformulate_question(question)
        if reformulated_questions:
            last_index = len(reformulated_questions) - 1
            for i, rq in enumerate(reformulated_questions):
                # Only the last reformulated attempt switches the prompt back to
                # the current/friendly fallback instruction, AND is the only
                # retry attempt where tools are enabled again — the middle
                # attempts are pure retrieval probes, not user-facing messages,
                # so we don't want a tool call sneaking in there.
                is_last_attempt = (i == last_index)
                answer = answer_question(
                    rq,
                    history,
                    use_current_fallback=is_last_attempt,
                    enable_tools=is_last_attempt,
                )

                if ANSWER_NOT_FOUND_MARKER not in answer:
                    print(f"[retry] reformulation #{i + 1} succeeded: '{rq}'")
                    break
    return answer

# None makes the parameter optional. in TS we use ? to make a parameter optional, but in Python we use None as the default value. If the caller doesn't provide a value for history, it will be None.
def answer_question(
    question: str,
    history: List[Message] = None,
    use_current_fallback: bool = False,
    enable_tools: bool = False,
) -> str:

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
        # Marker mode: a single, unambiguous instruction — no competing framing
        # instruction that could tempt the model to write prose instead of the
        # exact marker.
        fallback_instruction = (
            f"If the topic is genuinely absent from the context, respond with exactly the text "
            f"{ANSWER_NOT_FOUND_MARKER} and nothing else — no punctuation, no explanation, no extra words.\n\n"
        )
        if enable_tools:
            # Only mention notify_arin when the tool is actually being offered
            # on this call (enable_tools=True). On the middle reformulation
            # attempts, enable_tools is False and tools=None is sent to the
            # API — telling the model to call a tool it doesn't have access to
            # would just be confusing, unusable instruction.
            fallback_instruction += (
                "EXCEPTION: this fallback rule applies ONLY to factual questions about Arin's "
                "background, skills, or work. If the visitor is instead expressing interest in "
                "hiring or contacting Arin, that is NOT a context-lookup question — do not "
                "check the context or return the marker for it. If they have already provided "
                "contact info (in this message or earlier in the conversation), call the "
                "notify_arin tool now. If they have NOT provided contact info yet, do not call "
                "the tool — instead, ask them for their email so you can pass it along.\n\n"
            )

    # Same reasoning as the EXCEPTION block above: this instruction only makes
    # sense when the tool is actually available on this call. On the middle
    # reformulation attempts (enable_tools=False), leave it out entirely.
    tool_contact_instruction = (
        "IMPORTANT: Whenever a visitor asks about Arin's availability for hire, "
        "working with him, or expresses ANY interest related to hiring or contacting "
        "him — even indirectly, like asking 'is he available for hire?' — proactively "
        "invite them to share their email at the end of your reply, even while also "
        "answering informatively from context. Do NOT wait for them to explicitly ask "
        "to speak with him before offering this. Do not call the notify_arin tool until "
        "the visitor has actually provided contact info. If they've expressed interest "
        "but haven't given an email yet, ask for it in your reply instead of calling the "
        "tool — you'll get another chance to call it once they reply with their contact info.\n\n"
    ) if enable_tools else ""

    system_prompt = (
        "You are a helpful, friendly assistant on Arin Leviti's portfolio website. "
    "Your ONLY job is to answer questions about Arin himself — his background, skills, "
    "projects, and availability — using ONLY the context provided below. "
    "Do NOT use your own general knowledge to explain unrelated concepts, terms, "
    "technologies, or how-to questions, even if you know the answer confidently — that "
    "is out of scope for this assistant, no matter how helpful it might seem. If a "
    "visitor asks something unrelated to Arin, or something this context doesn't cover, "
    "treat it exactly like an absent topic under the rules below — never improvise an "
    "answer from outside knowledge.\n\n"
    + fallback_instruction +
    "Keep answers focused and skimmable — aim for 4 sentences maximum."
    "Only go longer if the visitor explicitly asks for more detail.\n\n"
    + tool_contact_instruction +
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

    # Build the create() arguments as a dict, so we can OMIT tools/tool_choice
    # entirely when disabled, rather than passing them as None. Groq's API
    # strictly requires tool_choice to be one of "none"/"auto"/"required" —
    # it rejects tool_choice=null outright, even though None seems like a
    # reasonable way to say "no tools." Omitting the key avoids this entirely.
    create_kwargs = {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
    }
    if enable_tools:
        create_kwargs["tools"] = TOOLS
        create_kwargs["tool_choice"] = "auto"

    response = groq_client.chat.completions.create(**create_kwargs)
    response_message = response.choices[0].message

    # If the model decided to call a tool, response_message.tool_calls will be
    # a non-empty list instead of None. This is the branch that actually runs
    # your real Python code and a real side effect (sending an email).
    if response_message.tool_calls:
        # Add the model's own tool-call request to the conversation history we're
        # building for this call — the API requires this so the follow-up call
        # below has the full back-and-forth, not just the result.
        # exclude_none=True matters here: .model_dump() by default includes
        # every field the Pydantic class defines, even ones we never set that
        # are just sitting at None (e.g. "annotations", "refusal"). Groq's SDK
        # reuses an OpenAI-compatible schema with more optional fields than
        # Groq's own API accepts as INPUT — sending one of those null fields
        # back causes a 400 "property is unsupported" error. Dropping None
        # fields entirely leaves only what we actually set: role, content,
        # tool_calls.
        messages.append(response_message.model_dump(exclude_none=True))

        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "notify_arin":
                # The model sends arguments back as a JSON string, not a dict —
                # same as JSON.parse() in TS, we have to decode it ourselves.
                args = json.loads(tool_call.function.arguments)
                success = notify_arin(
                    visitor_message=args.get("visitor_message", question),
                    visitor_contact=args.get("visitor_contact"),
                )
                # Every tool call must get a matching "tool" role message back,
                # linked by tool_call_id, so the model knows which call this
                # result belongs to (relevant if it called more than one tool).
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"success": success}),
                })

        # Second call: no tools this time, just asking the model to turn the
        # tool result into a natural final reply for the visitor.
        follow_up = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
        )
        return follow_up.choices[0].message.content

    return response_message.content
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
    # The model sometimes adds trailing spaces before the newline around the "---"
    # separator (Markdown's line-break syntax), e.g. "...moon?  \n---  \n...".
    # An exact "\n---\n" match misses that, so use a regex that tolerates any
    # whitespace (spaces, tabs) directly before/after the dashes on their own line.
    raw_split = re.split(r"\s*-{3,}\s*", raw)
    if len(raw_split) < 3:
        return []
    return [q.strip() for q in raw_split[:3]]


if __name__ == "__main__":
    question = "what AI agent work has arin done?"
    print(f"Question: {question}\n")
    print(f"Answer: {answer_question(question)}")