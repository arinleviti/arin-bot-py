#uvicorn app.main:app --reload 

from app.schemas import Message, ChatRequest
 
#--reload — tells Uvicorn to automatically restart the server whenever you save changes to your code — equivalent to running Express with nodemon, instead of manually stopping/restarting node server.js every time you edit something.

#fastapi is like Express in that it provides a framework for defining routes and handling HTTP requests, but it also has built-in support for data validation, serialization, and automatic API documentation generation. FastAPI is designed to be fast and efficient, leveraging Python's type hints to provide better developer experience and performance.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# BaseMOdel is a class from the Pydantic library, equivalent of a TS interface, except it also validates incoming data at runtime.
#this Pydantic version actually checks at runtime that incoming JSON has a message field that's genuinely a string
from app.rag.answer import answer_question
from typing import Dict, List
#creates the application object. Direct equivalent of const app = express().
app= FastAPI()

# Allow requests from the Angular dev server on port 4200.
# Same role as app.use(cors()) in Express.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory store: sessionId -> list of past messages
# {} means an empty dictionary, which is Python's built-in key-value store. In TypeScript, you'd use an object literal: const conversations: Record<string, any[]> = {};
conversations: Dict[str, List[Message]] = {}

#The @ symbol placed directly above a function is Python's way of "wrapping" that function with extra behavior 
#conceptually it's doing the same job as:  app.get('/', (req, res) => { ... });
#FastAPI is smart enough to automatically turn a returned Python dict into a JSON HTTP response unlike Express where you'd explicitly write res.json({ status: 'ok' })
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    #TS equivalent would be:
    #const history = conversations[request.sessionId] || [];
    history: List[Message] = conversations.get(request.sessionId, [])

    answer_text = answer_question(request.message, history)

    # Now constructing real Message objects instead of plain dicts, to match the List[Message] type above.
    history.append(Message(role="user", content=request.message))
    history.append(Message(role="assistant", content=answer_text))
    # completely replaces the old history with the new one, so that next time the user sends a message, the assistant will have the full conversation context.
    conversations[request.sessionId] = history

    return {
        "text": answer_text,
        "buttons": []
    }