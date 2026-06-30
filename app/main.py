
#uvicorn app.main:app --reload  
#--reload — tells Uvicorn to automatically restart the server whenever you save changes to your code — equivalent to running Express with nodemon, instead of manually stopping/restarting node server.js every time you edit something.

#fastapi is like Express in that it provides a framework for defining routes and handling HTTP requests, but it also has built-in support for data validation, serialization, and automatic API documentation generation. FastAPI is designed to be fast and efficient, leveraging Python's type hints to provide better developer experience and performance.
from fastapi import FastAPI
# BaseMOdel is a class from the Pydantic library, equivalent of a TS interface, except it also validates incoming data at runtime.
#this Pydantic version actually checks at runtime that incoming JSON has a message field that's genuinely a string
from pydantic import BaseModel
from app.rag.answer import answer_question

#creates the application object. Direct equivalent of const app = express().
app= FastAPI()

# Simple in-memory store: sessionId -> list of past messages
conversations = {}

class ChatRequest(BaseModel):
    message: str
    sessionId: str

#The @ symbol placed directly above a function is Python's way of "wrapping" that function with extra behavior 
#conceptually it's doing the same job as:  app.get('/', (req, res) => { ... });
#FastAPI is smart enough to automatically turn a returned Python dict into a JSON HTTP response unlike Express where you'd explicitly write res.json({ status: 'ok' })
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):

    history = conversations.get(request.sessionId, [])

    answer_text = answer_question(request.message, history)

    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer_text})
    conversations[request.sessionId] = history

    return {
        "text": answer_text,
        "buttons": []
    }