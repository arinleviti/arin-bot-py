import os
from dotenv import load_dotenv
from groq import Groq
# reads the env file and loads the variables into os.environ, so you can access them with os.environ["VAR_NAME"]
load_dotenv()  # Load environment variables from .env file

client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Say hello to the world in a fun way."}
    ]
)

print(response.choices[0].message.content)