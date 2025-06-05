from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import json
from duckduckgo_search import DDGS
import requests
from pinecone_plugins.assistant.models.chat import Message

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ✅ UPDATED: Allow Vercel frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chatbot-frontend-five-orpin.vercel.app",  # Vercel frontend
        "http://localhost:3000",                            # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key="pcsk_4jdvA1_HALsqbujDgxPWKpi9zjWwZk6UGYAeAbLdbvznuAHpyrwhCuwNC2apUZDZcUsTqG")
PINECONE_INDEX_NAME = "llama-pdf-chat"

assistant = pc.assistant.Assistant(assistant_name="example-assistant")

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating new Pinecone index '{PINECONE_INDEX_NAME}'")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

first_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the current exchange rate from base currency to target currency",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_currency": {"type": "string"},
                    "target_currency": {"type": "string"},
                    "date": {"type": "string", "default": "latest"}
                },
                "required": ["base_currency", "target_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Search the web for real-time information",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string"}
                },
                "required": ["search_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Searches through previously uploaded PDF content stored in Pinecone",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's question"},
                    "namespace": {"type": "string", "description": "Namespace corresponding to a specific uploaded PDF"}
                },
                "required": ["query", "namespace"]
            }
        }
    }
]

def get_exchange_rate(base_currency: str, target_currency: str, date: str = "latest") -> float:
    url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date}/v1/currencies/{base_currency.lower()}.json" 
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get(base_currency.lower(), {}).get(target_currency.lower(), None)
    else:
        raise Exception(f"Failed to fetch exchange rate: {response.status_code}")

def search_internet(search_query: str) -> list:
    results = DDGS().text(str(search_query), max_results=5)
    return [result['body'] for result in results]

def search_document(query: str, namespace: str = "llama-pdf-chat"):
    try:
        msg = Message(content="How old is the earth?")
        resp = assistant.chat(messages=[msg])
        return {
            "answer": resp["message"]["content"].strip(),
            "suggestions": []
        }
    except Exception as e:
        logger.error(f"Error searching document: {e}")
        return {"error": "⚠️ Failed to search document."}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info("Received PDF upload request")
    try:
        filename = os.path.splitext(file.filename)[0]
        namespace = "llama-pdf-chat"
        pdf_reader = PdfReader(file.file)
        raw_text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)
        for i, text in enumerate(texts):
            embedding = embeddings.embed_query(text)
            index.upsert(
                vectors=[
                    (f"{namespace}_doc_{i}", embedding, {
                        "text": text,
                        "source": filename
                    })
                ],
                namespace=namespace
            )
        logger.info(f"Uploaded {len(texts)} chunks from '{file.filename}' into namespace '{namespace}'")
        return {"message": f"PDF '{file.filename}' indexed successfully."}
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: Request):
    logger.info("Received chat request")
    data = await request.json()
    messages = data.get("messages", [])

    if not messages or not any(msg["sender"] == "user" for msg in messages):
        return {"response": "⚠️ No user message provided.", "suggestions": []}

    last_user_msg = [m["text"] for m in messages if m["sender"] == "user"][-1]

    is_document_query = False
    selected_namespace = None

    try:
        indices = pc.list_indexes().names()
        if indices and "llama-pdf-chat" in indices:
            doc_keywords = ["document", "pdf", "report", "assignment", "data", "study"]
            if any(kw in last_user_msg.lower() for kw in doc_keywords):
                is_document_query = True
                selected_namespace = "llama-pdf-chat"
    except Exception as e:
        logger.warning(f"Failed to list indices: {e}")

    if not is_document_query:
        prompt_for_intent = (
            "Classify the following question as 'document' or 'general':\n\n"
            f"Question: {last_user_msg}\n\nAnswer:"
        )
        detection_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_for_intent}],
            max_tokens=10,
            temperature=0
        )
        decision = detection_response.choices[0].message.content.strip().lower()
        if decision == "document":
            is_document_query = True
            selected_namespace = "llama-pdf-chat"

    system_prompt = (
        "You are a helpful assistant. Answer the user's question based on your own knowledge "
        "unless instructed to use document context. Always be concise and clear.\n\n"
        "Available tools:\n"
        "- get_exchange_rate: For real-time currency conversion\n"
        "- search_internet: For up-to-date news or facts\n"
        "- search_document: For answering from uploaded documents"
    )

    openai_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        openai_messages.append({
            "role": "user" if msg["sender"] == "user" else "assistant",
            "content": msg["text"]
        })

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=openai_messages,
        tools=first_tools,
        tool_choice="auto",
        max_tokens=300,
        temperature=0.7
    )
    choice = response.choices[0]

    tool_calls = choice.message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"\nTool called: {tool_name}")
            print(f"Arguments: {tool_args}")

            if tool_name == "get_exchange_rate":
                result = get_exchange_rate(**tool_args)
                return {
                    "response": f"1 {tool_args['base_currency'].upper()} = {result:.2f} {tool_args['target_currency'].upper()}",
                    "suggestions": []
                }

            elif tool_name == "search_internet":
                results = search_internet(tool_args["search_query"])
                return {
                    "response": "\n".join(f"- {r}" for r in results),
                    "suggestions": []
                }

            elif tool_name == "search_document":
                query = tool_args.get("query", "").strip()
                namespace = "llama-pdf-chat"

                try:
                    msg = Message(content=f"Question: {query}")
                    pinecone_response = assistant.chat(messages=[msg])
                    answer = pinecone_response["message"]["content"].strip()

                    return {
                        "response": answer,
                        "suggestions": []
                    }
                except Exception as e:
                    logger.error(f"Error calling Pinecone Assistant: {e}")
                    return {"response": "⚠️ Failed to get answer from document.", "suggestions": []}

    return {"response": "⚠️ Out of context question.", "suggestions": []}

@app.get("/list-indices")
async def list_indices():
    try:
        indices = pc.list_indexes().names()
        return {"indices": indices}
    except Exception as e:
        logger.error(f"Error fetching indices: {e}")
        return {"indices": []}
