from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import logging
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone("pcsk_4jdvA1_HALsqbujDgxPWKpi9zjWwZk6UGYAeAbLdbvznuAHpyrwhCuwNC2apUZDZcUsTqG")
PINECONE_INDEX_NAME = "openai-pdf-chat"

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating new Pinecone index '{PINECONE_INDEX_NAME}'")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=512,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info("Received PDF upload request")

    try:
        filename = os.path.splitext(file.filename)[0]
        cleaned_name = re.sub(r'[^a-zA-Z0-9\-]', '-', filename).lower()[:45]

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
            index.upsert(vectors=[(f"{cleaned_name}_doc_{i}", embedding, {"text": text})])

        logger.info(f"Uploaded {len(texts)} chunks to Pinecone under index '{PINECONE_INDEX_NAME}'")
        return {"message": f"PDF processed and stored in Pinecone under index '{PINECONE_INDEX_NAME}'."}

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: Request):
    logger.info("Received chat request")
    data = await request.json()
    messages = data.get("messages", [])
    use_document = data.get("use_document", False)
    selected_index = data.get("index_name", None)

    if not messages or not any(msg["sender"] == "user" for msg in messages):
        return {"response": "⚠️ No user message provided", "suggestions": []}

    last_user_msg = [m["text"] for m in messages if m["sender"] == "user"][-1]

    context = ""
    if use_document:
        try:
            query_embedding = embeddings.embed_query(last_user_msg)
            results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
            context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        except Exception as e:
            logger.warning(f"Failed to retrieve from Pinecone: {e}")

    full_prompt = f"{f'Context:\n{context}\n\n' if context else ''}User: {last_user_msg}\nAssistant:"
    
    try:
        response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ],
        max_tokens=200,
        temperature=0.7
        )
        assistant_response = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling OpenAI: {e}")
        assistant_response = "⚠️ Error getting response from OpenAI."

    suggestion_prompt = (
        "You are a helpful assistant. Based on the conversation below, suggest 3 "
        "short follow-up questions the user might ask next.\n\n"
        f"User: {last_user_msg}\n"
        f"Assistant: {assistant_response}\n\nSuggested questions:\n1."
    )

    try:
        suggestion_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": "You are a helpful assistant. Based on the conversation below, suggest 3 short follow-up questions the user might ask next."},
        {"role": "user", "content": f"User: {last_user_msg}\nAssistant: {assistant_response}\n\nSuggested questions:\n1."}
        ],
        max_tokens=100,
        temperature=0.7
        )

        suggestions_text = suggestion_response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        suggestions_text = ""

    suggestions = []
    for line in suggestions_text.split("\n"):
        line = line.strip()
        if line and line.startswith(("1.", "2.", "3.")):
            suggestions.append(line[2:].strip())

    return {
        "response": assistant_response,
        "suggestions": suggestions[:3],
        "used_index": selected_index
    }

@app.get("/list-indices")
async def list_indices():
    try:
        indices = pc.list_indexes().names()
        return {"indices": indices}
    except Exception as e:
        logger.error(f"Error fetching indices: {e}")
        return {"indices": []}