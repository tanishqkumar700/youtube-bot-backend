import os
import re
import time
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# ENV CHECK
# --------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is missing."
    )

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)

# --------------------------------------------------
# MODELS
# --------------------------------------------------

class VideoRequest(BaseModel):
    url: str
    transcript_text: str


class QuestionRequest(BaseModel):
    video_id: str
    question: str


# --------------------------------------------------
# MEMORY STORE
# --------------------------------------------------

retriever_store: Dict = {}

SESSION_EXPIRY_SECONDS = 1800

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def cleanup_old_sessions():
    current_time = time.time()

    expired = []

    for video_id, data in retriever_store.items():
        if current_time - data["created_at"] > SESSION_EXPIRY_SECONDS:
            expired.append(video_id)

    for video_id in expired:
        del retriever_store[video_id]

    if expired:
        print(f"🗑 Removed {len(expired)} expired sessions")


def extract_video_id(url: str):

    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise HTTPException(
        status_code=400,
        detail="Invalid YouTube URL"
    )


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "healthy",
        "message": "YouTube QA Backend Running"
    }


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

@app.post("/process_video")
async def process_video(request: VideoRequest):

    cleanup_old_sessions()

    transcript = request.transcript_text.strip()

    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="Transcript is empty."
        )

    video_id = extract_video_id(request.url)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.create_documents(
        [transcript]
    )

    retriever = BM25Retriever.from_documents(docs)

    retriever.k = 3

    retriever_store[video_id] = {
        "retriever": retriever,
        "created_at": time.time()
    }

    print(
        f"✅ Indexed video {video_id} "
        f"with {len(docs)} chunks"
    )

    return {
        "status": "success",
        "video_id": video_id,
        "chunks": len(docs)
    }


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):

    cleanup_old_sessions()

    if request.video_id not in retriever_store:
        raise HTTPException(
            status_code=404,
            detail="Video session not found. Reprocess the video."
        )

    retriever = retriever_store[
        request.video_id
    ]["retriever"]

    relevant_docs = retriever.invoke(
        request.question
    )

    if not relevant_docs:
        return {
            "status": "success",
            "answer": (
                "This information is not available "
                "in the video transcript."
            )
        }

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )

    prompt = PromptTemplate.from_template(
        """
You are a YouTube transcript question-answering assistant.

IMPORTANT RULES:

1. Answer ONLY using the provided transcript context.
2. Never invent information.
3. Never use outside knowledge.
4. If the answer cannot be found in the context,
   respond exactly:

This information is not available in the video transcript.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )

    final_prompt = prompt.format(
        context=context,
        question=request.question
    )

    response = llm.invoke(final_prompt)

    return {
        "status": "success",
        "answer": response.content
    }


# --------------------------------------------------
# STARTUP LOG
# --------------------------------------------------

print("🚀 YouTube QA Backend Ready")