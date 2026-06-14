import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Initializing Embeddings...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embeddings Ready.")

active_sessions = {}

class VideoRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    video_id: str
    question: str

def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path[1:]
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
    return None

@app.post("/process_video")
async def process_video(payload: VideoRequest):
    video_id = extract_video_id(payload.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    if video_id in active_sessions:
        return {"status": "success", "video_id": video_id}
    
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        raw_text = " ".join(chunk.text for chunk in transcript_list)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.create_documents([raw_text])
        
        vector_store = FAISS.from_documents(documents, embeddings_model)
        active_sessions[video_id] = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        return {"status": "success", "video_id": video_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_question")
async def ask_question(payload: QuestionRequest):
    video_retriever = active_sessions.get(payload.video_id)
    if not video_retriever:
        raise HTTPException(status_code=400, detail="Session not found")
    
    try:
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        rag_prompt_template = """You are a helpful assistant.
Answer the question ONLY using the provided context segments below.

Context:
{context}

Question: {question}
Answer:"""

        prompt = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": video_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | groq_llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(payload.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))