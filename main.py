import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
# SWITCHED: Using the online API inference instead of loading heavy local models
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
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

print("⏳ Initializing Lightweight Cloud Embeddings API...")
# This hits an external API endpoint instead of downloading heavy model weights to your server RAM
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("GROQ_API_KEY"), # We can pass your key or leave it blank for free tier limits
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Cloud Embeddings Ready.")

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
async def process_video(request: VideoRequest):
    try:
        url = request.url
        print(f"📥 Processing URL: {url}")
        
        # 1. Video ID parsing logic check
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
            
        print(f"🎥 Extracted Video ID: {video_id}")
        
        # 2. Fetch transcript with fallback language option
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
        except Exception as e:
            print(f"❌ Transcript API Failed: {str(e)}")
            return {"status": "error", "message": f"Transcript not available for this video: {str(e)}"}, 400

        full_text = " ".join([item['text'] for item in transcript_list])
        
        # 3. Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([full_text])
        
        # 4. Vector Store Generation
        global db
        db = FAISS.from_documents(docs, embeddings_model)
        
        print("✅ Vector Index Created Successfully!")
        return {"status": "success", "video_id": video_id}
        
    except Exception as main_err:
        print(f"💥 Critical Crash on Server: {str(main_err)}")
        raise HTTPException(status_code=500, detail=str(main_err))

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