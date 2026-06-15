import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Explicit clean library imports
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load Environment variables
load_dotenv()

app = FastAPI()

# Cross-Origin Resource Sharing (CORS) setup for Extension Authorization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Validation Models
# Validation model ko update karo taaki direct transcript accept ho sake
class VideoRequest(BaseModel):
    url: str
    transcript_text: str = None  # Extension se direct text lene ke liye

@app.post("/process_video")
async def process_video(request: VideoRequest):
    try:
        url = request.url
        print(f"📥 Processing URL request: {url}")
        
        # Parsing Video ID
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
            
        print(f"🎥 Video ID: {video_id}")
        
        # EXTENSION FALLBACK STEP: Agar text frontend se aaya hai toh wahi use karo
        if request.transcript_text:
            full_text = request.transcript_text
            print("📝 Received raw transcript directly from Chrome Extension context.")
        else:
            # Agar koi purana client hit kare toh low-level client execute karein
            try:
                api_client = YouTubeTranscriptApi()
                raw_transcript_data = api_client.list(video_id)
                transcript_list = raw_transcript_data.fetch()
                full_text = " ".join([item['text'] for item in transcript_list])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cloud IP is blocked by YouTube. Please update Extension frontend: {str(e)}")

        if not full_text or full_text.strip() == "":
            raise HTTPException(status_code=400, detail="Transcript content is empty.")

        # 3. Document Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([full_text])
        print(f"✂️ Total split chunks created: {len(docs)}")
        
        # 4. Building Vector Store Safely using Batches
        global db
        batch_size = 32
        
        print(f"🚀 Embedding initial block...")
        db = FAISS.from_documents(docs[:batch_size], embeddings_model)
        
        for i in range(batch_size, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            db.add_documents(batch)
            time.sleep(0.3)
        
        print("✅ Vector Index Active and Ready!")
        return {"status": "success", "video_id": video_id}
        
    except HTTPException as http_err:
        raise http_err
    except Exception as main_err:
        print(f"💥 Backend Crash: {str(main_err)}")
        raise HTTPException(status_code=500, detail=str(main_err))


class QuestionRequest(BaseModel):
    video_id: str
    question: str

# Global instance for single-user prototyping
db = None

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

print("⏳ Initializing Low-Memory Hugging Face Cloud Embeddings API...")
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_key=SecretStr(hf_token) if hf_token else None
)
print("✅ Hugging Face Cloud Embeddings Ready.")

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=groq_api_key
)

@app.get("/")
def home():
    return {"status": "healthy", "message": "YouTube RAG Bot Backend is active!"}

@app.post("/process_video")
async def process_video(request: VideoRequest):
    try:
        url = request.url
        print(f"📥 Processing URL: {url}")
        
        # 1. Parsing Video ID
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
            
        print(f"🎥 Extracted Video ID: {video_id}")
        
        # 2. Extract Transcript USING ONLY LIST AND FETCH
        transcript_list = None
        try:
            print("🔍 Fetching transcript via explicit list -> fetch mapping...")
            
            # Instance banao jiske paas list aur fetch hain
            api_client = YouTubeTranscriptApi()
            
            # Step A: Raw transcript metadata list uthao
            raw_transcript_data = api_client.list(video_id)
            
            # Step B: Direct content fetch karo jo list data object return karega
            transcript_list = raw_transcript_data.fetch()
            
            print("✅ Transcript pieces pulled successfully via direct mapping.")

        except Exception as e:
            print(f"❌ Low-Level Transcript Fetch Completely Failed: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Transcript method mismatch or unavailable: {str(e)}"
            )

        full_text = " ".join([item['text'] for item in transcript_list])
        print(f"📝 Transcript built successfully! Length: {len(full_text)} characters.")
        
        # 3. Document Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([full_text])
        print(f"✂️ Total split chunks created: {len(docs)}")
        
        # 4. Building Vector Store Safely using Batches
        global db
        batch_size = 32
        
        print(f"🚀 Embedding first batch (0 to {min(batch_size, len(docs))})...")
        db = FAISS.from_documents(docs[:batch_size], embeddings_model)
        
        for i in range(batch_size, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            print(f"⏳ Embedding batch ({i} to {min(i + batch_size, len(docs))})...")
            db.add_documents(batch)
            time.sleep(0.5)
        
        print("✅ Vector Index Created Successfully for the entire video!")
        return {"status": "success", "video_id": video_id}
        
    except HTTPException as http_err:
        raise http_err
    except Exception as main_err:
        print(f"💥 Critical Crash on Server: {str(main_err)}")
        raise HTTPException(status_code=500, detail=str(main_err))

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    global db
    if db is None:
        raise HTTPException(status_code=400, detail="No video has been processed yet.")
    
    try:
        print(f"❓ User Question: {request.question}")
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        template = """
        You are a helpful AI assistant that answers questions accurately based ONLY on the provided context transcript from a YouTube video.
        If you do not know the answer or if it's not mentioned in the context, say "This information is not available in the video transcript."
        Do not make up facts.

        Context:
        {context}

        Question: 
        {question}

        Helpful Answer:
        """
        custom_rag_prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | groq_llm
            | StrOutputParser()
        )
        
        response_text = rag_chain.invoke(request.question)
        print("⚡ Response generated from Llama model.")
        return {"status": "success", "answer": response_text}
        
    except Exception as e:
        print(f"💥 QA Pipeline Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))