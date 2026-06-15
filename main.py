import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# YouTube and LangChain imports
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
# FIXED IMPORT: Pulling GroqEmbeddings from community tools
from langchain_community.embeddings import GroqEmbeddings
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
class VideoRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    video_id: str
    question: str

# Global instance for single-user prototyping
db = None

groq_api_key = os.getenv("GROQ_API_KEY")

# FIXED: Correct initialization of lightweight Cloud Embeddings via Groq
print("⏳ Initializing Low-Memory Groq Cloud Embeddings...")
embeddings_model = GroqEmbeddings(
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)
print("✅ Groq Cloud Embeddings Ready.")

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
        
        # 2. Defensively Catching Transcript Library Variances
        transcript_list = None
        try:
            # Method A
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
        except Exception as e:
            print(f"⚠️ Method A failed ({str(e)}), switching to fallback Method B...")
            try:
                # Method B Fallback
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(['en', 'hi']).fetch()
            except Exception as inner_e:
                print(f"❌ Both Transcript Methods Failed: {str(inner_e)}")
                raise HTTPException(status_code=400, detail=f"Transcript not available: {str(inner_e)}")

        full_text = " ".join([item['text'] for item in transcript_list])
        print(f"📝 Transcript fetched successfully! Length: {len(full_text)} characters.")
        
        # 3. Document Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([full_text])
        
        # 4. Building Vector Store
        global db
        db = FAISS.from_documents(docs, embeddings_model)
        
        print("✅ Vector Index Created Successfully!")
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