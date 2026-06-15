import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str
    transcript_text: str = None

class QuestionRequest(BaseModel):
    video_id: str
    question: str

retriever_store = None

groq_api_key = os.getenv("GROQ_API_KEY")
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=groq_api_key
)

@app.get("/")
def home():
    return {"status": "healthy", "message": "YouTube Pure-Engine Bot Active!"}

@app.post("/process_video")
async def process_video(request: VideoRequest):
    global retriever_store
    try:
        url = request.url
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
            
        if request.transcript_text:
            full_text = request.transcript_text
        else:
            raise HTTPException(status_code=400, detail="Transcript text required from frontend.")

        if not full_text or full_text.strip() == "":
            raise HTTPException(status_code=400, detail="Transcript is empty.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([full_text])
        
        retriever_store = BM25Retriever.from_documents(docs)
        retriever_store.k = 3
        
        print("✅ Context Indexing Ready!")
        return {"status": "success", "video_id": video_id}
        
    except HTTPException as http_err:
        raise http_err
    except Exception as main_err:
        raise HTTPException(status_code=500, detail=str(main_err))

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    global retriever_store
    if retriever_store is None:
        raise HTTPException(status_code=400, detail="No video context mapped yet.")
    
    try:
        print(f"❓ Fetching context for question: {request.question}")
        relevant_docs = retriever_store.invoke(request.question)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        
        template = """
        You are a helpful AI assistant that answers questions accurately based ONLY on the provided context transcript from a YouTube video.
        If you do not know the answer, say "This information is not available in the video transcript." Do not invent facts.

        Context:
        {context}

        Question: 
        {question}

        Helpful Answer:
        """
        # Formatted string prompt layout
        prompt_content = TemplateContent = PromptTemplate.from_template(template).format(context=context, question=request.question)
        
        # FIXED: Using standard .invoke() wrapper to compile Groq response
        response = groq_llm.invoke(prompt_content)
        response_text = response.content
        
        print("⚡ Response compiled successfully via Groq LLM.")
        return {"status": "success", "answer": response_text}
        
    except Exception as e:
        print(f"💥 QA Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))