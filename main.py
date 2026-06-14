import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Core Document Processing, Embedding & AI Models
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Boot configuration variables
load_dotenv()

app = FastAPI(title="YouTube Bot Multi-Session RAG Backend")

# CRITICAL SECURITY RULE: Explicitly open up Cross-Origin Resource Sharing (CORS) 
# to let the background Chrome extension scripts query endpoints safely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the free embedding model locally on your device
print("⏳ Initializing Local HuggingFace Embedding matrices (all-MiniLM-L6-v2)...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embeddings engine successfully locked in.")

# Session dictionary storage mapping: { video_id: faiss_retriever_instance }
active_sessions = {}

# Validate incoming network request schemas
class VideoRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    video_id: str
    question: str

def extract_video_id(url: str) -> str:
    """Helper parsing function to isolate the unique 11-char YouTube tracking key."""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path[1:]
    if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
    return None

@app.post("/process_video")
async def process_video(payload: VideoRequest):
    """Fetches video transcript text on the server, builds a vector index, and stores it in memory."""
    video_id = extract_video_id(payload.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube watch URL link provided.")
    
    # Optimization: Skip reprocessing if the context index is already built in memory
    if video_id in active_sessions:
        return {"status": "success", "message": "Video layout loaded securely from memory cache.", "video_id": video_id}
    
    try:
        # Fetch subtitle sequences using the official instance method logic
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        raw_transcript_text = " ".join(chunk.text for chunk in transcript_list)
        
        # Segment data into manageable contextual chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.create_documents([raw_transcript_text])
        
        # Generate spatial numerical vectors and save to an isolated local FAISS instance
        vector_store = FAISS.from_documents(documents, embeddings_model)
        active_sessions[video_id] = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        return {"status": "success", "message": "Vector index built successfully.", "video_id": video_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed processing video transcript: {str(e)}")

@app.post("/ask_question")
async def ask_question(payload: QuestionRequest):
    """Retrieves context segments and streams a grounded question response through ChatGroq."""
    video_retriever = active_sessions.get(payload.video_id)
    if not video_retriever:
        raise HTTPException(status_code=400, detail="Active vector model context not found. Re-initialize video.")
    
    try:
        # Load modern ChatGroq configuration variables
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Strict Grounded RAG Instruction Template Framework
        rag_prompt_template = """You are a helpful assistant.
Answer the question ONLY using the provided context segments below. 
If the context information is missing or insufficient to answer, state clearly that you do not know.

Context:
{context}

Question: {question}
Answer:"""

        prompt = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Standard LangChain Expression Language execution pipeline
        rag_chain = (
            {"context": video_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | groq_llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(payload.question)
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Inference Exception: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)