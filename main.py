import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# FIX: Removed the incorrect Groq Key parameter so it safely leverages the public inference layer
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Cloud Embeddings Ready.")

# Active sessions storage dictionary
active_sessions = {}

class VideoRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    video_id: str
    question: str

@app.post("/process_video")
async def process_video(request: VideoRequest):
    try:
        url = request.url
        print(f"📥 Processing URL: {url}")
        
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
            
        print(f"🎥 Extracted Video ID: {video_id}")
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
        except AttributeError:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(['en', 'hi']).fetch()
            except Exception as inner_e:
                print(f"❌ Both Transcript Methods Failed: {str(inner_e)}")
                raise HTTPException(status_code=400, detail=f"Transcript fetching failed completely: {str(inner_e)}")
        except Exception as e:
            print(f"❌ Transcript API Network/Language Failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        full_text = " ".join([item['text'] for item in transcript_list])
        print(f"📝 Transcript fetched successfully! Length: {len(full_text)} characters.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([full_text])
        
        # FIX: Generate FAISS database and link it directly to active_sessions using the unique video_id
        db = FAISS.from_documents(docs, embeddings_model)
        active_sessions[video_id] = db.as_retriever(search_kwargs={"k": 3})
        
        print(f"✅ Vector Index Created and Saved for Session ID: {video_id}!")
        return {"status": "success", "video_id": video_id}
        
    except HTTPException as http_err:
        raise http_err
    except Exception as main_err:
        print(f"💥 Critical Crash on Server: {str(main_err)}")
        raise HTTPException(status_code=500, detail=str(main_err))
    
@app.post("/ask_question")
async def ask_question(payload: QuestionRequest):
    # Retrieve the correct indexed session vector block dynamically
    video_retriever = active_sessions.get(payload.video_id)
    if not video_retriever:
        raise HTTPException(status_code=400, detail="Session expired or not found. Please re-process the video.")
    
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
        print(f"💥 QA Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))