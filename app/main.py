from fastapi import FastAPI
from pydantic import BaseModel
from app.gemini import GeminiLLM
from app.retriever import get_relevant_chunks

app = FastAPI()
llm = GeminiLLM()

class Query(BaseModel):
    question: str
    
@app.get("/")
async def root():
    return {"message": "Welcome to the LLM API. Use the /ask endpoint to ask questions."}

@app.post("/ask")
async def ask_question(query: Query):
    chunks = get_relevant_chunks(query.question)
    context = "\n\n".join(chunks)
    
    prompt = f"""Use the context below to answer the question.
    
Context:
{context}

Question:
{query.question}
"""
    answer = llm.generate(prompt)
    return {"answer": answer}
