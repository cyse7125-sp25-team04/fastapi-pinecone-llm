import os
import re
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from typing import List, Dict, Any
import uvicorn
import pandas as pd


app = FastAPI()
openai_client = OpenAI(api_key="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")

# Create an instance of the Pinecone client using your API key.
pc = Pinecone(api_key='pcsk_5kwnzz_RjyAj464745fCCuz3S3CPF5Anf2Ktk9ww3CRifWXb1rYRhUvXmUNhtFxJthvnBj')

index_name = "trace-index"
dimension = 1536

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created index: {index_name}")
else:
    print(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)

# Sentiment model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

PDF_FOLDER = "trace_pdfs"
PROCESSED_CACHE_FILE = "processed_files.txt"

class AskRequest(BaseModel):
    question: str

class ReportRequest(BaseModel):
    threshold: float = 0.7
    course_name: Optional[str] = None
    top_k: Optional[int] = 10

@app.post("/ask")
def api_ask_question(req: AskRequest):
    return {"answer": ask_question(req.question, index, openai_client)}

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_qna_chunks(text):
    pattern = re.compile(r"(Q:.*?)(?=\nQ:|\Z)", re.DOTALL)
    matches = pattern.findall(text)
    return [chunk.strip() for chunk in matches if len(chunk.strip()) > 30]

def extract_question_title(chunk):
    return chunk.splitlines()[0].replace("Q:", "").strip()

def count_responses(chunk):
    return len(re.findall(r"\n\d{1,2}[\). ]", chunk)) or 1

def get_sentiment(text, max_tokens=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy()[0])
    labels = ["negative", "neutral", "positive"]
    return dict(zip(labels, map(float, probs)))

def chunk_already_indexed(index, chunk_id):
    try:
        result = index.fetch(ids=[chunk_id])
        return chunk_id in result.vectors
    except Exception as e:
        print(f"⚠️ Error checking chunk ID {chunk_id}: {e}")
        return False

def embed_and_upsert_chunks(chunks, pdf_name):
    batch = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{pdf_name}_chunk_{i}"
        if chunk_already_indexed(index, chunk_id):
            print(f"⏩ Chunk already in Pinecone: {chunk_id}")
            continue

        question = extract_question_title(chunk)
        comment_count = count_responses(chunk)
        sentiment_scores = get_sentiment(chunk)

        embedding = openai_client.embeddings.create(
            input=[chunk],
            model="text-embedding-ada-002"
        ).data[0].embedding

        metadata = {
            "question": question,
            "source_file": pdf_name,
            "chunk_id": i,
            "comment_count": comment_count,
            "sentiment_positive": round(sentiment_scores["positive"], 4),
            "sentiment_neutral": round(sentiment_scores["neutral"], 4),
            "sentiment_negative": round(sentiment_scores["negative"], 4),
            "text": chunk
        }

        batch.append((chunk_id, embedding, metadata))

        if len(batch) >= 100:
            index.upsert(batch)
            batch = []

    if batch:
        index.upsert(batch)

class SentimentRequest(BaseModel):
    years: List[int]
    
@app.post("/sentiment")
def get_avg_sentiment(request: SentimentRequest) -> Dict[int, Dict[str, Any]]:
    years = request.years
    sentiment_data = {}

    for year in years:
        try:
            results = index.query(
                vector=[0.0] * 1536,
                top_k=1000,
                include_metadata=True,
                filter={"year": year}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed for year {year}: {str(e)}")

        matches = results.get("matches", [])
        pos_vals = [m["metadata"]["sentiment_positive"] for m in matches if "sentiment_positive" in m["metadata"]]
        neg_vals = [m["metadata"]["sentiment_negative"] for m in matches if "sentiment_negative" in m["metadata"]]

        sentiment_data[year] = {
            "avg_positive": round(sum(pos_vals) / len(pos_vals), 4) if pos_vals else 0.0,
            "avg_negative": round(sum(neg_vals) / len(neg_vals), 4) if neg_vals else 0.0,
            "count": len(matches)
        }

        print(f"📅 Year {year}: {len(matches)} entries used for averaging.")

    return sentiment_data


def process_pdf_folder(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        print(f"📄 Processing {pdf_file}...")
        try:
            raw_text = extract_text_from_pdf(file_path)
            chunks = extract_qna_chunks(raw_text)
            embed_and_upsert_chunks(chunks, pdf_file)
            print(f"✅ Embedded {len(chunks)} chunks from {pdf_file}")
        except Exception as e:
            print(f"❌ Failed to process {pdf_file}: {e}")

def ask_question(question, index, openai_client, top_k=5):
    query_embedding = openai_client.embeddings.create(input=[question], model="text-embedding-ada-002").data[0].embedding
    query_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in query_response["matches"]:
        meta = match["metadata"]
        source = meta.get("source_file", "")
        question_text = meta.get("question", "")
        chunk_text = meta.get("text", "")
        contexts.append(f"[{source}]\nQ: {question_text}\n{chunk_text}")
    combined_context = "\n---\n".join(contexts)
    prompt = f"""
You are an AI assistant helping students make course decisions using real feedback.

Based on the context, answer the question with a helpful and honest summary. 
Do not say \"I don't know.\" Use your judgment and highlight relevant experiences students had.

Context:
{combined_context}

Question:
{question}

Answer:
""".strip()
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.2
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    process_pdf_folder(PDF_FOLDER)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
