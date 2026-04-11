import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from pymilvus import MilvusClient
from dotenv import load_dotenv

# --- CẤU HÌNH ---
load_dotenv()

app = FastAPI(title="Ninh Binh Search API", description="API chuyên dụng để truy vấn dữ liệu từ Vector DB")

# Biến môi trường
API_KEY = os.environ.get("EMBEDED_API_KEY")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/embeddings"
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
COLLECTION_NAME = "ninhbinh_v2"

# Khởi tạo Client (Dùng chung kết nối cho toàn bộ ứng dụng)
client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

# --- MODELS ---

class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    top_k: int = 5

class SearchResponse(BaseModel):
    score: float
    text: str
    title: str
    category: str
    url: Optional[str] = None

# --- UTILS ---

def get_query_embedding(text: str):
    """Chuyển đổi câu hỏi của người dùng thành vector."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    payload = {
        "input": [text],
        "model": "nvidia/nv-embedqa-e5-v5",
        "encoding_format": "float",
        "input_type": "query", # Rất quan trọng: Dùng 'query' thay vì 'passage' khi tìm kiếm
        "truncate": "NONE"
    }
    try:
        resp = requests.post(NVIDIA_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

# --- ENDPOINTS ---

@app.post("/search", response_model=List[SearchResponse])
async def search_vector_db(req: SearchRequest):
    """
    Thực hiện tìm kiếm ngữ nghĩa trong Milvus kèm lọc Category.
    """
    # 1. Chuyển Query thành Vector
    vector = get_query_embedding(req.query)
    if vector is None:
        raise HTTPException(status_code=500, detail="Không thể tạo embedding cho câu hỏi.")

    try:
        # 2. Xây dựng bộ lọc Filter (Milvus Expression)
        # Nếu có category thì lọc, nếu không thì tìm trên toàn bộ collection
        filter_expr = f'category == "{req.category}"' if req.category else ""

        # 3. Truy vấn Milvus
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            data=[vector], # Milvus nhận list các vector
            filter=filter_expr,
            limit=req.top_k,
            output_fields=["text", "title", "category", "url"], # Các trường metadata muốn lấy
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )

        # 4. Trích xuất và format kết quả
        final_results = []
        if search_results:
            for hit in search_results[0]: # Lấy kết quả của query đầu tiên
                final_results.append({
                    "score": hit["distance"],
                    "text": hit["entity"].get("text"),
                    "title": hit["entity"].get("title"),
                    "category": hit["entity"].get("category"),
                    "url": hit["entity"].get("url")
                })
        
        return final_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus Search Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ready", "collection": COLLECTION_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)