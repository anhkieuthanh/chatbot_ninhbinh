import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymilvus import MilvusClient

# ── Cấu hình đường dẫn ──────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Load .env ────────────────────────────────────────────────────────────────
env_path = os.path.join(script_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0]] = parts[1]

# ── Hằng số ──────────────────────────────────────────────────────────────────
API_KEY        = os.environ.get("EMBEDED_API_KEY")
NVIDIA_URL     = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_MODEL   = "nvidia/nv-embedqa-e5-v5"
COLLECTION     = "ninhbinh_kb"
DB_PATH        = os.path.join(script_dir, "chatbot.db")
DEFAULT_TOP_K  = 5

# Các loại nội dung hợp lệ
VALID_DOC_TYPES = {
    "destination", "cuisine", "festival", "hotel",
    "tour", "entertainment", "transport", "shopping",
    "support", "event", "craft_village", "virtual_guide"
}

# ── Kết nối Milvus (khởi tạo 1 lần khi server bắt đầu) ──────────────────────
milvus_client = MilvusClient(DB_PATH)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ninh Bình RAG API",
    description="API tra cứu thông tin du lịch Ninh Bình bằng Vector Search.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ───────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    doc_type: str | None = None  # Lọc theo chủ đề, VD: "cuisine", "hotel", "destination"


class SearchResult(BaseModel):
    title: str
    url: str
    doc_id: str
    doc_type: str
    score: float
    text_preview: str


class SearchResponse(BaseModel):
    query: str
    doc_type_filter: str | None
    results: list[SearchResult]
    total: int


# ── Helper: Gọi NVIDIA Embedding API ─────────────────────────────────────────
def get_query_embedding(query: str) -> list[float]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "input": [query],
        "model": NVIDIA_MODEL,
        "encoding_format": "float",
        "input_type": "query",
        "truncate": "NONE",
    }
    resp = requests.post(NVIDIA_URL, headers=headers, json=payload, timeout=15)
    if resp.status_code == 200:
        return resp.json()["data"][0]["embedding"]
    raise HTTPException(
        status_code=502,
        detail=f"NVIDIA API lỗi ({resp.status_code}): {resp.text}"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Kiểm tra server đang hoạt động."""
    return {"status": "ok", "message": "Ninh Bình RAG API đang chạy 🚀"}


@app.get("/health", tags=["Health"])
def health():
    """Kiểm tra chi tiết trạng thái server và DB."""
    collections = milvus_client.list_collections()
    total = 0
    if COLLECTION in collections:
        res = milvus_client.query(
            collection_name=COLLECTION, filter="", output_fields=["count(*)"]
        )
        total = res[0].get("count(*)", 0) if res else 0
    return {
        "status": "ok",
        "collection": COLLECTION,
        "total_chunks": total,
        "nvidia_model": NVIDIA_MODEL,
        "available_doc_types": sorted(VALID_DOC_TYPES),
    }


@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search(body: SearchRequest):
    """
    Tìm kiếm thông tin du lịch Ninh Bình.

    - **query**: Câu hỏi hoặc từ khóa tìm kiếm.
    - **top_k**: Số lượng kết quả trả về (mặc định 5, tối đa 20).
    - **doc_type**: Lọc theo chủ đề. VD: `cuisine`, `hotel`, `destination`, `festival`, `tour`, `entertainment`, `transport`, `shopping`, `event`, `craft_village`, `virtual_guide`, `support`.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    if body.doc_type and body.doc_type not in VALID_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"doc_type không hợp lệ: '{body.doc_type}'. Các giá trị hợp lệ: {sorted(VALID_DOC_TYPES)}"
        )
    top_k = min(body.top_k, 20)

    # Embed query
    vector = get_query_embedding(body.query)

    # Xây dựng filter expression cho Milvus
    milvus_filter = ""
    if body.doc_type:
        milvus_filter = f'doc_type == "{body.doc_type}"'

    # Vector search
    raw = milvus_client.search(
        collection_name=COLLECTION,
        data=[vector],
        limit=top_k,
        filter=milvus_filter,
        output_fields=["title", "url", "doc_id", "doc_type", "text"],
    )

    results = []
    for hit in raw[0]:
        entity = hit["entity"]
        results.append(SearchResult(
            title=entity.get("title", ""),
            url=entity.get("url", ""),
            doc_id=entity.get("doc_id", ""),
            doc_type=entity.get("doc_type", ""),
            score=round(hit["distance"], 4),
            text_preview=entity.get("text", "")[:400],
        ))

    return SearchResponse(
        query=body.query,
        doc_type_filter=body.doc_type,
        results=results,
        total=len(results)
    )
