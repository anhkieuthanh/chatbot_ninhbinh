import os
import re
import json
import glob
import requests
from pymilvus import MilvusClient, DataType

# --- CẤU HÌNH ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load .env
env_path = os.path.join(script_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0]] = parts[1]

API_KEY = os.environ.get("EMBEDED_API_KEY")
URL = "https://integrate.api.nvidia.com/v1/embeddings"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}
MODEL = "nvidia/nv-embedqa-e5-v5"

CHUNK_MAX_LEN = 800   
CHUNK_OVERLAP = 150   

# Regex nhận diện ranh giới câu
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}')

# --- HÀM BỔ TRỢ ---

def split_sentences(text: str) -> list[str]:
    """Tách văn bản thành danh sách câu."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]

def chunk_text(text: str, max_len: int = CHUNK_MAX_LEN, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chia nhỏ văn bản giữ nguyên ngữ cảnh câu."""
    sentences = split_sentences(text)
    chunks = []
    curr_sentences = []
    curr_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if curr_len + sent_len > max_len and curr_sentences:
            chunks.append(" ".join(curr_sentences))
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(curr_sentences):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            curr_sentences = overlap_sentences
            curr_len = overlap_len

        curr_sentences.append(sent)
        curr_len += sent_len + 1

    if curr_sentences:
        chunks.append(" ".join(curr_sentences))
    return chunks if chunks else [text]

def get_embeddings(texts):
    """Gọi API NVIDIA để lấy vector embedding."""
    if not texts: return []
    data = {
        "input": texts,
        "model": MODEL,
        "encoding_format": "float",
        "input_type": "passage",
        "truncate": "NONE"
    }
    try:
        resp = requests.post(URL, headers=HEADERS, json=data, timeout=30)
        resp.raise_for_status()
        embeddings = resp.json()["data"]
        embeddings = sorted(embeddings, key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

def validate_milvus_uri(uri: str) -> str:
    cleaned = (uri or "").strip()
    if not cleaned or cleaned.endswith(".db"):
        raise ValueError("Yêu cầu MILVUS_URI dạng http(s)://host:port")
    return cleaned

# --- CHƯƠNG TRÌNH CHÍNH ---

def main():
    collection_name = "ninhbinh_v2"
    raw_data_path = os.path.join(script_dir, "raw_data")

    # 1. Khởi tạo Milvus Client & Schema
    # 1. Khởi tạo Milvus Client
    MILVUS_URI = validate_milvus_uri(os.environ.get("MILVUS_URI", "http://localhost:19530"))
    MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # ĐỊNH NGHĨA SCHEMA THEO CÁCH CHUẨN NHẤT
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    
    # Trường Primary Key
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    
    # TRƯỜNG VECTOR: Lưu ý sử dụng tham số dimension rõ ràng
    # Nếu bản của bạn vẫn lỗi, hãy thử thay 'dimension' bằng 'dim'
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    
    # Các trường Metadata
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=255)

    # Chuẩn bị Index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", 
        metric_type="COSINE", 
        index_type="AUTOINDEX"
    )

    # TẠO COLLECTION
    client.create_collection(
        collection_name=collection_name, 
        schema=schema, 
        index_params=index_params
    )

    # 2. Quét file và xử lý
    data_pattern = os.path.join(raw_data_path, "**", "*.json")
    files = glob.glob(data_pattern, recursive=True)
    all_data = []

    print(f"🔍 Tìm thấy {len(files)} file dữ liệu.")

    for fpath in files:
        try:
            # Tách category từ folder name
            relative_path = os.path.relpath(fpath, raw_data_path)
            category = os.path.dirname(relative_path) or "general"

            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "No Title")
            text_content = f"Tiêu đề: {title}\nTóm tắt: {data.get('summary', '')}\nNội dung: {data.get('body_text', '')}"
            
            chunks = chunk_text(text_content)
            for c in chunks:
                all_data.append({
                    "doc_id": data.get("doc_id", "unknown"),
                    "doc_type": data.get("doc_type", "unknown"),
                    "text": c,
                    "url": data.get("source_url", ""),
                    "title": title,
                    "category": category
                })
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file {fpath}: {e}")

    # 3. Embedding và Insert
    BATCH_SIZE = 20
    inserted_count = 0
    
    for i in range(0, len(all_data), BATCH_SIZE):
        batch = all_data[i:i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        embeddings = get_embeddings(texts)

        if not embeddings or len(embeddings) != len(batch):
            print(f"❌ Bỏ qua batch {i//BATCH_SIZE + 1} do lỗi embedding.")
            continue

        insert_data = []
        for item, emb in zip(batch, embeddings):
            insert_data.append({
                "vector": emb,
                "text": item["text"],
                "url": item["url"],
                "title": item["title"],
                "doc_id": item["doc_id"],
                "doc_type": item["doc_type"],
                "category": item["category"]
            })

        try:
            client.insert(collection_name=collection_name, data=insert_data)
            inserted_count += len(insert_data)
            print(f"✅ Đã chèn: {inserted_count}/{len(all_data)} chunks")
        except Exception as e:
            print(f"❌ Lỗi khi insert batch: {e}")

    print(f"\n✨ Hoàn tất! Tổng cộng {inserted_count} bản ghi đã nằm trong Milvus.")

if __name__ == "__main__":
    main()