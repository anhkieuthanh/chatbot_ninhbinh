import os
import re
import json
import glob
import requests
from pymilvus import MilvusClient

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

CHUNK_MAX_LEN = 800   # Độ dài tối đa mỗi chunk (ký tự)
CHUNK_OVERLAP = 150   # Số ký tự overlap giữa hai chunk liên tiếp

# Regex nhận diện ranh giới câu: dấu chấm/hỏi/than + khoảng trắng HOẶC xuống dòng đôi
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}')

def split_sentences(text: str) -> list[str]:
    """Tách văn bản thành danh sách câu, giữ nguyên nội dung."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]

def chunk_text(text: str,
               max_len: int = CHUNK_MAX_LEN,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Sentence-aware chunking với overlap.
    - Cắt đúng ranh giới câu (không cắt giữa chừng).
    - Mỗi chunk mới bắt đầu bằng `overlap` ký tự cuối của chunk trước
      để giữ ngữ cảnh liên tiếp.
    """
    sentences = split_sentences(text)
    chunks = []
    curr_sentences: list[str] = []
    curr_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # Nếu thêm câu này vượt quá max_len → đóng chunk hiện tại
        if curr_len + sent_len > max_len and curr_sentences:
            chunk_text_str = " ".join(curr_sentences)
            chunks.append(chunk_text_str)

            # Tính overlap: giữ lại các câu cuối sao cho tổng <= overlap ký tự
            overlap_sentences: list[str] = []
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

    # Chunk cuối còn lại
    if curr_sentences:
        chunks.append(" ".join(curr_sentences))

    return chunks if chunks else [text]  # Fallback nếu text quá ngắn

def get_embeddings(texts):
    if not texts: return []
    data = {
        "input": texts,
        "model": MODEL,
        "encoding_format": "float",
        "input_type": "passage",
        "truncate": "NONE"
    }
    resp = requests.post(URL, headers=HEADERS, json=data)
    if resp.status_code == 200:
        embeddings = resp.json()["data"]
        # NVIDIA API returns in order, sort by index just in case
        embeddings = sorted(embeddings, key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]
    else:
        print(f"Error from NVIDIA API: {resp.text}")
        return []

def validate_milvus_uri(uri: str) -> str:
    """Chỉ chấp nhận Milvus Service qua HTTP/HTTPS, không dùng Milvus Lite (.db)."""
    cleaned = (uri or "").strip()
    if not cleaned:
        raise ValueError("Thiếu MILVUS_URI. Ví dụ: http://localhost:19530")
    if cleaned.endswith(".db"):
        raise ValueError(
            f"MILVUS_URI không hợp lệ: '{cleaned}'. Dự án chỉ hỗ trợ Milvus Service, không dùng Milvus Lite."
        )
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        raise ValueError(
            f"MILVUS_URI không hợp lệ: '{cleaned}'. Chỉ chấp nhận URL Milvus Service dạng http(s)://host:port."
        )
    return cleaned

def main():
    collection_name = "ninhbinh_kb"

    # 1. Setup DB
    MILVUS_URI = validate_milvus_uri(os.environ.get("MILVUS_URI", "http://localhost:19530"))
    MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=1024,
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True
    )
    print(f"Collection '{collection_name}' created in: {MILVUS_URI}")

    # 2. Process & Chunk dữ liệu
    data_pattern = os.path.join(script_dir, "raw_data", "**", "*.json")
    files = glob.glob(data_pattern, recursive=True)
    all_data = []

    print(f"Found {len(files)} source files.")

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = data.get("title", "")
        summary = data.get("summary", "")
        body = data.get("body_text", "")

        text_content = f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung: {body}"
        chunks = chunk_text(text_content)

        for c in chunks:
            all_data.append({
                "doc_id": data.get("doc_id", "unknown_id"),
                "doc_type": data.get("doc_type", "unknown"),
                "text": c,
                "url": data.get("source_url", ""),
                "title": title
            })

    print(f"Total chunks to embed: {len(all_data)}")

    # 3. Embed và Insert theo batch
    BATCH_SIZE = 20
    inserted_count = 0
    for i in range(0, len(all_data), BATCH_SIZE):
        batch = all_data[i:i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        embeddings = get_embeddings(texts)

        if not embeddings or len(embeddings) != len(batch):
            print(f"Skipping batch {i // BATCH_SIZE + 1} due to embedding failure.")
            continue

        insert_data = [
            {
                "vector": emb,
                "text": item["text"],
                "url": item["url"],
                "title": item["title"],
                "doc_id": item["doc_id"],
                "doc_type": item["doc_type"]
            }
            for item, emb in zip(batch, embeddings)
        ]

        client.insert(collection_name=collection_name, data=insert_data)
        inserted_count += len(insert_data)
        print(f"Inserted batch {i // BATCH_SIZE + 1}. Total Inserted: {inserted_count}")

    print(f"\nUpload to DB complete! {inserted_count}/{len(all_data)} chunks inserted.")


if __name__ == "__main__":
    main()
