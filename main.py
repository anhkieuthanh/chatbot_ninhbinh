import glob
import json
import os
import re

import requests
from pymilvus import DataType, MilvusClient


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "raw_data")


def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_env_file(ENV_PATH)


API_KEY = os.environ.get("EMBEDED_API_KEY", "").strip()
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_MODEL = "nvidia/nv-embedqa-e5-v5"
COLLECTION = "ninhbinh_kb"
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")

VECTOR_DIMENSION = 1024
CHUNK_MAX_LEN = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 20

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def validate_milvus_uri(uri: str) -> str:
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


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_RE.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_len: int = CHUNK_MAX_LEN, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_sentences and current_length + sentence_length > max_len:
            chunks.append(" ".join(current_sentences))

            overlap_sentences: list[str] = []
            overlap_length = 0
            for existing in reversed(current_sentences):
                projected = overlap_length + len(existing) + (1 if overlap_sentences else 0)
                if projected > overlap:
                    break
                overlap_sentences.insert(0, existing)
                overlap_length = projected

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length + (1 if current_length else 0)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def get_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not API_KEY:
        raise RuntimeError("Thiếu EMBEDED_API_KEY trong .env")

    payload = {
        "input": texts,
        "model": NVIDIA_MODEL,
        "encoding_format": "float",
        "input_type": "passage",
        "truncate": "NONE",
    }

    response = requests.post(NVIDIA_URL, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    embeddings = sorted(response.json()["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in embeddings]


def create_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=255)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=index_params,
    )


def build_records() -> list[dict]:
    data_pattern = os.path.join(RAW_DATA_DIR, "**", "*.json")
    files = sorted(glob.glob(data_pattern, recursive=True))
    all_records: list[dict] = []

    print(f"Found {len(files)} source files.")

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        title = data.get("title", "").strip()
        summary = data.get("summary", "").strip()
        body_text = data.get("body_text", "").strip()
        if not (title or summary or body_text):
            continue

        category = str(data.get("category") or "").strip()
        doc_type = str(data.get("doc_type") or "").strip()

        text_content = f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung: {body_text}"
        for chunk in chunk_text(text_content):
            all_records.append(
                {
                    "doc_id": str(data.get("doc_id", "unknown_id")),
                    "doc_type": doc_type or "unknown",
                    "category": category or "unknown",
                    "text": chunk,
                    "url": str(data.get("source_url", "")),
                    "title": title,
                }
            )

    return all_records


def insert_records(client: MilvusClient, all_records: list[dict]) -> int:
    inserted_count = 0

    for index in range(0, len(all_records), BATCH_SIZE):
        batch = all_records[index:index + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        embeddings = get_embeddings(texts)

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch ở batch {index // BATCH_SIZE + 1}: "
                f"nhận {len(embeddings)} vector cho {len(batch)} records."
            )

        insert_data = []
        for item, embedding in zip(batch, embeddings):
            insert_data.append(
                {
                    "vector": embedding,
                    "text": item["text"],
                    "url": item["url"],
                    "title": item["title"],
                    "doc_id": item["doc_id"],
                    "doc_type": item["doc_type"],
                    "category": item["category"],
                }
            )

        client.insert(collection_name=COLLECTION, data=insert_data)
        inserted_count += len(insert_data)
        print(f"Inserted {inserted_count}/{len(all_records)} chunks")

    return inserted_count


def main() -> None:
    milvus_uri = validate_milvus_uri(MILVUS_URI)
    client = MilvusClient(uri=milvus_uri, token=MILVUS_TOKEN)

    create_collection(client)
    print(f"Collection '{COLLECTION}' created in {milvus_uri}")

    all_records = build_records()
    print(f"Total chunks to embed: {len(all_records)}")

    inserted_count = insert_records(client, all_records)
    print(f"Upload complete: {inserted_count}/{len(all_records)} chunks inserted into Milvus Service.")


if __name__ == "__main__":
    main()
