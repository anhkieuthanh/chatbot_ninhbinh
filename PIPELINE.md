# 🔄 Pipeline: Chunking → Embedding → Insert DB

Tài liệu này mô tả chi tiết toàn bộ quá trình xử lý dữ liệu thô từ file JSON thành vector được lưu trong Milvus, tương ứng với code trong `main.py`.

---

## Tổng quan luồng xử lý

```
raw_data/**/*.json
       │
       ▼
  [Bước 1] Đọc & ghép nội dung
       │  title + summary + body_text
       ▼
  [Bước 2] Chunking (sentence-aware + overlap)
       │  1 bài → N chunks ngắn (≤ 800 ký tự)
       ▼
  [Bước 3] Embedding (NVIDIA NIM API)
       │  text → vector[1024 chiều]
       ▼
  [Bước 4] Insert vào Milvus Service DB
       │  vector + metadata → Milvus Service
       ▼
  Milvus Service → collection: ninhbinh_kb
```

---

## Bước 0 — Chuẩn bị DB

Trước khi xử lý dữ liệu, script tạo mới collection trong Milvus:

```python
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

if client.has_collection("ninhbinh_kb"):
    client.drop_collection("ninhbinh_kb")   # Xóa cũ nếu đã tồn tại

schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=2048)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=100)
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=255)

index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection(
    collection_name="ninhbinh_kb",
    schema=schema,
    index_params=index_params
)
```

> ⚠️ Lệnh `drop_collection` sẽ **xóa toàn bộ dữ liệu cũ** mỗi lần chạy lại `main.py`.

---

## Bước 1 — Đọc & ghép nội dung

Script quét đệ quy toàn bộ file `.json` trong thư mục `raw_data/` và ghép 3 trường thành một chuỗi văn bản thống nhất:

```python
files = glob.glob("raw_data/**/*.json", recursive=True)
# 181 files tương ứng với 181 bài viết

for fpath in files:
    data = json.load(open(fpath))

    title   = data.get("title", "")
    summary = data.get("summary", "")
    body    = data.get("body_text", "")

    text_content = f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung: {body}"
```

**Ví dụ đầu vào (file JSON):**
```json
{
  "title": "Nem Dê Ninh Bình – Hương vị truyền thống vùng Cố Đô",
  "summary": "Ninh Bình – vùng đất Cố đô Hoa Lư không chỉ được biết đến...",
  "body_text": "Nem dê Ninh Bình cùng với các món đặc sản khác như cơm cháy...",
  "doc_type": "cuisine",
  "source_url": "https://dulichninhbinh.com.vn/item/3385"
}
```

**Sau khi ghép:**
```
Tiêu đề: Nem Dê Ninh Bình – Hương vị truyền thống vùng Cố Đô
Tóm tắt: Ninh Bình – vùng đất Cố đô Hoa Lư không chỉ được biết đến...
Nội dung: Nem dê Ninh Bình cùng với các món đặc sản khác như cơm cháy...
```

---

## Bước 2 — Chunking

### Tại sao cần Chunking?

Mô hình embedding có giới hạn số token đầu vào. Văn bản dài cần được chia nhỏ để:
- Không vượt quá giới hạn token của API
- Mỗi chunk mang 1 ý chính rõ ràng → tìm kiếm chính xác hơn

### Thuật toán: Sentence-aware + Overlap

```python
CHUNK_MAX_LEN = 800   # Ký tự tối đa mỗi chunk
CHUNK_OVERLAP = 150   # Ký tự overlap giữa 2 chunk liên tiếp

# Bước 2a: Tách văn bản thành danh sách câu
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}')

def split_sentences(text):
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]
```

**Ví dụ tách câu:**
```
Input:  "Hoa Lư là cố đô. Tràng An là di sản thế giới. Bái Đính..."
Output: ["Hoa Lư là cố đô.", "Tràng An là di sản thế giới.", "Bái Đính..."]
        ───── câu 1 ──────  ────────── câu 2 ────────────   ── câu 3 ──
```

```python
# Bước 2b: Gom câu thành chunk
def chunk_text(text, max_len=800, overlap=150):
    sentences = split_sentences(text)
    chunks = []
    curr_sentences = []
    curr_len = 0

    for sent in sentences:
        if curr_len + len(sent) > max_len and curr_sentences:
            # Đóng chunk hiện tại
            chunks.append(" ".join(curr_sentences))

            # Giữ lại các câu cuối làm overlap (≤ 150 ký tự)
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(curr_sentences):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break

            curr_sentences = overlap_sentences   # Bắt đầu chunk mới từ overlap
            curr_len = overlap_len

        curr_sentences.append(sent)
        curr_len += len(sent) + 1

    if curr_sentences:
        chunks.append(" ".join(curr_sentences))  # Chunk cuối cùng

    return chunks
```

**Minh họa Overlap:**
```
Chunk 1: "[...câu A] [câu B] [câu C]"     ← cắt tại đây
                              └────────────────────────┐
Chunk 2:               "[câu C] [câu D] [câu E] ..."  ← câu C được lặp lại
                       ↑ overlap (≤ 150 ký tự)
```

> Overlap đảm bảo câu query span qua ranh giới 2 chunk vẫn có thể được tìm thấy đúng.

### Kết quả Chunking

| Số bài viết | Tổng chunks | Trung bình/bài |
|-------------|-------------|----------------|
| 181         | 596         | ~3.3 chunks    |

---

## Bước 3 — Embedding

Mỗi chunk text được chuyển thành **vector 1024 chiều** bằng NVIDIA NIM API:

```python
MODEL = "nvidia/nv-embedqa-e5-v5"
BATCH_SIZE = 20  # Gửi tối đa 20 chunks mỗi request

def get_embeddings(texts: list[str]) -> list[list[float]]:
    payload = {
        "input": texts,                    # Danh sách chunk
        "model": MODEL,
        "encoding_format": "float",
        "input_type": "passage",           # "passage" cho dữ liệu, "query" cho câu hỏi
        "truncate": "NONE"
    }
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload
    )
    embeddings = sorted(resp.json()["data"], key=lambda x: x["index"])
    return [e["embedding"] for e in embeddings]
```

**Tại sao dùng `input_type: "passage"`?**

Model `nv-embedqa-e5-v5` được huấn luyện với 2 loại input_type khác nhau:

| input_type | Dùng khi | Ví dụ |
|---|---|---|
| `passage` | Nhúng dữ liệu lưu trữ | Nội dung bài viết (chunking) |
| `query` | Nhúng câu truy vấn | Câu hỏi của người dùng |

> Dùng đúng `input_type` giúp tăng độ chính xác của Cosine Similarity đáng kể.

**Ví dụ kết quả embedding:**
```
Input:  "Nem dê Ninh Bình là món ăn đặc sản..."
Output: [-0.021, 0.045, 0.012, ..., -0.003]   ← 1024 số thực
         └──────────────────────────────────┘
                  vector[1024 chiều]
```

---

## Bước 4 — Insert vào Milvus DB

Sau khi có embeddings, dữ liệu được insert vào Milvus theo batch:

```python
insert_data = [
    {
        "vector":   emb,              # list[float] — 1024 chiều (schema field)
        "text":     item["text"],     # Nội dung chunk (dynamic field)
        "title":    item["title"],    # Tiêu đề bài gốc (dynamic field)
        "url":      item["url"],      # URL nguồn (dynamic field)
        "doc_id":   item["doc_id"],   # ID bài gốc (dynamic field)
        "doc_type": item["doc_type"]  # Loại nội dung (dynamic field, dùng để filter)
    }
    for item, emb in zip(batch, embeddings)
]

client.insert(collection_name="ninhbinh_kb", data=insert_data)
```

**Schema của mỗi record trong DB:**

| Trường | Kiểu dữ liệu | Mô tả |
|---|---|---|
| `id` | INT64 (auto) | Khóa chính tự tăng |
| `vector` | FLOAT_VECTOR[1024] | Embedding của chunk |
| `text` | string | Nội dung văn bản của chunk |
| `title` | string | Tiêu đề bài viết gốc |
| `url` | string | URL nguồn |
| `doc_id` | string | ID định danh bài gốc |
| `doc_type` | string | Loại nội dung (dùng để filter) |
| `category` | string | Category gốc đọc từ dữ liệu nguồn |

> `id`, `vector` và các metadata chính hiện được khai báo tường minh trong schema để API và pipeline luôn đồng bộ.

---

## Tóm tắt số liệu

```
181 file JSON
  └─ 596 chunks (sentence-aware, max 800 ký tự, overlap 150 ký tự)
       └─ 596 vectors (1024 chiều, model: nvidia/nv-embedqa-e5-v5)
           └─ records trong collection ninhbinh_kb (Milvus Service)
```

---

## Chạy lại pipeline

```bash
# 1. Rebuild DB
python3 chatbot/main.py
```
