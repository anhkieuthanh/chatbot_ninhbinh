# 🏔️ Ninh Bình RAG API

Hệ thống tra cứu thông tin du lịch Ninh Bình sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)**.  
Dữ liệu được crawl từ [dulichninhbinh.com.vn](https://dulichninhbinh.com.vn), nhúng vector bằng **NVIDIA NIM API**, lưu trữ tại **Milvus Service** và phục vụ qua **FastAPI**.

---

## 📁 Cấu trúc thư mục

```
chatbot/
├── .env              # API key (không commit lên git)
├── raw_data/         # Dữ liệu JSON thô đã crawl (181 bài viết)
│   ├── cuisine/
│   ├── destination/
│   ├── entertainment/
│   └── ...
├── main.py           # Pipeline: Chunking → Embedding → Insert DB
├── api.py            # FastAPI server (REST API)
├── retrieve.py       # CLI tool tìm kiếm thử qua terminal (gọi qua API)
├── check_db.py       # Kiểm tra trạng thái database (gọi qua API)
├── README.md         # Tài liệu hướng dẫn này
└── PIPELINE.md       # Giải thích chi tiết pipeline xử lý dữ liệu
```

---

## ⚙️ Yêu cầu

- Python **3.10+**
- Milvus chạy dạng **service** (Docker)
- Các thư viện cần thiết:

```bash
pip install fastapi uvicorn pymilvus requests
```

---

## 🔑 Cấu hình API Key

Tạo file `.env` trong thư mục `chatbot/`:

```env
EMBEDED_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxx
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
```

> Lấy API key miễn phí tại: [https://build.nvidia.com](https://build.nvidia.com)
> Dự án chỉ hỗ trợ Milvus Service qua `MILVUS_URI` (không dùng Milvus Lite `.db`).

---

## 🐳 Chạy Milvus Service

```bash
# Đứng tại thư mục chatbot/
docker compose up -d
```

Milvus service mặc định lắng nghe tại `http://localhost:19530`.

---

## 🚀 Chạy server

```bash
# Đứng tại thư mục chatbot/
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Server khởi động tại: `http://localhost:8000`

---

## 📡 API Endpoints

### `GET /`
Kiểm tra server đang hoạt động.

```bash
curl http://localhost:8000/
```

```json
{ "status": "ok", "message": "Ninh Bình RAG API đang chạy 🚀" }
```

---

### `GET /health`
Kiểm tra trạng thái chi tiết: kết nối DB, số chunk, danh sách doc_type hợp lệ.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "collection": "ninhbinh_kb",
  "total_chunks": 596,
  "nvidia_model": "nvidia/nv-embedqa-e5-v5",
  "available_doc_types": [
    "craft_village", "cuisine", "destination", "entertainment",
    "event", "festival", "hotel", "shopping",
    "support", "tour", "transport", "virtual_guide"
  ]
}
```

---

### `POST /search`
Tìm kiếm ngữ nghĩa trong cơ sở dữ liệu du lịch Ninh Bình.

**Request body:**

| Trường      | Kiểu          | Mặc định | Mô tả                                          |
|-------------|---------------|----------|------------------------------------------------|
| `query`     | string        | bắt buộc | Câu hỏi hoặc từ khóa tìm kiếm                 |
| `top_k`     | int           | `5`      | Số lượng kết quả trả về (tối đa 20)            |
| `doc_type`  | string / null | `null`   | Lọc kết quả theo chủ đề (xem bảng bên dưới)   |

**Tìm kiếm toàn bộ (không filter):**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "đặc sản ẩm thực nổi tiếng ở Ninh Bình", "top_k": 3}'
```

**Tìm kiếm có filter theo chủ đề:**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "món ngon đặc sản", "top_k": 3, "doc_type": "cuisine"}'
```

**Response:**

```json
{
  "query": "đặc sản ẩm thực nổi tiếng ở Ninh Bình",
  "doc_type_filter": "cuisine",
  "results": [
    {
      "title": "Nem Dê Ninh Bình – Hương vị truyền thống vùng Cố Đô",
      "url": "https://dulichninhbinh.com.vn/item/3385",
      "doc_id": "doc_b932edd57dfc82cf",
      "doc_type": "cuisine",
      "score": 0.5091,
      "text_preview": "đặc sản khác như cơm cháy, miến lươn, mắm tép Gia Viễn..."
    }
  ],
  "total": 3
}
```

**Các giá trị `doc_type` hợp lệ:**

| doc_type        | Chủ đề              |
|-----------------|---------------------|
| `destination`   | Điểm đến du lịch    |
| `cuisine`       | Ẩm thực             |
| `festival`      | Lễ hội              |
| `hotel`         | Khách sạn           |
| `tour`          | Tour du lịch        |
| `entertainment` | Vui chơi giải trí   |
| `transport`     | Vận chuyển          |
| `shopping`      | Mua sắm             |
| `event`         | Sự kiện nổi bật     |
| `craft_village` | Làng nghề           |
| `virtual_guide` | Thuyết minh ảo      |
| `support`       | Hỗ trợ du khách     |

---

## 🗄️ Tái tạo Database

```bash
# 1. Rebuild DB (từ thư mục gốc)
python3 chatbot/main.py
```

> Xem chi tiết quá trình xử lý trong [PIPELINE.md](./PIPELINE.md).

---

## 🔍 CLI: Kiểm tra nhanh

> Các tool này gọi qua HTTP đến server — **yêu cầu server đang chạy**.

**Tìm kiếm từ terminal:**

```bash
python3 chatbot/retrieve.py "khách sạn gần Tràng An"
```

**Kiểm tra trạng thái database:**

```bash
python3 chatbot/check_db.py
```

---

## 📊 Dữ liệu hiện có

| Chủ đề            | doc_type        | Số bài |
|-------------------|-----------------|--------|
| Điểm đến          | `destination`   | 27     |
| Ẩm thực           | `cuisine`       | 21     |
| Lễ hội            | `festival`      | 21     |
| Làng nghề         | `craft_village` | 21     |
| Khách sạn         | `hotel`         | 19     |
| Mua sắm           | `shopping`      | 19     |
| Vui chơi giải trí | `entertainment` | 16     |
| Tour du lịch      | `tour`          | 14     |
| Vận chuyển        | `transport`     | 12     |
| Hỗ trợ du khách   | `support`       | 6      |
| Thuyết minh ảo    | `virtual_guide` | 3      |
| Sự kiện nổi bật   | `event`         | 2      |
| **Tổng**          |                 | **181 bài / 596 chunks** |

---

## 📖 Tài liệu tương tác

Swagger UI tự động: [http://localhost:8000/docs](http://localhost:8000/docs)

ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
# chatbot_ninhbinh
