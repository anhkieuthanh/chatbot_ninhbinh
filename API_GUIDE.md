# 📖 Hướng dẫn sử dụng API — Ninh Bình RAG

Base URL: `http://[IP_ADDRESS]`  
Interactive Docs: [http://[IP_ADDRESS]/docs](http://[IP_ADDRESS]/docs)

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Endpoints](#endpoints)
  - [GET / — Kiểm tra server](#1-get----kiểm-tra-server)
  - [GET /health — Trạng thái chi tiết](#2-get-health--trạng-thái-chi-tiết)
  - [POST /search — Tìm kiếm thông tin](#3-post-search--tìm-kiếm-thông-tin)
- [Danh sách doc_type](#danh-sách-doc_type)
- [Ví dụ thực tế](#ví-dụ-thực-tế)
- [Xử lý lỗi](#xử-lý-lỗi)

---

## Tổng quan

API sử dụng **Vector Search (RAG)** để tìm kiếm thông tin du lịch Ninh Bình.  
Mỗi request tìm kiếm sẽ:
1. Embed câu hỏi thành vector qua NVIDIA Embedding API
2. Tìm các đoạn văn bản gần nhất trong cơ sở dữ liệu
3. Trả về kết quả kèm điểm tương đồng (`score`)

---

## Endpoints

### 1. `GET /` — Kiểm tra server

Kiểm tra server có đang hoạt động không.

**Request:**
```http
GET http://180.93.42.145/
```

**Response:**
```json
{
  "status": "ok",
  "message": "Ninh Bình RAG API đang chạy 🚀"
}
```

---

### 2. `GET /health` — Trạng thái chi tiết

Kiểm tra trạng thái server, database và số lượng chunks.

**Request:**
```http
GET http://180.93.42.145/health
```

**Response:**
```json
{
  "status": "ok",
  "collection": "ninhbinh_kb",
  "total_chunks": 1420,
  "nvidia_model": "nvidia/nv-embedqa-e5-v5",
  "available_doc_types": [
    "craft_village", "cuisine", "destination", "entertainment",
    "event", "festival", "hotel", "shopping",
    "support", "tour", "transport", "virtual_guide"
  ]
}
```

---

### 3. `POST /search` — Tìm kiếm thông tin

Endpoint chính để tìm kiếm thông tin du lịch Ninh Bình.

**Request:**
```http
POST http://180.93.42.145/search
Content-Type: application/json
```

**Body:**

| Trường | Kiểu | Bắt buộc | Mô tả |
|--------|------|----------|-------|
| `query` | string | ✅ | Câu hỏi hoặc từ khóa tìm kiếm |
| `top_k` | integer | ❌ | Số kết quả trả về (mặc định: `5`, tối đa: `20`) |
| `doc_type` | string | ❌ | Lọc theo chủ đề (xem danh sách bên dưới) |
| `category` | string | ❌ | Lọc theo category gốc nếu cần |

**Ví dụ body:**
```json
{
  "query": "Nhà hàng hải sản ngon ở Ninh Bình",
  "top_k": 5,
  "doc_type": "cuisine"
}
```

**Response:**
```json
{
  "query": "Nhà hàng hải sản ngon ở Ninh Bình",
  "doc_type_filter": "cuisine",
  "category_filter": null,
  "total": 5,
  "results": [
    {
      "title": "Nhà hàng Hoa Lư - Đặc sản dê núi Ninh Bình",
      "url": "https://...",
      "doc_id": "abc123",
      "doc_type": "cuisine",
      "category": "Ẩm thực",
      "score": 0.8921,
      "text_preview": "Nhà hàng Hoa Lư nổi tiếng với các món đặc sản..."
    }
  ]
}
```

**Mô tả các trường trong `results`:**

| Trường | Mô tả |
|--------|-------|
| `title` | Tiêu đề bài viết/địa điểm |
| `url` | Link nguồn gốc |
| `doc_id` | ID tài liệu trong database |
| `doc_type` | Chủ đề của tài liệu |
| `category` | Nhãn category gốc trong dữ liệu |
| `score` | Điểm tương đồng (0–1, càng cao càng liên quan) |
| `text_preview` | Đoạn trích nội dung (tối đa 400 ký tự) |

---

## Danh sách `doc_type`

| Giá trị | Ý nghĩa |
|---------|---------|
| `destination` | Địa điểm tham quan |
| `cuisine` | Ẩm thực, nhà hàng |
| `hotel` | Khách sạn, homestay, lưu trú |
| `tour` | Tour du lịch |
| `festival` | Lễ hội, sự kiện văn hóa |
| `entertainment` | Vui chơi, giải trí |
| `transport` | Phương tiện di chuyển |
| `shopping` | Mua sắm, đặc sản mang về |
| `event` | Sự kiện |
| `craft_village` | Làng nghề truyền thống |
| `virtual_guide` | Hướng dẫn tham quan ảo |
| `support` | Hỗ trợ du khách |

> Bỏ trống `doc_type` để tìm kiếm trên toàn bộ nội dung.

---

## Ví dụ thực tế

### cURL

```bash
# Tìm kiếm đơn giản
curl -X POST http://180.93.42.145/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Tràng An có gì đẹp?"}'

# Tìm kiếm với lọc theo chủ đề
curl -X POST http://180.93.42.145/search \
  -H "Content-Type: application/json" \
  -d '{"query": "khách sạn gần trung tâm", "top_k": 3, "doc_type": "hotel"}'

# Kiểm tra health
curl http://180.93.42.145/health
```

### Python

```python
import requests

BASE_URL = "http://180.93.42.145"

# Tìm kiếm thông tin
response = requests.post(f"{BASE_URL}/search", json={
    "query": "Lễ hội Hoa Lư diễn ra vào tháng mấy?",
    "top_k": 3,
    "doc_type": "festival"
})

data = response.json()
for item in data["results"]:
    print(f"[{item['score']:.4f}] {item['title']}")
    print(f"  → {item['text_preview'][:100]}...")
    print()
```

### JavaScript / Fetch

```javascript
const BASE_URL = "http://180.93.42.145";

async function search(query, docType = null, topK = 5) {
  const res = await fetch(`${BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, doc_type: docType, top_k: topK })
  });
  return await res.json();
}

// Sử dụng
const results = await search("Cuc Phuong có loài động vật gì?", "destination");
console.log(results);
```

### Node.js (axios)

```javascript
const axios = require("axios");

const api = axios.create({ baseURL: "http://180.93.42.145" });

const { data } = await api.post("/search", {
  query: "Đi từ Hà Nội đến Ninh Bình bằng gì?",
  doc_type: "transport",
  top_k: 5
});

console.log(`Tìm thấy ${data.total} kết quả`);
```

---

## Xử lý lỗi

| HTTP Code | Nguyên nhân | Xử lý |
|-----------|-------------|-------|
| `400` | `query` trống hoặc `doc_type` không hợp lệ | Kiểm tra lại body request |
| `502` | NVIDIA Embedding API lỗi | Thử lại sau |
| `500` | Lỗi server | Liên hệ quản trị |

**Ví dụ response lỗi:**
```json
{
  "detail": "doc_type không hợp lệ: 'food'. Các giá trị hợp lệ: ['craft_village', 'cuisine', ...]"
}
```

---

## Tích hợp vào Chatbot

Luồng điển hình khi dùng API này làm **RAG backend** cho chatbot:

```
User question
     ↓
POST /search  (query = câu hỏi, top_k = 3-5)
     ↓
Lấy text_preview từ results
     ↓
Ghép vào prompt: "Dựa vào thông tin sau: {context}\nTrả lời: {question}"
     ↓
Gửi đến LLM (GPT, Gemini, Claude...)
     ↓
Trả lời người dùng
```

---

*API Version: 1.1.0 — Cập nhật: 04/2026*
