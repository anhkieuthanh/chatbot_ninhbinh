import sys
import requests

API_BASE = "http://localhost:8000"

def main():
    query = "đặc sản ẩm thực ở ninh bình là gì?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    top_k = 3
    print(f"\n[?] Câu hỏi: '{query}'")
    print(f"Đang gọi API tại {API_BASE}/search ...\n")

    try:
        resp = requests.post(
            f"{API_BASE}/search",
            json={"query": query, "top_k": top_k},
            timeout=20
        )
    except requests.exceptions.ConnectionError:
        print(f"Lỗi: Không thể kết nối tới server tại {API_BASE}")
        print("Hãy chọn: python3 -m uvicorn api:app --host 0.0.0.0 --port 8000")
        return

    if resp.status_code != 200:
        print(f"Lỗi từ API: {resp.status_code} - {resp.text}")
        return

    data = resp.json()
    print("-" * 50)
    for i, result in enumerate(data["results"]):
        print(f"Kết quả {i+1}:")
        print(f">> Tiêu đề  : {result['title']}")
        print(f">> Độ tin cậy: {result['score']}")
        print(f">> URL      : {result['url']}")
        print(f">> Nội dung : {result['text_preview'][:300]}...\n")

if __name__ == "__main__":
    main()
