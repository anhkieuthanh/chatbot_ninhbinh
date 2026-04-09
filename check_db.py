import requests

API_BASE = "http://localhost:8000"

def check_database():
    print(f"Đang kiểm tra server tại: {API_BASE}\n")

    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"Lỗi: Không thể kết nối tới server tại {API_BASE}")
        print("Hãy chạy: python3 -m uvicorn api:app --host 0.0.0.0 --port 8000")
        return

    if resp.status_code != 200:
        print(f"Lỗi từ server: {resp.status_code} - {resp.text}")
        return

    data = resp.json()
    print("=" * 50)
    print("THÔNG TIN DATABASE")
    print("=" * 50)
    print(f"[1] Trạng thái server : {data.get('status', 'N/A').upper()}")
    print(f"[2] Collection        : {data.get('collection', 'N/A')}")
    print(f"[3] Tổng số chunks    : {data.get('total_chunks', 0)}")
    print(f"[4] Model embedding   : {data.get('nvidia_model', 'N/A')}")
    print("-" * 50)

if __name__ == "__main__":
    check_database()
