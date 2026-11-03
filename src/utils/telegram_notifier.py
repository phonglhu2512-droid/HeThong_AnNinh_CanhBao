import threading
import requests

BOT_TOKEN = "9267651925:AAFy_Tv_-986_1Oea14mK747jnsX5FRw210"
CHAT_ID = "5849584573"

def _send_photo_worker(image_path: str, caption: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as photo_file:
            payload = {"chat_id": CHAT_ID, "caption": caption}
            files = {"photo": photo_file}
            requests.post(url, data=payload, files=files, timeout=5)
    except Exception as e:
        print(f"[Telegram Alert] Lỗi gửi ảnh: {e}")

def notify_violation(image_path: str, timestamp_str: str, duration: float, device_name: str = "Hệ thống Cơ khí / Vùng nguy hiểm"):
    # Nếu vi phạm vừa chạm vạch (0s), hiển thị "Khởi phát xâm phạm" thay vì 0.00s
    status_time = f"{duration:.2f}s" if duration > 0 else "Phát hiện tức thời (Khởi phát)"

    caption = (
        f"🚨 CẢNH BÁO XÂM PHẠM AN TOÀN!\n"
        f"📍 Vị trí/Thiết bị: {device_name}\n"
        f"⏰ Thời điểm: {timestamp_str}\n"
        f"⏱ Trạng thái xâm nhập: {status_time}\n"
        f"🛑 Tác động an toàn: KÍCH HOẠT DỪNG KHẨN CẤP (E-STOP)"
    )
    thread = threading.Thread(target=_send_photo_worker, args=(image_path, caption), daemon=True)
    thread.start()