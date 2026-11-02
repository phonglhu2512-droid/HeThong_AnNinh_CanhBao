# VisionGuard AI — Industrial Safety Monitoring System (HMI ISA-101 Standard)

[![Release](https://img.shields.io/badge/Release-v1.0.0-crimson.svg)](https://github.com/phonglhu2512-droid/HeThong_AnNinh_CanhBao)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-PyQt6-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![AI Engine](https://img.shields.io/badge/AI-YOLOv8--Nano-purple.svg)](https://docs.ultralytics.com/)
[![Standard](https://img.shields.io/badge/Standard-ISA--101%20HMI-orange.svg)](#)

Hệ thống giám sát an ninh và an toàn lao động thời gian thực tích hợp Computer Vision (OpenCV) và Trí tuệ nhân tạo (YOLOv8-Nano). Giải pháp hỗ trợ kết nối trực tiếp với hệ thống Camera IP giám sát công nghiệp (giao thức RTSP) hoặc Webcam, phát hiện xâm nhập vùng nguy hiểm xung quanh các thiết bị cơ khí (máy dập thủy lực, robot công nghiệp), tự động ngắt khẩn cấp (E-STOP) và lưu trữ bằng chứng vi phạm.

---

## 📌 Các Tính Năng Nổi Bật

- **Mô hình Hybrid AI (YOLOv8n + Geometric Engine):**
  - Nhận diện người chính xác (`class 0: person`), loại bỏ hoàn toàn báo động giả do đồ vật tĩnh hoặc thay đổi ánh sáng.
  - Thuật toán kích hoạt đa điều kiện: Kiểm tra điểm tiếp xúc sàn ($P_{\text{foot}}$) và sự giao cắt giữa hộp bao Bounding Box với đa giác vùng cấm (ROI).
- **Giao diện Giám sát Chuẩn Công nghiệp (ISA-101 High-Performance HMI):**
  - Tối ưu hóa thị giác theo tiêu chuẩn ISA-101 với bảng màu tương phản nhẹ nhàng, điểm nhấn Đỏ đô `#9B1B30` và Xanh ngọc `#10B981`.
  - Bộ thẻ trạng thái vận hành trực quan, theo dõi số lượng người, số lượt cảnh báo theo ca và trạng thái máy dập.
- **Tương tác Vùng cấm (ROI) & Vùng che chắn (Machine Mask):**
  - Hỗ trợ công cụ click chuột chấm điểm vẽ đa giác vùng cấm linh hoạt trực tiếp trên màn hình Live Stream.
  - Tính năng vẽ vùng Mask che chắn các chi tiết cơ khí chuyển động lặp lại để loại trừ nhiễu quang học.
- **Quản lý Bằng chứng Vi phạm (Snapshot Carousel & Audit Logger):**
  - Tự động trích xuất và hiển thị các thẻ ảnh vi phạm kèm mốc thời gian (Timestamp) dạng cuộn ngang (Bottom Dock).
  - Tự động ghi nhật ký vi phạm ra file báo cáo `Violation_Report.csv` phục vụ kiểm toán an toàn lao động.
- **Xử lý Luồng Video & Giao thức Công nghiệp:**
  - Kiến trúc xử lý đa luồng bất đồng bộ (Multi-threading với `Queue(maxsize=1)`) bảo đảm UI không bao giờ bị giật lag.
  - Hỗ trợ đa dạng nguồn cấp: RTSP Stream (Camera Dahua, Imou, Hikvision...), Webcam USB/Laptop (tự động cân chỉnh lật gương) và Video File.

---

## 🏗️ Cấu Trúc Mã Nguồn

```text
HeThong_AnNinh_CanhBao/
├── config/
│   └── config.json              # Cấu hình nguồn video, tọa độ ROI/Mask, ngưỡng trễ
├── output_data/
│   ├── evidence/                # Lưu trữ ảnh chụp vi phạm thời gian thực (.jpg)
│   └── reports/                 # Lưu trữ báo cáo kiểm toán vi phạm (.csv)
├── src/
│   ├── core/
│   │   ├── cv_engine.py         # Module AI (YOLOv8n) & Thuật toán thị giác hình học
│   │   ├── safety_controller.py # Quản lý logic E-STOP, Smart Mute và còi cảnh báo
│   │   └── stream_manager.py    # Quản lý luồng RTSP/Webcam bất đồng bộ
│   ├── ui/
│   │   └── hmi_dashboard.py     # Giao diện giám sát ISA-101 Dashboard (PyQt6)
│   └── utils/
│       ├── audit_logger.py      # Module lưu vết, xuất báo cáo CSV và ảnh vi phạm
│       └── config_manager.py    # Đọc/ghi cấu hình JSON an toàn
├── main.py                      # Điểm khởi chạy ứng dụng chính
├── yolov8n.pt                   # Trọng số mô hình nhận diện người YOLOv8-Nano
└── requirements.txt             # Danh sách thư viện phụ thuộc
```

---

## ⚙️ Hướng Dẫn Cài Đặt & Khởi Chạy

### Yêu cầu tiên quyết
- Hệ điều hành: Windows 10/11 (khuyến nghị), Linux hoặc macOS.
- Python: Phiên bản 3.10 trở lên.
- Git: Công cụ quản lý mã nguồn.

### Các lệnh thực thi trong Terminal / PowerShell

```bash
# 1. Tải kho mã nguồn về máy
git clone [https://github.com/phonglhu2512-droid/HeThong_AnNinh_CanhBao.git](https://github.com/phonglhu2512-droid/HeThong_AnNinh_CanhBao.git)
cd HeThong_AnNinh_CanhBao

# 2. Khởi tạo môi trường ảo Python độc lập
python -m venv venv

# 3. Kích hoạt môi trường ảo
# Trên Windows PowerShell:
.\venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate

# 4. Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt

# 5. Khởi chạy phần mềm
python main.py
```

---

## 🔒 Cấu Hình Kết Nối Camera IP (Giao Thức RTSP)

Khi kết nối với các camera an ninh IP công nghiệp hoặc gia đình (Imou, Dahua, Ezviz, Hikvision):

### 1. Cú pháp URL RTSP chuẩn theo thiết bị

```text
# Dành cho Camera Imou / Dahua (Ví dụ: Imou Cruiser Dual 2):
rtsp://admin:<SAFETY_CODE>@<IP_ADDRESS>:554/cam/realmonitor?channel=1&subtype=0

# Dành cho Camera Hikvision / Ezviz:
rtsp://admin:<VERIFICATION_CODE>@<IP_ADDRESS>:554/h264/ch1/main/av_stream
```

### 2. Các tham số cấu hình:
- `<SAFETY_CODE>`: Mã an toàn gồm chữ in hoa/số in trên tem dưới đáy camera hoặc trong app cài đặt.
- `<IP_ADDRESS>`: Địa chỉ IP nội bộ của camera trong mạng Wi-Fi/LAN (ví dụ: `192.168.101.251`).
- `channel=1`: Mắt camera số 1 (đối với dòng camera kép 2 mắt như Cruiser Dual, đổi sang `channel=2` để xem mắt xoay).

### 3. Thao tác trên phần mềm:
1. Nhấn nút **`[Kết nối Camera IP]`** ở góc trên bên phải thanh Header giao diện.
2. Điền chính xác địa chỉ IP, tài khoản `admin`, mật khẩu an toàn và RTSP Path.
3. Nhấn **`Lưu _Áp dụng`** để chuyển luồng giám sát.