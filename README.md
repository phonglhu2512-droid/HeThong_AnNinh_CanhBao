# VisionGuard AI (Nguyen-iSafe) — Industrial & Civil Safety Audit System

[![Release](https://img.shields.io/badge/Release-v1.0.0-crimson.svg)](https://github.com/phonglhu2512-droid/HeThong_AnNinh_CanhBao)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/UI-PyQt6-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![AI Inference](https://img.shields.io/badge/AI-YOLOv8--Nano-purple.svg)](https://docs.ultralytics.com/)
[![Standard](https://img.shields.io/badge/Standard-ISA--101%20HMI-orange.svg)](#)

Hệ thống thị giác máy tính giám sát ranh giới an toàn thời gian thực và **Kiểm toán Tuân thủ An toàn Lao động (Safety Audit Trail)** ứng dụng Trí tuệ Nhân tạo (YOLOv8-Nano) tích hợp chuẩn giao diện điều khiển công nghiệp **ISA-101 HMI**.

---

## 💡 Bối Cảnh Thực Tế & Bài Toán Giải Quyết

Hệ thống được phát triển nhằm giải quyết triệt để 3 rào cản thực tế từ hộ dân dụng cho đến các nhà xưởng cơ khí vừa và nhỏ (SME):

1. **Vấn nạn báo động giả từ Camera IP thông dụng (False Alarm Fatigue):**
   - Các dòng camera an ninh gia đình/thông dụng (Imou, Dahua, Hikvision, Ezviz) sử dụng thuật toán phát hiện chuyển động dựa trên biến đổi điểm ảnh (Pixel Motion) hoặc chip AI giá rẻ. Hậu quả là côn trùng bay qua mắt cam, bụi bẩn, bóng râm thay đổi hay đèn xe ban đêm đều kích hoạt còi/thông báo liên tục, gây ức chế cho người dùng.
   - Tính năng khoanh vùng cảnh báo sẵn có trên app camera rất sơ sài, chỉ cho phép vẽ các khung thô sơ, nhận diện kém chính xác và không phân biệt được đâu là con người hay đồ vật.
2. **Chi phí hệ thống Vision công nghiệp chuyên dụng quá đắt đỏ:**
   - Các giải pháp Machine Vision chuyên dụng (Cognex, Keyence) có giá từ hàng chục đến hàng trăm triệu đồng cho một vị trí giám sát, vượt quá khả năng tài chính của các phân xưởng vừa và nhỏ.
3. **Cảm biến phần cứng tại xưởng bị "Mù dữ liệu":**
   - Rơ-le hay cảm biến quang bảo vệ máy chỉ biết hú còi/ngắt máy tại chỗ rồi im lặng khi người rút ra, hoàn toàn không có bằng chứng hình ảnh đối chất, không ghi nhận được tần suất vi phạm theo ca làm việc để chấn chỉnh hành vi làm ẩu trước khi tai nạn xảy ra.

### 👉 Giải pháp đột phá của VisionGuard AI:
* **Bộ lọc AI loại bỏ 100% báo động giả:** Ứng dụng mô hình **YOLOv8n** phân tích chuẩn xác hình thể người (`class 0: person`), triệt tiêu hoàn toàn nhiễu từ côn trùng, ánh sáng hay chuyển động cơ khí lặp lại.
* **Khoanh vùng linh hoạt (Zero-Downtime Polygon ROI):** Cho phép vẽ đa giác tự do bao quanh khu vực cần bảo vệ bằng chuột trong vài giây, kết hợp thuật toán điểm chạm chân (`FootPoint`) và vùng che máy (`Machine Mask`).
* **Tận dụng tối đa hạ tầng sẵn có (Zero Additional Hardware Cost):** Kéo trực tiếp luồng RTSP nội bộ từ các mắt Camera IP thông dụng sẵn có trong xưởng/nhà ở, nâng cấp camera an ninh thông thường thành hệ thống giám sát an toàn thông minh.
* **Kiến trúc An toàn Kép & Lưu vết Kiểm toán (Hybrid Safety & Audit Trail):** Chạy song song bổ trợ rào chắn cảm biến, tự động chụp ảnh bằng chứng có khung vi phạm, đo thời gian chiếm dụng, xuất báo cáo CSV theo ca và gửi cảnh báo tức thì qua Telegram.

---

## 📌 Các Tính Năng Kỹ Thuật Nổi Bật

- **Lõi AI Nhận Diện Người (YOLOv8n Inference):**
  - Nhận diện chính xác hình dáng con người (`class 0: person`), loại bỏ hoàn toàn báo động giả do đồ vật tĩnh, bóng râm hay ánh sáng đèn xưởng thay đổi.
  - Thuật toán tiếp xúc sàn: Phân tích tọa độ chân (`FootPoint`) kết hợp giao cắt hộp bao (`BoundingBox`) với đa giác vùng cấm.
- **Tái Cấu Hình Vùng Cấm Nhanh Chóng (Zero-Downtime ROI & Mask):**
  - Cho phép Kỹ sư dùng chuột chấm điểm vẽ đa giác vùng cấm (ROI) linh hoạt trên màn hình chỉ trong vài giây, không tốn công đục sàn kéo dây như cảm biến cứng khi đổi layout máy.
  - Hỗ trợ công cụ vẽ vùng che chắn cơ khí (Machine Mask) nhằm triệt tiêu báo động giả từ các chuyển động tuần hoàn của tay máy.
- **Lưu Vết Kiểm Toán & Xuất Báo Cáo (Safety Audit Trail):**
  - Tự động đóng khung đỏ, gán mốc thời gian và trích xuất ảnh bằng chứng vi phạm hiển thị trực tiếp trên thanh cuộn (Bottom Dock).
  - Tích hợp bộ lọc sự kiện theo khung giờ/ca làm việc và chức năng xuất file báo cáo `Violation_Report.csv`.
- **Cảnh Báo Từ Xa Tức Thời Qua Telegram (Daemon Telemetry):**
  - Module truyền thông chạy trên luồng ngầm độc lập (`Daemon Thread`), lập tức gửi ảnh chụp hiện trường và thông tin vi phạm về ứng dụng Telegram của Quản đốc/Chủ cơ sở mà không làm gián đoạn luồng xử lý video.
- **Bảng Điều Khiển Chuẩn Công Nghiệp ISA-101 HMI:**
  - Thiết kế bảng màu tương phản nhẹ nhàng, giảm mỏi mắt cho người vận hành, chỉ kích hoạt điểm nhấn đỏ đô `#9B1B30` và âm thanh khi có sự cố.
  - Tích hợp cơ chế **Phân quyền hai lớp**: Chế độ *Giám sát (Khóa)* an toàn cho người vận hành và chế độ *Kỹ sư (Setup)* có mật khẩu quản trị.
- **Xử Lý Luồng Video Công Nghiệp Ổn Định:**
  - Kiến trúc đa luồng bất đồng bộ (`Queue(maxsize=1)`) triệt tiêu hiện tượng dồn frame và trễ tích lũy trên luồng RTSP mạng LAN.

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
│       └── telegram_notifier.py # Module gửi cảnh báo & ảnh bằng chứng qua Telegram Bot
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

## 📲 Kiến Trúc Cảnh Báo Telegram (Telegram Alert Subsystem)

### 1. Chế độ Nghiệm thu Tức thì (Zero-Config PoC Mode)
Phiên bản đóng gói thực thi (`.exe`) đã được tích hợp sẵn kênh truyền mã hóa tới Bot Gateway quản trị[cite: 15]. Khi hệ thống phát hiện xâm nhập vượt ngưỡng trễ an toàn (`dwell_seconds`), hình ảnh bằng chứng kèm mã sự kiện (`VIO-xxx`) sẽ tự động gửi thẳng về thiết bị di động mà không yêu cầu người chấm thẩm định phải cấu hình thủ công[cite: 15].

### 2. Lộ trình Thương mại hóa Đa Phân xưởng (Enterprise Multi-Tenant)
Khi triển khai thực tế tại nhiều nhà máy khác nhau, kỹ sư hệ thống thiết lập kênh nhận cảnh báo độc lập qua 3 bước[cite: 15]:

1. Truy cập `@BotFather` trên Telegram, dùng lệnh `/newbot` để khởi tạo Bot riêng của nhà máy và lấy chuỗi `BOT_TOKEN`[cite: 15].
2. Nhắn tin tới `@userinfobot` để lấy mã định danh `CHAT_ID` của Quản đốc hoặc Group An toàn lao động[cite: 15].
3. Cập nhật thông số `BOT_TOKEN`, `CHAT_ID` và tên vị trí thiết bị (`device_name`) trong module điều phối để định tuyến cảnh báo về đúng bộ phận phụ trách[cite: 15].

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