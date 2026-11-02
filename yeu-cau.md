# TÀI LIỆU ĐẶC TẢ YÊU CẦU & KIẾN TRÚC PHẦN MỀM TOÀN DIỆN
## HỆ THỐNG GIÁM SÁT AN TOÀN VÀ CẢNH BÁO VÙNG NGUY HIỂM THỜI GIAN THỰC QUA CAMERA IP THÔNG DỤNG

* **Tên dự án:** VisionGuard AI — Industrial Safety & Virtual Fence Platform
* **Môn học:** Thị giác Máy tính (Computer Vision)
* **Phân loại sản phẩm:** Phần mềm Ứng dụng theo định hướng Giải pháp Thực tế (Solution-Oriented Application Software)
* **Ngôn ngữ & Công nghệ cốt lõi:** Python 3.8+, OpenCV (`cv2`), NumPy, PyQt6/PySide6, Multi-threading
* **Mốc thời gian hoàn tất báo cáo:** Trước ngày 11/11

---

# PHẦN 1: PHÂN TÍCH ĐỀ TÀI, MỤC TIÊU, CÔNG NGHỆ & TÍNH KHẢ THI

## 1.1. Mô tả Ý tưởng Hệ thống (Problem - Solution Fit)

```
[BỐI CẢNH & NỖI ĐAU THỰC TẾ]                    [GIẢI PHÁP VISIONGUARD AI]
• Camera AI chuyên dụng: 10 - 30 triệu VNĐ      ┌──────────────────────────────────────────────┐
  (Chi phí quá cao với doanh nghiệp vừa & nhỏ)   │ TẦNG PHẦN MỀM TRUNG GIAN (EDGE MIDDLEWARE)   │
• Camera IP dân dụng: 300k - 1 triệu VNĐ         │ • Kéo luồng RTSP/Webcam/Video đa chế độ      │
  (Chỉ ghi hình thụ động, không cảnh báo)         │ • Xử lý ảnh thời gian thực trên PC xưởng      │
                                                │ • Thiết lập rào chắn an toàn ảo (Dynamic ROI)│
                                                │ • Phát hiện xâm nhập qua điểm chân thực tế    │
                                                │ • Báo động tức thời, chụp ảnh, xuất CSV       │
                                                └──────────────────────────────────────────────┘
```

* **Bối cảnh & Vấn đề thực tế (Pain Point):** Trong môi trường sản xuất công nghiệp (xưởng cơ khí, máy dập thủy lực, băng chuyền tự động, khu vực xe nâng), các tai nạn lao động nghiêm trọng thường xảy ra trong tích tắc khi công nhân vô tình bước vào vùng nguy hiểm trong lúc máy đang vận hành. Các dòng camera AI chuyên dụng có giá thành quá đắt đỏ ($10 - 30$ triệu VNĐ/mắt), trong khi hệ thống camera IP giám sát thông dụng sẵn có tại xưởng ($300.000 - 1.000.000$ VNĐ/mắt) chỉ ghi hình thụ động để xem lại sau khi tai nạn đã xảy ra.
* **Giải pháp VisionGuard AI:** Xây dựng một **Tầng phần mềm trung gian xử lý biên (Edge Vision Middleware)** chạy trên máy tính tại xưởng. Hệ thống tận dụng trực tiếp luồng video thời gian thực từ camera IP thông dụng qua giao thức **RTSP** (hoặc Webcam/Video thực nghiệm), áp dụng chuỗi thuật toán Thị giác Máy tính để tạo **rào chắn an toàn ảo (Virtual Safety Fence)**. Khi phát hiện chân người lao động xâm nhập vùng cấm vượt quá độ trễ an toàn ($0.8\text{s}$), hệ thống lập tức hú còi cảnh báo, kích hoạt dừng máy khẩn cấp (Interlock), chụp ảnh bằng chứng và ghi nhận nhật ký phục vụ kiểm toán an toàn lao động.
* **Lộ trình phân kỳ (Roadmap):**
  * *Giai đoạn 1 (MVP Đồ án hiện tại — On-Premise LAN):* Xử lý cục bộ trong mạng LAN nội bộ nhà xưởng nhằm đạt độ trễ cực thấp ($< 200\text{ ms}$), bảo mật dữ liệu tuyệt đối và không phụ thuộc vào đường truyền Internet bên ngoài khi bảo vệ đồ án.
  * *Giai đoạn 2 (Tầm nhìn mở rộng — Distributed WAN/Cloud):* Mở rộng giám sát đa chi nhánh xuyên tỉnh qua VPN Site-to-Site, WebRTC Streaming Server và cơ chế P2P Relay vượt tường lửa NAT.

---

## 1.2. Hệ thống Mục tiêu Chi tiết

### 1.2.1. Mục tiêu Kỹ thuật & Định lượng (Technical Metrics)
* **Tốc độ xử lý thời gian thực:** Đạt và duy trì ổn định $\ge 20 - 25\text{ FPS}$ trên máy tính cấu hình phổ thông (CPU Core i5, không bắt buộc GPU rời).
* **Độ trễ toàn trình (End-to-End Latency):** Kiểm soát độ trễ từ lúc camera thu hình đến khi giao diện phát cảnh báo $< 200\text{ ms}$.
* **Khả năng chịu lỗi (Fault Tolerance):** Vận hành bền bỉ (`Crash-Free`), tự động bắt ngoại lệ khi đứt cáp/mất nguồn camera và tự động thử kết nối lại (`Auto-Reconnect`) sau mỗi $3.0\text{ giây}$.

### 1.2.2. Mục tiêu Nghiệp vụ & Chống Báo động giả
* **Chuyển đổi linh hoạt 3 nguồn đầu vào:** Hỗ trợ chuyển đổi mượt mà trực tiếp trên giao diện giữa Webcam (`Device ID: 0`), File Video thực nghiệm (`.mp4`, `.avi`) và Camera IP (`RTSP URL`) kèm cơ chế tự giải phóng tài nguyên RAM.
* **Hiệu chuẩn vùng cấm động (Dynamic ROI):** Vẽ đa giác vùng nguy hiểm ($N \ge 3$ đỉnh) trực tiếp bằng chuột trên màn hình; hỗ trợ xóa đỉnh cuối, khôi phục mặc định và lưu vào `config.json`.
* **Triệt tiêu báo động giả (False Positive Rejection):**
  * *Khử bóng đổ:* Chỉ kiểm tra điểm tiếp xúc chân người trên mặt sàn $P_{\text{foot}} = (x + w/2, y + h)$ thay vì dùng toàn bộ hộp bao thân trên.
  * *Chống nhiễu thoáng qua:* Tích hợp bộ đệm thời gian trễ ($t_{\text{delay}} = 0.8\text{s}$) và ngưỡng diện tích tối thiểu ($S_{\min} = 1500\text{ px}^2$).
  * *Bảo vệ sốc sáng:* Tự động reset mô hình nền khi diện tích biến đổi vượt quá $60\%$ khung hình do bật/tắt đèn xưởng.
* **Cảnh báo & Kiểm toán:** Khi xâm nhập vượt thời gian trễ, phát còi và giữ cảnh báo đỏ đến khi người bước ra ROI; người trực có thể ACK để tắt tiếng trong khi vẫn giữ cảnh báo đỏ. Tự động chốt thời lượng vi phạm khi thoát vùng; hỗ trợ lọc lịch sử theo Ca/Giờ, xem ảnh trên khay Snapshot Carousel và xuất báo cáo `.csv`.

---

## 1.3. Ánh xạ Kiến thức Môn học & Mô hình Toán học Cốt lõi

### 1.3.1. Ánh xạ 4 Tầng Kiến thức Thị giác Máy tính
* **Tầng 1 (Nền tảng & Công cụ):** Python 3, OpenCV, NumPy, quản lý mã nguồn nhóm trên GitHub theo mô hình Git Flow (`main`, `develop`, `feature/*`).
* **Tầng 2 (Xử lý ảnh cơ bản):** Thu nhận luồng `cv2.VideoCapture`, chuẩn hóa kích thước `cv2.resize` ($640 \times 360$), chuyển không gian màu `Grayscale`, lọc mịn khử nhiễu nén H.264 bằng `cv2.GaussianBlur` ($5 \times 5$).
* **Tầng 3 (Thuật toán bậc cao):** Phân tách chuyển động bằng mô hình hỗn hợp Gauss thích nghi `cv2.createBackgroundSubtractorMOG2`, phép toán hình thái học `cv2.morphologyEx` (`MORPH_OPEN` & `MORPH_CLOSE`), trích xuất đường bao `cv2.findContours` và hộp bao `cv2.boundingRect`.
* **Tầng 4 (Giải pháp thực tế):** Kiểm tra va chạm hình học đa giác `cv2.pointPolygonTest`, bộ đệm trễ an toàn, giao diện High-Performance HMI chuẩn ISA-101, lưu bằng chứng `cv2.imwrite` và trích xuất báo cáo CSV.

### 1.3.2. Mô hình Toán học Cốt lõi
1. **Mô hình Trừ nền Hỗn hợp Gauss Thích nghi (MOG2):**
   Xác suất quan sát được giá trị điểm ảnh $X_t$ tại thời điểm $t$ được mô hình hóa bởi tổng trọng số của $K$ phân phối Gauss ($K = 3 \dots 5$):
   $$P(X_t) = \sum_{i=1}^{K} \omega_{i, t} \cdot \eta(X_t, \mu_{i, t}, \Sigma_{i, t})$$
   Tham số `detectShadows=True` tách riêng điểm ảnh bóng đổ (gán giá trị xám $127$) khỏi đối tượng chuyển động thực (gán giá trị trắng $255$).
2. **Trích xuất Tọa độ Điểm Tiếp xúc Sàn (Foot-Point Extraction):**
   Từ hộp bao đối tượng $(x, y, w, h)$ thỏa mãn điều kiện lọc diện tích $S \ge S_{\min}$, tọa độ chân tiếp xúc mặt phẳng sàn được xác định tại:
   $$P_{\text{foot}} = \left(x + \frac{w}{2},\; y + h\right)$$
3. **Kiểm tra Giao cắt Hình học Đa giác (Point-in-Polygon Test):**
   Khoảng cách đại số từ điểm chân $P_{\text{foot}}$ tới đa giác vùng cấm $\mathcal{C}_{\text{ROI}}$:
   $$\text{result} = \text{cv2.pointPolygonTest}(\mathcal{C}_{\text{ROI}},\; P_{\text{foot}},\; \text{False})$$
   $$\text{Trạng thái} = \begin{cases} \text{Xâm nhập (Inside / On Edge)}, & \text{nếu } \text{result} \ge 0 \\ \text{An toàn (Outside)}, & \text{nếu } \text{result} < 0 \end{cases}$$

---

## 1.4. Đánh giá Tính Khả thi & Phạm vi Hoạt động (ODD)

* **Tính khả thi kỹ thuật:** Các thuật toán cổ điển trong OpenCV được tối ưu hóa bằng C/C++ dưới tầng thấp, cho phép xử lý khung hình $640 \times 360$ chỉ trong $12 - 25\text{ ms}$ trên CPU thông thường, hoàn toàn khả thi để đạt $\ge 25\text{ FPS}$.
* **Phạm vi Điều kiện Vận hành (ODD - Operational Design Domain):**
  * *Góc lắp đặt camera:* Cố định ở độ cao $3.5\text{m} - 5.5\text{m}$, góc chúc xuống mặt sàn $30^\circ - 45^\circ$.
  * *Môi trường chiếu sáng:* Trong nhà (nhà xưởng, kho bãi, phòng thực hành) có nguồn sáng ổn định.
  * *Vận tốc đối tượng:* Người lao động di chuyển ở tốc độ đi bộ bình thường ($v < 3\text{ m/s}$).

---

# PHẦN 2: XÁC ĐỊNH TÁC NHÂN (ACTORS) & MA TRẬN HÀNH VI

## 2.1. Phân tích Ranh giới Tác nhân & Hệ thống

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    SƠ ĐỒ TỔNG QUAN TÁC NHÂN & TƯƠNG TÁC (USE-CASE MAP)                 │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [TÁC NHÂN ĐẦU VÀO]                       [HỆ THỐNG TRUNG TÂM]             [TÁC NHÂN VẬN HÀNH]
 
  ┌─────────────────────┐                                                  ┌─────────────────────┐
  │ ACT-04: Nguồn Camera│──(Webcam / MP4 / RTSP)───┐    ┌──(Giám sát / ACK)│ ACT-01: Nhân viên   │
  │ & Bộ quản lý Luồng  │  (Auto Release & Retry)  │    │  (Lọc Ca / CSV)  │         Trực ca     │
  └─────────────────────┘                          ▼    ▼                  └─────────────────────┘
                                         ╔════════════════════════╗
  ┌─────────────────────┐                ║     VISIONGUARD AI     ║        ┌─────────────────────┐
  │ ACT-03: Công nhân   │──(Di chuyển /  ║   (HMI & ACT-05: Lõi   ║◄─(Setup│ ACT-02: Kỹ sư       │
  │         Nhà xưởng   │   Xâm nhập ROI)►║   VisionGuard Engine) ║   ROI) │         Hệ thống    │
  └─────────────────────┘                ╚═══════════╦════════════╝        └─────────────────────┘
                                                     │
                                      (Kích hoạt Còi hú / Lệnh E-Stop)
                                                     ▼
                                         ┌────────────────────────┐
                                         │ ACT-06: Cơ cấu Cảnh báo│
                                         │ & Liên động Thiết bị   │
                                         └────────────────────────┘
```

> **Lưu ý kiến trúc UML:** Khi biểu diễn trên sơ đồ Use-Case tổng quát, **ACT-01, ACT-02, ACT-03, ACT-04 (Phần cứng Camera), ACT-06 (Phần cứng Còi/Rơ-le)** là các **Tác nhân bên ngoài (External Actors)**. Riêng **Giao diện HMI** là lớp biên hiển thị (Boundary View) và **Lõi xử lý tự động (Stream Manager & CV Engine)** là các mô-đun thực thi ngầm bên trong hệ thống để phục vụ các System User Stories.

---

## 2.2. Bảng Ma trận Quyền hạn & Hành vi (Actor - Action Matrix)

| Mã Actor | Tên Tác nhân | Phân loại | Mô tả Hành vi & Trách nhiệm Cốt lõi | Tham chiếu SRS |
| :--- | :--- | :--- | :--- | :--- |
| **ACT-01** | **Nhân viên Trực ca (Safety Operator)** | Primary Human | • Giám sát Live Stream kèm HUD, ROI và lớp phủ điểm chân $P_{\text{foot}}$.<br>• Xác nhận sự cố và tắt còi báo động chốt giữ (`ACK ALARM`).<br>• **Lọc lịch sử theo ca/giờ và xem thumbnail Snapshot trực tiếp trên carousel.**<br>• Xuất báo cáo vi phạm ra tệp `.csv`. | FR-4.4, FR-5.2, FR-5.3, FR-5.4 |
| **ACT-02** | **Kỹ sư Hệ thống (System Admin)** | Primary Human | • **Mở modal cấu hình và chuyển đổi giữa Webcam, Video Offline và Camera IP (`RTSP URL`).**<br>• Vẽ đa giác vùng cấm ROI bằng chuột, xóa đỉnh cuối/khôi phục mặc định và lưu `config.json`.<br>• Tinh chỉnh ngưỡng diện tích $S_{\min}$ và độ trễ báo động $t_{\text{delay}}$.<br>• Chuyển đổi chế độ xem chẩn đoán (RGB / MOG2 Mask / Canny Edge) và lớp phủ HUD. | FR-1.1, FR-1.3, FR-2, FR-3.6, FR-6.2 |
| **ACT-03** | **Công nhân Nhà xưởng (Monitored Worker)** | External Human | • Di chuyển làm việc xung quanh máy móc trong vùng quan sát.<br>• Tạo bóng đổ nghiêng trên mặt sàn (kích hoạt bộ lọc khử bóng đổ).<br>• Bước chân thoáng qua ($< 0.8\text{s}$) hoặc xâm nhập vùng cấm ($\ge 0.8\text{s}$) kích hoạt báo động đỏ. | FR-3.7, FR-4.1, FR-4.2, Mục 6.1 |
| **ACT-04** | **Nguồn cấp Camera & Bộ quản lý Luồng** | External Hardware & System | • Cung cấp khung hình từ Webcam (`0`), File Video (`.mp4`/`.avi`) hoặc `RTSP`.<br>• **Tự động giải phóng bộ nhớ và thiết bị (`cap.release()`, xóa hàng đợi Queue, đóng Thread cũ) khi chuyển đổi mượt mà giữa các nguồn.**<br>• Tự động bắt lỗi mất tín hiệu và thực thi `Auto-Reconnect` mỗi $3.0\text{s}$. | FR-1.1, FR-1.2, NFR-2, NFR-3 |
| **ACT-05** | **Lõi Xử lý Thị giác (VisionGuard CV Engine)** | Internal Autonomous Engine | • Thực thi chuỗi tiền xử lý, trừ nền MOG2, Morphology, tìm Contour và tính điểm chân $P_{\text{foot}}$.<br>• Phát hiện biến thiên ánh sáng đột ngột ($> 60\%$ khung hình) để tự động reset nền (chống sốc sáng).<br>• Tự động chụp ảnh bằng chứng `cv2.imwrite()` và ghi bản ghi nhật ký. | FR-3.1 $\rightarrow$ FR-3.7, FR-5.1, Mục 6.2 |
| **ACT-06** | **Cơ cấu Cảnh báo & Liên động (Alarm/Interlock)** | Secondary Hardware | • Phát âm thanh còi hú cảnh báo công nghiệp bất đồng bộ.<br>• Kích hoạt tín hiệu dừng máy khẩn cấp (`E-STOP`) và giữ khóa liên động cho đến khi ACT-01 xác nhận (`ACK`). | FR-4.2, FR-4.3, FR-4.4 |

---

# PHẦN 3: DANH SÁCH 16 USER STORIES & ĐIỀU KIỆN CHẤP NHẬN (ACCEPTANCE CRITERIA)

## 3.1. Nhóm Nhân viên Trực ca An toàn (ACT-01: Safety Operator)

### US-01: Giám sát trực quan luồng video thời gian thực và lớp phủ an toàn
* **Câu chuyện:** *"Tôi là **Nhân viên Trực ca An toàn (Safety Operator)**, muốn làm việc **theo dõi luồng video trực tiếp từ khu vực máy móc kèm theo các lớp phủ đồ họa an toàn (đa giác vùng cấm ROI, khung bao đối tượng và tọa độ điểm chân)** trên màn hình HMI để **nhận biết tức thời vị trí của công nhân so với khu vực nguy hiểm mà không bị mỏi mắt trong suốt ca trực**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Khung hình Live Stream hiển thị ổn định ở độ phân giải chuẩn hóa $640 \times 360$ (tỷ lệ 16:9) với tốc độ khung hình đạt $\ge 20 - 25\text{ FPS}$ và độ trễ hiển thị $< 200\text{ ms}$.
  2. Giao diện tuân thủ chuẩn ISA-101 với tông nền sáng trung tính; ở trạng thái bình thường (`SAFE`), viền đa giác ROI hiển thị nét đứt màu xanh ngọc hoặc vàng nhạt, không xuất hiện mảng màu đỏ gây nhiễu thị giác.
  3. Mỗi đối tượng chuyển động hợp lệ (có diện tích $S \ge 1500\text{ px}^2$) phải được đóng khung chữ nhật (`Bounding Box`) kèm theo một chấm tròn đánh dấu chính xác tọa độ điểm chân tiếp xúc sàn $P_{\text{foot}} = (x + w/2, y + h)$.
  4. Góc màn hình hiển thị thời gian thực các chỉ số kỹ thuật gồm: Nhịp FPS hiện tại, trạng thái kết nối luồng và số lượng người trong khung hình.
  5. HMI dùng nền `#F1F5F9`, card trắng bo góc, header ViewLab Engineering; HUD bán trong suốt nằm trong ảnh hiển thị thời gian, nguồn và kết quả `cv2.pointPolygonTest`, không vẽ dòng FPS đè lên video.
  6. Cột vận hành có badge SAFE/WARNING/ALARM, số người, số cảnh báo trong ca, trạng thái E-STOP và progress thời gian xâm nhập; kỹ sư có công tắc hiển thị riêng cho Bounding Box và điểm chân.

### US-02: Xác nhận sự cố và ngắt còi báo động (ACK Alarm)
* **Câu chuyện:** *"Tôi là **Nhân viên Trực ca An toàn (Safety Operator)**, muốn nhấn nút `[Xác nhận sự cố / Tắt còi]` (ACK ALARM) để **xác nhận đã biết sự cố và tắt còi trong khi người vi phạm còn đứng trong ROI, nhưng vẫn giữ cảnh báo đỏ cho đến khi người đó bước ra**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Khi xâm nhập liên tục đạt ngưỡng trễ ($t_{\text{violation}} \ge t_{\text{delay}}$), hệ thống chuyển sang `ALARM`, kích hoạt E-STOP và phát còi hú liên tục nếu chưa mute.
  2. Khi tất cả điểm chân không còn trong ROI (hoặc không còn đối tượng trong vùng), hệ thống tự động dừng còi, xóa latch/E-STOP, chuyển sang `SAFE` và đặt `is_muted = False` mà không cần ACK.
  3. Khi người vẫn trong ROI, ACK đặt `is_muted = True` và dừng còi ngay; trạng thái `ALARM`, latch và E-STOP vẫn được giữ đến khi người bước ra.
  4. Khi thoát ROI, thời lượng vi phạm được chốt và cập nhật vào bản ghi kiểm toán.

### US-03: Lọc lịch sử vi phạm theo mốc thời gian và tra cứu ảnh bằng chứng (Snapshot)
* **Câu chuyện:** *"Tôi là **Nhân viên Trực ca An toàn (Safety Operator)**, muốn làm việc **lọc danh sách nhật ký vi phạm theo ca làm việc hoặc khoảng ngày/giờ và nhấp xem lại thẻ ảnh Snapshot tương ứng trên thanh cuộn Bottom Dock** để **nhanh chóng đối chiếu bằng chứng hình ảnh thực tế của từng vụ xâm nhập khi bàn giao ca**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Thanh công cụ lọc cho phép truy vấn theo 3 ca chuẩn (*Ca 1: 06:00–14:00*, *Ca 2: 14:00–22:00*, *Ca 3: 22:00–06:00*) hoặc nhập khoảng thời gian tùy chỉnh (`Từ HH:MM đến HH:MM`) với thời gian truy xuất $< 100\text{ ms}$.
  2. Khi áp dụng bộ lọc, cả Bảng Nhật ký Thời gian thực (40% bên phải) và Khay ảnh Snapshot Carousel cuộn ngang (60% bên trái) phải đồng bộ hiển thị đúng các bản ghi thuộc khung giờ đã chọn.
  3. Mỗi thẻ Snapshot hiển thị trực tiếp thumbnail ảnh `.jpg` cùng badge `[Ảnh HH:MM:SS]`, `Vi phạm: X.Xs` và `Xem →`; carousel cập nhật khi ảnh nền đã ghi xong.
  4. Click thẻ mở cửa sổ phóng to (Preview Modal) ảnh gốc từ `/evidence/`, có hộp bao, điểm chân và dấu thời gian.

### US-04: Trích xuất báo cáo vi phạm ra tệp CSV
* **Câu chuyện:** *"Tôi là **Nhân viên Trực ca An toàn (Safety Operator)**, muốn làm thao tác **xuất toàn bộ dữ liệu nhật ký vi phạm (hoặc danh sách đã lọc theo ca) ra tệp định dạng `.csv`** để **lưu trữ hồ sơ an toàn lao động và gửi báo cáo tổng hợp cho ban quản lý nhà xưởng cuối mỗi ca làm việc**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Nút `[Xuất File Báo Cáo CSV]` đặt tại khu vực Bottom Dock, khi nhấn sẽ mở hộp thoại lưu tệp với tên mặc định `Violation_Report_YYYYMMDD_HHMMSS.csv`.
  2. Tệp `.csv` xuất ra sử dụng bảng mã `UTF-8 with BOM` (đảm bảo mở trực tiếp bằng Microsoft Excel không bị lỗi font tiếng Việt).
  3. Cấu trúc tệp bao gồm đầy đủ 5 cột dữ liệu chuẩn: `Mã sự kiện (VIO-xxx)`, `Thời gian (YYYY-MM-DD HH:MM:SS)`, `Thời lượng vi phạm (giây)`, `Tọa độ chân (x, y)`, và `Đường dẫn ảnh bằng chứng (/evidence/...)`.
  4. Quá trình ghi tệp dưới 1.000 dòng bản ghi phải hoàn tất trong vòng $< 500\text{ ms}$ và hiển thị thông báo xác nhận xuất thành công trên giao diện.

---

## 3.2. Nhóm Kỹ sư Hệ thống / Quản trị viên (ACT-02: System Admin)

### US-05: Chuyển đổi linh hoạt giữa 3 chế độ nguồn đầu vào (Webcam, Video Offline, RTSP)
* **Câu chuyện:** *"Tôi là **Kỹ sư Hệ thống (System Admin)**, muốn làm thao tác **cấu hình và chuyển đổi trực tiếp trên giao diện giữa Webcam (dò các ID từ 0 đến 4, có chế độ tự động chọn), Tệp video thực nghiệm (`.mp4`, `.avi`) và Luồng Camera IP (`RTSP URL`)** để **linh hoạt kiểm thử thuật toán ngoại tuyến, demo báo cáo ổn định hoặc vận hành thực tế tại nhà xưởng**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Nút `[Kết nối Camera IP]` mở modal cho phép chọn Webcam Auto/ID `0–4`, video `.mp4/.avi`, hoặc RTSP; biểu mẫu RTSP có profile Dahua/Imou, Hikvision/Ezviz, ONVIF, IP, port `554`, tài khoản và mật khẩu, kèm chuỗi `cv2.VideoCapture`.
  2. Nút `[Kiểm tra kết nối (Ping)]` kiểm tra mở nguồn và đọc frame ngoài GUI thread; quét Webcam ưu tiên DirectShow trên Windows, giải phóng capture sau thử và thông báo rõ nếu thất bại.
  3. `[Lưu & Áp dụng]` gửi yêu cầu chuyển nguồn qua worker hiện có, không khóa GUI; cấu hình hợp lệ được lưu vào `config.json`.
  4. Toàn bộ thông số kết nối hợp lệ sau khi áp dụng phải được tự động ghi đè và lưu trữ bền vững vào tệp `config.json`.

### US-06: Thiết lập và hiệu chuẩn đa giác vùng nguy hiểm động (Dynamic ROI Calibration)
* **Câu chuyện:** *"Tôi là **Kỹ sư Hệ thống (System Admin)**, muốn dùng ba thao tác `[Vẽ lại Vùng cấm (Click chuột)]` / `[Xác nhận Vùng cấm mới]`, `[Xóa điểm]` và `[Mặc định]` để vẽ lại ROI trực tiếp trên khung hình mà vẫn đối chiếu được vùng đang hoạt động."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Nhấn lần đầu nút vẽ khởi tạo `draft_roi = []`, giữ ROI hiện hành hiển thị và cho phép chấm liên tiếp; từ 3 đỉnh, draft được nối thành polygon khép kín và các đỉnh tiếp theo cập nhật đa giác.
  2. Nhấn xác nhận chỉ lưu khi draft có ít nhất 3 đỉnh; sau khi lưu, draft thay thế ROI đang hoạt động và cập nhật ngay cho `cv2.pointPolygonTest`.
  3. `[Xóa điểm]` xóa đỉnh cuối của draft khi đang vẽ, hoặc của ROI đang hoạt động khi không vẽ; không cho phép xóa khiến đa giác còn ít hơn 3 đỉnh.
  4. `[Mặc định]` hủy draft, khôi phục và lưu `DEFAULT_ROI_POINTS` dùng chung với cấu hình mặc định.
  5. Tọa độ ROI và vùng che khuất thân máy đều dùng chung hệ tọa độ frame `640×360`; vùng che khuất có thể được hiệu chuẩn thành một hoặc nhiều đa giác riêng.
  6. Lớp phủ ROI sàn được trộn alpha, loại trừ foreground chuyển động và bị cắt tại các đa giác che khuất đã hiệu chuẩn để không phủ lên người hoặc thân máy.

### US-07: Tinh chỉnh tham số lọc nhiễu diện tích và độ trễ báo động
* **Câu chuyện:** *"Tôi là **Kỹ sư Hệ thống (System Admin)**, muốn làm việc **điều chỉnh thanh trượt ngưỡng diện tích tối thiểu ($S_{\min}$) và thời gian trễ kích hoạt báo động ($t_{\text{delay}}$)** trên bảng điều khiển kỹ thuật để **tối ưu hóa độ nhạy của hệ thống theo khoảng cách lắp đặt camera và đặc thù vận hành của từng phân xưởng**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Thanh trượt ngưỡng diện tích tối thiểu $S_{\min}$ cho phép điều chỉnh trong dải từ $500\text{ px}^2$ đến $5000\text{ px}^2$ (giá trị mặc định chuẩn là $1500\text{ px}^2$).
  2. Thanh trượt độ trễ báo động $t_{\text{delay}}$ cho phép điều chỉnh trong dải từ $0.0\text{s}$ đến $3.0\text{s}$ với bước nhảy $0.1\text{s}$ (giá trị mặc định chuẩn là $0.8\text{s}$).
  3. Mọi thay đổi trên thanh trượt phải có hiệu lực tức thời trên pipeline xử lý ảnh ở frame tiếp theo và tự động đồng bộ vào tệp `config.json`.
  4. Card cấu hình có nút `[Reset chuẩn]` khôi phục `minArea = 1500 px²` và `t_delay = 0.8 s`.

### US-08: Chuyển đổi bộ lọc hiển thị thuật toán phục vụ chẩn đoán (HUD Debug)
* **Câu chuyện:** *"Tôi là **Kỹ sư Hệ thống (System Admin)**, muốn làm thao tác **chuyển đổi nhanh giữa 3 lớp hiển thị gồm Ảnh thực tế (RGB), Mặt nạ nhị phân trừ nền (MOG2 Mask) và Đường biên (Canny Edge)** để **kiểm tra trực quan chất lượng tách nền, mức độ nhiễu hạt môi trường và chứng minh thuật toán thị giác máy tính đang hoạt động thực tế**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Cung cấp 3 nút gạt hoặc tab chuyển đổi chế độ xem: `RGB Thực tế`, `MOG2 Mask`, và `Canny Edge` ngay trên giao diện điều khiển.
  2. Chế độ `MOG2 Mask` hiển thị khung hình nhị phân đen-trắng-xám trích xuất từ bộ trừ nền (vật thể chuyển động màu trắng `255`, bóng đổ màu xám `127`, nền tĩnh màu đen `0`).
  3. Chế độ `Canny Edge` hiển thị bản đồ đường biên sắc nét của các đối tượng trong khung hình.
  4. Việc chuyển đổi qua lại giữa 3 chế độ hiển thị không làm gián đoạn tiến trình kiểm tra an toàn chạy ngầm và giữ nguyên tốc độ $\ge 20 - 25\text{ FPS}$.
  5. Hai checkbox `Khung Bounding Box` và `Tọa độ Điểm chân` bật/tắt độc lập lớp phủ tương ứng và lưu trạng thái vào `config.json`.

---

## 3.3. Nhóm Công nhân Nhà xưởng (ACT-03: Monitored Worker)

### US-09: Di chuyển làm việc an toàn bên ngoài vùng cấm mà không gây báo động giả do bóng đổ
* **Câu chuyện:** *"Tôi là **Công nhân Nhà xưởng (Monitored Worker)**, muốn làm việc **di chuyển và thao tác bình thường xung quanh bên ngoài đường biên vùng cấm, kể cả khi ánh đèn xưởng chiếu xiên làm bóng cơ thể tôi đổ dài vào trong vùng nguy hiểm** để **không bị hệ thống hú còi báo động giả gây gián đoạn quy trình sản xuất**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Khi công nhân di chuyển với vận tốc đi bộ bình thường ($v < 3\text{ m/s}$) bên ngoài đa giác ROI, hệ thống đóng khung bao màu xanh lá và hiển thị trạng thái `SAFE`.
  2. Khi bóng của thân trên công nhân đổ vươn vào bên trong vùng cấm ROI nhưng điểm tiếp xúc chân thực tế $P_{\text{foot}} = (x + w/2, y + h)$ vẫn nằm ngoài đa giác ($\text{cv2.pointPolygonTest} < 0$), hệ thống tuyệt đối không kích hoạt bộ đếm vi phạm và không phát còi báo động.

### US-10: Nhận cảnh báo tức thời khi bước chân xâm nhập vùng nguy hiểm vượt ngưỡng trễ
* **Câu chuyện:** *"Tôi là **Công nhân Nhà xưởng (Monitored Worker)**, muốn **hệ thống bỏ qua các pha quét chân thoáng qua vô hại ($< 0.8\text{s}$) nhưng phải phát còi cảnh báo tức thời ngay khi chân tôi đứng trong vùng nguy hiểm quá $0.8\text{ giây}$** để **tôi kịp thời nhận biết nguy cơ tai nạn và lùi ra khỏi phạm vi hoạt động của máy dập**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Nếu điểm chân $P_{\text{foot}}$ chạm vào biên hoặc nằm trong vùng ROI ($\text{pointPolygonTest} \ge 0$) nhưng công nhân rút chân ra ngay khi thời gian lưu lại $t_{\text{violation}} < 0.8\text{s}$, hệ thống chỉ hiển thị cảnh báo sớm màu vàng rồi tự động xóa bộ đếm về $0\text{s}$ mà không hú còi.
  2. Ngay tại thời điểm thời gian điểm chân nằm trong vùng cấm đạt ngưỡng $t_{\text{violation}} \ge 0.8\text{s}$, hệ thống phải chuyển sang trạng thái `BÁO ĐỘNG ĐỎ` trong vòng $< 200\text{ ms}$, phát còi cảnh báo và kích hoạt lệnh dừng máy.

---

## 3.4. Nhóm Nguồn cấp Camera & Bộ Quản lý Luồng (ACT-04: Stream Ingestion Engine)

### US-11: Thu nhận luồng đa luồng xóa trễ Buffer và tự động giải phóng tài nguyên khi chuyển nguồn
* **Câu chuyện:** *"Tôi là **Bộ quản lý Luồng Camera (Stream Ingestion Engine)**, muốn làm nhiệm vụ **đọc khung hình qua một luồng độc lập (`threading.Thread`) để xả liên tục bộ đệm FFmpeg và tự động giải phóng bộ nhớ (`release resource`) mỗi khi đổi nguồn** để **luồng chính luôn nhận được khung hình thời gian thực mới nhất mà không bị trễ tích lũy hay rò rỉ RAM**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Tiến trình đọc video chạy trên `threading.Thread` riêng biệt, liên tục cập nhật frame mới nhất vào biến chia sẻ hoặc hàng đợi giới hạn (`maxsize=1`), loại bỏ 100% hiện tượng trễ hình tích lũy (Buffer Lag) của giao thức RTSP.
  2. Khi nhận lệnh chuyển đổi giữa Webcam (`0`), File Video (`.mp4`, `.avi`) và `RTSP URL`, hệ thống phải dừng thread cũ, gọi lệnh `cap.release()` để giải phóng cổng thiết bị và dọn sạch hàng đợi bộ nhớ trước khi mở luồng mới.
  3. Đối với nguồn Tệp video thực nghiệm (`.mp4`, `.avi`), khi đọc tới frame cuối cùng, hệ thống tự động đặt lại con trỏ về frame đầu tiên (`cv2.CAP_PROP_POS_FRAMES = 0`) để duy trì luồng demo liên tục.

### US-12: Tự động bắt lỗi và phục hồi kết nối khi rớt mạng (Auto-Reconnect)
* **Câu chuyện:** *"Tôi là **Bộ quản lý Luồng Camera (Stream Ingestion Engine)**, muốn làm việc **tự động bắt ngoại lệ khi luồng RTSP bị đứt kết nối và thực thi cơ chế thử kết nối lại định kỳ** để **phần mềm không bao giờ bị sập (`Crash-Free`) khi camera nhà xưởng bị mất điện hoặc rớt mạng đột ngột**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Toàn bộ vòng lặp thu nhận khung hình phải được bọc trong khối xử lý ngoại lệ `try...except` an toàn.
  2. Khi không nhận được frame từ Camera IP (`ret == False` hoặc mất kết nối socket), giao diện hiển thị trạng thái cảnh báo vàng `"MẤT TÍN HIỆU - ĐANG KẾT NỐI LẠI..."` thay vì bị treo cứng hoặc văng ứng dụng.
  3. Hệ thống tự động giải phóng kết nối lỗi và kích hoạt chu kỳ thử kết nối lại (`Auto-Reconnect`) định kỳ sau mỗi $3.0\text{ giây}$ cho đến khi luồng RTSP hoạt động trở lại bình thường.

---

## 3.5. Nhóm Lõi Xử lý Thị giác Máy tính (ACT-05: VisionGuard CV Engine)

### US-13: Thực thi Pipeline xử lý ảnh phát hiện chuyển động và trích xuất điểm tiếp xúc chân
* **Câu chuyện:** *"Tôi là **Lõi Xử lý Thị giác Máy tính (VisionGuard CV Engine)**, muốn **kết hợp YOLOv8-Nano với OpenCV để phát hiện đúng người, đồng thời duy trì MOG2 và Canny cho các lớp Debug và fallback** để **xác định chính xác vị trí công nhân trên mặt sàn nhà xưởng trong thời gian thực mà không đóng khung máy móc, bàn ghế**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Mỗi khung hình đầu vào đều được chuẩn hóa kích thước về $640 \times 360$, chuyển sang ảnh xám `Grayscale` và làm mịn bằng bộ lọc `cv2.GaussianBlur` với kích thước nhân $5 \times 5$.
  2. Khởi tạo `YOLO("yolov8n.pt")`; mỗi frame gọi `predict(frame, classes=[0], conf=0.45, imgsz=640, verbose=False)` và chỉ nhận class `0` (person), với mọi bbox/foot-point dùng chung tọa độ frame `640×360`.
  3. Tính diện tích người bằng $S=w \times h$, loại bbox có $S < S_{\min}$ (mặc định $1500\text{ px}^2$), và tính $P_{\text{foot}}=(x+w//2,\ y+h)$; lớp phủ hiển thị nhãn `WORKER | S:{S}px²`.
  4. MOG2 (`history=500`, `varThreshold=16`, `detectShadows=True`) vẫn chạy nền để cung cấp tab `MOG2 Mask`; Canny vẫn tạo tab `Canny Edge`. Khi YOLO không thể import/tải model, ghi log và fallback về morphology/contour MOG2.
  5. Khi foreground lớn hoặc điểm chân người nằm trong ROI, giảm tốc độ học MOG2 xuống `0.0005` để hạn chế hấp thụ người đứng yên; giữ nguyên bảo vệ sốc sáng toàn cục và bỏ qua đánh giá an toàn ở frame sốc sáng.
  6. Lọc class người bằng YOLO giúp loại các vật thể không phải người khỏi detection khi mô hình hoạt động; fallback MOG2 ở chế độ suy giảm không bảo đảm phân loại ngữ nghĩa vật thể.

### US-14: Tự động bảo vệ hệ thống trước biến thiên ánh sáng đột ngột (Global Illumination Guard)
* **Câu chuyện:** *"Tôi là **Lõi Xử lý Thị giác Máy tính (VisionGuard CV Engine)**, muốn làm việc **tự động phát hiện hiện tượng sốc sáng toàn cục khi đèn nhà xưởng bật/tắt đột ngột và đặt lại mô hình nền** để **ngăn chặn mặt nạ MOG2 bị trắng toàn phần gây ra báo động giả hàng loạt**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Ở mỗi frame, hệ thống tính tỷ lệ diện tích các điểm ảnh chuyển động trên mặt nạ MOG2 so với tổng diện tích khung hình ($640 \times 360 = 230.400\text{ px}$).
  2. Nếu tỷ lệ diện tích biến đổi vượt quá ngưỡng $60\%$ tổng diện tích khung hình trong một frame đơn lẻ, hệ thống lập tức định danh đây là sự kiện biến thiên ánh sáng toàn cục (`Global Illumination Shift`).
  3. Khi phát hiện sốc sáng, lõi xử lý tự động bỏ qua kiểm tra vi phạm ở frame đó, tái khởi tạo hoặc cập nhật nhanh lịch sử mô hình nền (`reset background`) và giữ nguyên trạng thái an toàn, tuyệt đối không phát còi báo động sai.

### US-15: Tự động chụp lưu bằng chứng Snapshot và ghi nhận nhật ký sự kiện
* **Câu chuyện:** *"Tôi là **Lõi Xử lý Thị giác Máy tính (VisionGuard CV Engine)**, muốn làm việc **tự động trích xuất và lưu khung hình bằng chứng xuống ổ cứng kèm theo bản ghi nhật ký ngay tại thời điểm phát hiện vi phạm vượt ngưỡng trễ** để **đảm bảo mọi sự cố an toàn đều có hồ sơ hình ảnh và dữ liệu thời gian thực đối chứng**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Ngay khi bộ đếm vi phạm chạm ngưỡng $t_{\text{violation}} \ge t_{\text{delay}}$ ($0.8\text{s}$), hệ thống tự động gọi `cv2.imwrite()` để lưu khung hình hiện tại (đã vẽ đa giác ROI màu đỏ, hộp bao người vi phạm, tọa độ chân và nhãn timestamp) vào thư mục `/evidence/`.
  2. Tệp ảnh lưu dưới định dạng `.jpg` với quy chuẩn đặt tên `VIO_YYYYMMDD_HHMMSS.jpg` và đẩy thẻ ảnh mới nhất lên đầu khay Snapshot Carousel trên giao diện trong vòng $< 100\text{ ms}$.
  3. Khởi tạo bản ghi sự kiện mới vào bộ nhớ nhật ký với đầy đủ thông tin: Mã sự kiện tự tăng (`VIO-001`, `VIO-002`...), Thời gian bắt đầu, Tổng thời lượng xâm nhập (cập nhật khi đối tượng bước ra) và Tọa độ tiếp xúc.

---

## 3.6. Nhóm Cơ cấu Cảnh báo & Liên động Thiết bị (ACT-06: Alarm & Interlock System)

### US-16: Kích hoạt còi hú âm thanh và tín hiệu dừng máy khẩn cấp (Emergency Stop Interlock)
* **Câu chuyện:** *"Tôi là **Cơ cấu Cảnh báo & Liên động Thiết bị (Alarm & Interlock System)**, muốn **nhận lệnh kích hoạt tức thời từ lõi xử lý khi xác nhận vi phạm để phát âm thanh còi hú công nghiệp và gửi tín hiệu ngắt liên động dừng máy khẩn cấp** để **cảnh báo trực tiếp tại hiện trường và phanh hãm cơ cấu máy dập trước khi xảy ra tai nạn thương tích**."*
* **Điều kiện chấp nhận (Acceptance Criteria - AC):**
  1. Ngay khi trạng thái hệ thống chuyển sang `BÁO ĐỘNG ĐỎ` ($t_{\text{violation}} \ge 0.8\text{s}$), luồng âm thanh cảnh báo (Siren/Beep) được kích hoạt bất đồng bộ (không làm nghẽn luồng xử lý video chính) với độ trễ âm thanh $< 50\text{ ms}$.
  2. Hệ thống đồng thời xuất tín hiệu điều khiển dừng khẩn cấp (`E-STOP TRIGGERED` hiển thị trên giao diện HMI và xuất cờ tín hiệu ra cổng giao tiếp mô phỏng) để ngắt chu trình máy móc.
  3. Còi hú và khóa liên động được duy trì trong khi người còn ở ROI (trừ khi đã mute bằng ACK); khi người bước ra, hệ thống tự động tắt còi và nhả trạng thái báo động/E-STOP. ACK chỉ mute âm thanh, không đưa trạng thái về SAFE khi nguy cơ còn tồn tại.

---

# PHẦN 4: KIẾN TRÚC MODULE HÓA & ĐA LUỒNG DỮ LIỆU (MULTI-THREADING PIPELINE)

## 4.1. Phân rã Hệ thống thành 6 Module Chức năng Độc lập

Để đảm bảo nguyên lý **Single Responsibility Principle (SRP)** và giúp các thành viên trong nhóm làm việc song song trên các nhánh Git (`feature/*`) mà không bị xung đột mã nguồn (merge conflict), hệ thống được phân rã thành **6 Module cốt lõi**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       SƠ ĐỒ KIẾN TRÚC 6 MODULE CHỨC NĂNG ĐỘC LẬP                       │
└────────────────────────────────────────────────────────────────────────────────────────┘

                    ┌────────────────────────────────────────────────┐
                    │           MOD-06: HMI DASHBOARD & UI           │
                    │  (ViewLab Clean GUI, ROI Canvas, Bottom Dock)  │
                    │          [US-01, US-02, US-03, US-06]          │
                    └───────┬───────────────▲────────────────┬───────┘
                            │ (Đổi nguồn)   │ (Render Frame) │ (Lọc/Xuất CSV)
                            ▼               │                ▼
 ┌──────────────────────────────┐   ┌───────┴────────────────────────┐   ┌───────────────────────────┐
 │ MOD-01: STREAM INGESTION     │   │ MOD-02: VISION ENGINE          │   │ MOD-05: EVIDENCE & AUDIT  │
 │ • Đọc RTSP / Webcam / MP4    │──►│ • Resize, Blur, MOG2, Morph    │   │ • Chụp ảnh cv2.imwrite()  │
 │ • Xả Buffer & Auto-Reconnect │   │ • Contours, Foot-Point, Guard  │   │ • Quản lý Log & Lọc Ca    │
 │ • Giải phóng RAM (Release)   │   │   [US-08, US-09, US-13, US-14] │   │ • Xuất báo cáo UTF-8 CSV  │
 │     [US-05, US-11, US-12]    │   └───────────────┬────────────────┘   │   [US-03, US-04, US-15]   │
 └──────────────▲───────────────┘                   │ (P_foot, Box)      └─────────────▲─────────────┘
                │                                   ▼                                  │
 ┌──────────────┴───────────────┐   ┌────────────────────────────────┐                 │
 │ MOD-04: CONFIG & ROI MANAGER │   │ MOD-03: SAFETY & ALARM ENGINE  │                 │
 │ • Đọc/Ghi tệp config.json    │──►│ • Kiểm tra pointPolygonTest    ├─────────────────┘
 │ • Lưu đa giác ROI, S_min     │   │ • Bộ đếm trễ t_delay >= 0.8s   │ (Kích hoạt lưu bằng chứng)
 │         [US-06, US-07]       │   │ • Chốt giữ Latch, Còi, E-STOP  │
 └──────────────────────────────┘   │     [US-02, US-10, US-16]      │
                                    └────────────────────────────────┘
```

### Chi tiết Nhiệm vụ & Ánh xạ User Story của từng Module

| Mã Module | Tên Module & Tệp Python tương ứng | Nhiệm vụ Kỹ thuật Chi tiết | Các User Story thuộc Module |
| :--- | :--- | :--- | :--- |
| **MOD-01** | **Module Thu nhận & Quản lý Luồng Đa nguồn**<br>(`src/core/stream_manager.py`) | • Khởi tạo tiến trình `threading.Thread` đọc bất đồng bộ từ 3 chế độ nguồn: Webcam (dò ID `0–4`), File Video (`.mp4`, `.avi`) và Camera IP (`RTSP`).<br>• Quét camera bằng DirectShow trước trên Windows, xác minh frame và giải phóng capture; hỗ trợ lựa chọn webcam Auto.<br>• Liên tục xả bộ đệm FFmpeg, chỉ giữ lại 1 khung hình mới nhất (`Latest-Frame Queue`).<br>• Tự động gọi `cap.release()` giải phóng bộ nhớ khi chuyển nguồn và thực thi `Auto-Reconnect` sau mỗi $3.0\text{s}$ khi đứt mạng. | **US-05**<br>**US-11**<br>**US-12** |
| **MOD-02** | **Module Lõi Xử lý Thị giác Máy tính**<br>(`src/core/cv_engine.py`) | • Tiền xử lý: Resize khung hình về $640 \times 360$, chuyển đổi `Grayscale`, làm mịn `GaussianBlur(5, 5)`.<br>• Phát hiện người bằng `YOLO("yolov8n.pt")`, chỉ class `0`, `conf=0.45`, lọc bbox theo $S=w \times h$ và ngưỡng $S_{\min}$; fallback sang contour MOG2 nếu không tải được Ultralytics/model.<br>• Duy trì MOG2 (loại pixel bóng `127`) và Canny cho các lớp Debug; giảm learning rate khi có foreground/người trong ROI.<br>• Bảo vệ chống sốc sáng toàn cục khi diện tích biến đổi $> 60\%$.<br>• Tính $P_{\text{foot}}=(x+w//2,\ y+h)$ trong hệ tọa độ chuẩn; hiển thị nhãn `WORKER | S:{S}px²`.<br>• Trộn alpha ROI sàn, cắt lớp phủ tại vùng che khuất và xuất các lớp ảnh Debug (`RGB`, `MOG2 Mask`, `Canny Edge`). | **US-08**<br>**US-09**<br>**US-13**<br>**US-14** |
| **MOD-03** | **Module Đánh giá Quy tắc An toàn & Liên động**<br>(`src/core/safety_evaluator.py`) | • Thực thi hàm kiểm tra hình học `cv2.pointPolygonTest(roi, foot_point, False)`.<br>• Quản lý máy trạng thái an toàn (`SAFE` $\rightarrow$ `WARNING` $\rightarrow$ `ALARM`) và bộ đếm thời gian xâm nhập ($t_{\text{violation}} \ge t_{\text{delay}}$).<br>• Quản lý cờ chốt giữ báo động (`Alarm Latch`), kích hoạt luồng phát âm thanh còi hú bất đồng bộ, phát tín hiệu dừng máy `E-STOP` và xử lý lệnh xác nhận `ACK ALARM`. | **US-02**<br>**US-10**<br>**US-16** |
| **MOD-04** | **Module Quản lý Cấu hình & Hiệu chuẩn ROI**<br>(`src/utils/config_manager.py`) | • Đọc và ghi đồng bộ có khóa an toàn (`threading.Lock`) với tệp `config.json`.<br>• Quản lý danh sách tọa độ đỉnh đa giác vùng cấm ROI ($N \ge 3$), ngăn xếp hoàn tác (`Undo Stack`) và khôi phục mặc định (`Reset`).<br>• Lưu trữ thông số kết nối RTSP, ngưỡng diện tích $S_{\min}$ và độ trễ $t_{\text{delay}}$. | **US-05**<br>**US-06**<br>**US-07** |
| **MOD-05** | **Module Lưu trữ Bằng chứng & Kiểm toán Báo cáo**<br>(`src/utils/evidence_logger.py`) | • Đẩy tác vụ ghi ảnh `cv2.imwrite()` sang luồng I/O ngầm để lưu khung hình vi phạm vào `/evidence/VIO_YYYYMMDD_HHMMSS.jpg` mà không làm khựng luồng video.<br>• Quản lý danh sách bản ghi nhật ký trong bộ nhớ, hỗ trợ truy vấn và lọc nhanh theo Ca làm việc (Ca 1, Ca 2, Ca 3) hoặc khung giờ.<br>• Trích xuất dữ liệu nhật ký ra tệp `.csv` chuẩn `UTF-8 with BOM`. | **US-03**<br>**US-04**<br>**US-15** |
| **MOD-06** | **Module Giao diện Điều khiển High-Performance HMI**<br>(`src/ui/hmi_main_window.py`) | • Hiển thị giao diện chuẩn ISA-101 (tông nền sáng kết hợp màu nhấn Đỏ đô `#9B1B30`, bo góc chuẩn macOS).<br>• Bắt sự kiện click chuột trên Viewport để vẽ đa giác ROI trực tiếp.<br>• Hiển thị khay ảnh bằng chứng **Snapshot Carousel** cuộn ngang, bảng Nhật ký thời gian thực, cửa sổ phóng to ảnh (Modal) và các thanh trượt điều khiển. | **US-01**<br>**US-02**<br>**US-03**<br>**US-06**<br>**US-08** |

---

## 4.2. Kiến trúc Phân luồng Đa nhiệm (Multi-Threading Architecture)

Một trong những nguyên nhân lớn nhất khiến các đồ án OpenCV bị giật lag, trễ hình $5 - 10\text{ giây}$ hoặc báo lỗi `"Not Responding"` (đơ giao diện) là đặt cả 4 tác vụ: **Đọc RTSP + Xử lý ảnh + Phát âm thanh + Ghi file ảnh** vào chung một vòng lặp `while True` trên luồng giao diện chính.

Hệ thống **VisionGuard AI** giải quyết triệt để vấn đề này bằng kiến trúc **4 Luồng Độc lập (4-Thread Decoupled Architecture)** giao tiếp qua Hàng đợi giới hạn (`Bounded Queue maxsize=1`) và Tín hiệu sự kiện (`Signals / Thread-safe Locks`):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│            KIẾN TRÚC ĐA LUỒNG CHỐNG NGHẼN CAMERA & CHỐNG ĐƠ GIAO DIỆN (4 THREADS)      │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [THREAD 1: CAMERA I/O DAEMON] (Chạy ngầm liên tục tốc độ mạng)
  │ • Liên tục gọi cap.grab() & cap.retrieve() từ RTSP / Webcam / MP4
  │ • Nếu RawFrameQueue đã đầy (maxsize=1) -> Tự động vứt bỏ frame cũ, ghi đè frame mới nhất!
  │ • Khi mất mạng -> Tự động ngủ 3.0s rồi gọi Auto-Reconnect (Không làm treo UI)
  ▼
 〘 RawFrameQueue (maxsize = 1) 〙 ──(Luôn chỉ chứa duy nhất 1 frame tươi nhất, độ trễ = 0)
  │
  ▼
 [THREAD 2: CV & SAFETY WORKER] (Luồng xử lý tính toán ~25-30 FPS)
  │ • Lấy frame từ RawFrameQueue -> Resize 640x360 -> Blur -> MOG2 -> Morph -> Contours
  │ • Kiểm tra Global Illumination Guard (>60%) -> Tính điểm chân P_foot
  │ • Gọi pointPolygonTest(ROI, P_foot) -> Cập nhật bộ đếm t_violation
  │ • Vẽ lớp phủ HUD (Đa giác ROI, Bounding Box, Foot Point, FPS) lên frame
  ├─────────────────────────┬──────────────────────────────┐
  │ (Đẩy Frame đã vẽ HUD)   │ (Khi t_violation >= 0.8s)    │ (Khi ALARM kích hoạt)
  ▼                         ▼                              ▼
 〘 UIUpdateSignal 〙      〘 EvidenceIOQueue 〙          [THREAD 4: ASYNC AUDIO & ALARM]
  │                         │                            • Phát âm thanh còi hú (Beep/Siren)
  ▼                         ▼                              bất đồng bộ trên luồng riêng
 [THREAD 3: MAIN GUI]     [THREAD 5 / I/O WORKER]        • Dừng ngay lập tức khi nhận cờ ACK
  • Chỉ nhận QImage/Canvas  • Ghi ảnh cv2.imwrite()
    để vẽ lên màn hình HMI    xuống ổ cứng /evidence/
  • Bắt sự kiện click chuột • Ghi dòng mới vào Log
    vẽ ROI, kéo Slider,       & bắn tín hiệu cập nhật
    lọc Ca, bấm nút ACK       khay Snapshot Carousel
  ==> KHÔNG BAO GIỜ BỊ ĐƠ!  ==> KHÔNG LÀM KHỰNG FPS!
```

### Cơ chế Kỹ thuật Đảm bảo Không Nghẽn Camera & Không Đơ Giao diện

1. **Cơ chế Hàng đợi Đơn (`Drop-Frame / Latest-Frame Queue` với `maxsize=1`):**
   * Giữa **Thread 1 (Đọc Camera)** và **Thread 2 (Xử lý OpenCV)** là một hàng đợi `queue.Queue(maxsize=1)`.
   * Khi camera IP bắn frame về ở tốc độ $30\text{ FPS}$ nhưng luồng xử lý đang bận ở một frame nặng, **Thread 1** sẽ kiểm tra: nếu hàng đợi đã đầy (`q.full()`), nó lập tức lấy frame cũ ra vứt bỏ (`q.get_nowait()`) và nhét frame mới nhất vào (`q.put_nowait(frame)`).
   * **Hiệu quả:** Bộ đệm FFmpeg luôn được xả rỗng, luồng xử lý thị giác luôn lấy được hình ảnh của đúng tích tắc hiện tại, loại bỏ 100% hiện tượng trễ hình tích lũy (Buffer Lag).

2. **Tách biệt Luồng Ghi Ổ cứng & Phát Âm thanh (`Non-Blocking Disk I/O & Audio`):**
   * Lệnh ghi ảnh xuống đĩa cứng `cv2.imwrite()` và lệnh phát âm thanh còi báo động thường tốn từ $30\text{ ms} - 150\text{ ms}$. Nếu gọi trực tiếp trong luồng xử lý ảnh, video sẽ bị khựng lại mỗi khi có người vi phạm.
   * Bằng cách đẩy bản sao khung hình vi phạm (`frame.copy()`) sang `EvidenceIOQueue` để một luồng I/O phụ ghi xuống thư mục `/evidence/` và phát còi trên một luồng âm thanh riêng biệt, tốc độ khung hình trên màn hình chính luôn giữ vững ở mức $\ge 25\text{ FPS}$.

3. **Đồng bộ Tham số An toàn Luồng (`Thread-Safe Configuration Lock`):**
   * Khi Kỹ sư Hệ thống (**ACT-02**) kéo thanh trượt chỉnh $S_{\min}$, $t_{\text{delay}}$ hoặc chấm chuột vẽ lại đa giác ROI trên **Thread 3 (Main GUI)**, dữ liệu mới được cập nhật vào đối tượng `ConfigManager` thông qua khóa `threading.Lock()`.
   * Ở chu kỳ frame kế tiếp (chỉ sau $\sim 33\text{ ms}$), **Thread 2 (CV Worker)** đọc tọa độ ROI mới và áp dụng ngay lập tức mà không xảy ra xung đột vùng nhớ (`Race Condition`).

---

## 4.3. Cấu trúc Thư mục Dự án Chuẩn & Phân nhánh Git Flow

### 4.3.1. Cấu trúc Thư mục Mã nguồn (Project Directory Structure)
```
VisionGuard-AI/
├── config/
│   └── config.json                 # Lưu cấu hình nguồn RTSP, tọa độ đỉnh ROI, S_min, t_delay
├── datasets/
│   └── factory_demo.mp4            # Video thực nghiệm nhà xưởng phục vụ kiểm thử & demo offline
├── evidence/                       # Thư mục tự động lưu ảnh Snapshot vi phạm (VIO_YYYYMMDD_HHMMSS.jpg)
├── reports/                        # Thư mục chứa các tệp báo cáo xuất ra (.CSV)
├── src/
│   ├── core/
│   │   ├── stream_manager.py       # [MOD-01] Đa luồng đọc RTSP/Webcam/MP4, xả buffer, auto-reconnect
│   │   ├── vision_pipeline.py      # [MOD-02] Resize, Blur, MOG2, Morphology, Contours, Foot-Point
│   │   └── safety_evaluator.py     # [MOD-03] PointPolygonTest, Delay Timer, Latch Alarm, Audio E-Stop
│   ├── utils/
│   │   ├── config_manager.py       # [MOD-04] Đọc/ghi config.json, quản lý Undo/Reset đỉnh đa giác ROI
│   │   └── evidence_logger.py      # [MOD-05] Luồng ghi ảnh Snapshot, lọc lịch sử theo Ca, xuất CSV
│   └── ui/
│       └── hmi_main_window.py      # [MOD-06] Giao diện High-Performance HMI, Viewport, Snapshot Carousel
├── docs/
│   └── yeu-cau.md                  # Tài liệu Đặc tả Yêu cầu & Kiến trúc Hệ thống Toàn diện
├── main.py                         # Điểm khởi chạy ứng dụng (Khởi tạo các Thread và kết nối Signals)
└── requirements.txt                # Danh sách thư viện: opencv-python, numpy, PyQt6
```

### 4.3.2. Quy trình Phân nhánh GitHub (Git Flow 3 Tầng)
* `main`: Nhánh phát hành chính thức, luôn ở trạng thái chạy ổn định tuyệt đối phục vụ báo cáo ngày 11/11.
* `develop`: Nhánh tích hợp chung để ghép nối 6 module từ các thành viên.
* Các nhánh tính năng (`feature/*`):
  * `feature/mod01-stream-manager`: Phát triển thu nhận đa nguồn & tự phục hồi kết nối.
  * `feature/mod02-vision-pipeline`: Phát triển thuật toán MOG2, lọc bóng đổ, chống sốc sáng.
  * `feature/mod03-05-safety-audit`: Phát triển logic kiểm tra đa giác, còi báo động, lưu ảnh và xuất CSV.
  * `feature/mod06-hmi-dashboard`: Xây dựng giao diện HMI chuẩn ISA-101 và khay ảnh Snapshot Carousel.