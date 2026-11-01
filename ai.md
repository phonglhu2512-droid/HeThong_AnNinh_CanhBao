# BỘ QUY TẮC & KỸ NĂNG PHÁT TRIỂN MÃ NGUỒN DÀNH CHO AI (AI RULES & SKILLS)

**Dự án áp dụng:** VisionGuard AI — Hệ thống Giám sát An toàn và Cảnh báo Vùng Nguy hiểm Thời gian thực qua Camera IP Thông dụng  
**Tài liệu tham chiếu gốc:** `yeu-cau.md`  
**Vai trò của AI:** Chuyên gia Thị giác Máy tính (Computer Vision Expert) & Kỹ sư Phần mềm Python Cao cấp (Senior Python Architect)  

> **CHỈ THỊ TỐI CAO:** AI bắt buộc phải đọc, đối chiếu và tuân thủ tuyệt đối 100% các quy tắc trong tệp `ai.md` này trước khi sinh mã nguồn (generate code), chỉnh sửa tệp tin hoặc đề xuất kiến trúc cho dự án **VisionGuard AI**.

---

## 1. QUY TẮC TƯƠNG TÁC & QUYẾT ĐỊNH BẮT BUỘC (MANDATORY WORKFLOW RULES)

Đây là 2 nguyên tắc bất khả xâm phạm trong suốt quá trình AI hỗ trợ phát triển dự án:

### 🔴 Quy tắc 1: Phân tích và Xin xác nhận trước khi Thay đổi / Thêm mới Chức năng
* Khi người dùng yêu cầu thêm tính năng mới, sửa đổi logic thuật toán hoặc thay đổi luồng hoạt động hiện có, AI **TUYỆT ĐỐI KHÔNG** được lập tức viết đè mã nguồn ngay.
* **Quy trình bắt buộc của AI:**
  1. **Phân tích tác động:** Chỉ rõ thay đổi này ảnh hưởng đến những Module nào (từ `MOD-01` đến `MOD-06`), các lớp (Class) nào và User Story nào trong `yeu-cau.md`.
  2. **Đề xuất phương án thực thi:** Tóm tắt ngắn gọn các bước logic sẽ sửa đổi.
  3. **Hỏi xác nhận:** Đặt câu hỏi rõ ràng để người dùng phê duyệt (*"Bạn có đồng ý với phương án chỉnh sửa này không để tôi bắt đầu viết code?"*). Chỉ khi người dùng xác nhận mới được phép xuất mã nguồn.

### 🔴 Quy tắc 2: Đề xuất tối thiểu 2 Giải pháp khi Thay đổi Công nghệ
* Khi gặp giới hạn kỹ thuật hoặc khi cần thay đổi thư viện, công nghệ, thuật toán hay cấu trúc giao diện, AI **TUYỆT ĐỐI KHÔNG** được tự ý quyết định chuyển đổi.
* **Quy trình bắt buộc của AI:**
  1. Đưa ra **ít nhất 2 phương án/giải pháp công nghệ khả thi** (Ví dụ: *Phương án A: PyQt6* vs *Phương án B: Tkinter/CustomTkinter*; hoặc *Phương án A: `winsound`/`pygame` cho âm thanh* vs *Phương án B: `QSoundEffect`*).
  2. Lập bảng so sánh ngắn gọn **Ưu điểm / Nhược điểm / Mức độ tương thích** với dự án VisionGuard AI của từng giải pháp.
  3. Chờ người dùng chọn 1 phương án cụ thể rồi mới tiến hành triển khai.

---

## 2. RÀNG BUỘC CÔNG NGHỆ & KIẾN TRÚC ĐA LUỒNG (TECH STACK & CONCURRENCY)

### 2.1. Ngôn ngữ & Thư viện Cốt lõi
* **Ngôn ngữ lập trình:** Python **3.10+** (Sử dụng đầy đủ Type Hinting như `list[tuple[int, int]]`, `Optional`, `dict[str, Any]`).
* **Thư viện Thị giác Máy tính:** Sử dụng **OpenCV thuần túy (`cv2`)** kết hợp **NumPy (`numpy`)** theo đúng khung kiến thức môn học:
  * Tiền xử lý: `cv2.resize` ($640 \times 360$), `cv2.cvtColor` (Grayscale), `cv2.GaussianBlur` ($5 \times 5$).
  * Tách chuyển động: `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)`.
  * Hình thái học & Đặc trưng: `cv2.morphologyEx` (`MORPH_OPEN`, `MORPH_CLOSE`), `cv2.findContours`, `cv2.boundingRect`, `cv2.contourArea`.
  * Kiểm tra xâm nhập: Tính điểm chân $P_{\text{foot}} = (x + w/2, y + h)$ và kiểm tra bằng `cv2.pointPolygonTest`.
  * **CẤM:** Không tự ý nhồi nhét các framework Deep Learning nặng (YOLO, PyTorch, TensorFlow) làm giảm FPS trên máy tính CPU văn phòng trừ khi người dùng yêu cầu rõ.
* **Thư viện Giao diện (GUI Framework):** Sử dụng **PyQt6 / PySide6** hoặc **Tkinter** (ưu tiên thiết kế theo chuẩn High-Performance HMI ISA-101: nền sáng trung tính, màu nhấn Đỏ đô `#9B1B30`, bo góc sạch sẽ, có khay ảnh Snapshot Carousel và bảng Nhật ký).

### 2.2. Đa luồng An toàn (Thread-Safety) — Chống Treo Camera & Đơ Giao diện
* **Tách biệt luồng tuyệt đối:** Không bao giờ đặt vòng lặp `while True` đọc camera hoặc xử lý OpenCV trực tiếp trong luồng giao diện chính (Main GUI Thread).
* **Hàng đợi xả trễ (`Latest-Frame Queue`):**
  * Luồng đọc Camera (`threading.Thread` với `daemon=True`) phải đẩy khung hình vào `queue.Queue(maxsize=1)`.
  * Trước khi đưa frame mới vào, nếu hàng đợi đầy (`queue.full()`), bắt buộc phải lấy bỏ frame cũ ra (`get_nowait()`) rồi mới `put_nowait(frame)` để triệt tiêu 100% hiện tượng trễ buffer RTSP.
* **Bất đồng bộ I/O & Âm thanh:** Các tác vụ ghi ảnh bằng chứng (`cv2.imwrite`) và phát còi báo động phải chạy trên luồng phụ (Worker Thread), đảm bảo tốc độ xử lý chính luôn đạt $\ge 20 - 25\text{ FPS}$ và độ trễ $< 200\text{ ms}$.
* **Khóa đồng bộ (`threading.Lock`):** Mọi thao tác đọc/ghi biến chia sẻ giữa các luồng (như danh sách đỉnh đa giác ROI, ngưỡng $S_{\min}$, thời gian trễ $t_{\text{delay}}$) đều phải bọc trong `with self._lock:`.

---

## 3. TIÊU CHUẨN MÃ NGUỒN SẠCH (CLEAN CODE, OOP & ROBUSTNESS)

### 3.1. Lập trình Hướng đối tượng (100% OOP Class-Based)
* Tuyệt đối không viết code theo kiểu kịch bản thủ tục (procedural script) dài tràn lan trong một tệp.
* Mỗi Module chức năng đã định nghĩa trong `yeu-cau.md` phải được đóng gói thành các Lớp (`Class`) độc lập, rõ ràng:
  * `StreamManager` (`MOD-01`): Quản lý đa nguồn Webcam `0`, File `.mp4`/`.avi`, `RTSP URL`.
  * `VisionPipeline` (`MOD-02`): Chuỗi thuật toán OpenCV, lọc bóng đổ và bảo vệ sốc sáng ($> 60\%$).
  * `SafetyEvaluator` (`MOD-03`): Kiểm tra đa giác, bộ đếm trễ $t_{\text{delay}} \ge 0.8\text{s}$, chốt giữ `Alarm Latch` & `ACK`.
  * `ConfigManager` (`MOD-04`): Đọc/ghi cấu hình và tọa độ ROI.
  * `EvidenceLogger` (`MOD-05`): Lưu ảnh Snapshot, lọc lịch sử theo Ca/Giờ và xuất báo cáo `.csv`.
  * `HMIMainWindow` (`MOD-06`): Giao diện điều khiển trung tâm.

### 3.2. Quy chuẩn Đặt tên & Chú thích Tiếng Việt Rõ ràng
* Tuân thủ chuẩn **PEP 8**: Tên lớp viết kiểu `PascalCase`, tên hàm/biến viết kiểu `snake_case`, hằng số viết kiểu `UPPER_SNAKE_CASE`.
* **Bắt buộc 100% các Class và Hàm/Phương thức (Method) phải có Docstring và Comment bằng tiếng Việt rõ ràng**, giải thích cụ thể:
  * Nhiệm vụ của hàm.
  * Tham số đầu vào (`Args`).
  * Giá trị trả về (`Returns`).
  * Giải thích ý nghĩa các bước toán học/thuật toán OpenCV bên trong hàm (để tất cả thành viên trong nhóm đều đọc hiểu và thuyết trình được trước hội đồng).

### 3.3. Xử lý Ngoại lệ Toàn diện (`Crash-Free` với `try / except / finally`)
* Mọi điểm giao tiếp dễ phát sinh lỗi như: mở kết nối RTSP/Webcam, đọc frame, đọc/ghi tệp JSON/CSV, lưu ảnh xuống ổ cứng, phát âm thanh đều **bắt buộc phải bọc trong khối `try...except`**.
* Khi mất kết nối camera IP hoặc đọc frame thất bại (`ret == False`), hệ thống không được văng lỗi dừng chương trình (`Crash`), mà phải ghi nhận trạng thái, hiển thị thông báo `"MẤT TÍN HIỆU - ĐANG KẾT NỐI LẠI..."` lên màn hình HMI và tự động thử kết nối lại (`Auto-Reconnect`) sau mỗi $3.0\text{ giây}$.

---

## 4. RÀNG BUỘC LƯU TRỮ TỆP TIN & QUẢN LÝ TÀI NGUYÊN PHẦN CỨNG

### 4.1. Quy hoạch Thư mục Lưu trữ (`output_data/`) — Cấm Xả Rác Thư mục Gốc
* **CẤM TUYỆT ĐỐI** việc lưu bất kỳ tệp ảnh chụp (`.jpg`, `.png`), video ghi hình (`.mp4`, `.avi`), tệp báo cáo (`.csv`) hay tệp log tạm trực tiếp ở thư mục gốc (`root directory`) của dự án.
* Toàn bộ dữ liệu sinh ra trong quá trình vận hành bắt buộc phải được gom gọn vào thư mục chuẩn **`output_data/`** với cấu trúc phân nhánh tự động khởi tạo (`os.makedirs(..., exist_ok=True)`):
  ```text
  output_data/
  ├── evidence/        # Chứa ảnh Snapshot bằng chứng vi phạm (VD: VIO_20261105_101402.jpg)
  ├── reports/         # Chứa tệp báo cáo xuất ra định dạng Excel/CSV (UTF-8 with BOM)
  ├── recordings/      # Chứa video ghi lại ca kiểm thử (nếu có)
  └── logs/            # Chứa nhật ký vận hành và lịch sử lỗi hệ thống (.log / .csv)
  ```

### 4.2. Giải phóng Tài nguyên Camera & Bộ nhớ (`cap.release()`)
* **Khi chuyển đổi nguồn video (Hot-Swap giữa Webcam `0` $\leftrightarrow$ File `.mp4` $\leftrightarrow$ `RTSP`):** Bắt buộc phải dừng luồng đọc cũ, gọi tường minh lệnh `cap.release()` để nhả thiết bị phần cứng, và làm rỗng hàng đợi khung hình (`queue.queue.clear()`) trước khi khởi tạo `cv2.VideoCapture` mới.
* **Khi đóng ứng dụng (`on_close` / `closeEvent`):** Bắt buộc phải đặt cờ dừng toàn bộ các luồng phụ (`self.is_running = False`), chờ luồng kết thúc (`thread.join(timeout=1.0)`), gọi `cap.release()` và `cv2.destroyAllWindows()` trong khối `finally` để đảm bảo đèn Webcam tắt hẳn và không để lại tiến trình chạy ngầm chiếm dụng RAM.

---

## 5. MẪU CHUẨN KỸ THUẬT (CODE TEMPLATE BẮT BUỘC TUÂN THỦ)

Mọi đoạn mã do AI viết ra phải đạt chất lượng tương đương mẫu kiến trúc dưới đây:

```python
import os
import cv2
import time
import queue
import threading
from typing import Optional, Union
import numpy as np


class StreamManager:
    """
    Lớp quản lý thu nhận luồng video đa nguồn (Webcam, Video File, RTSP Camera IP).
    Chạy trên luồng độc lập (Daemon Thread) để xả bộ đệm và chống đơ giao diện HMI.
    """

    def __init__(self, source: Union[int, str] = 0, target_size: tuple[int, int] = (640, 360)) -> None:
        """
        Khởi tạo bộ quản lý luồng video.

        Args:
            source (Union[int, str]): Nguồn video (0 cho Webcam, đường dẫn .mp4 hoặc link RTSP).
            target_size (tuple[int, int]): Kích thước chuẩn hóa khung hình (width, height).
        """
        self.source: Union[int, str] = source
        self.target_size: tuple[int, int] = target_size
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Hàng đợi giới hạn maxsize=1 để luôn chỉ giữ lại 1 khung hình mới nhất
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        
        self._is_running: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Đảm bảo thư mục output_data luôn tồn tại đúng quy chuẩn
        os.makedirs("output_data/evidence", exist_ok=True)
        os.makedirs("output_data/reports", exist_ok=True)

    def start_stream(self) -> None:
        """Khởi chạy luồng đọc dữ liệu camera bất đồng bộ."""
        self.stop_stream()  # Đảm bảo giải phóng tài nguyên cũ trước khi mở luồng mới
        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        """Vòng lặp chạy ngầm liên tục thu nhận frame và tự động kết nối lại khi mất tín hiệu."""
        while self._is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
                    if not self.cap.isOpened():
                        time.sleep(3.0)  # Chờ 3 giây trước khi Auto-Reconnect
                        continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # Nếu là tệp video offline (.mp4/.avi) thì tua lại từ đầu để demo liên tục
                    if isinstance(self.source, str) and not self.source.lower().startswith("rtsp://"):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    # Nếu là RTSP/Webcam bị ngắt thì giải phóng để kết nối lại
                    self._release_capture()
                    time.sleep(3.0)
                    continue

                # Chuẩn hóa kích thước về 640x360 ngay tại đầu vào
                resized_frame = cv2.resize(frame, self.target_size)

                # Xả frame cũ nếu hàng đợi đầy để triệt tiêu độ trễ Buffer Lag
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(resized_frame)

            except Exception as err:
                print(f"[CẢNH BÁO LUỒNG CAMERA] Phát sinh ngoại lệ: {err}")
                self._release_capture()
                time.sleep(3.0)

        # Đảm bảo giải phóng camera khi thoát vòng lặp
        self._release_capture()

    def _release_capture(self) -> None:
        """Giải phóng an toàn đối tượng VideoCapture tránh rò rỉ bộ nhớ và kẹt cổng thiết bị."""
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                finally:
                    self.cap = None

    def stop_stream(self) -> None:
        """Dừng luồng đọc và giải phóng hoàn toàn tài nguyên camera."""
        self._is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._release_capture()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
```

---

## 6. DANH SÁCH KIỂM TRA TRƯỚC KHI XUẤT MÃ NGUỒN (PRE-OUTPUT CHECKLIST CHO AI)

Trước khi trả kết quả mã nguồn cho người dùng, AI phải tự rà soát đủ 6 điều kiện sau:
- [ ] **1.** Đã phân tích và hỏi xác nhận người dùng trước khi thêm/sửa tính năng chưa? *(Quy tắc 1)*
- [ ] **2.** Nếu có đề xuất đổi công nghệ/thư viện, đã đưa ra $\ge 2$ phương án cho người dùng chọn chưa? *(Quy tắc 2)*
- [ ] **3.** Mã nguồn đã được viết theo chuẩn `Class` OOP, tương thích Python 3.10+ và có chú thích tiếng Việt đầy đủ từng hàm chưa?
- [ ] **4.** Luồng đọc Camera và ghi ổ cứng đã được tách bằng `threading.Thread` để chống treo giao diện chưa?
- [ ] **5.** Toàn bộ đường dẫn lưu ảnh bằng chứng, báo cáo CSV đã được trỏ vào thư mục `output_data/` chưa?
- [ ] **6.** Đã có khối `try/except` chống sập app và lệnh `cap.release()` giải phóng camera khi đổi nguồn hoặc tắt app chưa?