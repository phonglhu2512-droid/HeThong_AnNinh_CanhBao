"""Giao diện HMI ISA-101 cho giám sát, hiệu chuẩn ROI và tra cứu bằng chứng."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeAlias
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QTime, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.cv_engine import CVProcessor, DisplayMode, FrameAnalysis
from src.core.safety_controller import SafetyController, SafetyResult, SafetyState
from src.core.stream_manager import StreamManager, StreamState, VideoSource
from src.utils.audit_logger import AuditEvent, AuditLogger
from src.utils.config_manager import ConfigManager


LOGGER = logging.getLogger(__name__)
ROIPoint: TypeAlias = tuple[int, int]
ROIColor: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class RenderPacket:
    """Gói dữ liệu đã xử lý do worker gửi đến GUI thread."""

    frame: np.ndarray | None
    analysis: FrameAnalysis | None
    safety: SafetyResult | None
    error_message: str | None = None


class VideoCanvas(QWidget):
    """Vẽ video đúng tỷ lệ và đổi vị trí click chuột sang tọa độ frame."""

    roi_point_added = pyqtSignal(int, int)

    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 360
    PLACEHOLDER_TEXT: str = "ĐANG CHỜ TÍN HIỆU VIDEO"

    def __init__(self, parent: QWidget | None = None) -> None:
        """Tạo vùng vẽ video tương tác, mặc định theo tỷ lệ 16:9.

        Args:
            parent: Widget cha quản lý vòng đời canvas.

        Returns:
            None.
        """
        super().__init__(parent)
        self._image: QImage | None = None
        self._roi_points: list[ROIPoint] = []
        self._editing_roi: bool = False
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)

    def set_frame(self, frame: np.ndarray) -> None:
        """Sao chép frame BGR thành QImage độc lập để GUI vẽ an toàn.

        Args:
            frame: Frame BGR 640×360 từ worker xử lý nền.

        Returns:
            None.

        Raises:
            ValueError: Khi frame không có kích thước 3 kênh hợp lệ.
            cv2.error: Khi chuyển đổi BGR sang RGB thất bại.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame hiển thị phải có ba kênh màu BGR.")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        )
        self._image = image.copy()
        self.update()

    def set_roi_overlay(
        self,
        points: list[ROIPoint],
        editing: bool,
    ) -> None:
        """Cập nhật lớp phủ ROI và trạng thái cho phép click vẽ đỉnh.

        Args:
            points: Các đỉnh ROI đã lưu hoặc các điểm đang vẽ.
            editing: True khi click trái được dùng để thêm đỉnh ROI.

        Returns:
            None.
        """
        self._roi_points = list(points)
        self._editing_roi = editing
        self.setCursor(
            Qt.CursorShape.CrossCursor if editing else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def _image_rect(self) -> QRectF:
        """Tính vùng hiển thị ảnh letterbox để giữ nguyên tỷ lệ 16:9.

        Args:
            None.

        Returns:
            Vùng chữ nhật của ảnh trong tọa độ local của canvas.
        """
        if self._image is None or self._image.height() == 0:
            return QRectF()
        scale = min(
            self.width() / self._image.width(),
            self.height() / self._image.height(),
        )
        width = self._image.width() * scale
        height = self._image.height() * scale
        return QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )

    def paintEvent(self, event: object) -> None:
        """Vẽ nền, frame hiện hành và ROI theo đúng tỷ lệ video.

        Args:
            event: Sự kiện vẽ do Qt gửi đến widget.

        Returns:
            None.
        """
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#17191D"))

        if self._image is None:
            painter.setPen(QColor("#D1D5DB"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self.PLACEHOLDER_TEXT,
            )
            return

        image_rect = self._image_rect()
        painter.drawImage(image_rect, self._image)
        if not self._roi_points:
            return

        scaled_points = [
            QPointF(
                image_rect.x() + point[0] * image_rect.width() / self.FRAME_WIDTH,
                image_rect.y() + point[1] * image_rect.height() / self.FRAME_HEIGHT,
            )
            for point in self._roi_points
        ]
        roi_color = QColor("#F59E0B") if self._editing_roi else QColor("#28B4A7")
        pen = QPen(roi_color, 2.0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        if len(scaled_points) >= 2:
            for index in range(len(scaled_points) - 1):
                painter.drawLine(scaled_points[index], scaled_points[index + 1])
            if not self._editing_roi and len(scaled_points) >= 3:
                painter.drawLine(scaled_points[-1], scaled_points[0])

        painter.setBrush(roi_color)
        painter.setPen(Qt.PenStyle.NoPen)
        for point in scaled_points:
            painter.drawEllipse(point, 4.0, 4.0)

    def mousePressEvent(self, event: object) -> None:
        """Phát tọa độ click hợp lệ trong frame khi đang hiệu chuẩn ROI.

        Args:
            event: Sự kiện chuột do Qt gửi đến widget.

        Returns:
            None.
        """
        if (
            self._editing_roi
            and event.button() == Qt.MouseButton.LeftButton
        ):
            image_rect = self._image_rect()
            click = event.position()
            if image_rect.contains(click):
                frame_x = int(
                    (click.x() - image_rect.x())
                    * self.FRAME_WIDTH
                    / image_rect.width()
                )
                frame_y = int(
                    (click.y() - image_rect.y())
                    * self.FRAME_HEIGHT
                    / image_rect.height()
                )
                frame_x = min(max(frame_x, 0), self.FRAME_WIDTH - 1)
                frame_y = min(max(frame_y, 0), self.FRAME_HEIGHT - 1)
                self.roi_point_added.emit(frame_x, frame_y)
                event.accept()
                return
        super().mousePressEvent(event)


class _CVSafetyWorker(threading.Thread):
    """Kết hợp CV và đánh giá an toàn ngoài GUI thread."""

    def __init__(
        self,
        stream_manager: StreamManager,
        cv_processor: CVProcessor,
        safety_controller: SafetyController,
        audit_logger: AuditLogger,
        roi_points: list[ROIPoint],
    ) -> None:
        """Khởi tạo worker xử lý frame và các phụ thuộc module dùng chung.

        Args:
            stream_manager: Nguồn frame latest từ MOD-01.
            cv_processor: Pipeline OpenCV từ MOD-02.
            safety_controller: Bộ đánh giá an toàn từ MOD-03.
            audit_logger: Bộ lưu snapshot từ MOD-05.
            roi_points: ROI ban đầu đã nạp từ MOD-04.

        Returns:
            None.
        """
        super().__init__(name="VisionGuard-CV-Safety", daemon=True)
        self._stream_manager = stream_manager
        self._cv_processor = cv_processor
        self._safety_controller = safety_controller
        self._audit_logger = audit_logger
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._roi_points = list(roi_points)
        self._display_mode = DisplayMode.RGB
        self._result_queue: queue.Queue[RenderPacket] = queue.Queue(maxsize=1)
        self._source_queue: queue.Queue[VideoSource] = queue.Queue(maxsize=1)
        self._active_event_id: str | None = None
        self._active_event_duration: float = 0.0
        self._previous_latched: bool = False
        self._last_fps: float = 0.0

    @property
    def fps(self) -> float:
        """Trả về FPS gần nhất để Header hiển thị mà không đọc field nội bộ.

        Args:
            None.

        Returns:
            Số frame xử lý trung bình gần nhất mỗi giây.
        """
        return self._last_fps

    def set_roi_points(self, points: list[ROIPoint]) -> None:
        """Cập nhật ROI đang sử dụng từ HMI mà không đụng vào frame đang đọc.

        Args:
            points: Tọa độ ROI mới theo kích thước 640×360.

        Returns:
            None.
        """
        with self._state_lock:
            self._roi_points = list(points)

    def set_display_mode(self, mode: DisplayMode) -> None:
        """Đặt chế độ xem áp dụng trên frame tiếp theo.

        Args:
            mode: Chế độ RGB, MOG2 Mask hoặc Canny Edge.

        Returns:
            None.
        """
        with self._state_lock:
            self._display_mode = mode

    def request_source_switch(self, source: VideoSource) -> None:
        """Gửi yêu cầu đổi nguồn để worker thực hiện ngoài GUI thread.

        Args:
            source: Webcam ID, đường dẫn video hoặc URL RTSP.

        Returns:
            None.
        """
        if self._source_queue.full():
            try:
                self._source_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._source_queue.put_nowait(source)
        except queue.Full:
            LOGGER.error("Không thể gửi yêu cầu chuyển nguồn video.")
            raise

    def take_latest_result(self) -> RenderPacket | None:
        """Lấy gói kết quả mới nhất; loại bỏ frame xử lý cũ nếu GUI bị trễ.

        Args:
            None.

        Returns:
            Gói frame mới nhất, hoặc None nếu chưa có kết quả.
        """
        newest: RenderPacket | None = None
        while True:
            try:
                newest = self._result_queue.get_nowait()
            except queue.Empty:
                return newest

    def stop(self) -> None:
        """Yêu cầu worker dừng và bỏ mọi frame đang chờ xử lý.

        Args:
            None.

        Returns:
            None.
        """
        self._stop_event.set()

    def run(self) -> None:
        """Đọc queue camera, xử lý CV/an toàn và gửi kết quả cho GUI.

        Args:
            None.

        Returns:
            None.
        """
        previous_frame_time: float | None = None
        while not self._stop_event.is_set():
            try:
                self._apply_pending_source_switch()
                frame = self._stream_manager.read_latest_frame(timeout=0.05)
                if frame is None:
                    continue
                with self._state_lock:
                    roi_points = list(self._roi_points)
                    display_mode = self._display_mode

                analysis = self._cv_processor.process_frame(
                    frame,
                    display_mode=display_mode,
                )
                foot_point, _ = self._select_detection(
                    analysis,
                    roi_points,
                )
                frame_time = time.monotonic()
                if previous_frame_time is not None:
                    elapsed = frame_time - previous_frame_time
                    if elapsed > 0:
                        self._last_fps = 1.0 / elapsed
                previous_frame_time = frame_time

                safety = self._safety_controller.evaluate(
                    foot_point,
                    roi_points,
                    timestamp=frame_time,
                    skip_evaluation=analysis.skip_safety_evaluation,
                )
                rendered_frame = self._render_frame(
                    analysis,
                    roi_points,
                    safety,
                    self._last_fps,
                )
                self._record_alarm_transition(
                    analysis,
                    safety,
                    foot_point,
                    roi_points,
                )
                self._publish_result(
                    RenderPacket(
                        frame=rendered_frame,
                        analysis=analysis,
                        safety=safety,
                    )
                )
            except Exception as error:
                LOGGER.exception("Worker xử lý CV/an toàn gặp lỗi.")
                self._publish_result(
                    RenderPacket(
                        frame=None,
                        analysis=None,
                        safety=None,
                        error_message=str(error),
                    )
                )
                self._stop_event.wait(0.05)

    def _apply_pending_source_switch(self) -> None:
        """Thực hiện lần đổi nguồn mới nhất ngoài luồng giao diện.

        Args:
            None.

        Returns:
            None.
        """
        newest_source: VideoSource | None = None
        while True:
            try:
                newest_source = self._source_queue.get_nowait()
            except queue.Empty:
                break
        if newest_source is not None:
            self._stream_manager.switch_source(newest_source)

    @staticmethod
    def _select_detection(
        analysis: FrameAnalysis,
        roi_points: list[ROIPoint],
    ) -> tuple[ROIPoint | None, tuple[int, int, int, int] | None]:
        """Chọn điểm chân trong ROI nếu có, nếu không dùng contour đầu tiên.

        Args:
            analysis: Kết quả CV frame hiện tại.
            roi_points: Đa giác ROI được dùng để ưu tiên đối tượng xâm nhập.

        Returns:
            Điểm chân và hộp bao tương ứng; cả hai None nếu không có contour.

        Raises:
            cv2.error: Khi pointPolygonTest thất bại.
        """
        if not analysis.detections:
            return None, None
        polygon = np.asarray(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        for detection in analysis.detections:
            if (
                cv2.pointPolygonTest(
                    polygon,
                    (
                        float(detection.foot_point[0]),
                        float(detection.foot_point[1]),
                    ),
                    False,
                )
                >= 0
            ):
                return detection.foot_point, detection.bounding_box
        detection = analysis.detections[0]
        return detection.foot_point, detection.bounding_box

    @staticmethod
    def _render_frame(
        analysis: FrameAnalysis,
        roi_points: list[ROIPoint],
        safety: SafetyResult,
        fps: float,
        base_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vẽ ROI, bounding box, foot-point và chỉ số HMI lên frame.

        Args:
            analysis: Ảnh theo chế độ xem và các phát hiện đã tính.
            roi_points: Đỉnh đa giác ROI cần hiển thị.
            safety: Trạng thái an toàn quyết định màu viền vùng cấm.
            fps: FPS worker tính từ thời gian xử lý frame.
            base_frame: Ảnh nguồn tùy chọn; None dùng chế độ hiển thị đang chọn.

        Returns:
            Bản sao frame BGR có lớp phủ phục vụ HMI và lưu bằng chứng.
        """
        rendered = (
            analysis.display_frame.copy()
            if base_frame is None
            else base_frame.copy()
        )
        colors: dict[SafetyState, ROIColor] = {
            SafetyState.SAFE: (40, 170, 150),
            SafetyState.WARNING: (0, 190, 245),
            SafetyState.ALARM: (30, 30, 240),
        }
        roi_array = np.asarray(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        if len(roi_points) >= 3:
            cv2.polylines(
                rendered,
                [roi_array],
                isClosed=True,
                color=colors[safety.state],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        for detection in analysis.detections:
            x, y, width, height = detection.bounding_box
            inside = safety.foot_point == detection.foot_point and safety.inside_roi
            object_color = (30, 30, 240) if inside else (35, 155, 75)
            cv2.rectangle(
                rendered,
                (x, y),
                (x + width, y + height),
                object_color,
                2,
            )
            cv2.circle(rendered, detection.foot_point, 5, object_color, -1)

        cv2.putText(
            rendered,
            f"FPS {fps:04.1f} | {safety.state.value} | "
            f"{len(analysis.detections)} doi tuong",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 45, 55),
            2,
            cv2.LINE_AA,
        )
        return rendered

    def _record_alarm_transition(
        self,
        analysis: FrameAnalysis,
        safety: SafetyResult,
        foot_point: ROIPoint | None,
        roi_points: list[ROIPoint],
    ) -> None:
        """Lưu snapshot một lần khi latch mới bật và chốt thời lượng khi ra ROI.

        Args:
            analysis: Kết quả các contour của frame hiện tại.
            safety: Trạng thái controller sau khi đánh giá frame.
            foot_point: Điểm chân được controller dùng trong frame.
            roi_points: ROI hiện hành dùng để chú thích ảnh bằng chứng.

        Returns:
            None.
        """
        if safety.inside_roi and self._active_event_id is not None:
            self._active_event_duration = max(
                self._active_event_duration,
                safety.violation_duration,
            )

        if safety.latched and not self._previous_latched:
            detection = next(
                (
                    candidate
                    for candidate in analysis.detections
                    if candidate.foot_point == foot_point
                ),
                None,
            )
            if detection is not None:
                try:
                    evidence_frame = self._render_frame(
                        analysis,
                        roi_points,
                        safety,
                        self._last_fps,
                        base_frame=analysis.rgb_frame,
                    )
                    event = self._audit_logger.log_violation(
                        evidence_frame,
                        detection.bounding_box,
                        detection.foot_point,
                        safety.violation_duration,
                        occurred_at=datetime.now(),
                    )
                    self._active_event_id = event.event_id
                    self._active_event_duration = safety.violation_duration
                except Exception:
                    LOGGER.exception("Không thể xếp snapshot vi phạm vào hàng đợi.")

        if (
            not safety.inside_roi
            and self._active_event_id is not None
        ):
            try:
                self._audit_logger.update_violation_duration(
                    self._active_event_id,
                    self._active_event_duration,
                )
            except Exception:
                LOGGER.exception(
                    "Không thể cập nhật thời lượng sự kiện %s.",
                    self._active_event_id,
                )
            finally:
                self._active_event_id = None
                self._active_event_duration = 0.0

        self._previous_latched = safety.latched

    def _publish_result(self, result: RenderPacket) -> None:
        """Thay kết quả cũ trong queue bằng packet mới nhất cho GUI.

        Args:
            result: Frame đã render hoặc thông tin lỗi worker.

        Returns:
            None.
        """
        if self._result_queue.full():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            LOGGER.warning("Không thể cập nhật frame mới nhất cho HMI.")


class SnapshotPreviewDialog(QDialog):
    """Hiển thị ảnh bằng chứng gốc trong cửa sổ phóng to."""

    def __init__(self, event: AuditEvent, parent: QWidget | None = None) -> None:
        """Nạp và trình bày ảnh Snapshot của sự kiện đã chọn.

        Args:
            event: Bản ghi chứa đường dẫn ảnh bằng chứng.
            parent: Cửa sổ HMI sở hữu modal.

        Returns:
            None.

        Raises:
            OSError: Khi tệp ảnh không thể đọc hoặc ảnh không hợp lệ.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Bằng chứng {event.event_id}")
        self.setModal(True)
        self.setMinimumSize(800, 500)
        layout = QVBoxLayout(self)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image = cv2.imread(str(event.image_path))
        if image is None:
            raise OSError(f"Không thể đọc ảnh bằng chứng: {event.image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        image_label.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                1200,
                760,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addWidget(image_label, 1)
        layout.addWidget(
            QLabel(
                f"{event.event_id}  |  "
                f"{event.occurred_at:%Y-%m-%d %H:%M:%S}  |  "
                f"Vi phạm: {event.violation_duration:.1f}s"
            )
        )


class HMIMainWindow(QMainWindow):
    """Cửa sổ HMI chính kết nối stream, vision, safety, config và audit."""

    def __init__(
        self,
        config_manager: ConfigManager,
        stream_manager: StreamManager,
        cv_processor: CVProcessor,
        safety_controller: SafetyController,
        audit_logger: AuditLogger,
    ) -> None:
        """Khởi tạo giao diện, tải config và khởi động các worker cần thiết.

        Args:
            config_manager: Quản lý cấu hình ROI, nguồn và ngưỡng.
            stream_manager: Quản lý nguồn video đa chế độ.
            cv_processor: Pipeline xử lý và phân tích frame.
            safety_controller: Bộ đánh giá trạng thái an toàn và ACK.
            audit_logger: Nhật ký sự kiện và Snapshot Carousel.

        Returns:
            None.
        """
        super().__init__()
        self._config_manager = config_manager
        self._stream_manager = stream_manager
        self._cv_processor = cv_processor
        self._safety_controller = safety_controller
        self._audit_logger = audit_logger
        self._config = config_manager.get_config()
        self._roi_points: list[ROIPoint] = [
            (int(point[0]), int(point[1]))
            for point in self._config["roi_points"]
        ]
        self._draft_roi_points: list[ROIPoint] = []
        self._setup_mode = False
        self._filtered_events: list[AuditEvent] = audit_logger.get_events()
        self._carousel_fingerprint: tuple[tuple[str, bool, float], ...] = ()
        self._last_canvas_state: tuple[SafetyState, StreamState, bool] | None = None
        self._shutdown_complete = False

        self.setWindowTitle("VisionGuard AI | Industrial Safety HMI")
        self.resize(1500, 940)
        self.setMinimumSize(1120, 760)
        self._build_ui()
        self._apply_styles()
        self._load_source_form()

        self._worker = _CVSafetyWorker(
            stream_manager,
            cv_processor,
            safety_controller,
            audit_logger,
            self._roi_points,
        )
        self._worker.start()
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._refresh_hmi)
        self._ui_timer.start()
        self._audit_timer = QTimer(self)
        self._audit_timer.setInterval(500)
        self._audit_timer.timeout.connect(self._refresh_audit_views)
        self._audit_timer.start()

    @property
    def shutdown_complete(self) -> bool:
        """Cho biết closeEvent đã hoàn tất quá trình đóng các dịch vụ nền.

        Args:
            None.

        Returns:
            True nếu worker và các dịch vụ liên quan đã được yêu cầu dừng.
        """
        return self._shutdown_complete

    def _build_ui(self) -> None:
        """Tạo Header, vùng video/Side Panel và Bottom Dock 60/40.

        Args:
            None.

        Returns:
            None.
        """
        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(18, 14, 18, 14)
        root_layout.setSpacing(12)
        central_widget.setObjectName("centralWidget")
        root_layout.addWidget(self._build_header())

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.addWidget(self._build_live_and_side_panel())
        vertical_splitter.addWidget(self._build_bottom_dock())
        vertical_splitter.setStretchFactor(0, 7)
        vertical_splitter.setStretchFactor(1, 3)
        vertical_splitter.setSizes([650, 270])
        root_layout.addWidget(vertical_splitter, 1)
        self.setCentralWidget(central_widget)

    def _build_header(self) -> QWidget:
        """Tạo thanh trạng thái, FPS và chuyển chế độ vận hành/cài đặt.

        Args:
            None.

        Returns:
            Widget Header hoàn chỉnh.
        """
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 10, 18, 10)

        title = QLabel("VISIONGUARD")
        title.setObjectName("brand")
        subtitle = QLabel("INDUSTRIAL SAFETY HMI")
        subtitle.setObjectName("muted")
        brand_stack = QVBoxLayout()
        brand_stack.addWidget(title)
        brand_stack.addWidget(subtitle)
        layout.addLayout(brand_stack)
        layout.addStretch(1)

        self._system_status = QLabel("ĐANG KHỞI TẠO")
        self._system_status.setObjectName("statusSafe")
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setObjectName("metric")
        self._people_label = QLabel("ĐỐI TƯỢNG: 0")
        self._people_label.setObjectName("metric")
        layout.addWidget(self._system_status)
        layout.addWidget(self._fps_label)
        layout.addWidget(self._people_label)

        layout.addWidget(QLabel("CHẾ ĐỘ"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["VẬN HÀNH", "CÀI ĐẶT"])
        self._mode_combo.setObjectName("modeCombo")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self._mode_combo)
        return header

    def _build_live_and_side_panel(self) -> QWidget:
        """Tạo vùng Live Stream 70% và bảng điều khiển bên phải.

        Args:
            None.

        Returns:
            Widget chứa canvas video và Side Panel.
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)
        video_panel = QFrame()
        video_panel.setObjectName("panel")
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("GIÁM SÁT TRỰC TIẾP"))
        title_row.addStretch(1)
        self._roi_hint = QLabel("ROI: ĐÃ LƯU")
        self._roi_hint.setObjectName("muted")
        title_row.addWidget(self._roi_hint)
        video_layout.addLayout(title_row)

        self._canvas = VideoCanvas()
        self._canvas.setObjectName("videoCanvas")
        self._canvas.roi_point_added.connect(self._add_roi_point)
        video_layout.addWidget(self._canvas, 1)
        self._roi_controls = self._build_roi_controls()
        video_layout.addWidget(self._roi_controls)
        splitter.addWidget(video_panel)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([1000, 420])
        return splitter

    def _build_roi_controls(self) -> QWidget:
        """Tạo Undo, Reset và Lưu ROI cho chế độ cài đặt.

        Args:
            None.

        Returns:
            Widget thanh nút hiệu chuẩn ROI.
        """
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        self._undo_roi_button = QPushButton("Undo đỉnh")
        self._reset_roi_button = QPushButton("Reset ROI")
        self._save_roi_button = QPushButton("Lưu vùng ROI")
        self._save_roi_button.setObjectName("primaryButton")
        self._undo_roi_button.clicked.connect(self._undo_roi_point)
        self._reset_roi_button.clicked.connect(self._reset_roi_draft)
        self._save_roi_button.clicked.connect(self._save_roi)
        layout.addWidget(QLabel("HIỆU CHUẨN ROI"))
        layout.addStretch(1)
        layout.addWidget(self._undo_roi_button)
        layout.addWidget(self._reset_roi_button)
        layout.addWidget(self._save_roi_button)
        return controls

    def _build_side_panel(self) -> QWidget:
        """Tạo ACK, bộ chọn view/source và sliders cấu hình kỹ thuật.

        Args:
            None.

        Returns:
            Side Panel cuộn được khi cửa sổ có chiều cao nhỏ.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        layout.addWidget(QLabel("ĐIỀU KHIỂN AN TOÀN"))
        self._ack_button = QPushButton("XÁC NHẬN SỰ CỐ\n/ TẮT CÒI (ACK)")
        self._ack_button.setObjectName("ackButton")
        self._ack_button.setMinimumHeight(78)
        self._ack_button.setEnabled(False)
        self._ack_button.clicked.connect(self._acknowledge_alarm)
        layout.addWidget(self._ack_button)
        self._estop_status = QLabel("E-STOP: KHÔNG KÍCH HOẠT")
        self._estop_status.setObjectName("muted")
        layout.addWidget(self._estop_status)

        layout.addWidget(self._section_separator())
        layout.addWidget(QLabel("CHẾ ĐỘ HIỂN THỊ"))
        self._display_combo = QComboBox()
        self._display_combo.addItem("RGB Thực tế", DisplayMode.RGB)
        self._display_combo.addItem("MOG2 Mask", DisplayMode.MOG2_MASK)
        self._display_combo.addItem("Canny Edge", DisplayMode.CANNY_EDGE)
        self._display_combo.currentIndexChanged.connect(self._on_display_changed)
        layout.addWidget(self._display_combo)

        layout.addWidget(self._section_separator())
        layout.addWidget(QLabel("NGUỒN VIDEO"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(
            ["Webcam 0", "File Video (.mp4/.avi)", "RTSP Stream"]
        )
        self._source_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        layout.addWidget(self._source_combo)

        self._file_path_edit = QLineEdit()
        self._file_path_edit.setPlaceholderText("Chọn video .mp4 hoặc .avi")
        self._browse_file_button = QPushButton("Chọn tệp video…")
        self._file_source_group = QWidget()
        file_row = QHBoxLayout(self._file_source_group)
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.addWidget(self._file_path_edit, 1)
        file_row.addWidget(self._browse_file_button)
        self._browse_file_button.clicked.connect(self._browse_video_file)
        layout.addWidget(self._file_source_group)

        self._rtsp_url_edit = QLineEdit()
        self._rtsp_url_edit.setPlaceholderText("rtsp:// hoặc địa chỉ camera")
        self._rtsp_port_edit = QLineEdit("554")
        self._rtsp_port_edit.setMaximumWidth(82)
        self._rtsp_user_edit = QLineEdit()
        self._rtsp_user_edit.setPlaceholderText("Username")
        self._rtsp_password_edit = QLineEdit()
        self._rtsp_password_edit.setPlaceholderText("Password")
        self._rtsp_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._rtsp_profile_combo = QComboBox()
        self._rtsp_profile_combo.addItems(["ONVIF", "Hikvision", "Dahua"])
        self._rtsp_path_edit = QLineEdit("/Streaming/Channels/101")
        self._rtsp_source_group = QWidget()
        rtsp_form = QFormLayout(self._rtsp_source_group)
        rtsp_form.addRow("URL / IP", self._rtsp_url_edit)
        rtsp_form.addRow("Port", self._rtsp_port_edit)
        rtsp_form.addRow("Username", self._rtsp_user_edit)
        rtsp_form.addRow("Password", self._rtsp_password_edit)
        rtsp_form.addRow("Profile", self._rtsp_profile_combo)
        rtsp_form.addRow("Path", self._rtsp_path_edit)
        layout.addWidget(self._rtsp_source_group)
        self._apply_source_button = QPushButton("ÁP DỤNG / CHUYỂN NGUỒN")
        self._apply_source_button.setObjectName("primaryButton")
        self._apply_source_button.clicked.connect(self._apply_video_source)
        layout.addWidget(self._apply_source_button)

        layout.addWidget(self._section_separator())
        layout.addWidget(QLabel("THAM SỐ PHÁT HIỆN"))
        self._min_area_value = QLabel()
        self._min_area_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_area_slider.setRange(500, 5000)
        self._min_area_slider.setSingleStep(100)
        self._min_area_slider.setValue(int(self._config["min_area"]))
        self._min_area_slider.valueChanged.connect(self._on_min_area_changed)
        layout.addWidget(QLabel("Diện tích tối thiểu S_min"))
        layout.addWidget(self._min_area_slider)
        layout.addWidget(self._min_area_value)

        self._delay_value = QLabel()
        self._delay_slider = QSlider(Qt.Orientation.Horizontal)
        self._delay_slider.setRange(0, 30)
        self._delay_slider.setSingleStep(1)
        self._delay_slider.setValue(
            round(float(self._config["violation_delay"]) * 10)
        )
        self._delay_slider.valueChanged.connect(self._on_delay_changed)
        self._tuning_save_timer = QTimer(self)
        self._tuning_save_timer.setSingleShot(True)
        self._tuning_save_timer.setInterval(300)
        self._tuning_save_timer.timeout.connect(self._save_tuning_config)
        layout.addWidget(QLabel("Độ trễ báo động"))
        layout.addWidget(self._delay_slider)
        layout.addWidget(self._delay_value)
        layout.addStretch(1)

        scroll.setWidget(panel)
        self._refresh_slider_labels()
        return scroll

    def _build_bottom_dock(self) -> QWidget:
        """Tạo Snapshot Carousel bên trái và bộ lọc/bảng audit bên phải.

        Args:
            None.

        Returns:
            Widget Bottom Dock theo tỷ lệ carousel 60%, nhật ký 40%.
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)
        carousel_panel = QFrame()
        carousel_panel.setObjectName("panel")
        carousel_layout = QVBoxLayout(carousel_panel)
        carousel_layout.setContentsMargins(14, 12, 14, 12)
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("BẰNG CHỨNG GẦN ĐÂY"))
        self._snapshot_count = QLabel("0 ảnh")
        self._snapshot_count.setObjectName("muted")
        title_row.addStretch(1)
        title_row.addWidget(self._snapshot_count)
        carousel_layout.addLayout(title_row)

        self._carousel_scroll = QScrollArea()
        self._carousel_scroll.setWidgetResizable(False)
        self._carousel_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._carousel_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._carousel_widget = QWidget()
        self._carousel_layout = QHBoxLayout(self._carousel_widget)
        self._carousel_layout.setContentsMargins(4, 4, 4, 4)
        self._carousel_layout.setSpacing(10)
        self._carousel_scroll.setWidget(self._carousel_widget)
        carousel_layout.addWidget(self._carousel_scroll, 1)
        splitter.addWidget(carousel_panel)
        splitter.addWidget(self._build_audit_panel())
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([720, 480])
        return splitter

    def _build_audit_panel(self) -> QWidget:
        """Tạo bộ lọc ca/giờ, bảng sự kiện và nút xuất báo cáo.

        Args:
            None.

        Returns:
            Widget bảng nhật ký và thao tác báo cáo.
        """
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("LỌC"))
        self._shift_combo = QComboBox()
        self._shift_combo.addItem("Tất cả", None)
        self._shift_combo.addItem("Ca 1 · 06–14", 1)
        self._shift_combo.addItem("Ca 2 · 14–22", 2)
        self._shift_combo.addItem("Ca 3 · 22–06", 3)
        self._shift_combo.addItem("Khoảng giờ", "custom")
        self._shift_combo.currentIndexChanged.connect(self._on_filter_mode_changed)
        filter_row.addWidget(self._shift_combo)
        self._start_time_edit = QTimeEdit()
        self._start_time_edit.setDisplayFormat("HH:mm")
        self._start_time_edit.setTime(QTime.fromString("06:00", "HH:mm"))
        self._end_time_edit = QTimeEdit()
        self._end_time_edit.setDisplayFormat("HH:mm")
        self._end_time_edit.setTime(QTime.fromString("14:00", "HH:mm"))
        filter_row.addWidget(self._start_time_edit)
        filter_row.addWidget(QLabel("–"))
        filter_row.addWidget(self._end_time_edit)
        self._apply_filter_button = QPushButton("Lọc")
        self._apply_filter_button.clicked.connect(self._apply_audit_filter)
        filter_row.addWidget(self._apply_filter_button)
        layout.addLayout(filter_row)

        self._audit_table = QTableWidget(0, 4)
        self._audit_table.setHorizontalHeaderLabels(
            ["Mã", "Thời gian", "Vi phạm", "Tọa độ chân"]
        )
        self._audit_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._audit_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._audit_table.setAlternatingRowColors(True)
        self._audit_table.horizontalHeader().setStretchLastSection(True)
        self._audit_table.cellDoubleClicked.connect(self._on_table_double_clicked)
        layout.addWidget(self._audit_table, 1)

        footer = QHBoxLayout()
        self._audit_count = QLabel("0 sự kiện")
        self._audit_count.setObjectName("muted")
        footer.addWidget(self._audit_count)
        footer.addStretch(1)
        self._export_button = QPushButton("Xuất File Báo Cáo CSV")
        self._export_button.setObjectName("primaryButton")
        self._export_button.clicked.connect(self._export_csv)
        footer.addWidget(self._export_button)
        layout.addLayout(footer)
        self._on_filter_mode_changed()
        return panel

    @staticmethod
    def _section_separator() -> QFrame:
        """Tạo đường kẻ phân nhóm nhẹ trong Side Panel.

        Args:
            None.

        Returns:
            QFrame ngang dùng làm separator.
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("separator")
        return separator

    def _apply_styles(self) -> None:
        """Áp dụng theme ISA-101 sáng, màu đỏ đô và bo góc hiện đại.

        Args:
            None.

        Returns:
            None.
        """
        self.setStyleSheet(
            """
            QWidget {
                color: #24262B;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QMainWindow, QWidget#centralWidget { background: #F5F5F7; }
            QFrame#header, QFrame#panel {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
            }
            QLabel#brand {
                color: #9B1B30;
                font-size: 18pt;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#muted { color: #747985; }
            QLabel#metric, QLabel#statusSafe, QLabel#statusWarning,
            QLabel#statusAlarm {
                padding: 7px 11px;
                border-radius: 8px;
                background: #F1F5F4;
                font-weight: 700;
            }
            QLabel#statusSafe { color: #087C67; background: #E8F5F1; }
            QLabel#statusWarning { color: #9A5A00; background: #FFF4D6; }
            QLabel#statusAlarm { color: #FFFFFF; background: #E11D48; }
            QLabel#videoCanvas {
                border: 1px solid #30343B;
                border-radius: 8px;
            }
            QPushButton, QToolButton, QComboBox, QLineEdit, QTimeEdit {
                min-height: 30px;
                border: 1px solid #D8DAE0;
                border-radius: 7px;
                padding: 3px 8px;
                background: #FFFFFF;
            }
            QPushButton:hover, QToolButton:hover { border-color: #9B1B30; }
            QPushButton:disabled { color: #9A9DA5; background: #ECEEF1; }
            QPushButton#primaryButton {
                color: #FFFFFF;
                background: #9B1B30;
                border: 0;
                font-weight: 700;
                padding: 7px 12px;
            }
            QPushButton#primaryButton:hover { background: #7F1427; }
            QPushButton#ackButton {
                color: #FFFFFF;
                background: #9B1B30;
                border: 0;
                border-radius: 10px;
                font-size: 12pt;
                font-weight: 800;
            }
            QPushButton#ackButton:disabled {
                color: #888B92;
                background: #E7E8EB;
            }
            QComboBox::drop-down, QTimeEdit::up-button,
            QTimeEdit::down-button { border: 0; width: 18px; }
            QSlider::groove:horizontal {
                height: 5px;
                background: #E4E6EA;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 15px;
                margin: -5px 0;
                border-radius: 7px;
                background: #9B1B30;
            }
            QTableWidget {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                gridline-color: #ECEEF1;
                alternate-background-color: #F8F9FA;
                background: #FFFFFF;
            }
            QHeaderView::section {
                background: #F0F1F3;
                border: 0;
                padding: 5px;
                font-weight: 700;
            }
            QFrame#separator { color: #E5E7EB; }
            QScrollArea { border: 0; background: transparent; }
            """
        )

    def _load_source_form(self) -> None:
        """Nạp cấu hình nguồn và RTSP vào các trường trên Side Panel.

        Args:
            None.

        Returns:
            None.
        """
        source_config = self._config.get("video_source", {})
        mode = source_config.get("mode", "webcam")
        mode_index = {"webcam": 0, "file": 1, "rtsp": 2}.get(mode, 0)
        self._source_combo.setCurrentIndex(mode_index)
        self._file_path_edit.setText(source_config.get("file_path", ""))
        self._rtsp_url_edit.setText(source_config.get("rtsp_url", ""))
        self._rtsp_port_edit.setText(str(source_config.get("port", 554)))
        self._rtsp_user_edit.setText(source_config.get("username", ""))
        self._rtsp_password_edit.setText(source_config.get("password", ""))
        profile = source_config.get("profile", "ONVIF")
        profile_index = self._rtsp_profile_combo.findText(profile)
        if profile_index >= 0:
            self._rtsp_profile_combo.setCurrentIndex(profile_index)
        self._on_source_mode_changed()

    def _on_mode_changed(self, index: int) -> None:
        """Chuyển giao diện giữa thao tác vận hành và cài đặt kỹ thuật.

        Args:
            index: 0 cho vận hành, 1 cho cài đặt.

        Returns:
            None.
        """
        self._setup_mode = index == 1
        self._canvas.set_roi_overlay(
            self._draft_roi_points if self._setup_mode else self._roi_points,
            self._setup_mode,
        )
        for widget in (
            self._roi_controls,
            self._source_combo,
            self._file_source_group,
            self._rtsp_source_group,
            self._apply_source_button,
            self._min_area_slider,
            self._delay_slider,
        ):
            widget.setEnabled(self._setup_mode)
        self._on_source_mode_changed()
        self._roi_hint.setText(
            "ROI: ĐANG HIỆU CHUẨN" if self._setup_mode else "ROI: ĐÃ LƯU"
        )

    def _on_source_mode_changed(self) -> None:
        """Chỉ bật trường cấu hình phù hợp với nguồn video được chọn.

        Args:
            None.

        Returns:
            None.
        """
        mode_index = self._source_combo.currentIndex()
        self._file_source_group.setVisible(mode_index == 1)
        self._rtsp_source_group.setVisible(mode_index == 2)
        self._apply_source_button.setEnabled(self._setup_mode)

    def _browse_video_file(self) -> None:
        """Mở hộp thoại chọn file video MP4/AVI cho nguồn offline.

        Args:
            None.

        Returns:
            None.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video kiểm thử",
            self._file_path_edit.text(),
            "Video (*.mp4 *.avi)",
        )
        if file_path:
            self._file_path_edit.setText(file_path)

    def _build_rtsp_url(self) -> str:
        """Tạo URL RTSP từ URL nhập sẵn hoặc các thông tin camera.

        Args:
            None.

        Returns:
            URL RTSP chứa host, thông tin đăng nhập và path profile.

        Raises:
            ValueError: Khi URL, host hoặc port RTSP không hợp lệ.
        """
        raw_url = self._rtsp_url_edit.text().strip()
        port_text = self._rtsp_port_edit.text().strip()
        try:
            port = int(port_text)
        except ValueError as error:
            raise ValueError("Port RTSP phải là số nguyên.") from error
        if not 1 <= port <= 65535:
            raise ValueError("Port RTSP phải nằm trong khoảng 1 đến 65535.")

        if raw_url.startswith(("rtsp://", "rtsps://")):
            parsed = urlsplit(raw_url)
            if not parsed.hostname:
                raise ValueError("URL RTSP không có địa chỉ host hợp lệ.")
            host = parsed.hostname
            selected_port = parsed.port or port
            username = self._rtsp_user_edit.text().strip()
            password = self._rtsp_password_edit.text()
            if username:
                credentials = quote(username, safe="")
                if password:
                    credentials += f":{quote(password, safe='')}"
                netloc = f"{credentials}@{host}:{selected_port}"
            else:
                netloc = f"{host}:{selected_port}"
            return urlunsplit(
                (
                    parsed.scheme,
                    netloc,
                    parsed.path or self._rtsp_path_edit.text().strip(),
                    parsed.query,
                    parsed.fragment,
                )
            )

        host = raw_url.strip().strip("/")
        if not host:
            raise ValueError("Hãy nhập URL RTSP hoặc địa chỉ IP camera.")
        username = self._rtsp_user_edit.text().strip()
        password = self._rtsp_password_edit.text()
        credentials = ""
        if username:
            credentials = quote(username, safe="")
            if password:
                credentials += f":{quote(password, safe='')}"
            credentials += "@"

        path = self._rtsp_path_edit.text().strip()
        if not path.startswith("/"):
            path = f"/{path}"
        if self._rtsp_profile_combo.currentText() == "Dahua" and (
            path == "/Streaming/Channels/101"
        ):
            path = "/cam/realmonitor?channel=1&subtype=0"
        return f"rtsp://{credentials}{host}:{port}{path}"

    def _apply_video_source(self) -> None:
        """Lưu lựa chọn nguồn vào config và yêu cầu worker đổi nguồn nền.

        Args:
            None.

        Returns:
            None.

        Raises:
            OSError: Khi không thể lưu thông số nguồn vào config.json.
            ValueError: Khi đường dẫn video hoặc URL camera không hợp lệ.
            RuntimeError: Khi worker không thể nhận yêu cầu đổi nguồn.
        """
        mode_index = self._source_combo.currentIndex()
        source_config = dict(self._config.get("video_source", {}))
        if mode_index == 0:
            source: VideoSource = 0
            source_config.update({"mode": "webcam", "device_id": 0})
        elif mode_index == 1:
            file_path = self._file_path_edit.text().strip()
            if not file_path or Path(file_path).suffix.lower() not in {
                ".mp4",
                ".avi",
            }:
                QMessageBox.warning(
                    self,
                    "Nguồn video chưa hợp lệ",
                    "Hãy chọn một file video .mp4 hoặc .avi.",
                )
                return
            source = file_path
            source_config.update({"mode": "file", "file_path": file_path})
        else:
            try:
                rtsp_url = self._build_rtsp_url()
            except ValueError as error:
                QMessageBox.warning(self, "Cấu hình RTSP", str(error))
                return
            source = rtsp_url
            source_config.update(
                {
                    "mode": "rtsp",
                    "rtsp_url": rtsp_url,
                    "port": int(self._rtsp_port_edit.text()),
                    "username": self._rtsp_user_edit.text().strip(),
                    "password": self._rtsp_password_edit.text(),
                    "profile": self._rtsp_profile_combo.currentText(),
                }
            )

        try:
            self._config = self._config_manager.update_config(
                {"video_source": source_config}
            )
            self._worker.request_source_switch(source)
        except Exception as error:
            LOGGER.exception("Không thể áp dụng nguồn video.")
            QMessageBox.critical(
                self,
                "Không thể chuyển nguồn",
                f"Không lưu hoặc gửi được cấu hình nguồn:\n{error}",
            )
            return
        self._system_status.setText("ĐANG CHUYỂN NGUỒN…")

    def _on_display_changed(self, index: int) -> None:
        """Chuyển bộ lọc ảnh mà không dừng tiến trình kiểm tra an toàn.

        Args:
            index: Chỉ số chế độ RGB, MOG2 Mask hoặc Canny Edge.

        Returns:
            None.
        """
        mode = self._display_combo.itemData(index)
        if isinstance(mode, DisplayMode):
            self._worker.set_display_mode(mode)

    def _on_min_area_changed(self, value: int) -> None:
        """Cập nhật ngưỡng contour và lưu đồng bộ vào config.

        Args:
            value: Diện tích tối thiểu từ 500 đến 5000 px².

        Returns:
            None.
        """
        self._min_area_value.setText(f"{value} px²")
        try:
            self._cv_processor.set_min_area(float(value))
            self._tuning_save_timer.start()
        except ValueError:
            LOGGER.exception("Không thể áp dụng giá trị min_area=%s.", value)
            self._system_status.setText("LỖI LƯU MIN_AREA")

    def _on_delay_changed(self, slider_value: int) -> None:
        """Cập nhật thời gian trễ theo bước 0,1 giây và lưu cấu hình.

        Args:
            slider_value: Giá trị thanh trượt từ 0 đến 30, tương ứng 0–3 giây.

        Returns:
            None.
        """
        delay = slider_value / 10.0
        self._delay_value.setText(f"{delay:.1f} s")
        try:
            self._safety_controller.set_violation_delay(delay)
            self._tuning_save_timer.start()
        except ValueError:
            LOGGER.exception("Không thể áp dụng violation_delay=%.1f.", delay)
            self._system_status.setText("LỖI LƯU VIOLATION_DELAY")

    def _save_tuning_config(self) -> None:
        """Lưu gộp các ngưỡng sau khi thao tác kéo thanh trượt kết thúc.

        Args:
            None.

        Returns:
            None.
        """
        try:
            self._config = self._config_manager.update_config(
                {
                    "min_area": self._min_area_slider.value(),
                    "violation_delay": self._delay_slider.value() / 10.0,
                }
            )
        except (OSError, TypeError, ValueError):
            LOGGER.exception("Không thể lưu ngưỡng xử lý vào config.json.")
            self._system_status.setText("LỖI LƯU THAM SỐ")

    def _refresh_slider_labels(self) -> None:
        """Hiển thị giá trị hiện tại của hai thanh trượt kỹ thuật.

        Args:
            None.

        Returns:
            None.
        """
        self._min_area_value.setText(f"{self._min_area_slider.value()} px²")
        self._delay_value.setText(
            f"{self._delay_slider.value() / 10.0:.1f} s"
        )

    def _add_roi_point(self, x: int, y: int) -> None:
        """Thêm tọa độ click vào polygon ROI đang hiệu chuẩn.

        Args:
            x: Tọa độ ngang trong frame chuẩn 640×360.
            y: Tọa độ dọc trong frame chuẩn 640×360.

        Returns:
            None.
        """
        if not self._setup_mode:
            return
        self._draft_roi_points.append((x, y))
        self._canvas.set_roi_overlay(self._draft_roi_points, editing=True)
        self._roi_hint.setText(f"ROI: {len(self._draft_roi_points)} ĐỈNH")

    def _undo_roi_point(self) -> None:
        """Xóa đỉnh cuối cùng vừa thêm trong bản nháp ROI.

        Args:
            None.

        Returns:
            None.
        """
        if self._draft_roi_points:
            self._draft_roi_points.pop()
        self._canvas.set_roi_overlay(self._draft_roi_points, editing=True)
        self._roi_hint.setText(f"ROI: {len(self._draft_roi_points)} ĐỈNH")

    def _reset_roi_draft(self) -> None:
        """Khôi phục bản nháp ROI về đa giác đã lưu trong cấu hình.

        Args:
            None.

        Returns:
            None.
        """
        self._draft_roi_points = list(self._roi_points)
        self._canvas.set_roi_overlay(self._draft_roi_points, editing=True)
        self._roi_hint.setText(f"ROI: {len(self._draft_roi_points)} ĐỈNH")

    def _save_roi(self) -> None:
        """Lưu ROI có ít nhất ba đỉnh cho config và worker CV.

        Args:
            None.

        Returns:
            None.

        Raises:
            OSError: Khi không thể ghi config.json.
            ValueError: Khi ROI không đủ ba đỉnh.
        """
        if len(self._draft_roi_points) < 3:
            QMessageBox.warning(
                self,
                "ROI chưa hợp lệ",
                "Đa giác vùng cấm cần tối thiểu 3 đỉnh.",
            )
            return
        try:
            self._config = self._config_manager.update_config(
                {"roi_points": [list(point) for point in self._draft_roi_points]}
            )
        except (OSError, ValueError):
            LOGGER.exception("Không thể lưu polygon ROI.")
            QMessageBox.critical(
                self,
                "Không thể lưu ROI",
                "Đã xảy ra lỗi khi ghi ROI vào config.json.",
            )
            return

        self._roi_points = list(self._draft_roi_points)
        self._worker.set_roi_points(self._roi_points)
        self._canvas.set_roi_overlay(self._roi_points, editing=False)
        self._roi_hint.setText(f"ROI: ĐÃ LƯU · {len(self._roi_points)} ĐỈNH")

    def _acknowledge_alarm(self) -> None:
        """Gửi ACK để ngắt còi và nhả latch theo trạng thái SafetyController.

        Args:
            None.

        Returns:
            None.
        """
        try:
            acknowledged = self._safety_controller.acknowledge_alarm()
        except RuntimeError as error:
            LOGGER.exception("Không thể ACK báo động.")
            QMessageBox.warning(self, "ACK thất bại", str(error))
            return
        if not acknowledged:
            return
        self._ack_button.setEnabled(False)
        self._system_status.setText("ĐÃ XÁC NHẬN SỰ CỐ")

    def _on_filter_mode_changed(self, index: int | None = None) -> None:
        """Bật trường nhập giờ khi chọn lọc khoảng giờ tùy chỉnh.

        Args:
            index: Chỉ số combo; None dùng lựa chọn đang hiển thị.

        Returns:
            None.
        """
        del index
        custom_range = self._shift_combo.currentData() == "custom"
        self._start_time_edit.setEnabled(custom_range)
        self._end_time_edit.setEnabled(custom_range)

    def _apply_audit_filter(self) -> None:
        """Lọc đồng bộ bảng nhật ký và Snapshot Carousel theo cùng điều kiện.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: Khi mốc giờ tùy chọn không hợp lệ.
        """
        all_events = self._audit_logger.get_events()
        shift = self._shift_combo.currentData()
        try:
            if shift is None:
                filtered = all_events
            elif shift == "custom":
                filtered = self._audit_logger.filter_by_time_range(
                    self._start_time_edit.time().toString("HH:mm"),
                    self._end_time_edit.time().toString("HH:mm"),
                    all_events,
                )
            else:
                filtered = self._audit_logger.filter_by_shift(shift, all_events)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Bộ lọc thời gian", str(error))
            return

        self._filtered_events = filtered
        self._render_audit_table(filtered)
        self._refresh_carousel(filtered)

    def _render_audit_table(self, events: list[AuditEvent]) -> None:
        """Cập nhật bảng nhật ký bằng danh sách đã lọc hiện hành.

        Args:
            events: Danh sách AuditEvent cần hiển thị.

        Returns:
            None.
        """
        self._audit_table.setRowCount(len(events))
        for row, event in enumerate(reversed(events)):
            values = (
                event.event_id,
                event.occurred_at.strftime("%H:%M:%S"),
                f"{event.violation_duration:.1f}s",
                f"({event.foot_point[0]}, {event.foot_point[1]})",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, event.event_id)
                self._audit_table.setItem(row, column, item)
        self._audit_count.setText(f"{len(events)} sự kiện")

    def _refresh_carousel(self, events: list[AuditEvent]) -> None:
        """Đồng bộ thumbnail và nhãn thời gian/thời lượng với bộ lọc đang dùng.

        Args:
            events: Danh sách sự kiện đã lọc.

        Returns:
            None.
        """
        visible_events = [event for event in events if event.image_path.is_file()]
        fingerprint = tuple(
            (
                event.event_id,
                event.image_path.is_file(),
                event.violation_duration,
            )
            for event in events
        )
        if fingerprint == self._carousel_fingerprint:
            return
        self._carousel_fingerprint = fingerprint

        while self._carousel_layout.count():
            item = self._carousel_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for event in reversed(visible_events):
            card = QPushButton()
            card.setObjectName("snapshotCard")
            card.setFixedSize(188, 132)
            card_layout = QVBoxLayout(card)
            thumbnail = QLabel()
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap = QPixmap(str(event.image_path))
            if not pixmap.isNull():
                thumbnail.setPixmap(
                    pixmap.scaled(
                        164,
                        82,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                thumbnail.setText("Không đọc được ảnh")
            card_layout.addWidget(thumbnail)
            card_layout.addWidget(
                QLabel(
                    f"Ảnh {event.occurred_at:%H:%M:%S} | "
                    f"Vi phạm: {event.violation_duration:.1f}s"
                )
            )
            card.clicked.connect(
                lambda checked=False, selected=event: self._open_snapshot(selected)
            )
            self._carousel_layout.addWidget(card)

        self._carousel_layout.addStretch(1)
        self._carousel_widget.adjustSize()
        self._snapshot_count.setText(f"{len(visible_events)} ảnh")

    def _open_snapshot(self, event: AuditEvent) -> None:
        """Mở modal hiển thị ảnh bằng chứng đã lưu trên đĩa.

        Args:
            event: Sự kiện chứa đường dẫn và nhãn thời gian Snapshot.

        Returns:
            None.
        """
        try:
            dialog = SnapshotPreviewDialog(event, self)
            dialog.exec()
        except (OSError, cv2.error) as error:
            LOGGER.exception("Không thể mở ảnh bằng chứng %s.", event.event_id)
            QMessageBox.warning(
                self,
                "Không thể mở ảnh",
                f"Ảnh bằng chứng không đọc được:\n{error}",
            )

    def _on_table_double_clicked(self, row: int, column: int) -> None:
        """Mở ảnh bằng chứng tương ứng khi double-click dòng nhật ký.

        Args:
            row: Hàng đang được kích hoạt.
            column: Cột nhận double-click.

        Returns:
            None.
        """
        del column
        item = self._audit_table.item(row, 0)
        if item is None:
            return
        event_id = item.data(Qt.ItemDataRole.UserRole)
        event = next(
            (
                candidate
                for candidate in self._filtered_events
                if candidate.event_id == event_id
            ),
            None,
        )
        if event is not None:
            self._open_snapshot(event)

    def _export_csv(self) -> None:
        """Mở hộp thoại lưu CSV và xuất danh sách đang hiển thị.

        Args:
            None.

        Returns:
            None.
        """
        default_name = (
            f"Violation_Report_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất báo cáo vi phạm",
            str(self._audit_logger.reports_dir / default_name),
            "CSV (*.csv)",
        )
        if not target_path:
            return
        destination = Path(target_path)
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        try:
            exported_path = self._audit_logger.export_csv(
                events=self._filtered_events,
                destination=destination,
            )
        except (OSError, ValueError, TypeError) as error:
            LOGGER.exception("Không thể xuất báo cáo vi phạm.")
            QMessageBox.critical(
                self,
                "Xuất báo cáo thất bại",
                f"Chỉ lưu báo cáo bên trong output_data/reports:\n{error}",
            )
            return
        QMessageBox.information(
            self,
            "Xuất báo cáo thành công",
            f"Đã xuất {len(self._filtered_events)} sự kiện:\n{exported_path}",
        )

    def _refresh_hmi(self) -> None:
        """Đọc queue worker và cập nhật canvas, FPS, trạng thái và nút ACK.

        Args:
            None.

        Returns:
            None.
        """
        status = self._stream_manager.read_latest_status(timeout=0.0)
        if status is not None:
            self._update_system_status(status.state, status.message)

        packet = self._worker.take_latest_result()
        if packet is not None:
            if packet.error_message:
                self._system_status.setText("LỖI XỬ LÝ FRAME")
                LOGGER.error("HMI nhận lỗi worker: %s", packet.error_message)
            if packet.frame is not None and packet.analysis is not None:
                self._canvas.set_frame(packet.frame)
                self._fps_label.setText(f"FPS: {self._worker.fps:.1f}")
                self._people_label.setText(
                    f"ĐỐI TƯỢNG: {len(packet.analysis.detections)}"
                )
                if packet.safety is not None:
                    self._refresh_safety_status(packet.safety)

        self._ack_button.setEnabled(self._safety_controller.latched)

    def _update_system_status(self, state: StreamState, message: str) -> None:
        """Chuyển trạng thái nguồn camera thành nhãn ISA-101 dễ nhận biết.

        Args:
            state: Trạng thái kết nối của StreamManager.
            message: Thông báo trạng thái tiếng Việt từ stream.

        Returns:
            None.
        """
        if state is StreamState.CONNECTED:
            self._system_status.setObjectName("statusSafe")
        elif state in (StreamState.RECONNECTING, StreamState.ERROR):
            self._system_status.setObjectName("statusWarning")
        else:
            self._system_status.setObjectName("metric")
        self._system_status.setText(message)
        self._system_status.style().unpolish(self._system_status)
        self._system_status.style().polish(self._system_status)

    def _refresh_safety_status(self, safety: SafetyResult) -> None:
        """Cập nhật Header, ACK và màu ROI theo máy trạng thái an toàn.

        Args:
            safety: Snapshot trạng thái từ SafetyController.

        Returns:
            None.
        """
        stream_state = self._stream_manager.status.state
        canvas_state = (safety.state, stream_state, self._setup_mode)
        if canvas_state != self._last_canvas_state:
            overlay_points = (
                self._draft_roi_points if self._setup_mode else self._roi_points
            )
            self._canvas.set_roi_overlay(overlay_points, self._setup_mode)
            self._last_canvas_state = canvas_state

        if safety.state is SafetyState.ALARM:
            self._system_status.setObjectName("statusAlarm")
            label = "BÁO ĐỘNG ĐỎ"
        elif safety.state is SafetyState.WARNING:
            self._system_status.setObjectName("statusWarning")
            label = f"CẢNH BÁO SỚM · {safety.violation_duration:.1f}s"
        else:
            self._system_status.setObjectName("statusSafe")
            label = "AN TOÀN · SAFE"
        self._system_status.setText(label)
        self._system_status.style().unpolish(self._system_status)
        self._system_status.style().polish(self._system_status)
        self._ack_button.setEnabled(safety.latched)
        self._estop_status.setText(
            "E-STOP: ĐÃ KÍCH HOẠT"
            if safety.estop_triggered
            else "E-STOP: KHÔNG KÍCH HOẠT"
        )
        self._estop_status.setObjectName(
            "statusAlarm" if safety.estop_triggered else "muted"
        )
        self._estop_status.style().unpolish(self._estop_status)
        self._estop_status.style().polish(self._estop_status)

    def _refresh_audit_views(self) -> None:
        """Đồng bộ bảng và Carousel khi có sự kiện hoặc ảnh mới được ghi xong.

        Args:
            None.

        Returns:
            None.
        """
        current_events = self._audit_logger.get_events()
        shift = self._shift_combo.currentData()
        try:
            if shift is None:
                filtered = current_events
            elif shift == "custom":
                filtered = self._audit_logger.filter_by_time_range(
                    self._start_time_edit.time().toString("HH:mm"),
                    self._end_time_edit.time().toString("HH:mm"),
                    current_events,
                )
            else:
                filtered = self._audit_logger.filter_by_shift(shift, current_events)
        except (TypeError, ValueError):
            LOGGER.exception("Không thể cập nhật bộ lọc audit hiện thời.")
            return

        self._filtered_events = filtered
        self._render_audit_table(filtered)
        self._refresh_carousel(filtered)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Dừng worker CV, camera, còi và ghi ảnh trước khi đóng cửa sổ.

        Args:
            event: Sự kiện đóng cửa sổ do Qt gửi.

        Returns:
            None.
        """
        if self._shutdown_complete:
            event.accept()
            return
        self._ui_timer.stop()
        self._audit_timer.stop()
        save_tuning = self._tuning_save_timer.isActive()
        self._tuning_save_timer.stop()
        if save_tuning:
            self._save_tuning_config()
        self._worker.stop()
        shutdown_errors: list[str] = []

        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
            if self._worker.is_alive():
                LOGGER.error("Worker CV không dừng trong 5 giây.")
                event.ignore()
                QMessageBox.warning(
                    self,
                    "Đang dừng xử lý video",
                    "Luồng xử lý vẫn đang hoàn tất frame hiện tại. "
                    "Hãy đóng cửa sổ lần nữa sau ít giây.",
                )
                return

        try:
            self._stream_manager.stop_stream()
        except Exception as error:
            LOGGER.exception("Không thể dừng an toàn StreamManager.")
            shutdown_errors.append(f"Camera: {error}")
        try:
            self._safety_controller.close()
        except Exception as error:
            LOGGER.exception("Không thể dừng an toàn SafetyController.")
            shutdown_errors.append(f"Âm thanh: {error}")
        try:
            self._audit_logger.close()
        except Exception as error:
            LOGGER.exception("Không thể dừng an toàn AuditLogger.")
            shutdown_errors.append(f"Lưu bằng chứng: {error}")

        self._shutdown_complete = True
        event.accept()
        if shutdown_errors:
            QMessageBox.warning(
                self,
                "Cảnh báo khi đóng",
                "Ứng dụng đã đóng nhưng có thành phần chưa dừng sạch:\n"
                + "\n".join(shutdown_errors),
            )
