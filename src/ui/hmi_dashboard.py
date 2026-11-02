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
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
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
from src.utils.config_manager import DEFAULT_ROI_POINTS, ConfigManager


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
    occlusion_point_added = pyqtSignal(int, int)

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
        self._hud_lines: tuple[str, str] = ("", "")
        self._roi_points: list[ROIPoint] = []
        self._draft_roi_points: list[ROIPoint] = []
        self._editing_roi: bool = False
        self._occlusion_polygons: list[list[ROIPoint]] = []
        self._draft_occlusion_points: list[ROIPoint] = []
        self._editing_occlusion: bool = False
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

    def set_hud_lines(self, first_line: str, second_line: str) -> None:
        """Cập nhật hai dòng HUD phủ trên góc video."""
        self._hud_lines = (first_line, second_line)
        self.update()

    def set_roi_overlay(
        self,
        points: list[ROIPoint],
        editing: bool,
        draft_points: list[ROIPoint] | None = None,
        occlusion_polygons: list[list[ROIPoint]] | None = None,
        draft_occlusion_points: list[ROIPoint] | None = None,
        editing_occlusion: bool = False,
    ) -> None:
        """Cập nhật ROI, vùng che khuất và chế độ hiệu chuẩn đang hoạt động.

        Args:
            points: Các đỉnh ROI đã lưu hoặc các điểm đang vẽ.
            editing: True khi click trái được dùng để thêm đỉnh ROI.
            draft_points: Các đỉnh ROI mới, hiển thị cùng ROI đang hoạt động.
            occlusion_polygons: Các đa giác thân máy che lớp ROI mặt sàn.
            draft_occlusion_points: Đa giác che khuất đang được hiệu chuẩn.
            editing_occlusion: True khi click trái thêm đỉnh vùng che khuất.

        Returns:
            None.
        """
        self._roi_points = list(points)
        self._draft_roi_points = list(draft_points or [])
        self._editing_roi = editing
        self._occlusion_polygons = [
            list(polygon) for polygon in (occlusion_polygons or [])
        ]
        self._draft_occlusion_points = list(draft_occlusion_points or [])
        self._editing_occlusion = editing_occlusion
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if editing or editing_occlusion
            else Qt.CursorShape.ArrowCursor
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
        if any(self._hud_lines):
            hud_rect = QRectF(
                image_rect.x() + 10,
                image_rect.y() + 10,
                min(410, image_rect.width() - 20),
                58,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(30, 41, 59, 220))
            painter.drawRoundedRect(hud_rect, 10, 10)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(
                hud_rect.adjusted(12, 7, -8, -30),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._hud_lines[0],
            )
            painter.setPen(QColor("#D1D5DB"))
            painter.drawText(
                hud_rect.adjusted(12, 28, -8, -7),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._hud_lines[1],
            )
        if not self._roi_points and not self._draft_roi_points:
            return

        if self._editing_roi or self._editing_occlusion:
            active_points = [
                QPointF(
                    image_rect.x()
                    + point[0] * image_rect.width() / self.FRAME_WIDTH,
                    image_rect.y()
                    + point[1] * image_rect.height() / self.FRAME_HEIGHT,
                )
                for point in self._roi_points
            ]
            draft_points = [
                QPointF(
                    image_rect.x()
                    + point[0] * image_rect.width() / self.FRAME_WIDTH,
                    image_rect.y()
                    + point[1] * image_rect.height() / self.FRAME_HEIGHT,
                )
                for point in self._draft_roi_points
            ]
            clip_path = QPainterPath()
            clip_path.addRect(image_rect)
            for polygon in self._occlusion_polygons:
                if len(polygon) < 3:
                    continue
                mask_path = QPainterPath()
                mask_points = [
                    QPointF(
                        image_rect.x()
                        + point[0] * image_rect.width() / self.FRAME_WIDTH,
                        image_rect.y()
                        + point[1] * image_rect.height() / self.FRAME_HEIGHT,
                    )
                    for point in polygon
                ]
                mask_path.moveTo(mask_points[0])
                for point in mask_points[1:]:
                    mask_path.lineTo(point)
                mask_path.closeSubpath()
                clip_path = clip_path.subtracted(mask_path)

            painter.save()
            painter.setClipPath(clip_path, Qt.ClipOperation.IntersectClip)
            active_color = QColor("#28B4A7")
            painter.setPen(
                QPen(active_color, 2.0, Qt.PenStyle.DashLine)
            )
            if len(active_points) >= 2:
                for index in range(len(active_points) - 1):
                    painter.drawLine(
                        active_points[index],
                        active_points[index + 1],
                    )
                painter.drawLine(active_points[-1], active_points[0])
            painter.setBrush(active_color)
            painter.setPen(Qt.PenStyle.NoPen)
            for point in active_points:
                painter.drawEllipse(point, 4.0, 4.0)

            draft_color = QColor("#F59E0B")
            painter.setPen(
                QPen(draft_color, 2.0, Qt.PenStyle.DashLine)
            )
            if len(draft_points) >= 2:
                for index in range(len(draft_points) - 1):
                    painter.drawLine(
                        draft_points[index],
                        draft_points[index + 1],
                    )
                if len(draft_points) >= 3:
                    painter.drawLine(draft_points[-1], draft_points[0])
            painter.setBrush(draft_color)
            painter.setPen(Qt.PenStyle.NoPen)
            for point in draft_points:
                painter.drawEllipse(point, 4.0, 4.0)
            painter.restore()

        if self._editing_roi or self._editing_occlusion:
            for polygon in self._occlusion_polygons:
                if len(polygon) < 3:
                    continue
                mask_points = [
                    QPointF(
                        image_rect.x()
                        + point[0] * image_rect.width() / self.FRAME_WIDTH,
                        image_rect.y()
                        + point[1] * image_rect.height() / self.FRAME_HEIGHT,
                    )
                    for point in polygon
                ]
                painter.setBrush(QColor(168, 85, 247, 45))
                painter.setPen(
                    QPen(QColor("#A855F7"), 1.5, Qt.PenStyle.DashLine)
                )
                painter.drawPolygon(*mask_points)

        if self._editing_occlusion and self._draft_occlusion_points:
            draft_points = [
                QPointF(
                    image_rect.x()
                    + point[0] * image_rect.width() / self.FRAME_WIDTH,
                    image_rect.y()
                    + point[1] * image_rect.height() / self.FRAME_HEIGHT,
                )
                for point in self._draft_occlusion_points
            ]
            painter.setPen(QPen(QColor("#F59E0B"), 2.0, Qt.PenStyle.DashLine))
            for index in range(len(draft_points) - 1):
                painter.drawLine(draft_points[index], draft_points[index + 1])
            painter.setBrush(QColor("#F59E0B"))
            painter.setPen(Qt.PenStyle.NoPen)
            for point in draft_points:
                painter.drawEllipse(point, 4.0, 4.0)

    def mousePressEvent(self, event: object) -> None:
        """Phát tọa độ click hợp lệ trong frame khi đang hiệu chuẩn ROI.

        Args:
            event: Sự kiện chuột do Qt gửi đến widget.

        Returns:
            None.
        """
        if (
            (self._editing_roi or self._editing_occlusion)
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
                signal = (
                    self.occlusion_point_added
                    if self._editing_occlusion
                    else self.roi_point_added
                )
                signal.emit(frame_x, frame_y)
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
        occlusion_polygons: list[list[ROIPoint]],
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
        self._occlusion_polygons = [
            list(polygon) for polygon in occlusion_polygons
        ]
        self._display_mode = DisplayMode.RGB
        self._result_queue: queue.Queue[RenderPacket] = queue.Queue(maxsize=1)
        self._source_queue: queue.Queue[VideoSource] = queue.Queue(maxsize=1)
        self._active_event_id: str | None = None
        self._active_event_duration: float = 0.0
        self._previous_latched: bool = False
        self._last_fps: float = 0.0
        self._last_processing_ms: float = 0.0
        self._show_bounding_boxes: bool = True
        self._show_foot_points: bool = True

    @property
    def fps(self) -> float:
        """Trả về FPS gần nhất để Header hiển thị mà không đọc field nội bộ.

        Args:
            None.

        Returns:
            Số frame xử lý trung bình gần nhất mỗi giây.
        """
        return self._last_fps

    @property
    def processing_ms(self) -> float:
        """Trả thời gian xử lý frame gần nhất tính bằng mili giây."""
        return self._last_processing_ms

    def set_roi_points(self, points: list[ROIPoint]) -> None:
        """Cập nhật ROI đang sử dụng từ HMI mà không đụng vào frame đang đọc.

        Args:
            points: Tọa độ ROI mới theo kích thước 640×360.

        Returns:
            None.
        """
        with self._state_lock:
            self._roi_points = list(points)

    def set_occlusion_polygons(
        self,
        polygons: list[list[ROIPoint]],
    ) -> None:
        """Cập nhật các vùng thân máy cần loại khỏi lớp phủ ROI.

        Args:
            polygons: Các đa giác che khuất theo hệ tọa độ 640×360.

        Returns:
            None.
        """
        with self._state_lock:
            self._occlusion_polygons = [
                list(polygon) for polygon in polygons
            ]

    def set_display_mode(self, mode: DisplayMode) -> None:
        """Đặt chế độ xem áp dụng trên frame tiếp theo.

        Args:
            mode: Chế độ RGB, MOG2 Mask hoặc Canny Edge.

        Returns:
            None.
        """
        with self._state_lock:
            self._display_mode = mode

    def set_overlay_options(
        self,
        show_bounding_boxes: bool,
        show_foot_points: bool,
    ) -> None:
        """Cập nhật độc lập trạng thái hiển thị hộp bao và điểm chân."""
        with self._state_lock:
            self._show_bounding_boxes = show_bounding_boxes
            self._show_foot_points = show_foot_points

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
                    occlusion_polygons = [
                        list(polygon) for polygon in self._occlusion_polygons
                    ]
                    display_mode = self._display_mode
                    show_bounding_boxes = self._show_bounding_boxes
                    show_foot_points = self._show_foot_points

                frame_started = time.perf_counter()
                analysis = self._cv_processor.process_frame(
                    frame,
                    display_mode=display_mode,
                    roi_points=roi_points,
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
                    occlusion_polygons=occlusion_polygons,
                    show_bounding_boxes=show_bounding_boxes,
                    show_foot_points=show_foot_points,
                )
                self._last_processing_ms = (
                    time.perf_counter() - frame_started
                ) * 1000.0
                self._record_alarm_transition(
                    analysis,
                    safety,
                    foot_point,
                    roi_points,
                    occlusion_polygons,
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
        base_frame: np.ndarray | None = None,
        occlusion_polygons: list[list[ROIPoint]] | None = None,
        show_bounding_boxes: bool = True,
        show_foot_points: bool = True,
    ) -> np.ndarray:
        """Vẽ ROI và các lớp phủ đối tượng được bật lên frame.

        Args:
            analysis: Ảnh theo chế độ xem và các phát hiện đã tính.
            roi_points: Đỉnh đa giác ROI cần hiển thị.
            safety: Trạng thái an toàn quyết định màu viền vùng cấm.
            base_frame: Ảnh nguồn tùy chọn; None dùng chế độ hiển thị đang chọn.
            occlusion_polygons: Các vùng thân máy không phủ màu ROI.
            show_bounding_boxes: Bật/tắt khung chữ nhật cho từng đối tượng.
            show_foot_points: Bật/tắt điểm chân của từng đối tượng.

        Returns:
            Bản sao frame BGR có ROI và các lớp phủ được bật.
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
        if len(roi_points) >= 3:
            rendered = CVProcessor.blend_floor_roi(
                rendered,
                roi_points,
                occlusion_polygons or [],
                colors[safety.state],
                foreground_mask=analysis.mog2_frame,
            )

        for detection in analysis.detections:
            x, y, width, height = detection.bounding_box
            inside = safety.foot_point == detection.foot_point and safety.inside_roi
            object_color = (30, 30, 240) if inside else (35, 155, 75)
            if show_bounding_boxes:
                cv2.rectangle(
                    rendered,
                    (x, y),
                    (x + width, y + height),
                    object_color,
                    2,
                )
            if show_foot_points:
                cv2.circle(rendered, detection.foot_point, 5, object_color, -1)

        return rendered

    def _record_alarm_transition(
        self,
        analysis: FrameAnalysis,
        safety: SafetyResult,
        foot_point: ROIPoint | None,
        roi_points: list[ROIPoint],
        occlusion_polygons: list[list[ROIPoint]],
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
                        base_frame=analysis.rgb_frame,
                        occlusion_polygons=occlusion_polygons,
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


class SnapshotCardWidget(QFrame):
    """Thẻ Snapshot có thumbnail trực tiếp và khả năng mở ảnh gốc."""

    activated = pyqtSignal(object)

    def __init__(
        self,
        event: AuditEvent,
        parent: QWidget | None = None,
    ) -> None:
        """Tạo thẻ ảnh kiểm toán với thời gian, thời lượng và liên kết xem."""
        super().__init__(parent)
        self._event = event
        self.setObjectName("snapshotCard")
        self.setFixedSize(180, 138)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(5)

        image_container = QWidget()
        image_container.setFixedSize(164, 84)
        image_container.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        image_layout = QGridLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image = QLabel()
        image.setObjectName("snapshotThumbnail")
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setFixedSize(164, 84)
        pixmap = QPixmap(str(event.image_path))
        if pixmap.isNull():
            image.setText("Không đọc được ảnh")
        else:
            image.setPixmap(
                pixmap.scaled(
                    image.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        image.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        image_layout.addWidget(image, 0, 0)
        badge = QLabel(f"[Ảnh {event.occurred_at:%H:%M:%S}]")
        badge.setObjectName("snapshotBadge")
        badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        badge.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        image_layout.addWidget(
            badge,
            0,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(image_container)

        footer = QHBoxLayout()
        duration = QLabel(f"Vi phạm: {event.violation_duration:.1f}s")
        duration.setObjectName("snapshotDuration")
        view_label = QLabel("Xem →")
        view_label.setObjectName("snapshotLink")
        duration.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        view_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        footer.addWidget(duration)
        footer.addStretch(1)
        footer.addWidget(view_label)
        layout.addLayout(footer)

    def mousePressEvent(self, event: object) -> None:
        """Chấp nhận thao tác nhấn trái để nhận click trên toàn bộ thẻ."""
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: object) -> None:
        """Mở ảnh khi người dùng click vào bất kỳ vị trí nào trên thẻ."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.activated.emit(self._event)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class HMIMainWindow(QMainWindow):
    """Cửa sổ HMI chính kết nối stream, vision, safety, config và audit."""

    webcams_scanned = pyqtSignal(list, str)
    camera_test_completed = pyqtSignal(bool, str)

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
        self._active_roi: list[ROIPoint] = [
            (int(point[0]), int(point[1]))
            for point in self._config["roi_points"]
        ]
        self._occlusion_polygons: list[list[ROIPoint]] = [
            [
                (int(point[0]), int(point[1]))
                for point in polygon
            ]
            for polygon in self._config.get("occlusion_polygons", [])
        ]
        self._draft_roi: list[ROIPoint] = []
        self._drawing_roi = False
        self._draft_occlusion_points: list[ROIPoint] = []
        self._editing_occlusion = False
        self._webcam_scan_running = False
        self._pending_auto_source_switch = False
        self._setup_mode = False
        self._filtered_events: list[AuditEvent] = audit_logger.get_events()
        self._carousel_fingerprint: tuple[tuple[str, bool, float], ...] = ()
        self._audit_table_fingerprint: tuple[tuple[str, float], ...] = ()
        self._last_canvas_state: tuple[SafetyState, StreamState, bool] | None = None
        self._shutdown_complete = False

        self.setWindowTitle("VisionGuard AI | Industrial Safety HMI")
        self.resize(1500, 940)
        self.setMinimumSize(1120, 760)
        self.webcams_scanned.connect(self._on_webcams_scanned)
        self.camera_test_completed.connect(self._on_camera_test_completed)
        self._build_ui()
        self._apply_styles()
        self._load_source_form()

        self._worker = _CVSafetyWorker(
            stream_manager,
            cv_processor,
            safety_controller,
            audit_logger,
            self._active_roi,
            self._occlusion_polygons,
        )
        self._worker.set_overlay_options(
            self._show_bounding_box_checkbox.isChecked(),
            self._show_foot_point_checkbox.isChecked(),
        )
        self._worker.start()
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._refresh_hmi)
        self._ui_timer.start()
        self._audit_timer = QTimer(self)
        self._audit_timer.setInterval(120)
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
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(10)
        central_widget.setObjectName("centralWidget")
        self._camera_dialog = self._build_camera_connection_dialog()
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
        """Tạo header ViewLab với trạng thái nguồn và các thao tác chính.

        Args:
            None.

        Returns:
            Widget Header hoàn chỉnh.
        """
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        logo = QLabel("VL")
        logo.setObjectName("viewLabLogo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(46, 46)
        layout.addWidget(logo)

        brand_stack = QVBoxLayout()
        brand_stack.setSpacing(2)
        title = QLabel(
            "VIEWLAB ENGINEERING | VISIONGUARD HMI ISA-101 Standard"
        )
        title.setObjectName("brand")
        self._system_status = QLabel("ĐANG KHỞI TẠO NGUỒN VIDEO")
        self._system_status.setObjectName("sourceStatus")
        brand_stack.addWidget(title)
        brand_stack.addWidget(self._system_status)
        layout.addLayout(brand_stack, 3)

        self._live_header_label = QLabel("Màn hình Live Stream (Mắt Camera IP)")
        self._live_header_label.setObjectName("liveHeader")
        self._live_header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._live_header_label, 2)

        self._ack_button = QPushButton("XÁC NHẬN / TẮT CÒI")
        self._ack_button.setObjectName("ackButton")
        self._ack_button.setMinimumHeight(40)
        self._ack_button.setEnabled(False)
        self._ack_button.clicked.connect(self._acknowledge_alarm)
        layout.addWidget(self._ack_button)

        self._mode_button = QPushButton("Chế độ: Giám sát (Khóa)")
        self._mode_button.setObjectName("modeButton")
        self._mode_button.setCheckable(True)
        self._mode_button.toggled.connect(
            lambda checked: self._on_mode_changed(1 if checked else 0)
        )
        layout.addWidget(self._mode_button)

        self._connect_camera_button = QPushButton("Kết nối Camera IP")
        self._connect_camera_button.setObjectName("primaryButton")
        self._connect_camera_button.clicked.connect(self._open_camera_dialog)
        layout.addWidget(self._connect_camera_button)
        return header

    def _build_live_and_side_panel(self) -> QWidget:
        """Tạo canvas Live Stream và ba thẻ điều khiển an toàn.

        Args:
            None.

        Returns:
            Widget chứa canvas video và Side Panel.
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)
        video_panel = QFrame()
        video_panel.setObjectName("panel")
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("LIVE STREAM"))
        title_row.addStretch(1)
        self._roi_hint = QLabel("ROI: ĐÃ LƯU")
        self._roi_hint.setObjectName("muted")
        title_row.addWidget(self._roi_hint)
        video_layout.addLayout(title_row)

        self._canvas = VideoCanvas()
        self._canvas.setObjectName("videoCanvas")
        self._canvas.roi_point_added.connect(self._add_roi_point)
        self._canvas.occlusion_point_added.connect(self._add_occlusion_point)
        video_layout.addWidget(self._canvas, 1)
        self._roi_controls = self._build_roi_controls()
        video_layout.addWidget(self._roi_controls)
        self._occlusion_controls = self._build_occlusion_controls()
        video_layout.addWidget(self._occlusion_controls)
        splitter.addWidget(video_panel)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([1000, 420])
        return splitter

    def _build_roi_controls(self) -> QWidget:
        """Tạo đúng ba nút thao tác ROI theo quy trình hiệu chuẩn.

        Args:
            None.

        Returns:
            Widget thanh nút hiệu chuẩn ROI.
        """
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        self._draw_roi_button = QPushButton(
            "Vẽ lại Vùng cấm (Click chuột)"
        )
        self._delete_roi_point_button = QPushButton("Xóa điểm")
        self._default_roi_button = QPushButton("Mặc định")
        self._draw_roi_button.setObjectName("primaryButton")
        self._draw_roi_button.clicked.connect(self._toggle_roi_drawing)
        self._delete_roi_point_button.clicked.connect(self._delete_roi_point)
        self._default_roi_button.clicked.connect(self._restore_default_roi)
        for button in (
            self._draw_roi_button,
            self._delete_roi_point_button,
            self._default_roi_button,
        ):
            button.setMinimumHeight(38)
            layout.addWidget(button, 1)
        return controls

    def _build_occlusion_controls(self) -> QWidget:
        """Tạo nhóm hiệu chuẩn mask máy riêng, không lẫn với ba nút ROI."""
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("MASK THÂN MÁY"))
        self._draw_occlusion_button = QPushButton("Vẽ vùng che máy")
        self._undo_occlusion_button = QPushButton("Undo mask")
        self._save_occlusion_button = QPushButton("Lưu mask")
        self._clear_occlusion_button = QPushButton("Xóa mask")
        self._save_occlusion_button.setObjectName("secondaryButton")
        self._draw_occlusion_button.clicked.connect(self._start_occlusion_draft)
        self._undo_occlusion_button.clicked.connect(self._undo_occlusion_point)
        self._save_occlusion_button.clicked.connect(self._save_occlusion_polygon)
        self._clear_occlusion_button.clicked.connect(self._clear_occlusion_polygons)
        for button in (
            self._draw_occlusion_button,
            self._undo_occlusion_button,
            self._save_occlusion_button,
            self._clear_occlusion_button,
        ):
            layout.addWidget(button)
        layout.addStretch(1)
        return controls

    def _build_side_panel(self) -> QWidget:
        """Tạo ba thẻ thông số, cấu hình thuật toán và tùy chọn HUD.

        Args:
            None.

        Returns:
            Side Panel cuộn được khi cửa sổ có chiều cao nhỏ.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        operations_card = QFrame()
        operations_card.setObjectName("controlCard")
        operations_layout = QVBoxLayout(operations_card)
        operations_layout.setContentsMargins(12, 10, 12, 10)
        operations_layout.setSpacing(8)
        state_header = QHBoxLayout()
        state_title = QLabel("1. BẢNG THÔNG SỐ VẬN HÀNH")
        state_title.setObjectName("cardTitle")
        state_title.setWordWrap(True)
        state_header.addWidget(state_title)
        state_header.addStretch(1)
        self._safety_badge = QLabel("AN TOÀN")
        self._safety_badge.setObjectName("statusSafe")
        state_header.addWidget(self._safety_badge)
        operations_layout.addLayout(state_header)

        metric_grid = QGridLayout()
        metric_grid.setSpacing(6)
        self._people_value = QLabel("00")
        self._shift_alarms_value = QLabel("00")
        self._machine_state_value = QLabel("ĐANG CHẠY")
        self._add_metric_box(
            metric_grid, 0, "NGƯỜI TRONG KHUNG", self._people_value
        )
        self._add_metric_box(
            metric_grid, 1, "CẢNH BÁO CA NÀY", self._shift_alarms_value
        )
        self._add_metric_box(
            metric_grid, 2, "TRẠNG THÁI MÁY", self._machine_state_value
        )
        operations_layout.addLayout(metric_grid)
        self._intrusion_buffer_label = QLabel(
            "Bộ đệm thời gian xâm nhập: 0.00s / 0.80s"
        )
        self._intrusion_buffer_label.setWordWrap(True)
        operations_layout.addWidget(self._intrusion_buffer_label)
        self._intrusion_progress = QProgressBar()
        self._intrusion_progress.setRange(0, 1000)
        self._intrusion_progress.setValue(0)
        self._intrusion_progress.setTextVisible(False)
        self._intrusion_progress.setFixedHeight(8)
        operations_layout.addWidget(self._intrusion_progress)
        self._estop_status = QLabel("E-STOP: KHÔNG KÍCH HOẠT")
        self._estop_status.setObjectName("muted")
        operations_layout.addWidget(self._estop_status)
        layout.addWidget(operations_card)

        tuning_card = QFrame()
        tuning_card.setObjectName("controlCard")
        tuning_layout = QVBoxLayout(tuning_card)
        tuning_layout.setContentsMargins(12, 10, 12, 10)
        tuning_layout.setSpacing(6)
        tuning_title = QLabel("2. CẤU HÌNH THUẬT TOÁN & CAMERA")
        tuning_title.setObjectName("cardTitle")
        tuning_title.setWordWrap(True)
        tuning_layout.addWidget(tuning_title)
        tuning_layout.addWidget(QLabel("Độ trễ báo động (s)"))
        self._delay_value = QLabel()
        self._delay_slider = QSlider(Qt.Orientation.Horizontal)
        self._delay_slider.setRange(0, 30)
        self._delay_slider.setSingleStep(1)
        self._delay_slider.setValue(
            round(float(self._config["violation_delay"]) * 10)
        )
        self._delay_slider.valueChanged.connect(self._on_delay_changed)
        tuning_layout.addWidget(self._delay_slider)
        tuning_layout.addWidget(self._delay_value)
        tuning_layout.addWidget(QLabel("Ngưỡng minArea (px²)"))
        self._min_area_value = QLabel()
        self._min_area_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_area_slider.setRange(500, 5000)
        self._min_area_slider.setSingleStep(100)
        self._min_area_slider.setValue(int(self._config["min_area"]))
        self._min_area_slider.valueChanged.connect(self._on_min_area_changed)
        tuning_layout.addWidget(self._min_area_slider)
        tuning_layout.addWidget(self._min_area_value)
        reset_parameters_button = QPushButton("Reset chuẩn")
        reset_parameters_button.setObjectName("secondaryButton")
        reset_parameters_button.clicked.connect(self._reset_standard_parameters)
        tuning_layout.addWidget(reset_parameters_button)
        self._tuning_save_timer = QTimer(self)
        self._tuning_save_timer.setSingleShot(True)
        self._tuning_save_timer.setInterval(300)
        self._tuning_save_timer.timeout.connect(self._save_tuning_config)
        layout.addWidget(tuning_card)

        overlay_card = QFrame()
        overlay_card.setObjectName("controlCard")
        overlay_layout = QVBoxLayout(overlay_card)
        overlay_layout.setContentsMargins(12, 10, 12, 10)
        overlay_layout.setSpacing(6)
        overlay_title = QLabel("3. BỘ LỌC HIỂN THỊ (HUD OVERLAY)")
        overlay_title.setObjectName("cardTitle")
        overlay_title.setWordWrap(True)
        overlay_layout.addWidget(overlay_title)
        overlay_options = self._config.get("display_overlays", {})
        self._show_bounding_box_checkbox = QCheckBox("Khung Bounding Box")
        self._show_bounding_box_checkbox.setChecked(
            bool(overlay_options.get("bounding_box", True))
        )
        self._show_foot_point_checkbox = QCheckBox("Tọa độ Điểm chân")
        self._show_foot_point_checkbox.setChecked(
            bool(overlay_options.get("foot_point", True))
        )
        self._show_bounding_box_checkbox.toggled.connect(
            self._on_overlay_options_changed
        )
        self._show_foot_point_checkbox.toggled.connect(
            self._on_overlay_options_changed
        )
        overlay_layout.addWidget(self._show_bounding_box_checkbox)
        overlay_layout.addWidget(self._show_foot_point_checkbox)
        self._display_combo = QComboBox()
        self._display_combo.addItem("RGB Thực tế", DisplayMode.RGB)
        self._display_combo.addItem("MOG2 Mask", DisplayMode.MOG2_MASK)
        self._display_combo.addItem("Canny Edge", DisplayMode.CANNY_EDGE)
        self._display_combo.currentIndexChanged.connect(self._on_display_changed)
        overlay_layout.addWidget(self._display_combo)
        layout.addWidget(overlay_card)
        layout.addStretch(1)
        self._refresh_slider_labels()
        self._side_panel = QScrollArea()
        self._side_panel.setWidgetResizable(True)
        self._side_panel.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._side_panel.setFrameShape(QFrame.Shape.NoFrame)
        self._side_panel.setWidget(panel)
        return self._side_panel

    @staticmethod
    def _add_metric_box(
        layout: QGridLayout,
        column: int,
        title: str,
        value_label: QLabel,
    ) -> None:
        """Đặt một chỉ số vận hành trong ô nền trung tính."""
        box = QFrame()
        box.setObjectName("metricBox")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(8, 6, 8, 6)
        box_layout.setSpacing(2)
        caption = QLabel(title)
        caption.setObjectName("metricCaption")
        caption.setWordWrap(True)
        value_label.setObjectName("metricValue")
        box_layout.addWidget(caption)
        box_layout.addWidget(value_label)
        layout.addWidget(box, column, 0)

    def _build_camera_connection_dialog(self) -> QDialog:
        """Tạo modal cấu hình Webcam, file video hoặc RTSP camera."""
        dialog = QDialog(self)
        dialog.setObjectName("cameraDialog")
        dialog.setWindowTitle("Cấu hình Kết nối Camera IP (RTSP Stream)")
        dialog.setModal(True)
        dialog.setMinimumWidth(620)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        title = QLabel("Cấu hình Kết nối Camera IP")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)

        self._source_combo = QComboBox()
        self._source_combo.addItem(
            "Webcam máy tính (Tự động dò 0, 1, 2...)", "webcam"
        )
        self._source_combo.addItem(
            "Tệp Video thực nghiệm (.mp4, .avi)", "file"
        )
        self._source_combo.addItem(
            "Camera IP qua RTSP (Mạng LAN nội bộ)", "rtsp"
        )
        source_form = QFormLayout()
        source_form.addRow("Chế độ Nguồn (INPUT_SOURCE)", self._source_combo)
        layout.addLayout(source_form)
        self._source_combo.currentIndexChanged.connect(
            self._on_source_mode_changed
        )

        self._webcam_device_combo = QComboBox()
        self._webcam_device_combo.addItem("Tự động dò (Auto)", "auto")
        configured_device = self._config.get("video_source", {}).get(
            "device_id", 0
        )
        if isinstance(configured_device, int) and not isinstance(
            configured_device, bool
        ):
            self._webcam_device_combo.addItem(
                f"Webcam {configured_device}", configured_device
            )
        self._scan_webcams_button = QPushButton("Dò webcam")
        self._scan_webcams_button.clicked.connect(self._request_webcam_scan)
        self._webcam_source_group = QWidget()
        webcam_row = QHBoxLayout(self._webcam_source_group)
        webcam_row.setContentsMargins(0, 0, 0, 0)
        webcam_row.addWidget(self._webcam_device_combo, 1)
        webcam_row.addWidget(self._scan_webcams_button)
        layout.addWidget(self._webcam_source_group)

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
        self._rtsp_url_edit.setPlaceholderText("Địa chỉ IP hoặc URL RTSP")
        self._rtsp_port_edit = QLineEdit("554")
        self._rtsp_port_edit.setMaximumWidth(110)
        self._rtsp_user_edit = QLineEdit()
        self._rtsp_user_edit.setPlaceholderText("Tài khoản")
        self._rtsp_password_edit = QLineEdit()
        self._rtsp_password_edit.setPlaceholderText("Mật khẩu")
        self._rtsp_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._rtsp_profile_combo = QComboBox()
        self._rtsp_profile_combo.addItems(
            ["Dahua/Imou", "Hikvision/Ezviz", "ONVIF"]
        )
        self._rtsp_path_edit = QLineEdit("/Streaming/Channels/101")
        self._rtsp_source_group = QWidget()
        rtsp_form = QFormLayout(self._rtsp_source_group)
        rtsp_form.addRow("Profile Hãng Camera", self._rtsp_profile_combo)
        rtsp_form.addRow("Địa chỉ IP Camera / URL", self._rtsp_url_edit)
        rtsp_form.addRow("Cổng RTSP", self._rtsp_port_edit)
        rtsp_form.addRow("Tài khoản", self._rtsp_user_edit)
        rtsp_form.addRow("Mật khẩu", self._rtsp_password_edit)
        rtsp_form.addRow("RTSP Path", self._rtsp_path_edit)
        for field in (
            self._rtsp_url_edit,
            self._rtsp_port_edit,
            self._rtsp_user_edit,
            self._rtsp_password_edit,
            self._rtsp_path_edit,
        ):
            field.textChanged.connect(self._update_rtsp_preview)
        self._rtsp_profile_combo.currentIndexChanged.connect(
            self._update_rtsp_preview
        )
        layout.addWidget(self._rtsp_source_group)

        layout.addWidget(QLabel("Chuỗi kết nối OpenCV (cv2.VideoCapture)"))
        self._rtsp_preview_edit = QLineEdit()
        self._rtsp_preview_edit.setReadOnly(True)
        self._rtsp_preview_edit.setObjectName("connectionPreview")
        layout.addWidget(self._rtsp_preview_edit)
        self._camera_test_status = QLabel("Chưa kiểm tra kết nối.")
        self._camera_test_status.setObjectName("muted")
        layout.addWidget(self._camera_test_status)

        buttons = QHBoxLayout()
        self._camera_test_button = QPushButton("Kiểm tra kết nối (Ping)")
        self._camera_test_button.setObjectName("secondaryButton")
        self._camera_test_button.clicked.connect(
            self._test_camera_connection
        )
        close_button = QPushButton("Đóng")
        close_button.clicked.connect(dialog.reject)
        self._apply_source_button = QPushButton("Lưu & Áp dụng")
        self._apply_source_button.setObjectName("primaryButton")
        self._apply_source_button.clicked.connect(self._apply_video_source)
        buttons.addWidget(self._camera_test_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        buttons.addWidget(self._apply_source_button)
        layout.addLayout(buttons)
        return dialog

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
        """Áp dụng bảng màu ViewLab sáng, phẳng và theo ISA-101."""
        self.setStyleSheet(
            """
            QWidget {
                color: #1E293B;
                font-family: "Segoe UI";
                font-size: 9pt;
            }
            QMainWindow, QWidget#centralWidget { background: #F1F5F9; }
            QFrame#header, QFrame#panel, QFrame#controlCard,
            QDialog#cameraDialog {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 14px;
            }
            QFrame#controlCard { border-radius: 12px; }
            QLabel#viewLabLogo {
                color: #FFFFFF;
                background: #9B1B30;
                border-radius: 23px;
                font-size: 16pt;
                font-weight: 800;
            }
            QLabel#brand {
                color: #9B1B30;
                font-size: 11pt;
                font-weight: 800;
            }
            QLabel#liveHeader {
                color: #334155;
                font-size: 11pt;
                font-weight: 700;
            }
            QLabel#cardTitle {
                color: #334155;
                font-weight: 800;
            }
            QLabel#sourceStatus, QLabel#muted { color: #64748B; }
            QLabel#sourceStatusOnline { color: #10B981; font-weight: 700; }
            QLabel#sourceStatusError { color: #B45309; font-weight: 700; }
            QLabel#statusSafe, QLabel#statusWarning, QLabel#statusAlarm {
                padding: 5px 10px;
                border-radius: 8px;
                font-weight: 700;
            }
            QLabel#statusSafe { color: #FFFFFF; background: #10B981; }
            QLabel#statusWarning { color: #B45309; background: #FEF3C7; }
            QLabel#statusAlarm { color: #FFFFFF; background: #E11D48; }
            QLabel#videoCanvas {
                border: 1px solid #CBD5E1;
                border-radius: 10px;
            }
            QPushButton, QToolButton, QComboBox, QLineEdit, QTimeEdit {
                min-height: 32px;
                border: 1px solid #CBD5E1;
                border-radius: 9px;
                padding: 4px 9px;
                background: #FFFFFF;
            }
            QPushButton:hover, QToolButton:hover { border-color: #9B1B30; }
            QPushButton:disabled { color: #94A3B8; background: #F1F5F9; }
            QPushButton#primaryButton {
                color: #FFFFFF;
                background: #9B1B30;
                border: 0;
                font-weight: 700;
                padding: 7px 11px;
            }
            QPushButton#primaryButton:hover { background: #7F1427; }
            QPushButton#ackButton {
                color: #FFFFFF;
                background: #D97706;
                border: 0;
                border-radius: 9px;
                font-weight: 800;
            }
            QPushButton#ackButton:hover { background: #B45309; }
            QPushButton#ackButton:disabled {
                color: #94A3B8;
                background: #E2E8F0;
            }
            QPushButton#modeButton:checked {
                color: #FFFFFF;
                background: #334155;
            }
            QPushButton#secondaryButton {
                color: #334155;
                background: #F8FAFC;
                border: 1px solid #CBD5E1;
            }
            QLabel#metricCaption {
                color: #64748B;
                font-size: 7pt;
                font-weight: 700;
            }
            QLabel#metricValue {
                color: #0F172A;
                font-size: 12pt;
                font-weight: 800;
            }
            QFrame#metricBox {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 9px;
            }
            QLabel#snapshotBadge {
                color: #FFFFFF;
                background: #9B1B30;
                border-radius: 6px;
                padding: 3px 6px;
                font-size: 8pt;
                font-weight: 700;
            }
            QFrame#snapshotCard {
                background: #FFFFFF;
                border: 1px solid #F3C7CE;
                border-radius: 12px;
            }
            QFrame#snapshotCard:hover {
                border: 1px solid #9B1B30;
                background: #FFF8F8;
            }
            QLabel#snapshotThumbnail {
                color: #64748B;
                background: #E2E8F0;
                border-radius: 8px;
            }
            QLabel#snapshotDuration {
                color: #334155;
                font-size: 8pt;
                font-weight: 700;
            }
            QLabel#snapshotLink {
                color: #9B1B30;
                font-size: 8pt;
                font-weight: 800;
            }
            QLabel#dialogTitle {
                color: #9B1B30;
                font-size: 14pt;
                font-weight: 800;
            }
            QLineEdit#connectionPreview {
                color: #E2E8F0;
                background: #0F172A;
                border: 0;
                font-family: Consolas;
            }
            QComboBox::drop-down, QTimeEdit::up-button,
            QTimeEdit::down-button { border: 0; width: 18px; }
            QSlider::groove:horizontal {
                height: 5px;
                background: #E2E8F0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 15px;
                margin: -5px 0;
                border-radius: 7px;
                background: #9B1B30;
            }
            QTableWidget {
                border: 1px solid #E2E8F0;
                border-radius: 9px;
                gridline-color: #F1F5F9;
                alternate-background-color: #F8FAFC;
                background: #FFFFFF;
            }
            QHeaderView::section {
                background: #F1F5F9;
                border: 0;
                padding: 5px;
                font-weight: 700;
            }
            QProgressBar {
                border: 0;
                border-radius: 4px;
                background: #E2E8F0;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: #D97706;
            }
            QScrollArea { border: 0; background: transparent; }
            """
        )

    def _load_source_form(self) -> None:
        """Nạp cấu hình nguồn vào modal cấu hình camera hiện hành.

        Args:
            None.

        Returns:
            None.
        """
        source_config = self._config.get("video_source", {})
        mode = source_config.get("mode", "webcam")
        mode_index = {"webcam": 0, "file": 1, "rtsp": 2}.get(mode, 0)
        self._source_combo.setCurrentIndex(mode_index)
        device_id = source_config.get("device_id", 0)
        device_index = self._webcam_device_combo.findData(device_id)
        if device_index >= 0:
            self._webcam_device_combo.setCurrentIndex(device_index)
        self._file_path_edit.setText(source_config.get("file_path", ""))
        self._rtsp_url_edit.setText(source_config.get("rtsp_url", ""))
        self._rtsp_port_edit.setText(str(source_config.get("port", 554)))
        self._rtsp_user_edit.setText(source_config.get("username", ""))
        self._rtsp_password_edit.setText(source_config.get("password", ""))
        profile = source_config.get("profile", "ONVIF")
        profile_labels = {
            "Dahua": "Dahua/Imou",
            "Hikvision": "Hikvision/Ezviz",
            "ONVIF": "ONVIF",
        }
        profile_index = self._rtsp_profile_combo.findText(
            profile_labels.get(profile, profile)
        )
        if profile_index >= 0:
            self._rtsp_profile_combo.setCurrentIndex(profile_index)
        self._on_source_mode_changed()
        self._update_rtsp_preview()

    def _open_camera_dialog(self) -> None:
        """Hiển thị hộp thoại cấu hình camera ở chế độ modal."""
        self._camera_dialog.exec()

    def _update_rtsp_preview(self, index: int | None = None) -> None:
        """Cập nhật chuỗi VideoCapture theo dữ liệu RTSP đang nhập."""
        del index
        if self._source_combo.currentData() != "rtsp":
            self._rtsp_preview_edit.setText(
                "Chọn Camera IP qua RTSP để tạo chuỗi kết nối."
            )
            return
        try:
            self._rtsp_preview_edit.setText(self._build_rtsp_url())
        except ValueError:
            self._rtsp_preview_edit.setText("Nhập IP/URL và cổng RTSP hợp lệ.")

    def _test_camera_connection(self) -> None:
        """Thử mở và đọc frame camera/file ngoài luồng giao diện."""
        mode = self._source_combo.currentData()
        if mode == "webcam":
            device = self._webcam_device_combo.currentData()
            source: VideoSource | None = (
                device if isinstance(device, int) and not isinstance(device, bool)
                else None
            )
            if device != "auto" and source is None:
                self._camera_test_status.setText(
                    "Chọn webcam hợp lệ để kiểm tra."
                )
                return
        elif mode == "file":
            source = self._file_path_edit.text().strip()
            if not source or not Path(source).is_file():
                self._camera_test_status.setText("Tệp video không tồn tại.")
                return
        else:
            try:
                source = self._build_rtsp_url()
            except ValueError as error:
                self._camera_test_status.setText(str(error))
                return

        self._camera_test_button.setEnabled(False)
        self._camera_test_status.setText("ĐANG KIỂM TRA NGUỒN VIDEO…")
        threading.Thread(
            target=self._test_camera_connection_in_background,
            args=(mode, source),
            name="VisionGuard-ConnectionTest",
            daemon=True,
        ).start()

    def _test_camera_connection_in_background(
        self,
        mode: str,
        source: VideoSource | None,
    ) -> None:
        """Mở thử nguồn camera/file, đọc một frame rồi luôn giải phóng."""
        if mode == "webcam":
            try:
                devices = self._stream_manager.scan_available_webcams(5)
            except Exception as error:
                LOGGER.exception("Không thể kiểm tra webcam tự động.")
                self.camera_test_completed.emit(False, str(error))
                return
            if source is None:
                success = bool(devices)
                message = (
                    "Webcam tự động hoạt động: "
                    f"{', '.join(map(str, devices))}."
                    if success
                    else "Không tìm thấy webcam hoạt động."
                )
            else:
                success = source in devices
                message = (
                    f"Webcam {source} kết nối thành công."
                    if success
                    else f"Không mở/đọc được frame từ webcam {source}."
                )
            self.camera_test_completed.emit(success, message)
            return

        capture: cv2.VideoCapture | None = None
        success = False
        message = "Không kiểm tra được nguồn video."
        try:
            if source is None:
                raise ValueError("Nguồn video chưa được cấu hình.")
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                message = (
                    "Không mở được nguồn. Kiểm tra địa chỉ, cổng "
                    "và thông tin đăng nhập."
                )
            else:
                read_success, frame = capture.read()
                success = read_success and frame is not None
                message = (
                    "Kết nối thành công — đã đọc được frame video."
                    if success
                    else "Nguồn đã mở nhưng không đọc được frame video."
                )
        except (cv2.error, OSError, TypeError, ValueError) as error:
            LOGGER.exception("Kiểm tra nguồn camera thất bại.")
            message = str(error)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except cv2.error as error:
                    LOGGER.exception("Không giải phóng được capture thử nghiệm.")
                    if success:
                        success = False
                        message = f"Không giải phóng được capture: {error}"
        self.camera_test_completed.emit(success, message)

    def _on_camera_test_completed(self, success: bool, message: str) -> None:
        """Hiển thị kết quả kiểm tra kết nối trong modal."""
        self._camera_test_button.setEnabled(True)
        self._camera_test_status.setText(message)
        self._camera_test_status.setObjectName(
            "statusSafe" if success else "statusWarning"
        )
        self._camera_test_status.style().unpolish(self._camera_test_status)
        self._camera_test_status.style().polish(self._camera_test_status)

    def _request_webcam_scan(self, checked: bool = False) -> None:
        """Khởi chạy dò webcam ở worker để không chặn luồng giao diện.

        Args:
            checked: Trạng thái nút gửi bởi tín hiệu clicked của Qt.

        Returns:
            None.
        """
        del checked
        self._begin_webcam_scan(auto_switch=False)

    def _begin_webcam_scan(self, auto_switch: bool) -> None:
        """Bắt đầu quét camera và ghi nhớ nếu kết quả cần áp dụng nguồn Auto.

        Args:
            auto_switch: Tự áp dụng webcam khả dụng đầu tiên sau khi quét.

        Returns:
            None.
        """
        self._pending_auto_source_switch = auto_switch
        if self._webcam_scan_running:
            return
        self._webcam_scan_running = True
        self._scan_webcams_button.setEnabled(False)
        self._system_status.setText("ĐANG DÒ WEBCAM…")
        scan_thread = threading.Thread(
            target=self._scan_webcams_in_background,
            name="VisionGuard-WebcamScan",
            daemon=True,
        )
        scan_thread.start()

    def _scan_webcams_in_background(self) -> None:
        """Dò webcam ngoài GUI và gửi kết quả qua Qt signal an toàn luồng.

        Args:
            None.

        Returns:
            None.
        """
        try:
            devices = self._stream_manager.scan_available_webcams(max_index=5)
        except Exception as error:
            LOGGER.exception("Không thể dò các webcam khả dụng.")
            self.webcams_scanned.emit([], str(error))
            return
        self.webcams_scanned.emit(devices, "")

    def _on_webcams_scanned(
        self,
        devices: list[int],
        error_message: str,
    ) -> None:
        """Đưa camera đã dò vào dropdown và xử lý yêu cầu Auto đang chờ.

        Args:
            devices: Các ID camera mở và đọc được frame thử.
            error_message: Lỗi nghiêm trọng khi quá trình quét không chạy được.

        Returns:
            None.
        """
        self._webcam_scan_running = False
        self._scan_webcams_button.setEnabled(True)
        selected_device = self._webcam_device_combo.currentData()
        self._webcam_device_combo.clear()
        self._webcam_device_combo.addItem("Tự động dò (Auto)", "auto")
        for device_id in devices:
            self._webcam_device_combo.addItem(
                f"Webcam {device_id}",
                device_id,
            )
        selected_index = self._webcam_device_combo.findData(selected_device)
        self._webcam_device_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )

        should_auto_switch = self._pending_auto_source_switch
        self._pending_auto_source_switch = False
        if error_message:
            QMessageBox.warning(
                self,
                "Không thể dò webcam",
                f"Quá trình dò camera thất bại:\n{error_message}",
            )
            return
        if not devices:
            if should_auto_switch:
                QMessageBox.warning(
                    self,
                    "Không tìm thấy webcam",
                    "Không có webcam nào mở được và đọc được frame.",
                )
            else:
                self._system_status.setText("KHÔNG TÌM THẤY WEBCAM")
            return

        self._system_status.setText(
            f"ĐÃ DÒ THẤY {len(devices)} WEBCAM"
        )
        if should_auto_switch:
            selected_index = self._webcam_device_combo.findData(devices[0])
            if selected_index >= 0:
                self._webcam_device_combo.setCurrentIndex(selected_index)
            self._apply_webcam_source(devices[0])

    def _apply_webcam_source(self, device_id: int) -> None:
        """Lưu và chuyển sang webcam đã chọn mà không chặn luồng GUI.

        Args:
            device_id: ID webcam đã chọn hoặc tìm thấy bằng chế độ Auto.

        Returns:
            None.
        """
        source_config = dict(self._config.get("video_source", {}))
        source_config.update({"mode": "webcam", "device_id": device_id})
        try:
            self._config = self._config_manager.update_config(
                {"video_source": source_config}
            )
            self._worker.request_source_switch(device_id)
        except Exception as error:
            LOGGER.exception("Không thể áp dụng webcam %s.", device_id)
            QMessageBox.critical(
                self,
                "Không thể chuyển webcam",
                f"Không lưu hoặc gửi được cấu hình webcam:\n{error}",
            )
            return
        self._system_status.setText(
            f"ĐANG CHUYỂN SANG WEBCAM {device_id}…"
        )
        self._camera_dialog.accept()

    def _on_mode_changed(self, index: int) -> None:
        """Chuyển giao diện giữa thao tác vận hành và cài đặt kỹ thuật.

        Args:
            index: 0 cho vận hành, 1 cho cài đặt.

        Returns:
            None.
        """
        self._setup_mode = index == 1
        if self._mode_button.isChecked() != self._setup_mode:
            self._mode_button.blockSignals(True)
            self._mode_button.setChecked(self._setup_mode)
            self._mode_button.blockSignals(False)
        self._mode_button.setText(
            "Chế độ: Kỹ sư (Setup)"
            if self._setup_mode
            else "Chế độ: Giám sát (Khóa)"
        )
        if not self._setup_mode:
            self._editing_occlusion = False
            self._drawing_roi = False
            self._draft_roi = []
            self._draw_roi_button.setText(
                "Vẽ lại Vùng cấm (Click chuột)"
            )
        self._refresh_canvas_overlay()
        for widget in (
            self._roi_controls,
            self._occlusion_controls,
            self._min_area_slider,
            self._delay_slider,
        ):
            widget.setEnabled(self._setup_mode)
        self._roi_hint.setText(
            "ROI: ĐANG HIỆU CHUẨN" if self._setup_mode else "ROI: ĐÃ LƯU"
        )

    def _refresh_canvas_overlay(self) -> None:
        """Đồng bộ ROI và các vùng che khuất lên canvas hiển thị.

        Args:
            None.

        Returns:
            None.
        """
        self._canvas.set_roi_overlay(
            self._active_roi,
            editing=self._setup_mode and self._drawing_roi,
            draft_points=self._draft_roi if self._drawing_roi else [],
            occlusion_polygons=self._occlusion_polygons,
            draft_occlusion_points=(
                self._draft_occlusion_points
                if self._setup_mode
                else []
            ),
            editing_occlusion=self._setup_mode and self._editing_occlusion,
        )

    def _on_source_mode_changed(self) -> None:
        """Chỉ bật trường cấu hình phù hợp với nguồn video được chọn.

        Args:
            None.

        Returns:
            None.
        """
        mode = self._source_combo.currentData()
        self._webcam_source_group.setVisible(mode == "webcam")
        self._file_source_group.setVisible(mode == "file")
        self._rtsp_source_group.setVisible(mode == "rtsp")
        self._webcam_device_combo.setEnabled(not self._webcam_scan_running)
        self._scan_webcams_button.setEnabled(not self._webcam_scan_running)
        self._update_rtsp_preview()

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
        if self._rtsp_profile_combo.currentText().startswith("Dahua") and (
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
            selected_device = self._webcam_device_combo.currentData()
            if selected_device == "auto":
                self._begin_webcam_scan(auto_switch=True)
                return
            if isinstance(selected_device, bool) or not isinstance(
                selected_device, int
            ):
                QMessageBox.warning(
                    self,
                    "Webcam chưa hợp lệ",
                    "Hãy chọn camera đã dò hoặc chế độ Tự động dò (Auto).",
                )
                return
            self._apply_webcam_source(selected_device)
            return
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
                    "profile": {
                        "Dahua/Imou": "Dahua",
                        "Hikvision/Ezviz": "Hikvision",
                    }.get(
                        self._rtsp_profile_combo.currentText(),
                        "ONVIF",
                    ),
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
        self._camera_dialog.accept()

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

    def _on_overlay_options_changed(self, checked: bool = False) -> None:
        """Áp dụng và lưu hai công tắc overlay độc lập."""
        del checked
        show_boxes = self._show_bounding_box_checkbox.isChecked()
        show_feet = self._show_foot_point_checkbox.isChecked()
        self._worker.set_overlay_options(show_boxes, show_feet)
        try:
            self._config = self._config_manager.update_config(
                {
                    "display_overlays": {
                        "bounding_box": show_boxes,
                        "foot_point": show_feet,
                    }
                }
            )
        except (OSError, TypeError, ValueError):
            LOGGER.exception("Không thể lưu tùy chọn HUD overlay.")
            QMessageBox.warning(
                self,
                "Không thể lưu tùy chọn",
                "Đã áp dụng tùy chọn cho phiên hiện tại nhưng chưa lưu được config.json.",
            )

    def _reset_standard_parameters(self) -> None:
        """Khôi phục minArea và độ trễ về thông số vận hành mặc định."""
        self._min_area_slider.setValue(1500)
        self._delay_slider.setValue(8)
        try:
            self._cv_processor.set_min_area(1500.0)
            self._safety_controller.set_violation_delay(0.8)
            self._config = self._config_manager.update_config(
                {"min_area": 1500, "violation_delay": 0.8}
            )
        except (OSError, TypeError, ValueError):
            LOGGER.exception("Không thể khôi phục tham số chuẩn.")
            QMessageBox.critical(
                self,
                "Không thể reset",
                "Không lưu được tham số chuẩn vào config.json.",
            )
            return
        self._refresh_slider_labels()

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
        """Thêm đỉnh mới vào đa giác ROI nháp khi toggle đang bật.

        Args:
            x: Tọa độ ngang trong frame chuẩn 640×360.
            y: Tọa độ dọc trong frame chuẩn 640×360.

        Returns:
            None.
        """
        if not self._setup_mode or not self._drawing_roi:
            return
        self._draft_roi.append((x, y))
        self._refresh_canvas_overlay()
        self._roi_hint.setText(f"ROI MỚI: {len(self._draft_roi)} ĐỈNH")

    def _toggle_roi_drawing(self, checked: bool = False) -> None:
        """Bật chế độ vẽ ROI mới hoặc xác nhận và lưu polygon nháp.

        Args:
            checked: Trạng thái nút gửi bởi tín hiệu clicked của Qt.

        Returns:
            None.
        """
        del checked
        if not self._setup_mode:
            return

        if not self._drawing_roi:
            self._draft_roi = []
            self._editing_occlusion = False
            self._drawing_roi = True
            self._draw_roi_button.setText("Xác nhận Vùng cấm mới")
            self._refresh_canvas_overlay()
            self._roi_hint.setText("ĐANG VẼ ROI MỚI · CHẤM TỐI THIỂU 3 ĐỈNH")
            return

        if len(self._draft_roi) < 3:
            QMessageBox.warning(
                self,
                "ROI chưa hợp lệ",
                "Hãy chấm ít nhất 3 đỉnh trước khi xác nhận vùng cấm.",
            )
            return
        try:
            self._config = self._config_manager.update_config(
                {"roi_points": [list(point) for point in self._draft_roi]}
            )
        except (OSError, ValueError):
            LOGGER.exception("Không thể lưu polygon ROI.")
            QMessageBox.critical(
                self,
                "Không thể lưu ROI",
                "Đã xảy ra lỗi khi ghi ROI vào config.json.",
            )
            return

        self._active_roi = list(self._draft_roi)
        self._worker.set_roi_points(self._active_roi)
        self._draft_roi = []
        self._drawing_roi = False
        self._draw_roi_button.setText("Vẽ lại Vùng cấm (Click chuột)")
        self._refresh_canvas_overlay()
        self._roi_hint.setText(f"ROI: ĐÃ LƯU · {len(self._active_roi)} ĐỈNH")

    def _delete_roi_point(self, checked: bool = False) -> None:
        """Xóa đỉnh cuối danh sách đang hiệu chỉnh nhưng không dưới ba đỉnh.

        Args:
            checked: Trạng thái nút gửi bởi tín hiệu clicked của Qt.

        Returns:
            None.
        """
        del checked
        if self._drawing_roi:
            if len(self._draft_roi) <= 3:
                self._roi_hint.setText("ROI NHÁP CẦN GIỮ TỐI THIỂU 3 ĐỈNH")
                return
            self._draft_roi.pop()
            self._refresh_canvas_overlay()
            self._roi_hint.setText(f"ROI MỚI: {len(self._draft_roi)} ĐỈNH")
            return

        if len(self._active_roi) <= 3:
            self._roi_hint.setText("ROI CẦN GIỮ TỐI THIỂU 3 ĐỈNH")
            return
        updated_roi = self._active_roi[:-1]
        try:
            self._config = self._config_manager.update_config(
                {"roi_points": [list(point) for point in updated_roi]}
            )
        except (OSError, ValueError):
            LOGGER.exception("Không thể lưu ROI sau khi xóa đỉnh.")
            QMessageBox.critical(
                self,
                "Không thể cập nhật ROI",
                "Đã xảy ra lỗi khi lưu vùng cấm đã chỉnh sửa.",
            )
            return

        self._active_roi = updated_roi
        self._worker.set_roi_points(self._active_roi)
        self._refresh_canvas_overlay()
        self._roi_hint.setText(
            f"ROI: ĐÃ CẬP NHẬT · {len(self._active_roi)} ĐỈNH"
        )

    def _restore_default_roi(self, checked: bool = False) -> None:
        """Hủy bản nháp và lưu lại ROI mặc định ban đầu của hệ thống.

        Args:
            checked: Trạng thái nút gửi bởi tín hiệu clicked của Qt.

        Returns:
            None.
        """
        del checked
        default_roi = list(DEFAULT_ROI_POINTS)
        try:
            self._config = self._config_manager.update_config(
                {"roi_points": [list(point) for point in default_roi]}
            )
        except (OSError, ValueError):
            LOGGER.exception("Không thể khôi phục ROI mặc định.")
            QMessageBox.critical(
                self,
                "Không thể khôi phục ROI",
                "Đã xảy ra lỗi khi lưu vùng cấm mặc định.",
            )
            return

        self._active_roi = default_roi
        self._draft_roi = []
        self._drawing_roi = False
        self._draw_roi_button.setText("Vẽ lại Vùng cấm (Click chuột)")
        self._worker.set_roi_points(self._active_roi)
        self._refresh_canvas_overlay()
        self._roi_hint.setText("ROI: ĐÃ KHÔI PHỤC MẶC ĐỊNH")

    def _start_occlusion_draft(self) -> None:
        """Bắt đầu vẽ đa giác loại trừ thân máy khỏi lớp phủ ROI.

        Args:
            None.

        Returns:
            None.
        """
        if not self._setup_mode:
            return
        self._drawing_roi = False
        self._draft_roi = []
        self._draw_roi_button.setText("Vẽ lại Vùng cấm (Click chuột)")
        self._editing_occlusion = True
        self._draft_occlusion_points = []
        self._refresh_canvas_overlay()
        self._roi_hint.setText("MASK MÁY: CHẤM TỐI THIỂU 3 ĐỈNH")

    def _add_occlusion_point(self, x: int, y: int) -> None:
        """Thêm một đỉnh che khuất đã quy đổi sang hệ tọa độ 640×360.

        Args:
            x: Tọa độ ngang trong frame chuẩn.
            y: Tọa độ dọc trong frame chuẩn.

        Returns:
            None.
        """
        if not self._setup_mode or not self._editing_occlusion:
            return
        self._draft_occlusion_points.append((x, y))
        self._refresh_canvas_overlay()
        self._roi_hint.setText(
            f"MASK MÁY: {len(self._draft_occlusion_points)} ĐỈNH"
        )

    def _undo_occlusion_point(self) -> None:
        """Hoàn tác đỉnh gần nhất của vùng che khuất đang vẽ.

        Args:
            None.

        Returns:
            None.
        """
        if self._draft_occlusion_points:
            self._draft_occlusion_points.pop()
        self._refresh_canvas_overlay()
        self._roi_hint.setText(
            f"MASK MÁY: {len(self._draft_occlusion_points)} ĐỈNH"
        )

    def _save_occlusion_polygon(self) -> None:
        """Lưu vùng che khuất đã hiệu chuẩn vào cấu hình và worker CV.

        Args:
            None.

        Returns:
            None.
        """
        if len(self._draft_occlusion_points) < 3:
            QMessageBox.warning(
                self,
                "Vùng che khuất chưa hợp lệ",
                "Đa giác vùng thân máy cần tối thiểu 3 đỉnh.",
            )
            return
        polygons = [
            [list(point) for point in polygon]
            for polygon in self._occlusion_polygons
        ]
        polygons.append(
            [list(point) for point in self._draft_occlusion_points]
        )
        try:
            self._config = self._config_manager.update_config(
                {"occlusion_polygons": polygons}
            )
        except (OSError, ValueError):
            LOGGER.exception("Không thể lưu vùng che khuất.")
            QMessageBox.critical(
                self,
                "Không thể lưu vùng che khuất",
                "Đã xảy ra lỗi khi ghi vùng che khuất vào config.json.",
            )
            return

        self._occlusion_polygons.append(
            list(self._draft_occlusion_points)
        )
        self._draft_occlusion_points = []
        self._editing_occlusion = False
        self._worker.set_occlusion_polygons(self._occlusion_polygons)
        self._refresh_canvas_overlay()
        self._roi_hint.setText(
            f"MASK MÁY: ĐÃ LƯU · {len(self._occlusion_polygons)} VÙNG"
        )

    def _clear_occlusion_polygons(self) -> None:
        """Xóa toàn bộ vùng che khuất đã lưu khỏi cấu hình và lớp phủ.

        Args:
            None.

        Returns:
            None.
        """
        try:
            self._config = self._config_manager.update_config(
                {"occlusion_polygons": []}
            )
        except (OSError, ValueError):
            LOGGER.exception("Không thể xóa vùng che khuất.")
            QMessageBox.critical(
                self,
                "Không thể xóa vùng che khuất",
                "Đã xảy ra lỗi khi cập nhật config.json.",
            )
            return

        self._occlusion_polygons = []
        self._draft_occlusion_points = []
        self._editing_occlusion = False
        self._worker.set_occlusion_polygons([])
        self._refresh_canvas_overlay()
        self._roi_hint.setText("MASK MÁY: ĐÃ XÓA")

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
        self._audit_table_fingerprint = tuple(
            (event.event_id, event.violation_duration)
            for event in filtered
        )
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
            card = SnapshotCardWidget(event)
            card.activated.connect(self._open_snapshot)
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
        """Cập nhật video, HUD, trạng thái an toàn và chỉ số vận hành.

        Args:
            None.

        Returns:
            None.
        """
        status = self._stream_manager.read_latest_status(timeout=0.0)
        if status is not None:
            self._update_system_status(status.state, status.message)
        else:
            status = self._stream_manager.status
            self._update_system_status(status.state, status.message)

        packet = self._worker.take_latest_result()
        if packet is not None:
            if packet.error_message:
                self._system_status.setText("LỖI XỬ LÝ FRAME")
                LOGGER.error("HMI nhận lỗi worker: %s", packet.error_message)
            if packet.frame is not None and packet.analysis is not None:
                self._canvas.set_frame(packet.frame)
                self._people_value.setText(
                    f"{len(packet.analysis.detections):02d}"
                )
                source_name = self._source_display_name()
                point_score = (
                    1.0
                    if packet.safety is not None
                    and packet.safety.inside_roi
                    else -1.0
                )
                point_state = (
                    "INTRUSION" if point_score >= 0 else "SAFE"
                )
                self._canvas.set_hud_lines(
                    f"REC • {datetime.now():%H:%M:%S} | {source_name}",
                    f"cv2.pointPolygonTest: {point_score:+.1f} "
                    f"({point_state})",
                )
                if packet.safety is not None:
                    self._refresh_safety_status(
                        packet.safety,
                        len(packet.analysis.detections),
                    )

        self._ack_button.setEnabled(self._safety_controller.latched)

    def _source_display_name(self) -> str:
        """Tạo nhãn nguồn ngắn, không đưa thông tin xác thực lên HUD."""
        source = self._config.get("video_source", {})
        mode = source.get("mode", "webcam")
        if mode == "webcam":
            return f"WEBCAM {source.get('device_id', 0)}"
        if mode == "file":
            return f"FILE {Path(source.get('file_path', '')).name}"
        if mode == "rtsp":
            return "RTSP CAMERA"
        return "VIDEO SOURCE"

    def _update_system_status(self, state: StreamState, message: str) -> None:
        """Hiển thị nguồn hiện hành, kết nối, FPS và thời gian xử lý.

        Args:
            state: Trạng thái kết nối của StreamManager.
            message: Thông báo trạng thái tiếng Việt từ stream.

        Returns:
            None.
        """
        source_name = self._source_display_name()
        if state is StreamState.CONNECTED:
            status_label = "ONLINE"
            self._system_status.setObjectName("sourceStatusOnline")
        elif state in (StreamState.RECONNECTING, StreamState.ERROR):
            status_label = message
            self._system_status.setObjectName("sourceStatusError")
        else:
            status_label = message
            self._system_status.setObjectName("sourceStatus")
        self._system_status.setText(
            f"{source_name} · {status_label} "
            f"({self._worker.fps:.1f} FPS / "
            f"{self._worker.processing_ms:.0f} ms)"
        )
        self._system_status.style().unpolish(self._system_status)
        self._system_status.style().polish(self._system_status)

    def _refresh_safety_status(
        self,
        safety: SafetyResult,
        detection_count: int,
    ) -> None:
        """Cập nhật badge, bộ đệm xâm nhập, E-STOP và số liệu ca.

        Args:
            safety: Snapshot trạng thái từ SafetyController.

        Returns:
            None.
        """
        stream_state = self._stream_manager.status.state
        canvas_state = (safety.state, stream_state, self._setup_mode)
        if canvas_state != self._last_canvas_state:
            self._refresh_canvas_overlay()
            self._last_canvas_state = canvas_state

        labels = {
            SafetyState.SAFE: ("AN TOÀN", "statusSafe"),
            SafetyState.WARNING: ("CẢNH BÁO SỚM", "statusWarning"),
            SafetyState.ALARM: ("BÁO ĐỘNG ĐỎ", "statusAlarm"),
        }
        label, object_name = labels[safety.state]
        self._safety_badge.setText(label)
        self._safety_badge.setObjectName(object_name)
        self._safety_badge.style().unpolish(self._safety_badge)
        self._safety_badge.style().polish(self._safety_badge)
        self._people_value.setText(f"{detection_count:02d}")
        self._machine_state_value.setText(
            "DỪNG KHẨN CẤP" if safety.estop_triggered else "ĐANG CHẠY"
        )
        delay = self._delay_slider.value() / 10.0
        duration = safety.violation_duration
        self._intrusion_buffer_label.setText(
            f"Bộ đệm thời gian xâm nhập (Chống nhiễu): "
            f"{duration:.2f}s / {delay:.2f}s"
        )
        progress = (
            1000
            if delay <= 0 and safety.state is SafetyState.ALARM
            else round(min(1.0, duration / delay) * 1000)
            if delay > 0
            else 0
        )
        self._intrusion_progress.setValue(progress)
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
        table_fingerprint = tuple(
            (event.event_id, event.violation_duration)
            for event in filtered
        )
        if table_fingerprint != self._audit_table_fingerprint:
            self._audit_table_fingerprint = table_fingerprint
            self._render_audit_table(filtered)
        self._refresh_carousel(filtered)
        current_hour = datetime.now().hour
        current_shift = (
            1 if 6 <= current_hour < 14
            else 2 if 14 <= current_hour < 22
            else 3
        )
        shift_events = self._audit_logger.filter_by_shift(
            current_shift,
            current_events,
        )
        self._shift_alarms_value.setText(f"{len(shift_events):02d}")

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
