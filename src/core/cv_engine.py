"""Pipeline OpenCV phát hiện chuyển động và tạo các chế độ hiển thị video."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, replace
from enum import Enum
from importlib import import_module
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


class DisplayMode(str, Enum):
    """Các lớp ảnh dùng để hiển thị trên màn hình HMI."""

    RGB = "rgb"
    MOG2_MASK = "mog2"
    CANNY_EDGE = "canny"


@dataclass(frozen=True)
class Detection:
    """Thông tin hình học của người được phát hiện trong frame chuẩn."""

    bounding_box: tuple[int, int, int, int]
    foot_point: tuple[int, int]
    area: float
    intersects_roi: bool = False


@dataclass(frozen=True)
class FrameAnalysis:
    """Kết quả xử lý frame, gồm các ảnh hiển thị và danh sách phát hiện."""

    rgb_frame: np.ndarray
    mog2_frame: np.ndarray
    canny_frame: np.ndarray
    display_frame: np.ndarray
    detections: tuple[Detection, ...]
    white_pixel_ratio: float
    illumination_shift: bool
    skip_safety_evaluation: bool


class CVProcessor:
    """Kết hợp YOLOv8 và OpenCV để phát hiện người, tạo các lớp ảnh Debug."""

    FRAME_SIZE: tuple[int, int] = (640, 360)
    DEFAULT_MIN_AREA: float = 1500.0
    ILLUMINATION_THRESHOLD: float = 0.60
    CANNY_THRESHOLDS: tuple[int, int] = (50, 150)
    MORPH_KERNEL_SIZE: tuple[int, int] = (3, 3)
    LARGE_MOTION_THRESHOLD: float = 0.10
    PROTECTED_LEARNING_RATE: float = 0.0005
    FLOOR_OVERLAY_ALPHA: float = 0.18

    def __init__(
        self,
        min_area: float = DEFAULT_MIN_AREA,
        display_mode: DisplayMode | str = DisplayMode.RGB,
    ) -> None:
        """Khởi tạo MOG2 và thử tải YOLOv8n, fallback nếu model không sẵn có.

        Args:
            min_area: Diện tích contour nhỏ nhất được giữ lại, từ 500 đến 5000 px².
            display_mode: Chế độ hiển thị RGB, mặt nạ MOG2 hoặc Canny Edge.

        Returns:
            None.

        Raises:
            TypeError: Khi min_area hoặc display_mode không đúng kiểu.
            ValueError: Khi min_area nằm ngoài phạm vi hoặc chế độ không hợp lệ.
            cv2.error: Khi OpenCV không thể khởi tạo bộ trừ nền.
        """
        self._lock: threading.RLock = threading.RLock()
        self._min_area = self._validate_min_area(min_area)
        self._display_mode = self._coerce_display_mode(display_mode)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.MORPH_KERNEL_SIZE
        )
        self._background_subtractor = self._create_background_subtractor()
        self._protect_background: bool = False
        self.yolo_model: Any | None = None
        try:
            yolo_module = import_module("ultralytics")
            yolo_factory = getattr(yolo_module, "YOLO")
            self.yolo_model = yolo_factory("yolov8n.pt")
        except Exception:
            LOGGER.exception(
                "Không tải được YOLOv8n; chuyển sang phát hiện contour MOG2."
            )

    @staticmethod
    def _validate_min_area(min_area: float) -> float:
        """Kiểm tra ngưỡng diện tích theo dải cấu hình được đặc tả.

        Args:
            min_area: Diện tích contour tối thiểu tính bằng pixel vuông.

        Returns:
            Ngưỡng diện tích đã chuyển về kiểu float.

        Raises:
            TypeError: Khi ngưỡng không phải số.
            ValueError: Khi ngưỡng không hữu hạn hoặc ngoài khoảng 500–5000.
        """
        if isinstance(min_area, bool) or not isinstance(min_area, (int, float)):
            raise TypeError("min_area phải là một số.")
        if not math.isfinite(min_area) or not 500 <= min_area <= 5000:
            raise ValueError("min_area phải nằm trong khoảng 500 đến 5000 px².")
        return float(min_area)

    @staticmethod
    def _coerce_display_mode(mode: DisplayMode | str) -> DisplayMode:
        """Chuyển chuỗi chế độ hiển thị sang enum hợp lệ.

        Args:
            mode: Giá trị DisplayMode hoặc chuỗi rgb, mog2 hay canny.

        Returns:
            Chế độ hiển thị đã chuẩn hóa.

        Raises:
            ValueError: Khi chuỗi không đại diện cho chế độ được hỗ trợ.
        """
        try:
            return mode if isinstance(mode, DisplayMode) else DisplayMode(mode)
        except ValueError as error:
            raise ValueError(
                "display_mode phải là 'rgb', 'mog2' hoặc 'canny'."
            ) from error

    @staticmethod
    def _create_background_subtractor() -> cv2.BackgroundSubtractorMOG2:
        """Khởi tạo MOG2 theo đúng tham số chuẩn của dự án.

        Args:
            None.

        Returns:
            Bộ trừ nền MOG2 có bật nhận diện bóng đổ.

        Raises:
            cv2.error: Khi OpenCV không khởi tạo được mô hình MOG2.
        """
        return cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True,
        )

    @property
    def min_area(self) -> float:
        """Đọc ngưỡng diện tích hiện tại một cách an toàn giữa các luồng.

        Args:
            None.

        Returns:
            Ngưỡng diện tích contour hiện tại.
        """
        with self._lock:
            return self._min_area

    @property
    def display_mode(self) -> DisplayMode:
        """Đọc chế độ hiển thị đang được chọn.

        Args:
            None.

        Returns:
            Enum biểu diễn RGB, MOG2 Mask hoặc Canny Edge.
        """
        with self._lock:
            return self._display_mode

    def set_min_area(self, min_area: float) -> None:
        """Cập nhật ngưỡng diện tích dùng ở frame tiếp theo.

        Args:
            min_area: Diện tích tối thiểu trong khoảng 500–5000 px².

        Returns:
            None.

        Raises:
            TypeError: Khi ngưỡng không phải số.
            ValueError: Khi ngưỡng nằm ngoài khoảng cho phép.
        """
        validated_area = self._validate_min_area(min_area)
        with self._lock:
            self._min_area = validated_area

    def set_display_mode(self, display_mode: DisplayMode | str) -> None:
        """Chọn lớp ảnh HMI sẽ nhận ở lần xử lý frame kế tiếp.

        Args:
            display_mode: DisplayMode hoặc chuỗi rgb, mog2 hay canny.

        Returns:
            None.

        Raises:
            ValueError: Khi chế độ hiển thị không được hỗ trợ.
        """
        validated_mode = self._coerce_display_mode(display_mode)
        with self._lock:
            self._display_mode = validated_mode

    def reset_background_model(self) -> None:
        """Tạo lại mô hình nền MOG2 để khôi phục sau sự kiện sốc sáng.

        Args:
            None.

        Returns:
            None.

        Raises:
            cv2.error: Khi OpenCV không thể khởi tạo mô hình nền mới.
        """
        with self._lock:
            self._background_subtractor = self._create_background_subtractor()
            self._protect_background = False

    def process_frame(
        self,
        frame: np.ndarray,
        display_mode: DisplayMode | str | None = None,
        roi_points: list[tuple[int, int]] | None = None,
    ) -> FrameAnalysis:
        """Phát hiện người bằng YOLO, giữ MOG2/Canny cho hiển thị và fallback.

        Args:
            frame: Frame BGR đầu vào từ StreamManager.
            display_mode: Chế độ hiển thị tùy chọn; None dùng chế độ hiện hành.
            roi_points: Đa giác ROI theo hệ tọa độ frame 640×360.

        Returns:
            FrameAnalysis chứa RGB, mặt nạ MOG2, Canny, detections và cờ sốc sáng.
            Khi YOLO không khả dụng, detections được tạo từ foreground MOG2;
            bóng xám 127 bị loại để không trở thành contour phát hiện.

        Raises:
            TypeError: Khi frame không phải numpy.ndarray.
            ValueError: Khi frame rỗng hoặc không có 3 kênh màu.
            cv2.error: Khi một phép toán OpenCV thất bại.
        """
        self._validate_frame(frame)
        with self._lock:
            active_mode = (
                self._display_mode
                if display_mode is None
                else self._coerce_display_mode(display_mode)
            )
            min_area = self._min_area

            rgb_frame = cv2.resize(frame, self.FRAME_SIZE)
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            learning_rate = (
                self.PROTECTED_LEARNING_RATE
                if self._protect_background
                else -1.0
            )
            mog2_frame = self._background_subtractor.apply(
                blurred_frame,
                learningRate=learning_rate,
            )

            object_mask = cv2.inRange(mog2_frame, 255, 255)
            total_pixels = object_mask.size
            white_pixel_ratio = cv2.countNonZero(object_mask) / total_pixels
            illumination_shift = (
                white_pixel_ratio > self.ILLUMINATION_THRESHOLD
            )
            yolo_detections: tuple[Detection, ...] | None = None
            if self.yolo_model is not None:
                results = self.yolo_model.predict(
                    rgb_frame,
                    classes=[0],
                    conf=0.45,
                    imgsz=640,
                    verbose=False,
                )
                yolo_detections = self._extract_yolo_detections(
                    results,
                    min_area,
                    rgb_frame.shape[1],
                    rgb_frame.shape[0],
                )

            if illumination_shift:
                # Không cho frame sốc sáng tạo contour; mô hình mới học lại nền
                # ngay trên frame hiện tại nhưng bỏ qua đánh giá an toàn của frame.
                self._background_subtractor = self._create_background_subtractor()
                self._background_subtractor.apply(
                    blurred_frame,
                    learningRate=1.0,
                )
                detections = yolo_detections or ()
                self._protect_background = False
            elif yolo_detections is not None:
                detections = yolo_detections
                person_in_roi = self._has_detection_in_roi(
                    detections,
                    roi_points,
                )
                self._protect_background = (
                    white_pixel_ratio >= self.LARGE_MOTION_THRESHOLD
                    or person_in_roi
                )
            else:
                opened_mask = cv2.morphologyEx(
                    object_mask, cv2.MORPH_OPEN, self._morph_kernel
                )
                cleaned_mask = cv2.morphologyEx(
                    opened_mask, cv2.MORPH_CLOSE, self._morph_kernel
                )
                contours, _ = cv2.findContours(
                    cleaned_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                detections = self._extract_detections(contours, min_area)
                person_in_roi = self._has_detection_in_roi(
                    detections,
                    roi_points,
                )
                self._protect_background = (
                    white_pixel_ratio >= self.LARGE_MOTION_THRESHOLD
                    or person_in_roi
                )

            detections = self._mark_roi_intersections(
                detections,
                roi_points,
            )
            canny_frame = cv2.Canny(
                blurred_frame,
                self.CANNY_THRESHOLDS[0],
                self.CANNY_THRESHOLDS[1],
            )
            display_frame = self._select_display_frame(
                active_mode,
                rgb_frame,
                mog2_frame,
                canny_frame,
            )

        return FrameAnalysis(
            rgb_frame=rgb_frame,
            mog2_frame=mog2_frame,
            canny_frame=canny_frame,
            display_frame=display_frame,
            detections=detections,
            white_pixel_ratio=white_pixel_ratio,
            illumination_shift=illumination_shift,
            skip_safety_evaluation=illumination_shift,
        )

    @staticmethod
    def _extract_yolo_detections(
        results: Any,
        min_area: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[Detection, ...]:
        """Lọc riêng class người và chuyển box YOLO thành tọa độ frame chuẩn.

        Args:
            results: Kết quả trả về từ Ultralytics YOLO.predict.
            min_area: Diện tích bounding box tối thiểu cần giữ lại.
            frame_width: Chiều rộng frame đã resize.
            frame_height: Chiều cao frame đã resize.

        Returns:
            Các Detection class 0 có diện tích bounding box đạt ngưỡng.
        """
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                if int(box.cls.item()) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x = max(0, min(int(x1), frame_width))
                y = max(0, min(int(y1), frame_height))
                right = max(x, min(int(x2), frame_width))
                bottom = max(y, min(int(y2), frame_height))
                width = right - x
                height = bottom - y
                area = width * height
                if area < min_area or width == 0 or height == 0:
                    continue
                detections.append(
                    Detection(
                        bounding_box=(x, y, width, height),
                        foot_point=(x + width // 2, y + height),
                        area=float(area),
                    )
                )
        return tuple(detections)

    @staticmethod
    def _has_detection_in_roi(
        detections: tuple[Detection, ...],
        roi_points: list[tuple[int, int]] | None,
    ) -> bool:
        """Kiểm tra foot hoặc bbox của detection có chạm ROI để giữ nền.

        Args:
            detections: Các đối tượng vừa trích xuất từ mask MOG2.
            roi_points: Đa giác ROI theo hệ tọa độ frame chuẩn 640×360.

        Returns:
            True khi điểm chân của ít nhất một đối tượng nằm trong ROI.
        """
        if not roi_points or len(roi_points) < 3:
            return False
        polygon = np.asarray(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        return any(
            detection.intersects_roi
            or CVProcessor._detection_intersects_roi(detection, roi_points, polygon)
            for detection in detections
        )

    @classmethod
    def _mark_roi_intersections(
        cls,
        detections: tuple[Detection, ...],
        roi_points: list[tuple[int, int]] | None,
    ) -> tuple[Detection, ...]:
        """Đánh dấu detection có foot hoặc bất kỳ cạnh bbox nào chạm ROI."""
        if not roi_points or len(roi_points) < 3:
            return detections
        polygon = np.asarray(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        return tuple(
            replace(
                detection,
                intersects_roi=cls._detection_intersects_roi(
                    detection,
                    roi_points,
                    polygon,
                ),
            )
            for detection in detections
        )

    @staticmethod
    def _detection_intersects_roi(
        detection: Detection,
        roi_points: list[tuple[int, int]],
        polygon: np.ndarray | None = None,
    ) -> bool:
        """Kiểm tra giao nhau đầy đủ giữa hình chữ nhật bbox và đa giác ROI."""
        if len(roi_points) < 3:
            return False
        roi_polygon = (
            np.asarray(roi_points, dtype=np.int32).reshape((-1, 1, 2))
            if polygon is None
            else polygon
        )
        x, y, width, height = detection.bounding_box
        right = x + width
        bottom = y + height
        box_corners = (
            (x, y),
            (right, y),
            (right, bottom),
            (x, bottom),
        )
        points_to_test = (detection.foot_point, *box_corners)
        if any(
            cv2.pointPolygonTest(
                roi_polygon,
                (float(point[0]), float(point[1])),
                False,
            )
            >= 0
            for point in points_to_test
        ):
            return True

        if any(
            x <= point[0] <= right and y <= point[1] <= bottom
            for point in roi_points
        ):
            return True

        box_edges = tuple(
            (box_corners[index], box_corners[(index + 1) % 4])
            for index in range(4)
        )
        roi_edges = tuple(
            (roi_points[index], roi_points[(index + 1) % len(roi_points)])
            for index in range(len(roi_points))
        )
        return any(
            CVProcessor._segments_intersect(box_edge, roi_edge)
            for box_edge in box_edges
            for roi_edge in roi_edges
        )

    @staticmethod
    def _segments_intersect(
        first: tuple[tuple[int, int], tuple[int, int]],
        second: tuple[tuple[int, int], tuple[int, int]],
    ) -> bool:
        """Kiểm tra hai đoạn thẳng đóng có cắt hoặc chạm nhau."""
        a, b = first
        c, d = second

        def orientation(
            p1: tuple[int, int],
            p2: tuple[int, int],
            p3: tuple[int, int],
        ) -> int:
            value = (
                (p2[0] - p1[0]) * (p3[1] - p1[1])
                - (p2[1] - p1[1]) * (p3[0] - p1[0])
            )
            return (value > 0) - (value < 0)

        def on_segment(
            p1: tuple[int, int],
            p2: tuple[int, int],
            point: tuple[int, int],
        ) -> bool:
            return (
                min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0])
                and min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1])
            )

        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)
        if o1 != o2 and o3 != o4:
            return True
        return (
            (o1 == 0 and on_segment(a, b, c))
            or (o2 == 0 and on_segment(a, b, d))
            or (o3 == 0 and on_segment(c, d, a))
            or (o4 == 0 and on_segment(c, d, b))
        )

    @classmethod
    def blend_floor_roi(
        cls,
        frame: np.ndarray,
        roi_points: list[tuple[int, int]],
        occlusion_polygons: list[list[tuple[int, int]]],
        color: tuple[int, int, int],
        foreground_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Trộn ROI bán trong suốt trên sàn và loại trừ vùng máy bị che khuất.

        Args:
            frame: Frame BGR chuẩn 640×360 cần vẽ lớp ROI.
            roi_points: Đa giác vùng sàn theo hệ tọa độ frame.
            occlusion_polygons: Các vùng thân máy không được phủ màu ROI.
            color: Màu BGR theo trạng thái an toàn hiện hành.
            foreground_mask: Mask MOG2 để giữ lớp ROI phía sau vật thể chuyển động.

        Returns:
            Frame mới có lớp ROI alpha-blend, không làm thay đổi frame đầu vào.

        Raises:
            ValueError: Khi foreground_mask không cùng kích thước với frame.
        """
        if len(roi_points) < 3:
            return frame.copy()

        height, width = frame.shape[:2]
        roi = np.asarray(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi], color)
        cv2.polylines(
            overlay,
            [roi],
            isClosed=True,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        visible_floor = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(visible_floor, [roi], 255)
        if foreground_mask is not None:
            if foreground_mask.shape != (height, width):
                raise ValueError(
                    "foreground_mask phải cùng kích thước với frame."
                )
            visible_floor[foreground_mask == 255] = 0
        for polygon_points in occlusion_polygons:
            if len(polygon_points) >= 3:
                polygon = np.asarray(polygon_points, dtype=np.int32).reshape(
                    (-1, 1, 2)
                )
                cv2.fillPoly(visible_floor, [polygon], 0)

        blended = cv2.addWeighted(
            overlay,
            cls.FLOOR_OVERLAY_ALPHA,
            frame,
            1.0 - cls.FLOOR_OVERLAY_ALPHA,
            0,
        )
        result = frame.copy()
        result[visible_floor > 0] = blended[visible_floor > 0]
        return result

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Xác thực cấu trúc frame trước khi gọi toán tử OpenCV.

        Args:
            frame: Mảng ảnh đầu vào cần kiểm tra.

        Returns:
            None.

        Raises:
            TypeError: Khi frame không phải numpy.ndarray.
            ValueError: Khi frame rỗng, không có kích thước hoặc không phải BGR.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame phải là numpy.ndarray.")
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame phải là ảnh BGR không rỗng có 3 kênh.")

    @staticmethod
    def _extract_detections(
        contours: list[np.ndarray],
        min_area: float,
    ) -> tuple[Detection, ...]:
        """Lọc contour theo diện tích và tính hộp bao cùng điểm tiếp xúc sàn.

        Args:
            contours: Các contour ngoài tìm được từ mặt nạ đã làm sạch.
            min_area: Ngưỡng diện tích tối thiểu.

        Returns:
            Tuple các Detection có diện tích không nhỏ hơn ngưỡng.

        Raises:
            cv2.error: Khi OpenCV không tính được diện tích hoặc hộp bao.
        """
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            foot_point = (x + width // 2, y + height)
            detections.append(
                Detection(
                    bounding_box=(x, y, width, height),
                    foot_point=foot_point,
                    area=area,
                )
            )
        return tuple(detections)

    @staticmethod
    def _select_display_frame(
        mode: DisplayMode,
        rgb_frame: np.ndarray,
        mog2_frame: np.ndarray,
        canny_frame: np.ndarray,
    ) -> np.ndarray:
        """Chọn ảnh hiển thị theo chế độ HMI mà không đổi kết quả phân tích.

        Args:
            mode: Chế độ hiển thị hiện hành.
            rgb_frame: Frame BGR đã chuẩn hóa.
            mog2_frame: Mặt nạ MOG2 giữ nền 0, bóng 127 và vật thể 255.
            canny_frame: Ảnh đường biên Canny dạng grayscale.

        Returns:
            Frame hiển thị ba kênh màu để GUI dùng thống nhất.
        """
        if mode is DisplayMode.MOG2_MASK:
            return cv2.cvtColor(mog2_frame, cv2.COLOR_GRAY2BGR)
        if mode is DisplayMode.CANNY_EDGE:
            return cv2.cvtColor(canny_frame, cv2.COLOR_GRAY2BGR)
        return rgb_frame
