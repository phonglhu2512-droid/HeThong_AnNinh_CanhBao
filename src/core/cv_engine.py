"""Pipeline OpenCV phát hiện chuyển động và tạo các chế độ hiển thị video."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class DisplayMode(str, Enum):
    """Các lớp ảnh dùng để hiển thị trên màn hình HMI."""

    RGB = "rgb"
    MOG2_MASK = "mog2"
    CANNY_EDGE = "canny"


@dataclass(frozen=True)
class Detection:
    """Thông tin hình học của một vùng chuyển động đủ ngưỡng diện tích."""

    bounding_box: tuple[int, int, int, int]
    foot_point: tuple[int, int]
    area: float


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
    """Xử lý frame bằng MOG2, morphology và trích xuất điểm chân."""

    FRAME_SIZE: tuple[int, int] = (640, 360)
    DEFAULT_MIN_AREA: float = 1500.0
    ILLUMINATION_THRESHOLD: float = 0.60
    CANNY_THRESHOLDS: tuple[int, int] = (50, 150)
    MORPH_KERNEL_SIZE: tuple[int, int] = (3, 3)

    def __init__(
        self,
        min_area: float = DEFAULT_MIN_AREA,
        display_mode: DisplayMode | str = DisplayMode.RGB,
    ) -> None:
        """Khởi tạo mô hình MOG2 và cấu hình xử lý ảnh.

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

    def process_frame(
        self,
        frame: np.ndarray,
        display_mode: DisplayMode | str | None = None,
    ) -> FrameAnalysis:
        """Phân tích chuyển động, lọc bóng và trả ảnh ở cả ba chế độ xem.

        Args:
            frame: Frame BGR đầu vào từ StreamManager.
            display_mode: Chế độ hiển thị tùy chọn; None dùng chế độ hiện hành.

        Returns:
            FrameAnalysis chứa RGB, mặt nạ MOG2, Canny, detections và cờ sốc sáng.
            Mặt nạ MOG2 hiển thị giữ pixel bóng xám 127; mặt nạ dùng tìm contour
            chỉ giữ pixel vật thể 255 để bóng không trở thành đối tượng phát hiện.

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
            mog2_frame = self._background_subtractor.apply(blurred_frame)

            object_mask = cv2.inRange(mog2_frame, 255, 255)
            total_pixels = object_mask.size
            white_pixel_ratio = cv2.countNonZero(object_mask) / total_pixels
            illumination_shift = (
                white_pixel_ratio > self.ILLUMINATION_THRESHOLD
            )

            if illumination_shift:
                # Không cho frame sốc sáng tạo contour; mô hình mới học lại nền
                # ngay trên frame hiện tại nhưng bỏ qua đánh giá an toàn của frame.
                self._background_subtractor = self._create_background_subtractor()
                self._background_subtractor.apply(
                    blurred_frame,
                    learningRate=1.0,
                )
                detections: tuple[Detection, ...] = ()
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
