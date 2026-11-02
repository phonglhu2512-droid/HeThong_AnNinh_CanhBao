"""Đánh giá xâm nhập ROI, chốt báo động và điều khiển còi bất đồng bộ."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeAlias

import cv2
import numpy as np

_winsound_play_sound: Callable[[str | None, int], None] | None = None
_winsound_snd_alias: int = 0
_winsound_snd_async: int = 0
_winsound_snd_loop: int = 0
try:
    from winsound import (
        PlaySound as _winsound_play_sound,
        SND_ALIAS as _winsound_snd_alias,
        SND_ASYNC as _winsound_snd_async,
        SND_LOOP as _winsound_snd_loop,
    )
except ImportError:  # Cho phép kiểm thử logic ở môi trường không phải Windows.
    pass


LOGGER = logging.getLogger(__name__)
FootPoint: TypeAlias = tuple[int, int]
ROIPoint: TypeAlias = tuple[int, int]


class SafetyState(str, Enum):
    """Các trạng thái vận hành an toàn mà HMI cần hiển thị."""

    SAFE = "SAFE"
    WARNING = "WARNING"
    ALARM = "ALARM"

    @property
    def display_color(self) -> str:
        """Trả về màu HMI tương ứng với trạng thái an toàn hiện hành.

        Args:
            None.

        Returns:
            Mã màu xanh cho SAFE, vàng cho WARNING hoặc đỏ cho ALARM.
        """
        colors = {
            SafetyState.SAFE: "#16A34A",
            SafetyState.WARNING: "#F59E0B",
            SafetyState.ALARM: "#E11D48",
        }
        return colors[self]


@dataclass(frozen=True)
class SafetyResult:
    """Ảnh chụp trạng thái của bộ điều khiển sau một lần đánh giá frame."""

    state: SafetyState
    inside_roi: bool
    violation_duration: float
    latched: bool
    estop_triggered: bool
    acknowledged: bool
    evaluation_skipped: bool
    foot_point: FootPoint | None


class SafetyController:
    """Quản lý đếm trễ ROI, còi báo động, chốt giữ và cờ E-STOP."""

    DEFAULT_VIOLATION_DELAY: float = 0.8
    ALARM_SOUND_ALIAS: str = "SystemExclamation"
    ALARM_PAUSE_SECONDS: float = 0.05
    AUDIO_JOIN_TIMEOUT_SECONDS: float = 1.0

    def __init__(
        self,
        violation_delay: float = DEFAULT_VIOLATION_DELAY,
        sound_callback: Callable[[], None] | None = None,
    ) -> None:
        """Khởi tạo bộ đánh giá an toàn và tùy chọn âm thanh còi.

        Args:
            violation_delay: Thời gian xâm nhập cần đạt, từ 0 đến 3 giây.
            sound_callback: Hàm phát một tiếng còi; None dùng winsound.Beep.

        Returns:
            None.

        Raises:
            TypeError: Khi thời gian trễ hoặc callback sai kiểu.
            ValueError: Khi thời gian trễ không hữu hạn hoặc ngoài khoảng 0–3.
        """
        self._validate_delay(violation_delay)
        if sound_callback is not None and not callable(sound_callback):
            raise TypeError("sound_callback phải là hàm có thể gọi hoặc None.")

        self._violation_delay = float(violation_delay)
        self._sound_callback = sound_callback
        self._lock: threading.RLock = threading.RLock()
        self._state: SafetyState = SafetyState.SAFE
        self._inside_since: float | None = None
        self._last_foot_point: FootPoint | None = None
        self._inside_roi: bool = False
        self._latched: bool = False
        self._estop_triggered: bool = False
        self._acknowledged: bool = False
        self._is_muted: bool = False
        self._entry_alarm_triggered: bool = False
        self._audio_stop_event: threading.Event | None = None
        self._audio_thread: threading.Thread | None = None
        self._audio_error: str | None = None
        self._closed: bool = False

    @staticmethod
    def _validate_delay(violation_delay: float) -> None:
        """Kiểm tra thời gian trễ trong dải điều chỉnh của cấu hình HMI.

        Args:
            violation_delay: Thời gian trễ báo động tính bằng giây.

        Returns:
            None.

        Raises:
            TypeError: Khi thời gian trễ không phải kiểu số.
            ValueError: Khi thời gian trễ không hữu hạn hoặc ngoài khoảng 0–3.
        """
        if isinstance(violation_delay, bool) or not isinstance(
            violation_delay, (int, float)
        ):
            raise TypeError("violation_delay phải là một số.")
        if not math.isfinite(violation_delay) or not 0 <= violation_delay <= 3:
            raise ValueError("violation_delay phải nằm trong khoảng 0 đến 3 giây.")

    @property
    def violation_delay(self) -> float:
        """Đọc ngưỡng thời gian trễ hiện hành.

        Args:
            None.

        Returns:
            Ngưỡng thời gian tính bằng giây.
        """
        with self._lock:
            return self._violation_delay

    @property
    def state(self) -> SafetyState:
        """Đọc trạng thái SAFE, WARNING hoặc ALARM hiện hành.

        Args:
            None.

        Returns:
            Trạng thái an toàn mới nhất.
        """
        with self._lock:
            return self._state

    @property
    def latched(self) -> bool:
        """Cho biết báo động đang được chốt và chờ nhân viên ACK hay không.

        Args:
            None.

        Returns:
            True khi báo động được chốt giữ.
        """
        with self._lock:
            return self._latched

    @property
    def is_muted(self) -> bool:
        """Cho biết còi báo động đã được nhân viên xác nhận tắt tiếng chưa.

        Args:
            None.

        Returns:
            True khi còi đã mute cho lần xâm nhập hiện tại.
        """
        with self._lock:
            return self._is_muted

    @property
    def estop_triggered(self) -> bool:
        """Đọc cờ mô phỏng tín hiệu dừng máy khẩn cấp.

        Args:
            None.

        Returns:
            True khi báo động đã kích hoạt E-STOP và chưa được xác nhận.
        """
        with self._lock:
            return self._estop_triggered

    @property
    def audio_error(self) -> str | None:
        """Trả về lỗi gần nhất của worker còi để HMI hoặc log xử lý.

        Args:
            None.

        Returns:
            Mô tả lỗi âm thanh, hoặc None nếu worker chưa gặp lỗi.
        """
        with self._lock:
            return self._audio_error

    def set_violation_delay(self, violation_delay: float) -> None:
        """Cập nhật ngưỡng thời gian để áp dụng từ lần đánh giá kế tiếp.

        Args:
            violation_delay: Thời gian trễ mới trong khoảng 0 đến 3 giây.

        Returns:
            None.

        Raises:
            TypeError: Khi thời gian trễ không phải số.
            ValueError: Khi thời gian trễ ngoài khoảng cho phép.
        """
        self._validate_delay(violation_delay)
        with self._lock:
            self._violation_delay = float(violation_delay)

    def evaluate(
        self,
        foot_point: FootPoint | None,
        roi_points: list[ROIPoint] | tuple[ROIPoint, ...],
        *,
        timestamp: float | None = None,
        skip_evaluation: bool = False,
    ) -> SafetyResult:
        """Đánh giá điểm chân với ROI và cập nhật trạng thái thời gian thực.

        Args:
            foot_point: Tọa độ chân cần kiểm tra; None biểu thị không có đối tượng.
            roi_points: Tối thiểu ba đỉnh đa giác ROI theo hệ tọa độ frame.
            timestamp: Mốc đơn điệu tính bằng giây; None dùng time.monotonic().
            skip_evaluation: Bỏ đánh giá frame sốc sáng, giữ trạng thái hiện hành.

        Returns:
            SafetyResult gồm trạng thái, thời lượng, latch và cờ E-STOP hiện tại.

        Raises:
            TypeError: Khi timestamp, điểm chân hoặc tọa độ sai kiểu.
            ValueError: Khi ROI không đủ đỉnh hoặc timestamp không hợp lệ.
            cv2.error: Khi OpenCV không thực hiện được pointPolygonTest.
        """
        validated_point = self._validate_foot_point(foot_point)
        polygon = self._build_roi_polygon(roi_points)
        current_time = time.monotonic() if timestamp is None else timestamp
        if (
            isinstance(current_time, bool)
            or not isinstance(current_time, (int, float))
            or not math.isfinite(current_time)
        ):
            raise ValueError("timestamp phải là thời gian đơn điệu hữu hạn.")

        with self._lock:
            if self._closed:
                raise RuntimeError("SafetyController đã đóng.")
            if skip_evaluation:
                return self._build_result(
                    foot_point=validated_point,
                    timestamp=float(current_time),
                    evaluation_skipped=True,
                )

            inside_roi = self._is_point_inside(validated_point, polygon)
            self._last_foot_point = validated_point
            self._inside_roi = inside_roi

            if inside_roi:
                if self._inside_since is None:
                    self._inside_since = float(current_time)
                    self._entry_alarm_triggered = False
                    self._acknowledged = False
                    self._is_muted = False

                duration = max(0.0, float(current_time) - self._inside_since)
                delay_reached = duration >= self._violation_delay or math.isclose(
                    duration,
                    self._violation_delay,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                if delay_reached:
                    self._state = SafetyState.ALARM
                    if not self._entry_alarm_triggered:
                        self._entry_alarm_triggered = True
                        self._latched = True
                        self._estop_triggered = True
                        self._acknowledged = False
                        if not self._is_muted:
                            self._start_alarm_audio_locked()
                elif not self._latched:
                    self._state = SafetyState.WARNING
            else:
                self._inside_since = None
                self._entry_alarm_triggered = False
                self._state = SafetyState.SAFE
                self._latched = False
                self._estop_triggered = False
                self._acknowledged = False
                self._is_muted = False
                self._silence_alarm_locked()

            return self._build_result(
                foot_point=validated_point,
                timestamp=float(current_time),
                evaluation_skipped=False,
            )

    def acknowledge_alarm(self) -> bool:
        """Xác nhận đã biết sự cố và tắt còi trong khi người còn trong ROI.

        Returns:
            True nếu báo động đang được giữ và đã được mute; False nếu chưa có.

        Raises:
            RuntimeError: Khi controller đã đóng.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("SafetyController đã đóng.")
            if not self._latched:
                return False

            if not self._inside_roi:
                self._state = SafetyState.SAFE
                self._latched = False
                self._estop_triggered = False
                self._acknowledged = False
                self._is_muted = False
                self._silence_alarm_locked()
                return True

            self._is_muted = True
            self._acknowledged = True
            self._silence_alarm_locked()
            self._state = SafetyState.ALARM

        return True

    def close(self) -> None:
        """Dừng worker âm thanh khi ứng dụng đóng.

        Args:
            None.

        Returns:
            None.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            stop_event = self._audio_stop_event
            audio_thread = self._audio_thread
            if stop_event is not None:
                stop_event.set()
            self._stop_alarm_sound_locked()

        self._join_audio_thread(audio_thread)
        with self._lock:
            self._audio_thread = None
            self._audio_stop_event = None

    @staticmethod
    def _validate_foot_point(
        foot_point: FootPoint | None,
    ) -> FootPoint | None:
        """Kiểm tra tọa độ chân là một cặp số nguyên hợp lệ.

        Args:
            foot_point: Tọa độ pixel (x, y) hoặc None.

        Returns:
            Tọa độ nguyên đã chuẩn hóa, hoặc None.

        Raises:
            TypeError: Khi foot_point không phải cặp hai số nguyên.
        """
        if foot_point is None:
            return None
        if (
            not isinstance(foot_point, (tuple, list))
            or len(foot_point) != 2
            or any(
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, np.integer))
                for coordinate in foot_point
            )
        ):
            raise TypeError("foot_point phải là cặp tọa độ nguyên (x, y) hoặc None.")
        return int(foot_point[0]), int(foot_point[1])

    @staticmethod
    def _build_roi_polygon(
        roi_points: list[ROIPoint] | tuple[ROIPoint, ...],
    ) -> np.ndarray:
        """Chuẩn hóa các đỉnh ROI thành contour OpenCV và kiểm tra đầu vào.

        Args:
            roi_points: Danh sách tọa độ nguyên tạo thành đa giác ROI.

        Returns:
            Mảng contour int32 có hình dạng N×1×2 cho pointPolygonTest.

        Raises:
            TypeError: Khi danh sách hoặc tọa độ đỉnh sai kiểu.
            ValueError: Khi ROI có ít hơn ba đỉnh.
        """
        if not isinstance(roi_points, (list, tuple)):
            raise TypeError("roi_points phải là list hoặc tuple.")
        if len(roi_points) < 3:
            raise ValueError("ROI phải có ít nhất 3 đỉnh.")
        normalized_points: list[tuple[int, int]] = []
        for point in roi_points:
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
                or any(
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, (int, np.integer))
                    for coordinate in point
                )
            ):
                raise TypeError("Mỗi đỉnh ROI phải là cặp tọa độ nguyên (x, y).")
            normalized_points.append((int(point[0]), int(point[1])))
        return np.asarray(normalized_points, dtype=np.int32).reshape((-1, 1, 2))

    @staticmethod
    def _is_point_inside(
        foot_point: FootPoint | None,
        polygon: np.ndarray,
    ) -> bool:
        """Kiểm tra điểm chân nằm trong hoặc trên biên vùng ROI.

        Args:
            foot_point: Tọa độ chân cần kiểm tra.
            polygon: Contour ROI đã chuẩn hóa bằng _build_roi_polygon.

        Returns:
            True nếu pointPolygonTest trả về giá trị không âm.
        """
        if foot_point is None:
            return False
        result = cv2.pointPolygonTest(
            polygon,
            (float(foot_point[0]), float(foot_point[1])),
            False,
        )
        return result >= 0

    def _build_result(
        self,
        foot_point: FootPoint | None,
        timestamp: float,
        evaluation_skipped: bool,
    ) -> SafetyResult:
        """Tạo snapshot bất biến của trạng thái controller trong lock.

        Args:
            foot_point: Điểm chân frame hiện tại.
            timestamp: Mốc đơn điệu hiện hành.
            evaluation_skipped: Cho biết frame có bị bỏ qua do sốc sáng không.

        Returns:
            SafetyResult phản ánh trạng thái và thời lượng xâm nhập hiện tại.
        """
        duration = (
            max(0.0, timestamp - self._inside_since)
            if self._inside_since is not None and self._inside_roi
            else 0.0
        )
        return SafetyResult(
            state=self._state,
            inside_roi=self._inside_roi,
            violation_duration=duration,
            latched=self._latched,
            estop_triggered=self._estop_triggered,
            acknowledged=self._acknowledged,
            evaluation_skipped=evaluation_skipped,
            foot_point=foot_point,
        )

    def _start_alarm_audio_locked(self) -> None:
        """Khởi chạy còi trên daemon thread khi có lần kích hoạt ALARM mới.

        Phương thức chỉ được gọi khi đã giữ self._lock.

        Args:
            None.

        Returns:
            None.
        """
        if (
            self._audio_thread is not None
            and self._audio_thread.is_alive()
            and self._audio_stop_event is not None
            and not self._audio_stop_event.is_set()
        ):
            return

        stop_event = threading.Event()
        audio_thread = threading.Thread(
            target=self._alarm_audio_loop,
            args=(stop_event,),
            name="VisionGuard-AlarmAudio",
            daemon=True,
        )
        self._audio_stop_event = stop_event
        self._audio_thread = audio_thread
        self._audio_error = None
        try:
            audio_thread.start()
        except RuntimeError as error:
            self._audio_error = str(error)
            self._audio_thread = None
            self._audio_stop_event = None
            LOGGER.exception("Không thể khởi chạy worker âm thanh báo động.")

    def _alarm_audio_loop(self, stop_event: threading.Event) -> None:
        """Phát lặp tiếng còi ở worker riêng, không chặn xử lý frame.

        Args:
            stop_event: Cờ dừng được kích hoạt bởi ACK, thoát ROI hoặc đóng ứng dụng.

        Returns:
            None.
        """
        try:
            while not stop_event.is_set():
                if self._sound_callback is not None:
                    self._sound_callback()
                elif _winsound_play_sound is not None:
                    _winsound_play_sound(
                        self.ALARM_SOUND_ALIAS,
                        _winsound_snd_alias
                        | _winsound_snd_async
                        | _winsound_snd_loop,
                    )
                    stop_event.wait()
                else:
                    raise RuntimeError(
                        "winsound.PlaySound chỉ khả dụng trên Windows; "
                        "hãy cung cấp sound_callback cho nền tảng khác."
                    )
                if self._sound_callback is not None:
                    stop_event.wait(self.ALARM_PAUSE_SECONDS)
        except Exception as error:
            with self._lock:
                self._audio_error = str(error)
            LOGGER.exception("Worker phát còi báo động gặp lỗi.")

    def _stop_alarm_sound_locked(self) -> None:
        """Dừng playback winsound bất đồng bộ mà không chờ worker kết thúc.

        Phương thức chỉ được gọi khi đã giữ self._lock.

        Args:
            None.

        Returns:
            None.
        """
        if _winsound_play_sound is None:
            return
        try:
            _winsound_play_sound(None, 0)
        except Exception as error:
            self._audio_error = str(error)
            LOGGER.exception("Không thể dừng playback còi báo động.")

    def _silence_alarm_locked(self) -> None:
        """Ngắt worker và playback còi khi đang giữ self._lock.

        Args:
            None.

        Returns:
            None.
        """
        stop_event = self._audio_stop_event
        if stop_event is not None and not stop_event.is_set():
            stop_event.set()
            self._stop_alarm_sound_locked()

    @staticmethod
    def _join_audio_thread(thread: threading.Thread | None) -> None:
        """Chờ worker âm thanh kết thúc sau khi cờ dừng đã được bật.

        Args:
            thread: Worker âm thanh cần chờ; None nếu chưa khởi chạy.

        Returns:
            None.
        """
        if thread is not None and thread.is_alive():
            thread.join(timeout=SafetyController.AUDIO_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                LOGGER.error("Worker âm thanh không dừng trong thời gian chờ.")
