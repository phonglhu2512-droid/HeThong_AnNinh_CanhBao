"""Thu nhận video đa nguồn bằng luồng nền và hàng đợi frame mới nhất."""

from __future__ import annotations

import argparse
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
VideoSource: TypeAlias = int | str


class StreamState(str, Enum):
    """Các trạng thái kết nối luồng video có thể hiển thị trên HMI."""

    STOPPED = "stopped"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass(frozen=True)
class StreamStatus:
    """Thông tin trạng thái luồng để GUI đọc mà không truy cập worker thread."""

    state: StreamState
    message: str


class StreamManager:
    """Quản lý Webcam, video offline và RTSP mà không chặn luồng giao diện."""

    FRAME_SIZE: tuple[int, int] = (640, 360)
    RECONNECT_DELAY_SECONDS: float = 3.0
    JOIN_TIMEOUT_SECONDS: float = 1.0
    SIGNAL_LOST_MESSAGE: str = "MẤT TÍN HIỆU - ĐANG KẾT NỐI LẠI..."

    def scan_available_webcams(self, max_index: int = 5) -> list[int]:
        """Dò webcam; tạm nhả webcam đang chạy rồi khôi phục luồng sau khi quét.

        Args:
            max_index: Cận trên không bao gồm của dải ID webcam cần thử.

        Returns:
            Danh sách ID webcam mở được và đọc thành công ít nhất một frame.

        Raises:
            TypeError: Khi max_index không phải số nguyên.
            ValueError: Khi max_index âm.
            RuntimeError: Khi luồng webcam đang chạy không thể dừng an toàn.
        """
        if isinstance(max_index, bool) or not isinstance(max_index, int):
            raise TypeError("max_index phải là số nguyên.")
        if max_index < 0:
            raise ValueError("max_index không được âm.")

        with self._lifecycle_lock:
            active_source = self.source
            resume_webcam = self.is_running and isinstance(active_source, int)
            if resume_webcam:
                self.stop_stream()
            try:
                return self._probe_available_webcams(max_index)
            finally:
                if resume_webcam:
                    self.start_stream(active_source)

    @staticmethod
    def _probe_available_webcams(max_index: int) -> list[int]:
        """Mở tạm từng webcam, thử frame rồi giải phóng capture ngay.

        Args:
            max_index: Cận trên không bao gồm của dải ID webcam cần thử.

        Returns:
            Danh sách ID webcam đọc được frame thử.
        """
        available: list[int] = []
        for device_id in range(max_index):
            backends: tuple[int | None, ...] = (
                (cv2.CAP_DSHOW, None)
                if sys.platform == "win32"
                else (None,)
            )
            for backend in backends:
                capture: cv2.VideoCapture | None = None
                try:
                    capture = (
                        cv2.VideoCapture(device_id, backend)
                        if backend is not None
                        else cv2.VideoCapture(device_id)
                    )
                    if capture.isOpened():
                        success, frame = capture.read()
                        if success and frame is not None:
                            available.append(device_id)
                            break
                except cv2.error:
                    LOGGER.debug(
                        "Không thể kiểm tra webcam %s bằng backend %s.",
                        device_id,
                        backend,
                        exc_info=True,
                    )
                finally:
                    if capture is not None:
                        try:
                            capture.release()
                        except cv2.error:
                            LOGGER.warning(
                                "Không thể giải phóng webcam %s sau khi dò.",
                                device_id,
                                exc_info=True,
                            )
        return available

    def __init__(
        self,
        source: VideoSource = 0,
        target_size: tuple[int, int] = FRAME_SIZE,
        reconnect_interval: float = RECONNECT_DELAY_SECONDS,
    ) -> None:
        """Tạo bộ quản lý nguồn video và hàng đợi nhận frame/status.

        Args:
            source: Webcam ID, đường dẫn video hoặc URL RTSP/RTSPS.
            target_size: Kích thước frame chuẩn hóa theo thứ tự (rộng, cao).
            reconnect_interval: Thời gian chờ giữa hai lần kết nối lại, tính bằng giây.

        Returns:
            None.

        Raises:
            TypeError: Khi nguồn hoặc kích thước frame sai kiểu dữ liệu.
            ValueError: Khi nguồn, kích thước hoặc thời gian kết nối lại không hợp lệ.
        """
        self._validate_source(source)
        if (
            not isinstance(target_size, tuple)
            or len(target_size) != 2
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int)
                for dimension in target_size
            )
        ):
            raise TypeError("target_size phải là tuple gồm hai số nguyên.")
        if any(dimension <= 0 for dimension in target_size):
            raise ValueError("Chiều rộng và chiều cao frame phải lớn hơn 0.")
        if (
            isinstance(reconnect_interval, bool)
            or not isinstance(reconnect_interval, (int, float))
        ):
            raise TypeError("reconnect_interval phải là một số.")
        if not math.isfinite(reconnect_interval) or reconnect_interval <= 0:
            raise ValueError("reconnect_interval phải là số hữu hạn lớn hơn 0.")

        self._source: VideoSource = source
        self.target_size: tuple[int, int] = target_size
        self.reconnect_interval: float = float(reconnect_interval)
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue[StreamStatus] = queue.Queue(maxsize=1)

        self.cap: cv2.VideoCapture | None = None
        self._stop_event: threading.Event = threading.Event()
        self._stop_event.set()
        self._state_lock: threading.RLock = threading.RLock()
        self._capture_lock: threading.Lock = threading.Lock()
        self._lifecycle_lock: threading.RLock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._status: StreamStatus = StreamStatus(
            StreamState.STOPPED, "ĐÃ DỪNG LUỒNG VIDEO"
        )
        self.status_queue.put_nowait(self._status)

    @staticmethod
    def _validate_source(source: VideoSource) -> None:
        """Xác thực nguồn video trước khi mở VideoCapture.

        Args:
            source: Webcam ID, đường dẫn video hoặc URL luồng mạng.

        Returns:
            None.

        Raises:
            TypeError: Khi nguồn không phải số nguyên hoặc chuỗi.
            ValueError: Khi ID webcam âm hoặc chuỗi nguồn rỗng.
        """
        if isinstance(source, bool) or not isinstance(source, (int, str)):
            raise TypeError("source phải là ID webcam dạng số nguyên hoặc chuỗi.")
        if isinstance(source, int) and source < 0:
            raise ValueError("ID webcam không được âm.")
        if isinstance(source, str) and not source.strip():
            raise ValueError("Đường dẫn hoặc URL nguồn video không được để trống.")

    @property
    def source(self) -> VideoSource:
        """Trả về nguồn video hiện tại theo cách an toàn giữa các luồng.

        Args:
            None.

        Returns:
            ID webcam, đường dẫn video hoặc URL RTSP hiện tại.
        """
        with self._state_lock:
            return self._source

    @property
    def status(self) -> StreamStatus:
        """Trả về trạng thái kết nối hiện tại để HMI cập nhật nội dung hiển thị.

        Args:
            None.

        Returns:
            Trạng thái luồng và thông báo tiếng Việt tương ứng.
        """
        with self._state_lock:
            return self._status

    @property
    def is_running(self) -> bool:
        """Cho biết worker thread đang hoạt động hay không.

        Args:
            None.

        Returns:
            True nếu worker thread còn sống và chưa nhận yêu cầu dừng.
        """
        with self._state_lock:
            thread = self._thread
            return (
                thread is not None
                and thread.is_alive()
                and not self._stop_event.is_set()
            )

    @property
    def signal_lost(self) -> bool:
        """Cho biết nguồn video đang mất tín hiệu hoặc trong chu kỳ kết nối lại.

        Args:
            None.

        Returns:
            True khi luồng đang kết nối lại hoặc gặp lỗi; ngược lại là False.
        """
        with self._state_lock:
            return self._status.state in (
                StreamState.RECONNECTING,
                StreamState.ERROR,
            )

    def start_stream(self, source: VideoSource | None = None) -> None:
        """Khởi chạy luồng đọc mới, giải phóng luồng cũ trước khi đổi nguồn.

        Args:
            source: Nguồn mới; bỏ trống để tiếp tục dùng nguồn hiện tại.

        Returns:
            None.

        Raises:
            TypeError: Khi nguồn mới sai kiểu dữ liệu.
            ValueError: Khi nguồn mới không hợp lệ.
            RuntimeError: Khi worker cũ không dừng được hoặc không tạo được thread mới.
        """
        if source is not None:
            self._validate_source(source)

        with self._lifecycle_lock:
            self.stop_stream()
            with self._state_lock:
                if source is not None:
                    self._source = source
                active_source = self._source

            self._clear_frame_queue()
            self._stop_event.clear()
            self._set_status(
                StreamState.CONNECTING, "ĐANG KẾT NỐI TÍN HIỆU CAMERA..."
            )
            thread = threading.Thread(
                target=self._capture_loop,
                args=(active_source,),
                name="VisionGuard-CameraIO",
                daemon=True,
            )
            with self._state_lock:
                self._thread = thread
            try:
                thread.start()
            except RuntimeError:
                self._stop_event.set()
                self._set_status(
                    StreamState.ERROR, "KHÔNG THỂ KHỞI CHẠY LUỒNG CAMERA."
                )
                LOGGER.exception("Không thể khởi chạy worker thread đọc camera.")
                with self._state_lock:
                    self._thread = None
                raise

    def switch_source(self, source: VideoSource) -> None:
        """Chuyển nguồn video mà không cần khởi động lại ứng dụng.

        Args:
            source: Webcam ID, đường dẫn video hoặc URL RTSP mới.

        Returns:
            None.

        Raises:
            TypeError: Khi nguồn mới sai kiểu dữ liệu.
            ValueError: Khi nguồn mới không hợp lệ.
            RuntimeError: Khi worker hiện tại không dừng được.
        """
        self.start_stream(source)

    def stop_stream(self) -> None:
        """Dừng worker, giải phóng camera và xóa frame tồn đọng khỏi hàng đợi.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: Khi thread vẫn hoạt động sau khi được yêu cầu dừng.
        """
        with self._lifecycle_lock:
            self._stop_event.set()
            with self._state_lock:
                thread = self._thread

            if thread is not None and thread.is_alive():
                thread.join(timeout=self.JOIN_TIMEOUT_SECONDS)
            if thread is not None and thread.is_alive():
                # release() giúp giải phóng một số backend đang chặn ở lệnh read().
                self._release_capture()
                thread.join(timeout=self.JOIN_TIMEOUT_SECONDS)

            if thread is not None and thread.is_alive():
                self._set_status(
                    StreamState.ERROR,
                    "KHÔNG THỂ DỪNG LUỒNG CAMERA AN TOÀN.",
                )
                LOGGER.error("Worker thread camera vẫn hoạt động sau khi dừng.")
                raise RuntimeError("Worker thread camera không dừng đúng hạn.")

            self._release_capture()
            self._clear_frame_queue()
            self._set_status(StreamState.STOPPED, "ĐÃ DỪNG LUỒNG VIDEO")
            with self._state_lock:
                self._thread = None

    def read_latest_frame(self, timeout: float = 0.0) -> np.ndarray | None:
        """Lấy frame mới nhất; trả về None khi hàng đợi chưa có frame.

        Args:
            timeout: Thời gian chờ frame mới, tính bằng giây; 0 là không chờ.

        Returns:
            Frame BGR đã chuẩn hóa kích thước, hoặc None nếu hết thời gian chờ.

        Raises:
            ValueError: Khi timeout âm hoặc không hữu hạn.
        """
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout phải là số hữu hạn không âm.")
        try:
            if timeout == 0:
                return self.frame_queue.get_nowait()
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_latest_status(self, timeout: float = 0.0) -> StreamStatus | None:
        """Lấy trạng thái mới nhất để HMI cập nhật mà không gọi UI từ worker.

        Args:
            timeout: Thời gian chờ trạng thái mới, tính bằng giây; 0 là không chờ.

        Returns:
            Trạng thái mới nhất, hoặc None nếu hết thời gian chờ.

        Raises:
            ValueError: Khi timeout âm hoặc không hữu hạn.
        """
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout phải là số hữu hạn không âm.")
        try:
            if timeout == 0:
                return self.status_queue.get_nowait()
            return self.status_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self, source: VideoSource) -> None:
        """Đọc, chuẩn hóa và đẩy frame vào hàng đợi trong worker daemon.

        Args:
            source: Nguồn video cố định cho vòng đời của worker hiện tại.

        Returns:
            None.
        """
        connected_before = False
        is_file = self._is_file_source(source)
        frame_interval = 1.0 / 25.0

        try:
            while not self._stop_event.is_set():
                try:
                    capture = self._get_capture()
                    if capture is None:
                        capture = self._open_capture(source)
                        if capture is None:
                            self._set_status(
                                StreamState.RECONNECTING, self.SIGNAL_LOST_MESSAGE
                            )
                            self._stop_event.wait(self.reconnect_interval)
                            continue

                        connected_before = True
                        if is_file:
                            frame_interval = self._get_file_frame_interval(capture)
                        self._set_status(
                            StreamState.CONNECTED, "ĐÃ KẾT NỐI TÍN HIỆU VIDEO"
                        )

                    frame_started = time.monotonic()
                    success, frame = capture.read()
                    if not success or frame is None:
                        if is_file and capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                            # Tạm nghỉ để tệp rỗng/hỏng không tạo vòng lặp chiếm CPU.
                            self._stop_event.wait(min(frame_interval, 0.05))
                            continue

                        message = (
                            self.SIGNAL_LOST_MESSAGE
                            if connected_before
                            else "KHÔNG THỂ ĐỌC FRAME VIDEO."
                        )
                        self._set_status(StreamState.RECONNECTING, message)
                        self._clear_frame_queue()
                        self._release_capture()
                        self._stop_event.wait(self.reconnect_interval)
                        continue

                    resized_frame = cv2.resize(frame, self.target_size)
                    self._enqueue_latest_frame(resized_frame)

                    if is_file:
                        remaining = frame_interval - (
                            time.monotonic() - frame_started
                        )
                        if remaining > 0:
                            self._stop_event.wait(remaining)
                except Exception:
                    LOGGER.exception(
                        "Worker camera gặp lỗi khi đọc nguồn %r.", source
                    )
                    if not self._stop_event.is_set():
                        self._set_status(
                            StreamState.RECONNECTING, self.SIGNAL_LOST_MESSAGE
                        )
                        self._clear_frame_queue()
                        self._release_capture()
                        self._stop_event.wait(self.reconnect_interval)
        finally:
            self._release_capture()
            if self._stop_event.is_set():
                self._set_status(StreamState.STOPPED, "ĐÃ DỪNG LUỒNG VIDEO")

    @staticmethod
    def _is_file_source(source: VideoSource) -> bool:
        """Phân biệt URL camera mạng với đường dẫn tệp video.

        Args:
            source: Nguồn video cần phân loại.

        Returns:
            True nếu nguồn là đường dẫn tệp, False nếu là webcam hoặc RTSP.
        """
        return isinstance(source, str) and not source.lower().startswith(
            ("rtsp://", "rtsps://")
        )

    @staticmethod
    def _get_file_frame_interval(capture: cv2.VideoCapture) -> float:
        """Tính khoảng cách giữa frame của video offline theo FPS nguồn.

        Args:
            capture: Đối tượng VideoCapture đang mở tệp video.

        Returns:
            Khoảng thời gian phát mỗi frame; mặc định 1/25 giây nếu FPS lỗi.
        """
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0 or fps > 120:
            fps = 25.0
        return 1.0 / fps

    def _open_capture(self, source: VideoSource) -> cv2.VideoCapture | None:
        """Mở nguồn bằng OpenCV và công bố capture để stop có thể giải phóng.

        Args:
            source: Webcam ID, đường dẫn video hoặc URL RTSP/RTSPS.

        Returns:
            Đối tượng VideoCapture đang mở, hoặc None nếu mở nguồn thất bại.

        Raises:
            cv2.error: Khi OpenCV phát sinh lỗi trong quá trình mở nguồn.
        """
        capture = cv2.VideoCapture(source)
        with self._capture_lock:
            self.cap = capture

        if self._stop_event.is_set():
            self._release_capture()
            return None
        if not capture.isOpened():
            self._release_capture()
            return None
        return capture

    def _get_capture(self) -> cv2.VideoCapture | None:
        """Đọc tham chiếu capture hiện tại an toàn giữa worker và luồng gọi.

        Args:
            None.

        Returns:
            Đối tượng VideoCapture hiện tại, hoặc None nếu chưa mở.
        """
        with self._capture_lock:
            return self.cap

    def _release_capture(self) -> None:
        """Giải phóng capture hiện tại và xóa tham chiếu dùng chung.

        Args:
            None.

        Returns:
            None.
        """
        with self._capture_lock:
            capture = self.cap
            self.cap = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                LOGGER.exception("Không thể giải phóng VideoCapture.")

    def _enqueue_latest_frame(self, frame: np.ndarray) -> None:
        """Thay frame cũ bằng frame mới nhất trong queue có kích thước một.

        Args:
            frame: Frame BGR đã được chuẩn hóa kích thước.

        Returns:
            None.

        Raises:
            queue.Full: Khi không thể chèn frame mới do tranh chấp queue.
        """
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Giữ ưu tiên frame mới nhất nếu consumer đọc đúng lúc queue thay đổi.
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(frame)

    def _clear_frame_queue(self) -> None:
        """Xóa frame tồn đọng khi đổi nguồn hoặc dừng luồng.

        Args:
            None.

        Returns:
            None.
        """
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
            self.frame_queue.not_full.notify_all()

    def _set_status(self, state: StreamState, message: str) -> None:
        """Cập nhật trạng thái và phát sự kiện mới nhất cho GUI qua queue.

        Args:
            state: Trạng thái kết nối hiện tại.
            message: Nội dung tiếng Việt hiển thị trên HMI.

        Returns:
            None.
        """
        status = StreamStatus(state, message)
        with self._state_lock:
            if status == self._status:
                return
            self._status = status

            if self.status_queue.full():
                try:
                    self.status_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.status_queue.put_nowait(status)
            except queue.Full:
                LOGGER.warning("Không thể cập nhật status queue của luồng camera.")


def main() -> int:
    """Chạy màn hình thử độc lập để kiểm tra nguồn camera hoặc video.

    Args:
        None. Tùy chọn dòng lệnh ``--source`` nhận webcam ID, đường dẫn video
        hoặc URL RTSP; nhấn ``q`` trên cửa sổ video để thoát.

    Returns:
        Mã thoát 0 nếu chạy và dừng bình thường, 1 nếu gặp lỗi vận hành.
    """
    for output_stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(output_stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Kiểm tra độc lập StreamManager của VisionGuard AI."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam ID, tệp .mp4/.avi hoặc URL RTSP (mặc định: 0).",
    )
    arguments = parser.parse_args()
    source: VideoSource = (
        int(arguments.source)
        if arguments.source.isdecimal()
        else arguments.source
    )

    manager = StreamManager(source=source)
    last_status: StreamStatus | None = None
    exit_code = 0
    try:
        manager.start_stream()
        while manager.is_running:
            status = manager.read_latest_status(timeout=0.01)
            if status is not None and status != last_status:
                print(status.message, flush=True)
                last_status = status

            frame = manager.read_latest_frame(timeout=0.05)
            if frame is not None:
                cv2.imshow("VisionGuard - StreamManager Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception:
        LOGGER.exception("Lỗi khi chạy kiểm tra độc lập StreamManager.")
        exit_code = 1
    finally:
        try:
            manager.stop_stream()
        except Exception:
            LOGGER.exception("Không thể dừng luồng camera trong chế độ kiểm tra.")
            exit_code = 1
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            LOGGER.exception("Không thể đóng cửa sổ kiểm tra OpenCV.")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
