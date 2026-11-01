"""Lưu ảnh vi phạm bất đồng bộ và quản lý nhật ký kiểm toán trong bộ nhớ."""

from __future__ import annotations

import csv
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, time as datetime_time
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
FootPoint: TypeAlias = tuple[int, int]
BoundingBox: TypeAlias = tuple[int, int, int, int]
TimeFilter: TypeAlias = str | datetime_time
_STOP_WORKER = object()


@dataclass(frozen=True)
class AuditEvent:
    """Bản ghi kiểm toán của một sự kiện xâm nhập vùng nguy hiểm."""

    event_id: str
    occurred_at: datetime
    violation_duration: float
    foot_point: FootPoint
    image_path: Path


@dataclass(frozen=True)
class _EvidenceJob:
    """Công việc lưu ảnh đã được sao chép và chú thích trước khi xếp hàng."""

    event: AuditEvent
    annotated_frame: np.ndarray


class AuditLogger:
    """Quản lý sự kiện, ảnh bằng chứng, lọc ca và xuất CSV cho Excel."""

    QUEUE_MAX_SIZE: int = 128
    WORKER_JOIN_TIMEOUT_SECONDS: float = 5.0
    CSV_HEADERS: tuple[str, ...] = (
        "Mã sự kiện",
        "Thời gian",
        "Thời lượng vi phạm (s)",
        "Tọa độ chân",
        "Đường dẫn ảnh",
    )
    SHIFT_RANGES: dict[int, tuple[datetime_time, datetime_time]] = {
        1: (datetime_time(6, 0), datetime_time(14, 0)),
        2: (datetime_time(14, 0), datetime_time(22, 0)),
        3: (datetime_time(22, 0), datetime_time(6, 0)),
    }

    def __init__(
        self,
        output_root: str | Path = "output_data",
        queue_max_size: int = QUEUE_MAX_SIZE,
    ) -> None:
        """Tạo thư mục bằng chứng/báo cáo và khởi động worker ghi ảnh.

        Args:
            output_root: Thư mục gốc lưu dữ liệu chạy ứng dụng.
            queue_max_size: Số ảnh tối đa đang chờ ghi để giới hạn RAM.

        Returns:
            None.

        Raises:
            TypeError: Khi queue_max_size không phải số nguyên.
            ValueError: Khi queue_max_size không lớn hơn 0.
            OSError: Khi không thể tạo thư mục bằng chứng hoặc báo cáo.
            RuntimeError: Khi không thể khởi chạy worker ghi ảnh.
        """
        if isinstance(queue_max_size, bool) or not isinstance(queue_max_size, int):
            raise TypeError("queue_max_size phải là số nguyên.")
        if queue_max_size <= 0:
            raise ValueError("queue_max_size phải lớn hơn 0.")

        self.output_root: Path = Path(output_root)
        self.evidence_dir: Path = self.output_root / "evidence"
        self.reports_dir: Path = self.output_root / "reports"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.evidence_io_queue: queue.Queue[_EvidenceJob | object] = queue.Queue(
            maxsize=queue_max_size
        )
        self._lock: threading.RLock = threading.RLock()
        self._events: list[AuditEvent] = []
        self._write_errors: dict[str, str] = {}
        self._reserved_image_paths: set[Path] = set()
        self._next_event_number: int = 1
        self._closed: bool = False
        self._sentinel_enqueued: bool = False
        self._close_lock: threading.Lock = threading.Lock()
        self._worker: threading.Thread = threading.Thread(
            target=self._evidence_worker,
            name="VisionGuard-EvidenceIO",
            daemon=True,
        )
        try:
            self._worker.start()
        except RuntimeError:
            LOGGER.exception("Không thể khởi chạy worker ghi ảnh bằng chứng.")
            raise

    @property
    def write_errors(self) -> dict[str, str]:
        """Trả về bản sao các lỗi lưu ảnh theo mã sự kiện.

        Args:
            None.

        Returns:
            Dictionary event_id -> thông báo lỗi lưu ảnh.
        """
        with self._lock:
            return dict(self._write_errors)

    @property
    def is_running(self) -> bool:
        """Cho biết worker ghi ảnh còn hoạt động hay không.

        Args:
            None.

        Returns:
            True nếu worker daemon còn sống.
        """
        return self._worker.is_alive()

    def log_violation(
        self,
        frame: np.ndarray,
        bounding_box: BoundingBox,
        foot_point: FootPoint,
        violation_duration: float,
        occurred_at: datetime | None = None,
    ) -> AuditEvent:
        """Tạo bản ghi vi phạm và xếp ảnh chú thích vào worker ghi đĩa.

        Args:
            frame: Frame BGR cần lưu làm bằng chứng.
            bounding_box: Hộp bao (x, y, width, height) theo tọa độ frame.
            foot_point: Tọa độ chân tiếp xúc sàn (x, y).
            violation_duration: Thời lượng xâm nhập tính bằng giây.
            occurred_at: Thời điểm sự kiện; None dùng giờ địa phương hiện tại.

        Returns:
            Bản ghi AuditEvent đã được thêm vào nhật ký trong bộ nhớ.

        Raises:
            TypeError: Khi frame, bounding box, foot point hoặc thời lượng sai kiểu.
            ValueError: Khi frame rỗng, tọa độ/box không hợp lệ hoặc thời lượng âm.
            queue.Full: Khi EvidenceIOQueue đầy; không thêm bản ghi thành công giả.
            RuntimeError: Khi logger đã đóng hoặc worker không còn hoạt động.
            cv2.error: Khi OpenCV không thể vẽ annotation.
        """
        self._validate_frame(frame)
        validated_box = self._validate_bounding_box(bounding_box)
        validated_foot = self._validate_point(foot_point, "foot_point")
        validated_duration = self._validate_duration(violation_duration)
        event_time = datetime.now() if occurred_at is None else occurred_at
        if not isinstance(event_time, datetime):
            raise TypeError("occurred_at phải là datetime hoặc None.")

        annotated_frame = self._annotate_frame(
            frame.copy(),
            validated_box,
            validated_foot,
            event_time,
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("AuditLogger đã đóng.")
            if not self._worker.is_alive():
                raise RuntimeError("Worker ghi ảnh bằng chứng không còn hoạt động.")

            event_id = f"VIO-{self._next_event_number:03d}"
            image_path = self._build_unique_image_path(event_time, event_id)
            event = AuditEvent(
                event_id=event_id,
                occurred_at=event_time,
                violation_duration=validated_duration,
                foot_point=validated_foot,
                image_path=image_path,
            )
            job = _EvidenceJob(event, annotated_frame)
            try:
                self.evidence_io_queue.put_nowait(job)
            except queue.Full:
                LOGGER.error(
                    "EvidenceIOQueue đầy; từ chối ghi nhận sự kiện %s.",
                    event_id,
                )
                raise

            self._reserved_image_paths.add(image_path)
            self._events.append(event)
            self._next_event_number += 1
            return event

    def update_violation_duration(
        self,
        event_id: str,
        violation_duration: float,
    ) -> AuditEvent:
        """Cập nhật tổng thời lượng khi đối tượng rời vùng cấm.

        Args:
            event_id: Mã sự kiện cần cập nhật.
            violation_duration: Tổng thời lượng xâm nhập cuối cùng tính bằng giây.

        Returns:
            Bản ghi đã cập nhật.

        Raises:
            TypeError: Khi mã sự kiện không phải chuỗi hoặc thời lượng sai kiểu.
            ValueError: Khi thời lượng âm hoặc không hữu hạn.
            KeyError: Khi không tìm thấy mã sự kiện trong nhật ký.
        """
        if not isinstance(event_id, str):
            raise TypeError("event_id phải là chuỗi.")
        duration = self._validate_duration(violation_duration)
        with self._lock:
            for index, event in enumerate(self._events):
                if event.event_id == event_id:
                    updated_event = replace(
                        event,
                        violation_duration=duration,
                    )
                    self._events[index] = updated_event
                    return updated_event
        raise KeyError(f"Không tìm thấy sự kiện {event_id}.")

    def get_events(self) -> list[AuditEvent]:
        """Lấy snapshot nhật ký trong bộ nhớ theo thứ tự phát sinh.

        Args:
            None.

        Returns:
            Danh sách mới chứa các bản ghi AuditEvent hiện có.
        """
        with self._lock:
            return list(self._events)

    def filter_by_shift(
        self,
        shift: int | str,
        events: list[AuditEvent] | None = None,
    ) -> list[AuditEvent]:
        """Lọc sự kiện theo ca 1, 2 hoặc 3; ca 3 đi qua nửa đêm.

        Args:
            shift: Số ca 1–3 hoặc chuỗi như "Ca 1".
            events: Danh sách cần lọc; None dùng toàn bộ nhật ký nội bộ.

        Returns:
            Danh sách sự kiện có giờ địa phương nằm trong ca đã chọn.

        Raises:
            TypeError: Khi shift không phải số hoặc chuỗi.
            ValueError: Khi shift không thuộc Ca 1, Ca 2 hoặc Ca 3.
        """
        shift_number = self._parse_shift(shift)
        start_time, end_time = self.SHIFT_RANGES[shift_number]
        source_events = self._resolve_events(events)
        return [
            event
            for event in source_events
            if self._time_in_range(
                event.occurred_at.time(),
                start_time,
                end_time,
            )
        ]

    def filter_by_time_range(
        self,
        start_time: TimeFilter,
        end_time: TimeFilter,
        events: list[AuditEvent] | None = None,
    ) -> list[AuditEvent]:
        """Lọc sự kiện theo khoảng giờ, hỗ trợ khoảng qua nửa đêm.

        Args:
            start_time: Giờ bắt đầu dạng datetime.time hoặc chuỗi HH:MM[:SS].
            end_time: Giờ kết thúc dạng datetime.time hoặc chuỗi HH:MM[:SS].
            events: Danh sách cần lọc; None dùng toàn bộ nhật ký nội bộ.

        Returns:
            Sự kiện từ giờ bắt đầu (bao gồm) đến giờ kết thúc (không bao gồm).

        Raises:
            TypeError: Khi thời điểm có kiểu không hỗ trợ.
            ValueError: Khi chuỗi giờ không hợp lệ hoặc hai mốc bằng nhau.
        """
        start = self._parse_time(start_time)
        end = self._parse_time(end_time)
        if start == end:
            raise ValueError("Giờ bắt đầu và kết thúc không được trùng nhau.")
        source_events = self._resolve_events(events)
        return [
            event
            for event in source_events
            if self._time_in_range(event.occurred_at.time(), start, end)
        ]

    def export_csv(
        self,
        events: list[AuditEvent] | None = None,
        exported_at: datetime | None = None,
    ) -> Path:
        """Xuất nhật ký ra CSV UTF-8 with BOM trong thư mục reports.

        Args:
            events: Danh sách cần xuất; None xuất toàn bộ nhật ký nội bộ.
            exported_at: Thời gian dùng đặt tên; None dùng giờ địa phương hiện tại.

        Returns:
            Đường dẫn tuyệt đối của báo cáo CSV đã ghi thành công.

        Raises:
            TypeError: Khi danh sách sự kiện hoặc exported_at sai kiểu.
            OSError: Khi không thể tạo hoặc ghi tệp CSV.
            csv.Error: Khi writer gặp lỗi định dạng dữ liệu.
        """
        source_events = self._resolve_events(events)
        timestamp = datetime.now() if exported_at is None else exported_at
        if not isinstance(timestamp, datetime):
            raise TypeError("exported_at phải là datetime hoặc None.")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._build_unique_report_path(timestamp)
        temporary_path = report_path.with_suffix(".csv.tmp")

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as report_file:
                writer = csv.writer(report_file)
                writer.writerow(self.CSV_HEADERS)
                for event in source_events:
                    writer.writerow(self._event_to_csv_row(event))
                report_file.flush()
            temporary_path.replace(report_path)
        except (OSError, csv.Error):
            LOGGER.exception("Không thể xuất báo cáo CSV tại %s.", report_path)
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Không thể dọn tệp CSV tạm %s.", temporary_path)
            raise
        return report_path

    def wait_until_idle(self, timeout: float = 5.0) -> bool:
        """Chờ queue ghi ảnh hoàn tất trong tối đa timeout giây.

        Args:
            timeout: Thời lượng chờ tối đa, tính bằng giây.

        Returns:
            True nếu queue đã xử lý hết công việc; False nếu hết thời gian.

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
        deadline = time.monotonic() + timeout
        while self.evidence_io_queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    def close(self, timeout: float = WORKER_JOIN_TIMEOUT_SECONDS) -> None:
        """Dừng worker sau khi xử lý hết ảnh đang chờ và chờ thread kết thúc.

        Args:
            timeout: Thời gian tối đa chờ worker kết thúc, tính bằng giây.

        Returns:
            None.

        Raises:
            ValueError: Khi timeout âm hoặc không hữu hạn.
            RuntimeError: Khi queue đầy không nhận sentinel hoặc worker không dừng.
        """
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout phải là số hữu hạn không âm.")
        with self._close_lock:
            with self._lock:
                self._closed = True
                worker = self._worker

            if worker.is_alive() and not self._sentinel_enqueued:
                try:
                    self.evidence_io_queue.put(
                        _STOP_WORKER,
                        timeout=timeout,
                    )
                except queue.Full as error:
                    LOGGER.error("Không thể dừng worker vì EvidenceIOQueue đầy.")
                    raise RuntimeError(
                        "Không thể đưa tín hiệu dừng vào EvidenceIOQueue."
                    ) from error
                self._sentinel_enqueued = True

            worker.join(timeout=timeout)
            if worker.is_alive():
                LOGGER.error("Worker ghi ảnh chưa dừng sau %.2f giây.", timeout)
                raise RuntimeError("Worker ghi ảnh không dừng đúng thời hạn.")

    def _evidence_worker(self) -> None:
        """Tiêu thụ EvidenceIOQueue và ghi từng ảnh bằng cv2.imwrite.

        Args:
            None.

        Returns:
            None.
        """
        while True:
            job = self.evidence_io_queue.get()
            try:
                if job is _STOP_WORKER:
                    return
                if not isinstance(job, _EvidenceJob):
                    LOGGER.error("Bỏ qua đối tượng không hợp lệ trong EvidenceIOQueue.")
                    continue
                self._write_evidence(job)
            except Exception:
                LOGGER.exception("Worker xử lý tác vụ EvidenceIOQueue thất bại.")
            finally:
                self.evidence_io_queue.task_done()

    def _write_evidence(self, job: _EvidenceJob) -> None:
        """Ghi một frame bằng chứng; lưu lỗi để caller không hiểu nhầm thành công.

        Args:
            job: Công việc gồm metadata và frame đã annotate.

        Returns:
            None.
        """
        try:
            job.event.image_path.parent.mkdir(parents=True, exist_ok=True)
            saved = cv2.imwrite(
                str(job.event.image_path),
                job.annotated_frame,
            )
            if not saved:
                raise OSError("cv2.imwrite trả về False.")
        except Exception as error:
            message = str(error)
            with self._lock:
                self._write_errors[job.event.event_id] = message
            LOGGER.exception(
                "Không thể lưu ảnh bằng chứng của sự kiện %s tại %s.",
                job.event.event_id,
                job.event.image_path,
            )
        else:
            with self._lock:
                self._write_errors.pop(job.event.event_id, None)

    @staticmethod
    def _annotate_frame(
        frame: np.ndarray,
        bounding_box: BoundingBox,
        foot_point: FootPoint,
        occurred_at: datetime,
    ) -> np.ndarray:
        """Vẽ hộp đỏ, điểm chân và dấu thời gian lên bản sao frame.

        Args:
            frame: Frame BGR đã được sao chép khỏi buffer của caller.
            bounding_box: Hộp bao theo dạng x, y, rộng, cao.
            foot_point: Tọa độ pixel của điểm chân.
            occurred_at: Thời điểm cần in lên ảnh.

        Returns:
            Frame BGR đã gắn annotation.

        Raises:
            cv2.error: Khi một thao tác vẽ OpenCV thất bại.
        """
        x, y, width, height = bounding_box
        red = (0, 0, 255)
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            red,
            2,
        )
        cv2.circle(frame, foot_point, 5, red, thickness=-1)
        timestamp_text = occurred_at.strftime("%Y-%m-%d %H:%M:%S")
        text_origin = (10, max(24, min(frame.shape[0] - 8, 28)))
        cv2.putText(
            frame,
            timestamp_text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            red,
            2,
            cv2.LINE_AA,
        )
        return frame

    def _build_unique_image_path(
        self,
        occurred_at: datetime,
        event_id: str,
    ) -> Path:
        """Tạo đường dẫn ảnh theo chuẩn giây và thêm hậu tố khi trùng tên.

        Args:
            occurred_at: Thời điểm sự kiện dùng tạo timestamp tên ảnh.
            event_id: Mã sự kiện dùng phân biệt khi nhiều sự cố cùng giây.

        Returns:
            Đường dẫn chưa tồn tại bên trong output_data/evidence.
        """
        base_name = f"VIO_{occurred_at:%Y%m%d_%H%M%S}"
        image_path = self.evidence_dir / f"{base_name}.jpg"
        if (
            not image_path.exists()
            and image_path not in self._reserved_image_paths
        ):
            return image_path
        candidate = self.evidence_dir / f"{base_name}_{event_id}.jpg"
        if candidate.exists() or candidate in self._reserved_image_paths:
            suffix = 2
            while True:
                candidate = self.evidence_dir / (
                    f"{base_name}_{event_id}_{suffix:02d}.jpg"
                )
                if (
                    not candidate.exists()
                    and candidate not in self._reserved_image_paths
                ):
                    break
                suffix += 1
        return candidate

    def _build_unique_report_path(self, exported_at: datetime) -> Path:
        """Tạo tên báo cáo chuẩn, tránh ghi đè báo cáo cùng giây.

        Args:
            exported_at: Thời gian xuất dùng tạo tên tệp.

        Returns:
            Đường dẫn CSV chưa tồn tại trong thư mục reports.
        """
        base_name = f"Violation_Report_{exported_at:%Y%m%d_%H%M%S}"
        report_path = self.reports_dir / f"{base_name}.csv"
        if not report_path.exists():
            return report_path
        suffix = 2
        while True:
            candidate = self.reports_dir / f"{base_name}_{suffix:02d}.csv"
            if not candidate.exists():
                return candidate
            suffix += 1

    def _resolve_events(
        self,
        events: list[AuditEvent] | None,
    ) -> list[AuditEvent]:
        """Lấy nhật ký nội bộ hoặc sao chép danh sách do caller chỉ định.

        Args:
            events: Danh sách sự kiện tùy chọn.

        Returns:
            Danh sách snapshot để lọc hoặc xuất.

        Raises:
            TypeError: Khi events không phải list AuditEvent hoặc None.
        """
        if events is None:
            return self.get_events()
        if not isinstance(events, list) or any(
            not isinstance(event, AuditEvent) for event in events
        ):
            raise TypeError("events phải là list[AuditEvent] hoặc None.")
        return list(events)

    @staticmethod
    def _parse_shift(shift: int | str) -> int:
        """Chuẩn hóa số ca hoặc chuỗi Ca N về số nguyên 1–3.

        Args:
            shift: Số ca hoặc chuỗi như 'Ca 1'.

        Returns:
            Số ca 1, 2 hoặc 3.

        Raises:
            TypeError: Khi shift không phải số nguyên hoặc chuỗi.
            ValueError: Khi ca không thuộc 1, 2 hoặc 3.
        """
        if isinstance(shift, bool) or not isinstance(shift, (int, str)):
            raise TypeError("shift phải là số nguyên hoặc chuỗi 'Ca N'.")
        if isinstance(shift, str):
            normalized = shift.strip().lower().replace("ca", "").strip()
            if not normalized.isdecimal():
                raise ValueError("shift phải là Ca 1, Ca 2 hoặc Ca 3.")
            shift_number = int(normalized)
        else:
            shift_number = shift
        if shift_number not in AuditLogger.SHIFT_RANGES:
            raise ValueError("shift phải là Ca 1, Ca 2 hoặc Ca 3.")
        return shift_number

    @staticmethod
    def _parse_time(value: TimeFilter) -> datetime_time:
        """Chuyển datetime.time hoặc chuỗi HH:MM[:SS] thành giờ chuẩn.

        Args:
            value: Mốc giờ kiểu datetime.time hoặc chuỗi 24 giờ.

        Returns:
            datetime.time dùng để so sánh giờ trong ngày.

        Raises:
            TypeError: Khi value không phải datetime.time hoặc chuỗi.
            ValueError: Khi chuỗi giờ sai định dạng hoặc nằm ngoài phạm vi.
        """
        if isinstance(value, datetime_time):
            return value.replace(tzinfo=None)
        if not isinstance(value, str):
            raise TypeError("Mốc giờ phải là datetime.time hoặc chuỗi.")
        try:
            return datetime_time.fromisoformat(value.strip())
        except ValueError as error:
            raise ValueError(
                "Mốc giờ phải theo định dạng HH:MM hoặc HH:MM:SS."
            ) from error

    @staticmethod
    def _time_in_range(
        value: datetime_time,
        start_time: datetime_time,
        end_time: datetime_time,
    ) -> bool:
        """Kiểm tra mốc giờ theo khoảng thường hoặc khoảng qua nửa đêm.

        Args:
            value: Giờ địa phương của bản ghi.
            start_time: Biên bắt đầu, được tính vào khoảng.
            end_time: Biên kết thúc, không được tính vào khoảng.

        Returns:
            True nếu giờ thuộc khoảng [start, end), kể cả ca qua nửa đêm.
        """
        local_time = value.replace(tzinfo=None)
        if start_time < end_time:
            return start_time <= local_time < end_time
        return local_time >= start_time or local_time < end_time

    @staticmethod
    def _event_to_csv_row(event: AuditEvent) -> tuple[str, str, str, str, str]:
        """Định dạng một bản ghi theo năm cột CSV dành cho báo cáo Excel.

        Args:
            event: Sự kiện kiểm toán cần chuyển sang dòng CSV.

        Returns:
            Tuple chuỗi gồm mã, thời gian, thời lượng, tọa độ và đường dẫn ảnh.
        """
        return (
            event.event_id,
            event.occurred_at.strftime("%Y-%m-%d %H:%M:%S"),
            f"{event.violation_duration:.3f}",
            f"({event.foot_point[0]}, {event.foot_point[1]})",
            event.image_path.as_posix(),
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Đảm bảo frame đầu vào là ảnh BGR hợp lệ trước khi sao chép/vẽ.

        Args:
            frame: Ảnh do pipeline video cung cấp.

        Returns:
            None.

        Raises:
            TypeError: Khi frame không phải numpy.ndarray.
            ValueError: Khi frame rỗng hoặc không có đúng ba kênh màu.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame phải là numpy.ndarray.")
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame phải là ảnh BGR không rỗng có 3 kênh.")

    @staticmethod
    def _validate_point(point: FootPoint, name: str) -> FootPoint:
        """Kiểm tra một tọa độ pixel gồm hai số nguyên.

        Args:
            point: Tọa độ (x, y) cần kiểm tra.
            name: Tên tham số để tạo thông báo lỗi dễ hiểu.

        Returns:
            Cặp tọa độ số nguyên đã chuẩn hóa.

        Raises:
            TypeError: Khi point không phải cặp số nguyên.
        """
        if (
            not isinstance(point, (tuple, list))
            or len(point) != 2
            or any(
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, np.integer))
                for coordinate in point
            )
        ):
            raise TypeError(f"{name} phải là cặp tọa độ nguyên (x, y).")
        return int(point[0]), int(point[1])

    @classmethod
    def _validate_bounding_box(cls, bounding_box: BoundingBox) -> BoundingBox:
        """Xác thực hộp bao không âm, có diện tích và nằm trong frame.

        Args:
            bounding_box: Bộ bốn giá trị (x, y, rộng, cao).

        Returns:
            Hộp bao đã chuẩn hóa thành tuple số nguyên.

        Raises:
            TypeError: Khi hộp bao không phải bốn số nguyên.
            ValueError: Khi vị trí âm hoặc kích thước không dương.
        """
        if not isinstance(bounding_box, (tuple, list)) or len(bounding_box) != 4:
            raise TypeError("bounding_box phải có dạng (x, y, width, height).")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in bounding_box
        ):
            raise TypeError("Các thành phần bounding_box phải là số nguyên.")
        x, y, width, height = (int(value) for value in bounding_box)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("bounding_box phải không âm và có rộng/cao dương.")
        return x, y, width, height

    @staticmethod
    def _validate_duration(violation_duration: float) -> float:
        """Xác thực thời lượng vi phạm hữu hạn và không âm.

        Args:
            violation_duration: Thời lượng xâm nhập tính bằng giây.

        Returns:
            Thời lượng đã chuyển thành float.

        Raises:
            TypeError: Khi thời lượng không phải số.
            ValueError: Khi thời lượng âm hoặc không hữu hạn.
        """
        if isinstance(violation_duration, bool) or not isinstance(
            violation_duration, (int, float)
        ):
            raise TypeError("violation_duration phải là một số.")
        if not math.isfinite(violation_duration) or violation_duration < 0:
            raise ValueError("violation_duration phải hữu hạn và không âm.")
        return float(violation_duration)
