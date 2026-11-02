"""Quản lý cấu hình hệ thống và thư mục dữ liệu vận hành."""

from __future__ import annotations

import copy
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
DEFAULT_ROI_POINTS: tuple[tuple[int, int], ...] = (
    (160, 120),
    (480, 120),
    (480, 320),
    (160, 320),
)


class ConfigManager:
    """Đọc, ghi cấu hình VisionGuard và khởi tạo các thư mục lưu dữ liệu."""

    def __init__(
        self,
        config_path: str | Path = "config/config.json",
        output_root: str | Path = "output_data",
    ) -> None:
        """Khởi tạo trình quản lý, tạo thư mục cần thiết và nạp cấu hình.

        Args:
            config_path: Đường dẫn tệp JSON cấu hình.
            output_root: Thư mục gốc chứa dữ liệu sinh ra khi vận hành.

        Returns:
            None.

        Raises:
            OSError: Khi không thể tạo thư mục lưu trữ hoặc thư mục cấu hình.
            ValueError: Khi tệp cấu hình hiện có không hợp lệ.
            json.JSONDecodeError: Khi tệp cấu hình không phải JSON hợp lệ.
        """
        self.config_path: Path = Path(config_path)
        self.output_root: Path = Path(output_root)
        self._lock: threading.RLock = threading.RLock()
        self._config: dict[str, Any] = {}

        try:
            (self.output_root / "evidence").mkdir(parents=True, exist_ok=True)
            (self.output_root / "reports").mkdir(parents=True, exist_ok=True)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.load_config()
        except (OSError, ValueError, json.JSONDecodeError):
            LOGGER.exception("Không thể khởi tạo ConfigManager.")
            raise

    @staticmethod
    def _default_config() -> dict[str, Any]:
        """Tạo cấu hình mặc định cho nguồn video và vùng ROI ban đầu.

        Args:
            None.

        Returns:
            Bản cấu hình mới, độc lập để tránh thay đổi dữ liệu mặc định dùng chung.
        """
        return {
            "video_source": {
                "mode": "webcam",
                "device_id": 0,
                "file_path": "",
                "rtsp_url": "",
                "port": 554,
                "username": "",
                "password": "",
                "profile": "ONVIF",
            },
            "roi_points": [list(point) for point in DEFAULT_ROI_POINTS],
            "occlusion_polygons": [],
            "min_area": 1500,
            "violation_delay": 0.8,
        }

    @classmethod
    def _merge_defaults(
        cls,
        defaults: dict[str, Any],
        configured: dict[str, Any],
    ) -> dict[str, Any]:
        """Bổ sung các khóa mặc định còn thiếu mà vẫn giữ cấu hình mở rộng.

        Args:
            defaults: Cấu hình mặc định cho các khóa được hỗ trợ.
            configured: Dữ liệu đã đọc từ tệp JSON.

        Returns:
            Cấu hình kết hợp, không làm thay đổi hai đối số đầu vào.
        """
        merged = copy.deepcopy(defaults)
        for key, value in configured.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = cls._merge_defaults(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> None:
        """Kiểm tra các trường cấu hình cốt lõi trước khi dùng hoặc ghi tệp.

        Args:
            config: Cấu hình cần kiểm tra.

        Returns:
            None.

        Raises:
            ValueError: Khi cấu trúc nguồn video, ROI hoặc ngưỡng không hợp lệ.
        """
        video_source = config.get("video_source")
        if not isinstance(video_source, dict):
            raise ValueError("'video_source' phải là một đối tượng JSON.")

        roi_points = config.get("roi_points")
        if not isinstance(roi_points, list) or len(roi_points) < 3:
            raise ValueError("'roi_points' phải có ít nhất 3 đỉnh.")
        for point in roi_points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, (int, float))
                    for coordinate in point
                )
            ):
                raise ValueError("Mỗi đỉnh ROI phải có dạng [x, y] là số.")
            if any(
                not math.isfinite(coordinate)
                or not 0 <= coordinate <= limit
                for coordinate, limit in zip(point, (640, 360))
            ):
                raise ValueError(
                    "Tọa độ ROI phải nằm trong hệ frame 640x360."
                )

        occlusion_polygons = config.get("occlusion_polygons", [])
        if not isinstance(occlusion_polygons, list):
            raise ValueError("'occlusion_polygons' phải là danh sách đa giác.")
        for polygon in occlusion_polygons:
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError("Mỗi vùng che khuất cần tối thiểu 3 đỉnh.")
            for point in polygon:
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or any(
                        isinstance(coordinate, bool)
                        or not isinstance(coordinate, int)
                        or not 0 <= coordinate < limit
                        for coordinate, limit in zip(point, (640, 360))
                    )
                ):
                    raise ValueError(
                        "Đỉnh vùng che khuất phải là [x, y] nguyên trong frame 640x360."
                    )

        min_area = config.get("min_area")
        if isinstance(min_area, bool) or not isinstance(min_area, (int, float)):
            raise ValueError("'min_area' phải là một số.")
        if not math.isfinite(min_area) or not 500 <= min_area <= 5000:
            raise ValueError("'min_area' phải nằm trong khoảng 500 đến 5000.")

        violation_delay = config.get("violation_delay")
        if isinstance(violation_delay, bool) or not isinstance(
            violation_delay, (int, float)
        ):
            raise ValueError("'violation_delay' phải là một số.")
        if not math.isfinite(violation_delay) or not 0 <= violation_delay <= 3:
            raise ValueError("'violation_delay' phải nằm trong khoảng 0 đến 3 giây.")

    def load_config(self) -> dict[str, Any]:
        """Đọc cấu hình JSON, tạo tệp mặc định nếu chưa tồn tại.

        Args:
            None.

        Returns:
            Bản sao cấu hình đã được kiểm tra và bổ sung khóa mặc định.

        Raises:
            OSError: Khi không thể đọc hoặc ghi tệp cấu hình.
            ValueError: Khi nội dung JSON không có cấu trúc cấu hình hợp lệ.
            json.JSONDecodeError: Khi tệp hiện có không phải JSON hợp lệ.
        """
        with self._lock:
            try:
                defaults = self._default_config()
                if self.config_path.exists():
                    with self.config_path.open("r", encoding="utf-8") as config_file:
                        configured = json.load(config_file)
                    if not isinstance(configured, dict):
                        raise ValueError("Cấu hình gốc phải là một đối tượng JSON.")
                    config = self._merge_defaults(defaults, configured)
                else:
                    config = defaults

                self._validate_config(config)
                self._write_config(config)
                self._config = copy.deepcopy(config)
                return copy.deepcopy(self._config)
            except (OSError, ValueError, json.JSONDecodeError):
                LOGGER.exception("Không thể đọc cấu hình tại %s.", self.config_path)
                raise

    def save_config(self, config: dict[str, Any] | None = None) -> None:
        """Ghi cấu hình ra JSON theo cách thay thế tệp an toàn.

        Args:
            config: Cấu hình cần ghi; nếu bỏ trống thì ghi cấu hình đang dùng.

        Returns:
            None.

        Raises:
            TypeError: Khi đối số cấu hình không phải dictionary.
            OSError: Khi không thể ghi hoặc thay thế tệp cấu hình.
            ValueError: Khi dữ liệu cấu hình không hợp lệ.
        """
        with self._lock:
            candidate = copy.deepcopy(self._config if config is None else config)
            if not isinstance(candidate, dict):
                raise TypeError("Cấu hình cần ghi phải là dictionary.")

            try:
                candidate = self._merge_defaults(self._default_config(), candidate)
                self._validate_config(candidate)
                self._write_config(candidate)
                self._config = copy.deepcopy(candidate)
            except (OSError, TypeError, ValueError):
                LOGGER.exception("Không thể ghi cấu hình tại %s.", self.config_path)
                raise

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Cập nhật một phần cấu hình và lưu thay đổi xuống tệp JSON.

        Args:
            updates: Các khóa cấu hình cần cập nhật.

        Returns:
            Bản sao cấu hình sau khi cập nhật thành công.

        Raises:
            TypeError: Khi updates không phải dictionary.
            OSError: Khi không thể ghi tệp cấu hình.
            ValueError: Khi cấu hình kết quả không hợp lệ.
        """
        if not isinstance(updates, dict):
            raise TypeError("'updates' phải là dictionary.")

        with self._lock:
            updated = self._merge_defaults(self._config, updates)
            self.save_config(updated)
            return copy.deepcopy(self._config)

    def get_config(self) -> dict[str, Any]:
        """Trả về bản sao cấu hình hiện tại để tránh sửa dữ liệu dùng chung trực tiếp.

        Args:
            None.

        Returns:
            Bản sao độc lập của cấu hình hiện tại.
        """
        with self._lock:
            return copy.deepcopy(self._config)

    def _write_config(self, config: dict[str, Any]) -> None:
        """Ghi JSON vào tệp tạm rồi thay thế tệp đích để tránh ghi dở dang.

        Args:
            config: Cấu hình đã được kiểm tra.

        Returns:
            None.

        Raises:
            OSError: Khi không thể tạo tệp tạm, ghi dữ liệu hoặc thay thế tệp.
        """
        temporary_path = self.config_path.with_name(
            f"{self.config_path.name}.tmp"
        )
        try:
            with temporary_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as config_file:
                json.dump(config, config_file, ensure_ascii=False, indent=2)
                config_file.write("\n")
                config_file.flush()
            temporary_path.replace(self.config_path)
        except (OSError, TypeError):
            LOGGER.exception("Không thể ghi tệp tạm cấu hình %s.", temporary_path)
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Không thể dọn tệp tạm %s.", temporary_path)
            raise
