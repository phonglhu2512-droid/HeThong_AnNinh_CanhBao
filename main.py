"""Entrypoint tích hợp các module và khởi chạy VisionGuard AI HMI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QApplication, QMessageBox

from src.core.cv_engine import CVProcessor
from src.core.safety_controller import SafetyController
from src.core.stream_manager import StreamManager, VideoSource
from src.ui.hmi_dashboard import HMIMainWindow
from src.utils.audit_logger import AuditLogger
from src.utils.config_manager import ConfigManager


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def _configured_video_source(config: dict[str, Any]) -> VideoSource:
    """Chuyển cấu hình nguồn video thành kiểu mà StreamManager hỗ trợ.

    Args:
        config: Cấu hình đã được ConfigManager nạp và kiểm tra.

    Returns:
        Webcam ID, đường dẫn video hoặc URL RTSP đã cấu hình.

    Raises:
        TypeError: Khi cấu trúc thông tin nguồn không phải dictionary.
        ValueError: Khi chế độ hoặc thông tin nguồn tương ứng không hợp lệ.
    """
    source_config = config.get("video_source")
    if not isinstance(source_config, dict):
        raise TypeError("'video_source' phải là một đối tượng cấu hình.")

    mode = source_config.get("mode", "webcam")
    if mode == "webcam":
        device_id = source_config.get("device_id", 0)
        if isinstance(device_id, bool) or not isinstance(device_id, int):
            raise ValueError("'device_id' của webcam phải là số nguyên.")
        return device_id
    if mode == "file":
        file_path = source_config.get("file_path", "")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("Chế độ file cần có đường dẫn video.")
        if Path(file_path).suffix.lower() not in {".mp4", ".avi"}:
            raise ValueError("Chỉ hỗ trợ nguồn video .mp4 hoặc .avi.")
        return file_path
    if mode == "rtsp":
        rtsp_url = source_config.get("rtsp_url", "")
        if (
            not isinstance(rtsp_url, str)
            or not rtsp_url.startswith(("rtsp://", "rtsps://"))
        ):
            raise ValueError("Chế độ RTSP cần có URL rtsp:// hoặc rtsps:// hợp lệ.")
        return rtsp_url
    raise ValueError(f"Chế độ nguồn video không được hỗ trợ: {mode!r}.")


def main() -> int:
    """Khởi tạo các dịch vụ, chạy event loop Qt và dọn tài nguyên khi thoát.

    Args:
        None.

    Returns:
        Mã thoát của ứng dụng Qt; trả về 1 nếu không thể khởi động hệ thống.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    application = QApplication(sys.argv)
    application.setApplicationName("VisionGuard AI")
    application.setStyle("Fusion")

    config_manager: ConfigManager | None = None
    stream_manager: StreamManager | None = None
    safety_controller: SafetyController | None = None
    audit_logger: AuditLogger | None = None
    window: HMIMainWindow | None = None

    try:
        config_manager = ConfigManager(
            config_path=PROJECT_ROOT / "config" / "config.json",
            output_root=PROJECT_ROOT / "output_data",
        )
        config = config_manager.get_config()
        video_source = _configured_video_source(config)
        stream_manager = StreamManager(source=video_source)
        cv_processor = CVProcessor(min_area=config["min_area"])
        safety_controller = SafetyController(
            violation_delay=config["violation_delay"]
        )
        audit_logger = AuditLogger(output_root=PROJECT_ROOT / "output_data")
        stream_manager.start_stream()

        window = HMIMainWindow(
            config_manager=config_manager,
            stream_manager=stream_manager,
            cv_processor=cv_processor,
            safety_controller=safety_controller,
            audit_logger=audit_logger,
        )
        window.show()
        exit_code = application.exec()
        if not window.shutdown_complete:
            window.close()
        return exit_code
    except Exception as error:
        LOGGER.exception("Không thể khởi động VisionGuard AI.")
        QMessageBox.critical(
            None,
            "Không thể khởi động VisionGuard AI",
            f"Ứng dụng gặp lỗi khi khởi tạo:\n{error}",
        )
        return 1
    finally:
        if window is None or not window.shutdown_complete:
            if stream_manager is not None:
                try:
                    stream_manager.stop_stream()
                except Exception:
                    LOGGER.exception(
                        "Không thể dừng luồng camera khi thoát ứng dụng."
                    )
            if safety_controller is not None:
                try:
                    safety_controller.close()
                except Exception:
                    LOGGER.exception("Không thể dừng dịch vụ âm thanh khi thoát.")
            if audit_logger is not None:
                try:
                    audit_logger.close()
                except Exception:
                    LOGGER.exception(
                        "Không thể hoàn tất ghi bằng chứng khi thoát."
                    )


if __name__ == "__main__":
    sys.exit(main())