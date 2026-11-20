#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_gui.py - Khởi chạy giao diện Web Studio thời gian thực cho AI Video Upscaler & Subtrans.

Cách dùng:
  python web_gui.py                      # Chạy trên http://127.0.0.1:8765 và tự mở trình duyệt
  python web_gui.py --no-browser         # Chạy máy chủ không tự mở tab trình duyệt
  python web_gui.py --lan --port 8765    # Cho phép xem từ điện thoại/máy tính bảng cùng mạng LAN
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env_isolation import configure_e_drive_isolation
configure_e_drive_isolation(project_root=ROOT_DIR)


def _configure_utf8_console() -> None:
    if sys.platform == "win32":
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass


def _detect_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="web_gui.py",
        description="Giao diện điều khiển & theo dõi thời gian thực cho 4K Anime Video Upscaler + Subtrans.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Địa chỉ IP lắng nghe (mặc định: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Cổng HTTP cục bộ (mặc định: 8765).",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Mở lắng nghe trên toàn mạng nội bộ (0.0.0.0) để theo dõi chỉ-đọc từ điện thoại cùng Wi-Fi.",
    )
    parser.add_argument(
        "--lan-control",
        action="store_true",
        help="Cho phép máy khác trong mạng LAN điều khiển tác vụ, duyệt tệp và sửa cấu hình (cần bật cùng --lan).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Không tự động mở trình duyệt khi khởi động máy chủ.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with deterministic simulated telemetry, jobs, and SSE ticks (no CUDA/TensorRT/external drives).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warning",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Mức ghi log của Uvicorn (mặc định: warning để giữ gọn console).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    import uvicorn
    from web.app import create_app

    bind_host = "0.0.0.0" if args.lan else args.host
    normalized_bind = bind_host.strip().strip("[]").lower()
    is_loopback_bind = normalized_bind in {"localhost", "127.0.0.1", "::1"} or normalized_bind.startswith("127.")
    allow_lan = bool(args.lan or not is_loopback_bind)
    allow_lan_control = bool(args.lan_control)
    local_url = f"http://127.0.0.1:{args.port}"

    print("=" * 72)
    print("🎬 AI VIDEO UPSCALER & SUBTRANS — REAL-TIME WEB STUDIO")
    print("=" * 72)
    print(f"  • Địa chỉ máy trạm : {local_url}")
    if args.lan or bind_host == "0.0.0.0":
        lan_mode_note = "Điều khiển đầy đủ (--lan-control)" if allow_lan_control else "Chỉ-đọc an toàn (mặc định; thêm --lan-control để điều khiển)"
        print(f"  • Địa chỉ mạng LAN : http://{_detect_lan_ip()}:{args.port} [{lan_mode_note}]")
    print(f"  • Thư mục dự án    : {ROOT_DIR}")
    if args.mock:
        print("  • Mode             : MOCK (--mock deterministic telemetry/jobs; no GPU/CUDA)")
    print("  • Nhấn Ctrl+C để đóng máy chủ Web (các tiến trình CLI độc lập không bị ảnh hưởng).")
    print("=" * 72)

    if not args.no_browser:
        def _open_later() -> None:
            time.sleep(0.9)
            try:
                webbrowser.open(local_url)
            except Exception:
                pass

        threading.Thread(target=_open_later, name="WebGuiOpenBrowser", daemon=True).start()

    app = create_app(
        workspace_root=ROOT_DIR,
        allow_lan=allow_lan,
        allow_lan_control=allow_lan_control,
        mock_mode=bool(args.mock),
    )
    uvicorn.run(app, host=bind_host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
