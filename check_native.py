#!/usr/bin/env python3
"""
Entry point thuận tiện ở thư mục gốc để gọi công cụ kiểm tra độ phân giải video
trong thư mục tools/check_native.py.

Cách dùng:
    python3 check_native.py sample.mkv
    python3 check_native.py --web
"""

import sys
from pathlib import Path

# Thêm thư mục tools vào Python path
tools_dir = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(tools_dir))

import check_native

if __name__ == "__main__":
    check_native.main()
