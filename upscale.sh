#!/bin/bash
# ==============================================================================
#  AI VIDEO UPSCALER 4K - SHELL LAUNCHER CHO MACOS (APPLE SILICON M3 PRO)
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Sử dụng Python trong virtualenv nếu có, fallback sang python3
if [ -f "$DIR/venv/bin/python3" ]; then
    PYTHON_EXEC="$DIR/venv/bin/python3"
else
    PYTHON_EXEC="python3"
fi

if [ -z "$1" ]; then
    echo "❌ Vui lòng cung cấp đường dẫn video!"
    echo "💡 Cách dùng: ./upscale.sh <đường_dẫn_video.mkv> [tùy_chọn]"
    echo ""
    "$PYTHON_EXEC" "$DIR/upscale.py" --help
    exit 1
fi

# Bọc toàn bộ phiên chạy bằng caffeinate (-ims) để chống sleep máy trong suốt ca qua đêm
if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -ims "$PYTHON_EXEC" "$DIR/upscale.py" "$@"
else
    exec "$PYTHON_EXEC" "$DIR/upscale.py" "$@"
fi
