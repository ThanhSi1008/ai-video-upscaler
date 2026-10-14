#!/bin/bash
# ==============================================================================
#  AI VIDEO UPSCALER 4K - SHELL LAUNCHER CHO MACOS / LINUX
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

# Kiểm tra nếu có cờ --queue để tự động bật soak_monitor.sh và ghi mốc SESSION
HAS_QUEUE=0
for arg in "$@"; do
    if [ "$arg" = "--queue" ]; then
        HAS_QUEUE=1
        break
    fi
done

if [ "$HAS_QUEUE" -eq 1 ]; then
    SESSION_NAME="queue_$(date +%Y%m%d_%H%M%S)"
    TELEMETRY_CSV="logs/soak_telemetry_${SESSION_NAME}.csv"
    mkdir -p logs

    echo "[SESSION START] $(date +'%Y-%m-%d %H:%M:%S') (epoch $(date +%s)) - Session: ${SESSION_NAME}"

    # Khởi động scripts/soak_monitor.sh nền
    ./scripts/soak_monitor.sh "$TELEMETRY_CSV" 60 >/dev/null 2>&1 &
    MONITOR_PID=$!

    cleanup() {
        echo "[SESSION END] $(date +'%Y-%m-%d %H:%M:%S') (epoch $(date +%s))"
        if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
            kill "$MONITOR_PID" 2>/dev/null || true
            wait "$MONITOR_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    if command -v caffeinate >/dev/null 2>&1; then
        caffeinate -ims "$PYTHON_EXEC" "$DIR/upscale.py" "$@"
        EXIT_CODE=$?
    else
        "$PYTHON_EXEC" "$DIR/upscale.py" "$@"
        EXIT_CODE=$?
    fi
    exit $EXIT_CODE
else
    # Bọc toàn bộ phiên chạy bằng caffeinate (-ims) để chống sleep máy trong suốt ca qua đêm
    if command -v caffeinate >/dev/null 2>&1; then
        exec caffeinate -ims "$PYTHON_EXEC" "$DIR/upscale.py" "$@"
    else
        exec "$PYTHON_EXEC" "$DIR/upscale.py" "$@"
    fi
fi
