#!/usr/bin/env bash
# scripts/soak_monitor.sh
# Giám sát viễn trắc hệ thống mỗi 60 giây (zero-impact, ghi CSV phục vụ phân tích soak test)
# Đo đạc: vm_stat (pageouts), sysctl vm.swapusage, memory_pressure, iostat -Id disk0,
# dung lượng RAM disk, pmset therm, và powermetrics (nếu có quyền sudo).

set -u

OUTPUT_CSV="${1:-logs/soak_telemetry_$(date +%Y%m%d_%H%M%S).csv}"
INTERVAL="${2:-60}"

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "=== SOAK MONITORING STARTED ==="
echo "Output CSV: $OUTPUT_CSV"
echo "Sampling Interval: ${INTERVAL}s"

# Kiểm tra quyền sudo cho powermetrics
HAS_SUDO=0
if sudo -n true 2>/dev/null; then
    HAS_SUDO=1
    echo "[INFO] Đã có quyền sudo: sẽ tự động trích xuất GPU Power và Thermal qua powermetrics."
else
    echo "[NOTE] Chưa có quyền sudo không mật khẩu. Các cột GPU Power / Temp sẽ để 'NA'."
    echo "       Nếu bạn muốn ghi chi tiết GPU Power / Nhiệt độ, hãy mở 1 Terminal riêng và chạy:"
    echo "       sudo powermetrics -i 10000 --samplers thermal,gpu_power | tee logs/powermetrics.log"
fi

# Kiểm tra smartctl
if command -v smartctl >/dev/null 2>&1; then
    echo "[INFO] Tìm thấy smartctl. Đang ghi baseline Data Units Written vào logs/smartctl_baseline.txt..."
    smartctl -a disk0 > logs/smartctl_baseline.txt 2>&1 || true
else
    echo "[NOTE] Không tìm thấy smartctl (có thể cài qua: brew install smartmontools)."
    echo "       Hệ thống sẽ đo tổng dung lượng I/O qua 'iostat -Id disk0' (chuẩn macOS)."
fi

# Ghi Header CSV
echo "timestamp,epoch,pageouts,pageins,swap_used_mb,swap_free_mb,mem_free_pct,disk0_mb_total,ioreg_write_bytes,ioreg_read_bytes,ramdisk_free_mb,thermal_status,gpu_power_w,gpu_temp_c" > "$OUTPUT_CSV"

echo "Đang theo dõi... Bấm Ctrl+C để dừng."

while true; do
    TS=$(date +"%Y-%m-%d %H:%M:%S")
    EPOCH=$(date +%s)

    # 1. vm_stat (pageouts, pageins)
    PAGEOUTS=$(vm_stat | awk '/Pageouts:/ {gsub(/\./, "", $2); print $2}')
    PAGEINS=$(vm_stat | awk '/Pageins:/ {gsub(/\./, "", $2); print $2}')

    # 2. sysctl vm.swapusage
    SWAP_RAW=$(sysctl -n vm.swapusage 2>/dev/null || echo "total = 0M used = 0M free = 0M")
    SWAP_USED=$(echo "$SWAP_RAW" | awk -F'used = ' '{print $2}' | awk '{gsub(/M/, "", $1); print $1}')
    SWAP_FREE=$(echo "$SWAP_RAW" | awk -F'free = ' '{print $2}' | awk '{gsub(/M/, "", $1); print $1}')

    # 3. memory_pressure
    MEM_FREE_PCT=$(memory_pressure 2>/dev/null | awk '/System-wide memory free percentage:/ {gsub(/%/, "", $5); print $5}')
    if [ -z "$MEM_FREE_PCT" ]; then
        MEM_FREE_PCT="NA"
    fi

    # 4. iostat -Id disk0 (Dung lượng chuyển đọc+ghi MB từ lúc boot)
    DISK0_MB=$(iostat -Id disk0 2>/dev/null | tail -n 1 | awk '{print $3}')
    if [ -z "$DISK0_MB" ]; then
        DISK0_MB="NA"
    fi

    # 4b. ioreg IOBlockStorageDriver disk0 (Bytes Write và Read riêng biệt)
    IOREG_STATS=$(python3 -c '
import subprocess, re
try:
    out = subprocess.check_output(["ioreg", "-c", "IOBlockStorageDriver", "-k", "Statistics", "-r", "-l"], text=True)
    for block in out.split("+-o IOBlockStorageDriver"):
        if "\"BSD Name\" = \"disk0\"" in block:
            bw = re.search(r"\"Bytes \(Write\)\"\s*=\s*(\d+)", block)
            br = re.search(r"\"Bytes \(Read\)\"\s*=\s*(\d+)", block)
            w = bw.group(1) if bw else "NA"
            r = br.group(1) if br else "NA"
            print(f"{w},{r}")
            break
    else:
        print("NA,NA")
except Exception:
    print("NA,NA")
' 2>/dev/null || echo "NA,NA")
    IOREG_WRITE=$(echo "$IOREG_STATS" | awk -F',' '{print $1}')
    IOREG_READ=$(echo "$IOREG_STATS" | awk -F',' '{print $2}')
    [ -z "$IOREG_WRITE" ] && IOREG_WRITE="NA"
    [ -z "$IOREG_READ" ] && IOREG_READ="NA"

    # 5. RAM disk free MB
    RAMDISK_FREE="0"
    RAMDISK_PATH=$(df -m | grep -i "/Volumes/.*RAM.*" | head -n 1 | awk '{print $9}')
    if [ -n "$RAMDISK_PATH" ] && [ -d "$RAMDISK_PATH" ]; then
        RAMDISK_FREE=$(df -m "$RAMDISK_PATH" | tail -n 1 | awk '{print $4}')
    fi

    # 6. pmset -g therm
    THERM_RAW=$(pmset -g therm 2>/dev/null | grep -i "warning level" | head -n 1 || echo "Normal")
    if echo "$THERM_RAW" | grep -qi "No thermal warning"; then
        THERM_STATUS="Normal"
    else
        THERM_STATUS=$(echo "$THERM_RAW" | tr -s ' ' '_' | tr -d ',')
    fi

    # 7. GPU Power & Temp
    GPU_POWER="NA"
    GPU_TEMP="NA"
    if [ "$HAS_SUDO" -eq 1 ]; then
        # Lấy mẫu 500ms
        PM_SAMPLE=$(sudo powermetrics -n 1 -i 500 --samplers thermal,gpu_power 2>/dev/null || true)
        GPU_POWER=$(echo "$PM_SAMPLE" | awk -F'GPU Power: ' '/GPU Power: / {print $2}' | awk '{gsub(/mW/, "", $1); print $1/1000}')
        GPU_TEMP=$(echo "$PM_SAMPLE" | awk -F'GPU die temperature: ' '/GPU die temperature: / {print $2}' | awk '{print $1}')
        [ -z "$GPU_POWER" ] && GPU_POWER="NA"
        [ -z "$GPU_TEMP" ] && GPU_TEMP="NA"
    fi

    echo "$TS,$EPOCH,$PAGEOUTS,$PAGEINS,$SWAP_USED,$SWAP_FREE,$MEM_FREE_PCT,$DISK0_MB,$IOREG_WRITE,$IOREG_READ,$RAMDISK_FREE,$THERM_STATUS,$GPU_POWER,$GPU_TEMP" >> "$OUTPUT_CSV"

    sleep "$INTERVAL"
done
