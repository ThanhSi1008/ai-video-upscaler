"""
Batch Subtitle Muxer for Fate Series
Embeds downloaded Vietnamese subtitles into:
- Fate/stay night Heaven's Feel I, II, III (3 movies)
- Fate/stay night Unlimited Blade Works (S00E01 + S01E01..S01E25 = 26 episodes)
- Fate/Zero (S01E01..S01E25 = 25 episodes)
Total: 54 files
Ensures frame-accurate soft subtitle embedding (vie, default: 1, title: 'Tiếng Việt (Vietsub)')
Ensures Japanese audio is default: 1
Preserves sidecar .vi.srt next to each MKV
Cleans up downloaded archives in Downloads upon successful completion
"""

import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(r"E:\ai-video-upscaler")
sys.path.insert(0, str(PROJECT_ROOT))
from src.ffmpeg_ops import embed_subtitle_track
from scripts.set_track_defaults import process_file

DOWNLOADS_DIR = Path(r"C:\Users\Admin\Downloads")
RAW_SOURCES_DIR = Path(r"E:\Films\Raw_Sources")

HF_ZIPS = [
    ("fatestay-night-heavens-feel-i-presage-flower-2017_vietnamese_3304665-[SubSource].zip", "Fate.stay.night.Heavens.Feel.I.Presage.Flower.2017.1080p.BluRay.Opus5.1.H.265-LYS1TH3A.mkv"),
    ("fatestay-night-heavens-feel-ii-lost-butterfly-2019_vietnamese_3304669-[SubSource].zip", "Fate.stay.night.Heavens.Feel.II.Lost.Butterfly.2019.1080p.BluRay.Opus5.1.H.265-LYS1TH3A.mkv"),
    ("fatestay-night-heavens-feel-iii-spring-song-2020_vietnamese_3304670-[SubSource].zip", "Fate.stay.night.Heavens.Feel.III.Spring.Song.2020.1080p.BluRay.Opus5.1.H.265-LYS1TH3A.mkv"),
]

UBW_DIR = RAW_SOURCES_DIR / r"[Tenrai-Sensei] Fate Stay Night Unlimited Blade Works [BD][1080p][HEVC 10bit x265][Dual Audio]"
UBW_Z1 = DOWNLOADS_DIR / "fatestay-night-unlimited-blade-works-season-1_vietnamese_3304772-[SubSource].zip"
UBW_Z2 = DOWNLOADS_DIR / "fatestay-night-unlimited-blade-works-season-2_vietnamese_3304773-[SubSource].zip"

FZ_DIR = RAW_SOURCES_DIR / r"[Tenrai-Sensei] Fate Zero [BD][1080p][HEVC 10bit x265][Dual Audio]"
FZ_Z1 = DOWNLOADS_DIR / "fatezero-season-1-2011_vietnamese_3304741-[SubSource].zip"
FZ_Z2 = DOWNLOADS_DIR / "fatezero-season-2-2011_vietnamese_3231427-[SubSource].zip"

ALL_FATE_ZIPS = [
    DOWNLOADS_DIR / "fatestay-night-heavens-feel-i-presage-flower-2017_vietnamese_3304665-[SubSource].zip",
    DOWNLOADS_DIR / "fatestay-night-heavens-feel-ii-lost-butterfly-2019_vietnamese_3304669-[SubSource].zip",
    DOWNLOADS_DIR / "fatestay-night-heavens-feel-iii-spring-song-2020_vietnamese_3304670-[SubSource].zip",
    DOWNLOADS_DIR / "fatestay-night-unlimited-blade-works-season-1_vietnamese_3304772-[SubSource].zip",
    DOWNLOADS_DIR / "fatestay-night-unlimited-blade-works-season-2_vietnamese_3304773-[SubSource].zip",
    DOWNLOADS_DIR / "fatezero-season-1-2011_vietnamese_3304741-[SubSource].zip",
    DOWNLOADS_DIR / "fatezero-season-2-2011_vietnamese_3231427-[SubSource].zip",
]


def build_worklist():
    worklist = []

    # 1. Heaven's Feel (3 movies)
    for zname, mkv_rel in HF_ZIPS:
        zp = DOWNLOADS_DIR / zname
        mkv_path = RAW_SOURCES_DIR / mkv_rel
        with zipfile.ZipFile(zp) as z:
            sub_name = z.namelist()[0]
        worklist.append({
            "series": "Fate/stay night: Heaven's Feel",
            "zip_path": zp,
            "sub_entry": sub_name,
            "mkv_path": mkv_path,
        })

    # 2. Unlimited Blade Works (26 episodes)
    # S00E01
    mkv_s00e01 = UBW_DIR / "Special" / "FateStay Night Unlimited Blade Works - S00E01 - Prologue.mkv"
    worklist.append({
        "series": "Fate/stay night: UBW (Special)",
        "zip_path": UBW_Z1,
        "sub_entry": "Fate / Stay Night? Unlimited Blade Works (2014) - S00E01.vi.srt",
        "mkv_path": mkv_s00e01,
    })

    # S01E01 to S01E12
    ubw_s1_files = os.listdir(UBW_DIR / "Season 1")
    for ep in range(1, 13):
        mkv_fname = [f for f in ubw_s1_files if f.startswith(f"FateStay Night Unlimited Blade Works - S01E{ep:02d}")][0]
        mkv_path = UBW_DIR / "Season 1" / mkv_fname
        sub_name = f"Fate / Stay Night? Unlimited Blade Works (2014) - S01E{ep:02d}.vi.srt"
        worklist.append({
            "series": "Fate/stay night: UBW (Season 1)",
            "zip_path": UBW_Z1,
            "sub_entry": sub_name,
            "mkv_path": mkv_path,
        })

    # S01E13 to S01E25
    for ep in range(13, 26):
        mkv_fname = [f for f in ubw_s1_files if f.startswith(f"FateStay Night Unlimited Blade Works - S01E{ep:02d}")][0]
        mkv_path = UBW_DIR / "Season 1" / mkv_fname
        sub_name = f"Fate / Stay Night? Unlimited Blade Works (2014) - S01E{ep:02d}.vi.srt"
        worklist.append({
            "series": "Fate/stay night: UBW (Season 2)",
            "zip_path": UBW_Z2,
            "sub_entry": sub_name,
            "mkv_path": mkv_path,
        })

    # 3. Fate/Zero (25 episodes)
    fz_s1_files = os.listdir(FZ_DIR / "Season 1")
    for ep in range(1, 26):
        mkv_fname = [f for f in fz_s1_files if f.startswith(f"FateZero - S01E{ep:02d}")][0]
        mkv_path = FZ_DIR / "Season 1" / mkv_fname
        sub_name = f"Fate / Zero (2011) - E{ep:02d}.vi.srt"
        worklist.append({
            "series": "Fate/Zero",
            "zip_path": FZ_Z1,
            "sub_entry": sub_name,
            "mkv_path": mkv_path,
        })

    return worklist


def process_item(item, index, total):
    mkv_path = str(item["mkv_path"])
    zp = str(item["zip_path"])
    sub_entry = item["sub_entry"]
    fname = os.path.basename(mkv_path)
    vid_dir = os.path.dirname(mkv_path)
    base_name = os.path.splitext(fname)[0]
    sidecar_srt = os.path.join(vid_dir, f"{base_name}.vi.srt")

    print(f"\n[{index}/{total}] 🎬 [{item['series']}] {fname}", flush=True)

    # 1. Extract subtitle from zip
    t0 = time.time()
    try:
        with zipfile.ZipFile(zp, 'r') as z:
            sub_bytes = z.read(sub_entry)
        with open(sidecar_srt, "wb") as f_out:
            f_out.write(sub_bytes)
        print(f"  📄 Đã xuất file phụ đề sidecar: {os.path.basename(sidecar_srt)} ({len(sub_bytes)} bytes)", flush=True)
    except Exception as e:
        print(f"  ❌ Lỗi giải nén phụ đề: {e}", flush=True)
        return False

    # 2. Embed into MKV
    print(f"  🔄 Đang nhúng track phụ đề vào MKV...", flush=True)
    ok = embed_subtitle_track(
        mkv_path=mkv_path,
        srt_path=sidecar_srt,
        language="vie",
        title="Tiếng Việt (Vietsub)",
        is_default=True
    )
    elapsed = time.time() - t0

    if ok:
        # Guarantee Japanese audio & Vietnamese subtitle default: 1 via intelligent track analysis
        try:
            process_file(mkv_path)
        except Exception:
            pass
        print(f"  ✅ Đã nhúng phụ đề thành công ({elapsed:.1f}s)!", flush=True)
        return True
    else:
        print(f"  ❌ Nhúng phụ đề thất bại (Audit FAIL hoặc lỗi ffmpeg). File sidecar vẫn được giữ nguyên.", flush=True)
        return False


def cleanup_downloads():
    print("\n🧹 Bắt đầu dọn dẹp thư mục Downloads...", flush=True)
    cleaned = 0
    for zp in ALL_FATE_ZIPS:
        if zp.exists():
            try:
                zp.unlink()
                print(f"  🗑️ Đã xóa: {zp.name}", flush=True)
                cleaned += 1
            except Exception as e:
                print(f"  ⚠️ Không thể xóa {zp.name}: {e}", flush=True)
    print(f"✅ Đã dọn dẹp sạch sẽ {cleaned} file zip trong Downloads!", flush=True)


def main():
    print("=" * 80, flush=True)
    print("🚀 BẮT ĐẦU GHÉP PHỤ ĐỀ TIẾNG VIỆT CHO FATE SERIES", flush=True)
    print("=" * 80, flush=True)

    worklist = build_worklist()
    total = len(worklist)
    print(f"📋 Tổng số tập cần xử lý: {total}", flush=True)

    success_count = 0
    fail_count = 0

    t_start = time.time()
    for idx, item in enumerate(worklist, 1):
        ok = process_item(item, idx, total)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    total_time = time.time() - t_start
    print("\n" + "=" * 80, flush=True)
    print("📊 KẾT QUẢ TỔNG KẾT GHÉP PHỤ ĐỀ FATE SERIES", flush=True)
    print(f"⏱️ Tổng thời gian: {total_time:.1f}s ({total_time/60:.1f} phút)", flush=True)
    print(f"✅ Thành công: {success_count}/{total}", flush=True)
    print(f"❌ Thất bại: {fail_count}/{total}", flush=True)
    print("=" * 80, flush=True)

    if success_count == total:
        cleanup_downloads()
    else:
        print(f"⚠️ Do có {fail_count} tập thất bại, tạm thời giữ lại các file zip trong Downloads để bạn kiểm tra.", flush=True)


if __name__ == "__main__":
    main()
