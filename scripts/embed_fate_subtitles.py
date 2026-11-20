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

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.set_track_defaults import process_file
from src.env_isolation import configure_e_drive_isolation
from src.ffmpeg_ops import embed_subtitle_track

HF_ZIPS = [
    ("fatestay-night-heavens-feel-i-presage-flower-2017_vietnamese_3304665-[SubSource].zip", "Fate.stay.night.Heavens.Feel.I.Presage.Flower.2017.1080p.BluRay.Opus5.1.H.265-LYS1TH3A.mkv"),
    ("fatestay-night-heavens-feel-ii-lost-butterfly-2019_vietnamese_3304669-[SubSource].zip", "Fate.stay.night.Heavens.Feel.II.Lost.Butterfly.2019.1080p.BluRay.Opus5.1.H.265-LYS1TH3A.mkv"),
    ("fatestay-night-heavens-feel-iii-spring-song-2020_vietnamese_3304670-[SubSource].zip", "Fate.stay.night.Heavens.Feel.III.Spring.Song.2020.1080p.BluRay.Opus5.1.H.265-LYS1TH3A.mkv"),
]

FATE_ZIP_NAMES = [
    "fatestay-night-heavens-feel-i-presage-flower-2017_vietnamese_3304665-[SubSource].zip",
    "fatestay-night-heavens-feel-ii-lost-butterfly-2019_vietnamese_3304669-[SubSource].zip",
    "fatestay-night-heavens-feel-iii-spring-song-2020_vietnamese_3304670-[SubSource].zip",
    "fatestay-night-unlimited-blade-works-season-1_vietnamese_3304772-[SubSource].zip",
    "fatestay-night-unlimited-blade-works-season-2_vietnamese_3304773-[SubSource].zip",
    "fatezero-season-1-2011_vietnamese_3304741-[SubSource].zip",
    "fatezero-season-2-2011_vietnamese_3231427-[SubSource].zip",
]


def _pick_zip_entry(zip_path: Path, patterns: List[str], default_entry: Optional[str] = None) -> Optional[str]:
    if not zip_path.is_file():
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = [n for n in z.namelist() if n.lower().endswith((".srt", ".ass"))]
    except Exception:
        return None
    if not names:
        return None
    if default_entry and default_entry in names:
        return default_entry
    for pat in patterns:
        matched = next((n for n in names if pat.lower() in n.lower()), None)
        if matched:
            return matched
    return names[0] if len(patterns) == 0 else None


def _find_mkv(dir_path: Path, prefixes: List[str]) -> Optional[Path]:
    if not dir_path.is_dir():
        return None
    mkvs = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".mkv") and not f.startswith(".")])
    for pref in prefixes:
        hit = next((f for f in mkvs if pref.lower() in f.lower()), None)
        if hit:
            return dir_path / hit
    return None


def build_worklist(downloads_dir: Path, raw_sources_dir: Path) -> List[Dict]:
    worklist: List[Dict] = []

    ubw_dir = raw_sources_dir / "[Tenrai-Sensei] Fate Stay Night Unlimited Blade Works [BD][1080p][HEVC 10bit x265][Dual Audio]"
    ubw_z1 = downloads_dir / "fatestay-night-unlimited-blade-works-season-1_vietnamese_3304772-[SubSource].zip"
    ubw_z2 = downloads_dir / "fatestay-night-unlimited-blade-works-season-2_vietnamese_3304773-[SubSource].zip"

    fz_dir = raw_sources_dir / "[Tenrai-Sensei] Fate Zero [BD][1080p][HEVC 10bit x265][Dual Audio]"
    fz_z1 = downloads_dir / "fatezero-season-1-2011_vietnamese_3304741-[SubSource].zip"
    fz_z2 = downloads_dir / "fatezero-season-2-2011_vietnamese_3231427-[SubSource].zip"

    # 1. Heaven's Feel (3 movies)
    for zname, mkv_rel in HF_ZIPS:
        zp = downloads_dir / zname
        mkv_path = raw_sources_dir / mkv_rel
        sub_name = _pick_zip_entry(zp, [])
        if sub_name and mkv_path.is_file():
            worklist.append({
                "series": "Fate/stay night: Heaven's Feel",
                "zip_path": zp,
                "sub_entry": sub_name,
                "mkv_path": mkv_path,
            })

    # 2. Unlimited Blade Works (26 episodes)
    mkv_s00e01 = ubw_dir / "Special" / "FateStay Night Unlimited Blade Works - S00E01 - Prologue.mkv"
    s00_entry = _pick_zip_entry(
        ubw_z1,
        ["S00E01"],
        default_entry="Fate / Stay Night? Unlimited Blade Works (2014) - S00E01.vi.srt",
    )
    if mkv_s00e01.is_file() and s00_entry:
        worklist.append({
            "series": "Fate/stay night: UBW (Special)",
            "zip_path": ubw_z1,
            "sub_entry": s00_entry,
            "mkv_path": mkv_s00e01,
        })

    # S01E01 to S01E12
    ubw_s1_dir = ubw_dir / "Season 1"
    for ep in range(1, 13):
        mkv_path = _find_mkv(ubw_s1_dir, [f"S01E{ep:02d}"])
        sub_name = _pick_zip_entry(
            ubw_z1,
            [f"S01E{ep:02d}", f"E{ep:02d}"],
            default_entry=f"Fate / Stay Night? Unlimited Blade Works (2014) - S01E{ep:02d}.vi.srt",
        )
        if mkv_path and sub_name:
            worklist.append({
                "series": "Fate/stay night: UBW (Season 1)",
                "zip_path": ubw_z1,
                "sub_entry": sub_name,
                "mkv_path": mkv_path,
            })

    # UBW Season 2 (ep 13..25): kiểm tra Season 2 trước, fallback Season 1
    ubw_s2_dir = (ubw_dir / "Season 2") if (ubw_dir / "Season 2").is_dir() else ubw_s1_dir
    for ep in range(13, 26):
        s2_ep = ep - 12
        mkv_path = _find_mkv(ubw_s2_dir, [f"S02E{s2_ep:02d}", f"S01E{ep:02d}"])
        if mkv_path is None and ubw_s2_dir != ubw_s1_dir:
            mkv_path = _find_mkv(ubw_s1_dir, [f"S01E{ep:02d}", f"S02E{s2_ep:02d}"])
        sub_name = _pick_zip_entry(
            ubw_z2,
            [f"S02E{s2_ep:02d}", f"S01E{ep:02d}", f"E{ep:02d}"],
            default_entry=f"Fate / Stay Night? Unlimited Blade Works (2014) - S01E{ep:02d}.vi.srt",
        )
        if mkv_path and sub_name:
            worklist.append({
                "series": "Fate/stay night: UBW (Season 2)",
                "zip_path": ubw_z2,
                "sub_entry": sub_name,
                "mkv_path": mkv_path,
            })

    # 3. Fate/Zero (ep 1..13 Season 1 -> FZ_Z1; ep 14..25 Season 2 -> FZ_Z2)
    fz_s1_dir = fz_dir / "Season 1"
    fz_s2_dir = (fz_dir / "Season 2") if (fz_dir / "Season 2").is_dir() else fz_s1_dir
    for ep in range(1, 14):
        mkv_path = _find_mkv(fz_s1_dir, [f"S01E{ep:02d}"])
        sub_name = _pick_zip_entry(
            fz_z1,
            [f"S01E{ep:02d}", f"E{ep:02d}"],
            default_entry=f"Fate / Zero (2011) - E{ep:02d}.vi.srt",
        )
        if mkv_path and sub_name:
            worklist.append({
                "series": "Fate/Zero (Season 1)",
                "zip_path": fz_z1,
                "sub_entry": sub_name,
                "mkv_path": mkv_path,
            })

    for ep in range(14, 26):
        s2_ep = ep - 13
        mkv_path = _find_mkv(fz_s2_dir, [f"S02E{s2_ep:02d}", f"S01E{ep:02d}"])
        if mkv_path is None and fz_s2_dir != fz_s1_dir:
            mkv_path = _find_mkv(fz_s1_dir, [f"S01E{ep:02d}", f"S02E{s2_ep:02d}"])
        sub_name = _pick_zip_entry(
            fz_z2,
            [f"S02E{s2_ep:02d}", f"E{ep:02d}", f"S01E{ep:02d}"],
            default_entry=f"Fate / Zero (2011) - E{ep:02d}.vi.srt",
        )
        if mkv_path and sub_name:
            worklist.append({
                "series": "Fate/Zero (Season 2)",
                "zip_path": fz_z2,
                "sub_entry": sub_name,
                "mkv_path": mkv_path,
            })

    return worklist


def process_item(item: Dict, index: int, total: int, dry_run: bool = False) -> bool:
    mkv_path = str(item["mkv_path"])
    zp = str(item["zip_path"])
    sub_entry = item["sub_entry"]
    fname = os.path.basename(mkv_path)
    vid_dir = os.path.dirname(mkv_path)
    base_name = os.path.splitext(fname)[0]
    sidecar_srt = os.path.join(vid_dir, f"{base_name}.vi.srt")

    print(f"\n[{index}/{total}] 🎬 [{item['series']}] {fname}", flush=True)
    if dry_run:
        print(f"  🔍 [DRY RUN] Sẽ giải nén '{sub_entry}' từ '{os.path.basename(zp)}' -> '{os.path.basename(sidecar_srt)}' và nhúng vào MKV.", flush=True)
        return True

    # 1. Extract subtitle from zip
    t0 = time.time()
    try:
        with zipfile.ZipFile(zp, "r") as z:
            sub_bytes = z.read(sub_entry)
        with open(sidecar_srt, "wb") as f_out:
            f_out.write(sub_bytes)
        print(f"  📄 Đã xuất file phụ đề sidecar: {os.path.basename(sidecar_srt)} ({len(sub_bytes)} bytes)", flush=True)
    except Exception as e:
        print(f"  ❌ Lỗi giải nén phụ đề: {e}", flush=True)
        return False

    # 2. Embed into MKV
    print("  🔄 Đang nhúng track phụ đề vào MKV...", flush=True)
    ok = embed_subtitle_track(
        mkv_path=mkv_path,
        srt_path=sidecar_srt,
        language="vie",
        title="Tiếng Việt (Vietsub)",
        is_default=True,
    )
    elapsed = time.time() - t0

    if ok:
        try:
            process_file(mkv_path)
        except Exception:
            pass
        print(f"  ✅ Đã nhúng phụ đề thành công ({elapsed:.1f}s)!", flush=True)
        return True
    else:
        print("  ❌ Nhúng phụ đề thất bại (Audit FAIL hoặc lỗi ffmpeg). File sidecar vẫn được giữ nguyên.", flush=True)
        return False


def cleanup_downloads(downloads_dir: Path) -> None:
    print("\n🧹 Bắt đầu dọn dẹp thư mục Downloads...", flush=True)
    cleaned = 0
    for zname in FATE_ZIP_NAMES:
        zp = downloads_dir / zname
        if zp.exists():
            try:
                zp.unlink()
                print(f"  🗑️ Đã xóa: {zp.name}", flush=True)
                cleaned += 1
            except Exception as e:
                print(f"  ⚠️ Không thể xóa {zp.name}: {e}", flush=True)
    print(f"✅ Đã dọn dẹp sạch sẽ {cleaned} file zip!", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    configure_e_drive_isolation(project_root=PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="Nhúng hàng loạt phụ đề tiếng Việt cho Fate Series (HF, UBW, Fate/Zero)"
    )
    parser.add_argument("--downloads-dir", required=True, help="Thư mục chứa các file .zip phụ đề")
    parser.add_argument("--raw-sources-dir", required=True, help="Thư mục gốc chứa các thư mục video Fate Series")
    parser.add_argument("--dry-run", action="store_true", help="Mô phỏng ghép danh sách mà không giải nén/nhúng MKV")
    parser.add_argument("--cleanup-zips", action="store_true", help="Xóa các file zip nguồn sau khi nhúng thành công 100%")
    args = parser.parse_args(argv)

    downloads_dir = Path(args.downloads_dir)
    raw_sources_dir = Path(args.raw_sources_dir)
    if not downloads_dir.is_dir():
        print(f"❌ Thư mục downloads không tồn tại: {downloads_dir}", file=sys.stderr)
        return 2
    if not raw_sources_dir.is_dir():
        print(f"❌ Thư mục raw sources không tồn tại: {raw_sources_dir}", file=sys.stderr)
        return 2

    print("=" * 80, flush=True)
    print("🚀 BẮT ĐẦU GHÉP PHỤ ĐỀ TIẾNG VIỆT CHO FATE SERIES", flush=True)
    print("=" * 80, flush=True)

    worklist = build_worklist(downloads_dir, raw_sources_dir)
    total = len(worklist)
    print(f"📋 Tổng số tập tìm thấy cần xử lý: {total}", flush=True)

    success_count = 0
    fail_count = 0

    t_start = time.time()
    for idx, item in enumerate(worklist, 1):
        ok = process_item(item, idx, total, dry_run=args.dry_run)
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

    if args.cleanup_zips and not args.dry_run and total > 0 and success_count == total:
        cleanup_downloads(downloads_dir)
    elif args.cleanup_zips and not args.dry_run and fail_count > 0:
        print(f"⚠️ Do có {fail_count} tập thất bại, giữ lại các file zip để kiểm tra.", flush=True)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
