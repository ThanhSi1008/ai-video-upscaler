# -*- coding: utf-8 -*-
"""
tests/test_subtrans_stub_pipeline.py
Test tích hợp end-to-end cho Stage B phụ đề KHÔNG gọi API (Zero API calls, Zero GPU):
- Quét chọn track JA/EN qua find_or_extract_subtitles
- Tự động phát hiện và áp dụng auto-offset riêng từng tập trong hàng đợi
  (Ep 1: không áp dụng [-148ms], Ep 2: không áp dụng [-98ms], Ep 3: áp dụng [-11141ms])
- Sử dụng stub backend (dry_run=True) sinh phụ đề mẫu đúng quy chuẩn đặt tên <tập>_4K.vi.srt
- Nhúng phụ đề vào bản sao container MKV thành phẩm và kiểm định toàn vẹn bằng verify_mux_integrity.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from upscale import find_or_extract_subtitles
from subtrans.pipeline import TranslationPipeline
from subtrans.parser import detect_auto_offset
from src.ffmpeg_ops import embed_subtitle_track, probe_video
from src.mux_audit import verify_mux_integrity


RAW_DIR = Path("/Users/xis108/Documents/RawSources")
UPSCALED_DIR = Path("/Users/xis108/Documents/Upscaled")
TMP_TEST_DIR = Path("/tmp/test_subtrans_integration")


EPISODES = [
    {
        "name": "EP1 (Kimi ga Shinu made Koi wo Shitai 13)",
        "source": RAW_DIR / "Kimi_ga_Shinu_made_Koi_wo_Shitai" / "Kimi_ga_Shinu_made_Koi_wo_Shitai_S01E13.mkv",
        "upscaled": UPSCALED_DIR / "Kimi_ga_Shinu_made_Koi_wo_Shitai" / "Kimi_ga_Shinu_made_Koi_wo_Shitai_S01E13_4K.mkv",
        "expected_offset_applied": False,
        "expected_offset_min": -300,
        "expected_offset_max": 0,
        "expected_fonts": 11,
    },
    {
        "name": "EP2 (Re Zero 84)",
        "source": RAW_DIR / "Re_Zero_kara_Hajimeru_Isekai_Seikatsu" / "Re_Zero_kara_Hajimeru_Isekai_Seikatsu_S04E84.mkv",
        "upscaled": UPSCALED_DIR / "Re_Zero_kara_Hajimeru_Isekai_Seikatsu" / "Re_Zero_kara_Hajimeru_Isekai_Seikatsu_S04E84_4K.mkv",
        "expected_offset_applied": False,
        "expected_offset_min": -200,
        "expected_offset_max": 0,
        "expected_fonts": 24,
    },
    {
        "name": "EP3 (Slime S4-24)",
        "source": RAW_DIR / "Tensei_Shitara_Slime_Datta_Ken" / "Tensei_Shitara_Slime_Datta_Ken_S04E24.mkv",
        "upscaled": None,  # Tập 3 chưa hoàn tất upscale ở Đêm 1
        "expected_offset_applied": True,
        "expected_offset_min": -11300,
        "expected_offset_max": -11000,
        "expected_fonts": 0,
    },
]


def test_subtrans_queue_stub_integration():
    """Chạy toàn bộ pipeline phụ đề hàng đợi bằng stub backend trên bản sao độc lập."""
    if not RAW_DIR.exists():
        pytest.skip("Thư mục RawSources không khả dụng trên môi trường này.")

    if TMP_TEST_DIR.exists():
        shutil.rmtree(TMP_TEST_DIR)
    TMP_TEST_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    try:
        for ep in EPISODES:
            ep_name = ep["name"]
            src_vid = str(ep["source"])
            if not os.path.exists(src_vid):
                pytest.skip(f"File nguồn {src_vid} không tồn tại.")

            base_name = os.path.splitext(os.path.basename(src_vid))[0]
            target_4k_base = f"{base_name}_4K"

            # 1. Quét tìm track JA/EN
            ja_sub, en_sub = find_or_extract_subtitles(src_vid)
            assert ja_sub is not None, f"[{ep_name}] Không tìm thấy phụ đề JA"
            assert en_sub is not None, f"[{ep_name}] Không tìm thấy phụ đề EN"

            # 2. Kiểm tra auto-offset độc lập từng tập trong hàng đợi
            det_offset, det_applied, det_reason, det_meta = detect_auto_offset(ja_sub, en_sub)
            assert det_applied == ep["expected_offset_applied"], (
                f"[{ep_name}] Lỗi auto-offset applied: mong đợi {ep['expected_offset_applied']}, thực tế {det_applied} ({det_reason})"
            )
            assert ep["expected_offset_min"] <= det_offset <= ep["expected_offset_max"], (
                f"[{ep_name}] Offset {det_offset} nằm ngoài khoảng mong đợi [{ep['expected_offset_min']}, {ep['expected_offset_max']}]"
            )

            # 3. Dịch bằng stub backend (dry_run=True, KHÔNG gọi API)
            pipeline = TranslationPipeline(
                local_only=False,
                output_dir=TMP_TEST_DIR,
                output_format="srt",
                require_approved_cast=False,
                allow_latest_alias=False,
                dry_run=True,
                limit_cues=25,
            )

            applied_offset = det_offset if det_applied else 0
            res = pipeline.run(
                ja_path=Path(ja_sub),
                en_path=Path(en_sub),
                output_basename=target_4k_base,
                sub_offset_ms=applied_offset,
                auto_offset=True,
            )

            srt_path = res.get("subtitle_path") or res.get("output_path")
            assert srt_path is not None, f"[{ep_name}] Không tạo được file SRT"
            srt_path = Path(srt_path)
            assert srt_path.exists(), f"[{ep_name}] File SRT {srt_path} không tồn tại trên đĩa"
            assert srt_path.name == f"{target_4k_base}.vi.srt", (
                f"[{ep_name}] Tên file SRT sai định dạng: mong đợi {target_4k_base}.vi.srt, thực tế {srt_path.name}"
            )

            # Kiểm tra file audit.md
            audit_md_path = TMP_TEST_DIR / f"{target_4k_base}.audit.md"
            assert audit_md_path.exists(), f"[{ep_name}] File audit.md {audit_md_path} không tồn tại"
            audit_md_content = audit_md_path.read_text(encoding="utf-8")
            assert "Căn Chỉnh Thời Gian Phụ Đề" in audit_md_content or "Offset đã áp dụng" in audit_md_content, f"[{ep_name}] Audit report thiếu mục offset"

            # 4. Kiểm tra Embed + Audit trên bản sao MKV
            target_4k_clone = TMP_TEST_DIR / f"{target_4k_base}.mkv"
            if ep["upscaled"] and ep["upscaled"].exists():
                subprocess.run(["cp", "-c", str(ep["upscaled"]), str(target_4k_clone)], check=True)
            else:
                # Đối với tập 3 (chưa upscale 4K), sao chép nguồn để giả lập container thành phẩm
                subprocess.run(["cp", "-c", src_vid, str(target_4k_clone)], check=True)

            embed_ok = embed_subtitle_track(
                mkv_path=str(target_4k_clone),
                srt_path=str(srt_path),
                original_source_path=src_vid,
            )
            assert embed_ok, f"[{ep_name}] Hàm embed_subtitle_track trả về False"

            probe_res = probe_video(str(target_4k_clone))
            mux_audit = verify_mux_integrity(src_vid, str(target_4k_clone))

            assert mux_audit["status"] == "PASS", (
                f"[{ep_name}] Mux audit không PASS: {mux_audit.get('failure_reasons')}"
            )
            if ep["expected_fonts"] > 0:
                assert probe_res["attachment_count"] == ep["expected_fonts"], (
                    f"[{ep_name}] Số font không khớp: mong đợi {ep['expected_fonts']}, thực tế {probe_res['attachment_count']}"
                )

            results.append({
                "name": ep_name,
                "ja_track": os.path.basename(ja_sub),
                "en_track": os.path.basename(en_sub),
                "offset_detected": det_offset,
                "offset_applied": det_applied,
                "srt_name": srt_path.name,
                "embed_status": "PASS",
                "fonts_verified": probe_res["attachment_count"],
                "audio_verified": probe_res["audio_count"],
                "subs_verified": probe_res["subtitle_count"],
            })

        # In bảng kết quả chi tiết
        print("\n" + "=" * 80)
        print("📊 KẾT QUẢ TEST TÍCH HỢP STAGE PHỤ ĐỀ TRÊN BẢN SAO (KHÔNG GỌI API)")
        print("=" * 80)
        for r in results:
            print(f"• Tập: {r['name']}")
            print(f"  - Track JA: {r['ja_track']} | Track EN: {r['en_track']}")
            print(f"  - Auto-offset đo được: {r['offset_detected']} ms (Áp dụng: {r['offset_applied']})")
            print(f"  - Tên file SRT: {r['srt_name']}")
            print(f"  - Embed + Mux Audit: {r['embed_status']} (Fonts: {r['fonts_verified']}, Audio: {r['audio_verified']}, Subs: {r['subs_verified']})")
        print("=" * 80 + "\n")

    finally:
        if TMP_TEST_DIR.exists():
            shutil.rmtree(TMP_TEST_DIR)


if __name__ == "__main__":
    test_subtrans_queue_stub_integration()
