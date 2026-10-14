"""
Unit tests for automatic cleanup and organization of temporary artifacts.
Kiểm thử tính năng tự động dọn dẹp thư mục checkpoint tạm, symlinks và gom gọn báo cáo audit.
"""

import os
import shutil
import pytest
from pathlib import Path
from src.cleanup import (
    cleanup_checkpoint_dir,
    cleanup_subtitle_extracts,
    organize_output_artifacts,
    clean_all_temporary_data,
    get_path_size
)


def test_cleanup_regular_checkpoint_dir(tmp_path: Path):
    """Kiểm tra xóa sạch thư mục checkpoint tạm thông thường và trả về dung lượng."""
    work_dir = tmp_path / ".temp_sample_ep"
    work_dir.mkdir()
    (work_dir / "up_0001.mp4").write_bytes(b"A" * 1024 * 1024)  # 1 MB
    (work_dir / "up_0002.mp4").write_bytes(b"B" * 1024 * 1024)  # 1 MB
    (work_dir / "plan.json").write_text("{}", encoding="utf-8")

    assert work_dir.exists()
    freed = cleanup_checkpoint_dir(str(work_dir))

    assert not work_dir.exists()
    assert freed >= 2 * 1024 * 1024


def test_cleanup_symlink_checkpoint_dir(tmp_path: Path):
    """Kiểm tra xử lý triệt để khi work_dir là symbolic link trỏ đến thư mục con trên SSD."""
    actual_parent = tmp_path / "SubFolder"
    actual_parent.mkdir()
    actual_dir = actual_parent / ".temp_actual_ep"
    actual_dir.mkdir()
    (actual_dir / "up_0001.mp4").write_bytes(b"X" * 1024 * 512)

    # Tạo symlink trong output_dir
    symlink_path = tmp_path / ".temp_actual_ep"
    try:
        os.symlink(str(actual_dir), str(symlink_path))
    except OSError as e:
        pytest.skip(f"Bỏ qua kiểm tra symlink trên Windows (thiếu quyền Administrator/Developer Mode): {e}")

    assert symlink_path.is_symlink()
    assert actual_dir.exists()

    freed = cleanup_checkpoint_dir(str(symlink_path), output_dir=str(tmp_path))

    # Cả symlink lẫn thư mục thực tế phải được dọn sạch
    assert not os.path.lexists(str(symlink_path))
    assert not actual_dir.exists()
    # actual_parent rỗng nên cũng được dọn sạch
    assert not actual_parent.exists()
    assert freed >= 512 * 1024


def test_cleanup_orphan_symlinks_in_output_dir(tmp_path: Path):
    """Kiểm tra dọn dẹp các symlink mồ côi hoặc liên quan trong output_dir."""
    real_temp = tmp_path / "somewhere" / ".temp_target"
    real_temp.mkdir(parents=True)
    (real_temp / "seg.mp4").write_bytes(b"data")

    # Tạo symlink trỏ tới real_temp và một broken symlink
    link1 = tmp_path / ".temp_target"
    broken_link = tmp_path / ".temp_target_broken"
    try:
        os.symlink(str(real_temp), str(link1))
        os.symlink("/nonexistent/path/temp", str(broken_link))
    except OSError as e:
        pytest.skip(f"Bỏ qua kiểm tra symlink trên Windows: {e}")

    cleanup_checkpoint_dir(str(link1), output_dir=str(tmp_path))

    assert not os.path.lexists(str(link1))
    assert not os.path.lexists(str(broken_link))


def test_cleanup_subtitle_extracts(tmp_path: Path):
    """Kiểm tra dọn dẹp các tệp trích xuất phụ đề tạm thời."""
    vid_dir = tmp_path / "sources"
    vid_dir.mkdir()
    temp_dir = vid_dir / ".temp_episode1"
    temp_dir.mkdir()
    (temp_dir / "extracted_en.ass").write_text("sub en", encoding="utf-8")
    (temp_dir / "extracted_ja.ass").write_text("sub ja", encoding="utf-8")

    freed = cleanup_subtitle_extracts(str(vid_dir), "episode1")
    assert freed > 0
    assert not (temp_dir / "extracted_en.ass").exists()
    assert not (temp_dir / "extracted_ja.ass").exists()
    assert not temp_dir.exists()


def test_organize_output_artifacts(tmp_path: Path):
    """Kiểm tra gom các file kỹ thuật (.audit.json, .audit.md, .cast.yaml) vào audit/."""
    # Tạo các file video thành phẩm và phụ đề (không được di chuyển)
    mkv_file = tmp_path / "anime_S01E01_4K.mkv"
    mkv_file.write_bytes(b"video 4k")
    srt_file = tmp_path / "anime_S01E01_4K.vi.srt"
    srt_file.write_text("phu de", encoding="utf-8")

    # Tạo các file audit/cast (cần gom vào audit/)
    audit_json = tmp_path / "anime_S01E01_4K.audit.json"
    audit_json.write_text("{}", encoding="utf-8")
    audit_md = tmp_path / "anime_S01E01_4K.audit.md"
    audit_md.write_text("# report", encoding="utf-8")
    cast_yaml = tmp_path / "anime_S01E01_4K.cast.yaml"
    cast_yaml.write_text("cast: []", encoding="utf-8")

    moved = organize_output_artifacts(str(tmp_path), base_name="anime_S01E01")
    assert len(moved) == 3

    # Các file thành phẩm vẫn ở nguyên ngoài root
    assert mkv_file.exists()
    assert srt_file.exists()

    # Các file audit được chuyển vào thư mục audit/
    audit_dir = tmp_path / "audit"
    assert audit_dir.is_dir()
    assert (audit_dir / "anime_S01E01_4K.audit.json").exists()
    assert (audit_dir / "anime_S01E01_4K.audit.md").exists()
    assert (audit_dir / "anime_S01E01_4K.cast.yaml").exists()

    # Không còn nằm ở root
    assert not audit_json.exists()
    assert not audit_md.exists()
    assert not cast_yaml.exists()


def test_clean_all_temporary_data(tmp_path: Path):
    """Kiểm tra quét toàn diện và dọn sạch toàn bộ thư mục."""
    temp_dir = tmp_path / ".temp_ep01"
    temp_dir.mkdir()
    (temp_dir / "dummy.mp4").write_bytes(b"dummy")

    audit_json = tmp_path / "ep01_4K.audit.json"
    audit_json.write_text("{}", encoding="utf-8")

    res = clean_all_temporary_data(str(tmp_path))
    assert len(res["cleaned_dirs"]) >= 1
    assert len(res["organized_files"]) >= 1
    assert not temp_dir.exists()
    assert (tmp_path / "audit" / "ep01_4K.audit.json").exists()
