"""
Module: src/cleanup.py
Chức năng: Quản lý tự động dọn dẹp các artifact tạm thời sau khi hoàn tất upscale,
giải phóng dung lượng SSD và sắp xếp thư mục xuất xưởng ngăn nắp, tinh tươm.
"""

import os
import shutil
from typing import Optional, List, Dict, Any


def get_path_size(path: str) -> int:
    """Tính tổng dung lượng (bytes) của một file hoặc thư mục trên đĩa."""
    if not os.path.lexists(path):
        return 0
    if os.path.islink(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except Exception:
            return 0
    elif os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if not os.path.islink(fp):
                        total += os.path.getsize(fp)
                except Exception:
                    pass
        return total
    return 0


def cleanup_checkpoint_dir(work_dir: str, output_dir: Optional[str] = None, logger=None) -> int:
    """
    Dọn dẹp an toàn và triệt để thư mục checkpoint tạm (.temp_<basename>)
    cùng các symlink và tệp đệm trên SSD sau khi hoàn tất muxing & Mux Audit PASS.
    
    Hỗ trợ xử lý:
    - Thư mục thông thường trên SSD.
    - Symlinks trỏ đến thư mục checkpoint ở vị trí khác (resolve target và xóa cả hai).
    - Dọn dẹp thư mục cha nếu sau khi xóa thư mục tạm thì thư mục cha rỗng (trừ .DS_Store).
    - Quét dọn các symlink mồ côi dạng .temp_* liên quan trong output_dir.
    
    Trả về số byte đã giải phóng.
    """
    if not work_dir or not os.path.lexists(work_dir):
        return 0

    bytes_freed = 0
    real_work_dir = None

    if os.path.islink(work_dir):
        try:
            target = os.path.realpath(work_dir)
            if os.path.isdir(target):
                real_work_dir = target
        except Exception:
            pass
    elif os.path.isdir(work_dir):
        real_work_dir = work_dir

    # 1. Xóa thư mục thực tế nếu tồn tại
    if real_work_dir and os.path.isdir(real_work_dir):
        bytes_freed += get_path_size(real_work_dir)
        try:
            shutil.rmtree(real_work_dir, ignore_errors=True)
        except Exception as e:
            if logger:
                logger.warn(f"Không thể xóa thư mục tạm '{real_work_dir}': {e}")

        # Dọn dẹp thư mục cha nếu rỗng (ví dụ ~/Documents/Upscaled/SeriesName chỉ chứa .temp)
        parent_real = os.path.dirname(real_work_dir)
        if (
            parent_real
            and os.path.isdir(parent_real)
            and parent_real != output_dir
            and parent_real != os.path.dirname(output_dir or "")
        ):
            try:
                remaining = [f for f in os.listdir(parent_real) if f != ".DS_Store"]
                if not remaining:
                    ds = os.path.join(parent_real, ".DS_Store")
                    if os.path.exists(ds):
                        os.remove(ds)
                    os.rmdir(parent_real)
            except Exception:
                pass

    # 2. Xóa chính symlink work_dir nếu nó là symlink
    if os.path.islink(work_dir):
        try:
            os.unlink(work_dir)
        except Exception:
            pass
    elif os.path.exists(work_dir) and not os.path.isdir(work_dir):
        try:
            os.remove(work_dir)
        except Exception:
            pass

    # 3. Quét dọn các symlink .temp_* mồ côi hoặc liên quan trong output_dir
    if output_dir and os.path.isdir(output_dir):
        work_base = os.path.basename(work_dir)
        try:
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.islink(item_path) and item.startswith(".temp_"):
                    should_remove = False
                    try:
                        resolved = os.path.realpath(item_path)
                        # Nếu trỏ tới real_work_dir vừa xóa, hoặc link bị đứt, hoặc chứa work_base
                        if real_work_dir and resolved == real_work_dir:
                            should_remove = True
                        elif not os.path.exists(resolved):
                            should_remove = True
                        elif work_base and work_base in item:
                            should_remove = True
                    except Exception:
                        should_remove = True
                    if should_remove:
                        try:
                            os.unlink(item_path)
                        except Exception:
                            pass
        except Exception:
            pass

    if bytes_freed > 0 and logger:
        from src.logger import format_bytes
        logger.info(
            f"🧹 [TỰ ĐỘNG DỌN DẸP] Đã giải phóng {format_bytes(bytes_freed)} dung lượng SSD từ thư mục checkpoint tạm.",
            to_console=True
        )

    return bytes_freed


def cleanup_subtitle_extracts(vid_dir: str, base_name: str, logger=None) -> int:
    """
    Dọn dẹp các tệp phụ đề tạm (extracted_en.ass, extracted_ja.ass) được trích xuất
    trong quá trình xử lý subtrans trong thư mục .temp_<base_name>.
    """
    safe_base = base_name.rstrip('. ')
    tmp_dir = os.path.join(vid_dir, f".temp_{safe_base}")
    if not os.path.lexists(tmp_dir):
        return 0

    freed = 0
    for fname in ["extracted_en.ass", "extracted_ja.ass", "extracted_en.srt", "extracted_ja.srt"]:
        fp = os.path.join(tmp_dir, fname)
        if os.path.exists(fp):
            try:
                freed += os.path.getsize(fp)
                os.remove(fp)
            except Exception:
                pass

    # Nếu tmp_dir sau đó rỗng, xóa luôn tmp_dir
    if os.path.isdir(tmp_dir):
        try:
            remaining = [f for f in os.listdir(tmp_dir) if f != ".DS_Store"]
            if not remaining:
                ds = os.path.join(tmp_dir, ".DS_Store")
                if os.path.exists(ds):
                    os.remove(ds)
                os.rmdir(tmp_dir)
        except Exception:
            pass

    return freed


def organize_output_artifacts(
    output_dir: str,
    base_name: Optional[str] = None,
    audit_subdir_name: str = "audit",
    logger=None
) -> List[str]:
    """
    Gom các file kỹ thuật/kiểm định (.audit.json, .audit.md, .cast.yaml)
    vào thư mục con 'audit/' bên trong output_dir.
    Giữ cho thư mục output_dir chỉ chứa các file thành phẩm xem được (.mkv, .mp4, .srt).
    """
    if not output_dir or not os.path.isdir(output_dir):
        return []

    audit_dir = os.path.join(output_dir, audit_subdir_name)
    moved_files = []

    target_exts = [".audit.json", ".audit.md", ".cast.yaml"]

    try:
        entries = os.listdir(output_dir)
    except Exception:
        return []

    for item in entries:
        item_path = os.path.join(output_dir, item)
        if not os.path.isfile(item_path):
            continue

        is_target = False
        for ext in target_exts:
            if item.endswith(ext):
                if base_name:
                    if base_name in item or item.endswith(".cast.yaml"):
                        is_target = True
                else:
                    is_target = True
                break

        if is_target:
            os.makedirs(audit_dir, exist_ok=True)
            dest_path = os.path.join(audit_dir, item)
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(item_path, dest_path)
                moved_files.append(item)
            except Exception as e:
                if logger:
                    logger.warn(f"Không thể di chuyển '{item}' vào '{audit_subdir_name}/': {e}")

    if moved_files and logger:
        logger.info(
            f"📁 [GOM GỌN THƯ MỤC] Đã chuyển {len(moved_files)} file báo cáo kỹ thuật vào '{audit_subdir_name}/' "
            f"để giữ thư mục xuất xưởng gọn gàng tinh tươm.",
            to_console=True
        )

    return moved_files


def clean_all_temporary_data(output_dir: str, logger=None) -> Dict[str, Any]:
    """
    Quét và dọn dẹp toàn diện mọi thư mục tạm, symlinks gãy, và gom các file audit
    trong output_dir.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return {"bytes_freed": 0, "cleaned_dirs": [], "organized_files": []}

    total_freed = 0
    cleaned_dirs = []

    try:
        entries = os.listdir(output_dir)
    except Exception:
        entries = []

    # 1. Quét tìm các thư mục và symlink .temp_*
    for item in entries:
        item_path = os.path.join(output_dir, item)
        if item.startswith(".temp_"):
            freed = cleanup_checkpoint_dir(item_path, output_dir=output_dir, logger=logger)
            if freed > 0 or not os.path.lexists(item_path):
                total_freed += freed
                cleaned_dirs.append(item)

    # 2. Quét sâu 1 cấp vào các thư mục con xem có .temp_* không
    for item in entries:
        sub_dir = os.path.join(output_dir, item)
        if os.path.isdir(sub_dir) and not item.startswith("."):
            try:
                sub_entries = os.listdir(sub_dir)
                for sub_item in sub_entries:
                    if sub_item.startswith(".temp_"):
                        sub_path = os.path.join(sub_dir, sub_item)
                        freed = cleanup_checkpoint_dir(sub_path, output_dir=output_dir, logger=logger)
                        if freed > 0 or not os.path.lexists(sub_path):
                            total_freed += freed
                            cleaned_dirs.append(f"{item}/{sub_item}")
            except Exception:
                pass

    # 3. Gom gọn các file audit
    organized_files = organize_output_artifacts(output_dir, logger=logger)

    return {
        "bytes_freed": total_freed,
        "cleaned_dirs": cleaned_dirs,
        "organized_files": organized_files
    }
