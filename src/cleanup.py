"""Temporary checkpoint cleanup and final output artifact organization helpers."""

import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional, List, Dict, Any


def _is_symlink_or_reparse(path: str) -> bool:
    """Return True before resolve()/realpath() when path is a symlink, junction, or Win32 reparse point."""
    if not path:
        return False
    try:
        if os.path.islink(path) or (hasattr(os.path, "isjunction") and os.path.isjunction(path)):
            return True
        st = os.lstat(path)
        return bool(getattr(st, "st_file_attributes", 0) & 0x0400)
    except OSError:
        return False


def _has_reparse_in_path(path: str, stop_at: Optional[str] = None) -> bool:
    """Return True if path or any non-system ancestor segment is a symlink, junction, or reparse point."""
    if not path:
        return False
    if _is_symlink_or_reparse(path):
        return True
    safe_roots: set[str] = set()
    for candidate_root in (stop_at, os.getcwd(), tempfile.gettempdir()):
        if candidate_root:
            cur_r = os.path.normcase(os.path.abspath(candidate_root)).rstrip(os.sep)
            while cur_r:
                safe_roots.add(cur_r)
                par_r = os.path.dirname(cur_r).rstrip(os.sep)
                if par_r == cur_r:
                    break
                cur_r = par_r

    p_obj = Path(path)
    if ".." in p_obj.parts:
        acc = Path(p_obj.anchor) if p_obj.is_absolute() else Path(os.getcwd())
        rel_parts = p_obj.parts[1:] if p_obj.is_absolute() else p_obj.parts
        for part in rel_parts:
            if part == "..":
                acc = acc.parent
            elif part != ".":
                acc = acc / part
                acc_norm = os.path.normcase(os.path.abspath(str(acc))).rstrip(os.sep)
                if acc_norm not in safe_roots and _is_symlink_or_reparse(str(acc)):
                    return True

    cur = os.path.abspath(path)
    while cur:
        cur_norm = os.path.normcase(cur).rstrip(os.sep)
        if cur_norm in safe_roots:
            break
        if _is_symlink_or_reparse(cur):
            return True
        par = os.path.dirname(cur)
        if par == cur:
            break
        cur = par
    return False


def _remove_reparse_entry_only(path: str) -> bool:
    """Remove a symlink or Win32 directory junction entry itself without touching its target."""
    try:
        os.unlink(path)
        return True
    except OSError:
        try:
            os.rmdir(path)
            return True
        except OSError:
            return False


def get_path_size(path: str) -> int:
    """Tính tổng dung lượng (bytes) của một file hoặc thư mục trên đĩa."""
    if not os.path.lexists(path) or _is_symlink_or_reparse(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except Exception:
            return 0
    elif os.path.isdir(path):
        total = 0
        for root, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [d for d in dirs if not _is_symlink_or_reparse(os.path.join(root, d))]
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if not _is_symlink_or_reparse(fp):
                        total += os.path.getsize(fp)
                except Exception:
                    pass
        return total
    return 0


def cleanup_checkpoint_dir(
    work_dir: str,
    output_dir: Optional[str] = None,
    logger=None,
    follow_internal_symlink: bool = True,
) -> int:
    """Clean temporary .temp_<basename> checkpoint directory while blocking external symlink/reparse traversal."""
    if not work_dir or (not os.path.lexists(work_dir) and not _is_symlink_or_reparse(work_dir)):
        return 0

    bytes_freed = 0
    real_work_dir = None
    work_is_reparse = _is_symlink_or_reparse(work_dir)

    if work_is_reparse:
        if (
            follow_internal_symlink
            and output_dir
            and not _is_symlink_or_reparse(output_dir)
            and os.path.isdir(output_dir)
        ):
            try:
                real_out = Path(os.path.realpath(output_dir))
                target = Path(os.path.realpath(work_dir))
                if (
                    target.is_dir()
                    and not _is_symlink_or_reparse(str(target))
                    and target != real_out
                    and target.is_relative_to(real_out)
                    and target.name.startswith(".temp_")
                ):
                    real_work_dir = str(target)
            except Exception:
                real_work_dir = None
    elif os.path.isdir(work_dir):
        real_work_dir = work_dir

    # 1. Xóa thư mục thực tế nếu tồn tại (và không phải reparse point)
    if real_work_dir and not _is_symlink_or_reparse(real_work_dir) and os.path.isdir(real_work_dir):
        bytes_freed += get_path_size(real_work_dir)
        try:
            shutil.rmtree(real_work_dir, ignore_errors=True)
        except Exception as e:
            if logger:
                logger.warn(f"Không thể xóa thư mục tạm '{real_work_dir}': {e}")

        parent_real = os.path.dirname(real_work_dir)
        real_out_str = os.path.realpath(output_dir) if (output_dir and os.path.isdir(output_dir)) else (output_dir or "")
        if (
            parent_real
            and not _is_symlink_or_reparse(parent_real)
            and os.path.isdir(parent_real)
            and parent_real != output_dir
            and parent_real != real_out_str
            and parent_real != os.path.dirname(output_dir or "")
            and parent_real != os.path.dirname(real_out_str)
        ):
            try:
                remaining = [f for f in os.listdir(parent_real) if f != ".DS_Store"]
                if not remaining:
                    ds = os.path.join(parent_real, ".DS_Store")
                    if os.path.exists(ds) and not _is_symlink_or_reparse(ds):
                        os.remove(ds)
                    os.rmdir(parent_real)
            except Exception:
                pass

    # 2. Xóa chính symlink/junction work_dir mà không đụng tới đích bên ngoài
    if work_is_reparse:
        _remove_reparse_entry_only(work_dir)
    elif os.path.exists(work_dir) and not os.path.isdir(work_dir):
        try:
            os.remove(work_dir)
        except Exception:
            pass

    # 3. Quét dọn các symlink/reparse .temp_* mồ côi hoặc liên quan trong output_dir
    if output_dir and not _is_symlink_or_reparse(output_dir) and os.path.isdir(output_dir):
        work_base = os.path.basename(work_dir)
        try:
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if item.startswith(".temp_") and _is_symlink_or_reparse(item_path):
                    should_remove = False
                    try:
                        resolved = os.path.realpath(item_path)
                        if real_work_dir and resolved == real_work_dir:
                            should_remove = True
                        elif not os.path.exists(resolved):
                            should_remove = True
                        elif work_base and item == work_base:
                            should_remove = True
                    except Exception:
                        should_remove = True
                    if should_remove:
                        _remove_reparse_entry_only(item_path)
        except Exception:
            pass

    if bytes_freed > 0 and logger:
        from src.logger import format_bytes
        logger.info(
            f"🧹 [TỰ ĐỘNG DỌN DẸP] Đã giải phóng {format_bytes(bytes_freed)} dung lượng SSD từ thư mục checkpoint tạm.",
            to_console=True
        )

    return bytes_freed


def cleanup_subtitle_extracts(
    vid_dir: str,
    base_name: str,
    logger=None,
    keep_whisper: bool = False,
) -> int:
    """
    Dọn dẹp các tệp phụ đề tạm (extracted_en.ass, extracted_ja.ass, *.whisper_ja.srt)
    được trích xuất trong quá trình xử lý subtrans trong thư mục .temp_<base_name>.
    """
    safe_base = base_name.rstrip('. ')
    tmp_dir = os.path.join(vid_dir, f".temp_{safe_base}")
    if not os.path.lexists(tmp_dir):
        return 0

    freed = 0
    static_names = {"extracted_en.ass", "extracted_ja.ass", "extracted_en.srt", "extracted_ja.srt"}
    candidate_files = list(static_names)
    if not keep_whisper and os.path.isdir(tmp_dir):
        try:
            for entry in os.listdir(tmp_dir):
                if entry.endswith(".whisper_ja.srt") and entry not in static_names:
                    candidate_files.append(entry)
        except Exception:
            pass

    for fname in candidate_files:
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
    logger=None,
    out_stem: Optional[str] = None,
) -> List[str]:
    """
    Gom các file kỹ thuật/kiểm định (.audit.json, .audit.md, .cast.yaml)
    vào thư mục con 'audit/' bên trong output_dir.
    Giữ cho thư mục output_dir chỉ chứa các file thành phẩm xem được (.mkv, .mp4, .srt).
    """
    if not output_dir or _has_reparse_in_path(output_dir) or not os.path.isdir(output_dir):
        return []

    audit_dir = os.path.join(output_dir, audit_subdir_name)
    if _is_symlink_or_reparse(audit_dir):
        return []
    moved_files = []

    target_exts = [".audit.json", ".audit.md", ".cast.yaml"]

    try:
        entries = os.listdir(output_dir)
    except Exception:
        return []

    for item in entries:
        item_path = os.path.join(output_dir, item)
        if _is_symlink_or_reparse(item_path) or not os.path.isfile(item_path):
            continue

        is_target = False
        for ext in target_exts:
            if item.endswith(ext):
                if base_name:
                    safe_base = base_name.rstrip(". ")
                    stem_no_ext = item[:-len(ext)]
                    exact_prefixes = [
                        f"{base_name}.",
                        f"{base_name}_4K.",
                        f"{base_name}_4K_animejanai.",
                        f"{base_name}_preview_",
                        f"{safe_base}.",
                        f"{safe_base}_4K.",
                        f"{safe_base}_4K_animejanai.",
                        f"{safe_base}_preview_",
                    ]
                    exact_stems = {
                        base_name,
                        f"{base_name}_4K",
                        f"{base_name}_4K_animejanai",
                        safe_base,
                        f"{safe_base}_4K",
                        f"{safe_base}_4K_animejanai",
                    }
                    if out_stem:
                        safe_out = out_stem.rstrip(". ")
                        exact_prefixes.extend([f"{out_stem}.", f"{safe_out}."])
                        exact_stems.update({out_stem, safe_out})
                    if item.startswith(tuple(exact_prefixes)) or stem_no_ext in exact_stems:
                        is_target = True
                else:
                    is_target = True
                break

        if is_target:
            if _is_symlink_or_reparse(audit_dir):
                continue
            os.makedirs(audit_dir, exist_ok=True)
            dest_path = os.path.join(audit_dir, item)
            if _is_symlink_or_reparse(dest_path):
                continue
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


def _has_resumable_checkpoint_state(temp_dir: str) -> bool:
    """Return True when a regular .temp_* directory holds checkpoint state rather than transient extracts."""
    if not temp_dir or _is_symlink_or_reparse(temp_dir) or not os.path.isdir(temp_dir):
        return False
    base = os.path.basename(temp_dir)
    if base.startswith(".temp_.preview_clip_"):
        return False
    for marker in ("manifest.json", "plan.json", "upscaled_segs", "raw_segs"):
        if os.path.lexists(os.path.join(temp_dir, marker)):
            return True
    try:
        for fn in os.listdir(temp_dir):
            if fn.endswith((".mp4", ".mkv", ".whisper_ja.srt")):
                return True
    except OSError:
        return True
    return False


def clean_all_temporary_data(
    output_dir: str,
    logger=None,
    include_checkpoints: bool = True,
) -> Dict[str, Any]:
    """Clean temporary artifacts and organize audits inside output_dir without following symlinks/reparse points."""
    if not output_dir or _has_reparse_in_path(output_dir) or not os.path.isdir(output_dir):
        return {
            "bytes_freed": 0,
            "cleaned_dirs": [],
            "organized_files": [],
            "preserved_checkpoints": [],
        }

    total_freed = 0
    cleaned_dirs: List[str] = []
    preserved_checkpoints: List[str] = []

    try:
        entries = os.listdir(output_dir)
    except Exception:
        entries = []

    def _clean_temp_candidate(cand_path: str, label: str) -> None:
        nonlocal total_freed
        if _is_symlink_or_reparse(cand_path):
            if _remove_reparse_entry_only(cand_path):
                cleaned_dirs.append(label)
            return
        if not include_checkpoints and _has_resumable_checkpoint_state(cand_path):
            preserved_checkpoints.append(label)
            return
        freed = cleanup_checkpoint_dir(
            cand_path,
            output_dir=output_dir,
            logger=logger,
            follow_internal_symlink=False,
        )
        if freed > 0 or not os.path.lexists(cand_path):
            total_freed += freed
            cleaned_dirs.append(label)

    # 1. Clean top-level .temp_* and transient .preview_clip_* artifacts
    for item in entries:
        item_path = os.path.join(output_dir, item)
        if item.startswith(".temp_"):
            _clean_temp_candidate(item_path, item)
        elif item.startswith(".preview_clip_") and not _is_symlink_or_reparse(item_path) and os.path.isfile(item_path):
            try:
                sz = os.path.getsize(item_path)
                os.remove(item_path)
                total_freed += sz
                cleaned_dirs.append(item)
            except OSError:
                pass

    # 2. Scan 1 level into non-reparse subdirectories for .temp_*
    for item in entries:
        sub_dir = os.path.join(output_dir, item)
        if not item.startswith(".") and not _is_symlink_or_reparse(sub_dir) and os.path.isdir(sub_dir):
            try:
                for sub_item in os.listdir(sub_dir):
                    if sub_item.startswith(".temp_"):
                        _clean_temp_candidate(os.path.join(sub_dir, sub_item), f"{item}/{sub_item}")
            except Exception:
                pass

    # 3. Gom gọn các file audit
    organized_files = organize_output_artifacts(output_dir, logger=logger)

    return {
        "bytes_freed": total_freed,
        "cleaned_dirs": cleaned_dirs,
        "organized_files": organized_files,
        "preserved_checkpoints": preserved_checkpoints,
    }
