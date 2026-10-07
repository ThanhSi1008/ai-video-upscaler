"""
src/output_sanity.py - Chốt chặn kiểm định chất lượng hình ảnh đầu ra (Output Sanity Guard)
Ngăn chặn hoàn toàn lỗi mô hình xuất video đen, suy biến màu hoặc sai lệch cấu trúc.
Tuân thủ yêu cầu C3 (sau chunk đầu tiên và mỗi 10 chunk sau).
"""

import os
import time
import json
import cv2
import numpy as np


class OutputSanityError(RuntimeError):
    """Ngoại lệ khi frame đầu ra không đạt ngưỡng chốt chặn chất lượng (màn hình đen, lệch màu)."""
    def __init__(self, message: str, metrics: dict = None):
        super().__init__(message)
        self.metrics = metrics or {}


def check_frame_sanity(
    src_frame_path: str,
    out_frame_path: str,
    max_delta_y: float = 6.0,
    min_corr: float = 0.98
) -> dict:
    """
    Kiểm định độ sáng và tương quan giữa 1 frame nguồn và 1 frame đầu ra sau AI:
    - Thu nhỏ cả hai về 480x270 bằng cv2.INTER_AREA.
    - Tính độ sáng BT.709: Y = 0.2126*R + 0.7152*G + 0.0722*B.
    - So sánh chênh lệch độ sáng trung bình: |ΔY| <= max_delta_y (trên thang 255).
    - So sánh hệ số tương quan Pearson: corr >= min_corr.
    - Đo thời gian thực thi (chi phí < 1s).
    """
    t0 = time.time()
    
    if not os.path.exists(src_frame_path):
        raise FileNotFoundError(f"Không tìm thấy frame nguồn: {src_frame_path}")
    if not os.path.exists(out_frame_path):
        raise FileNotFoundError(f"Không tìm thấy frame đầu ra: {out_frame_path}")

    src_bgr = cv2.imread(src_frame_path)
    out_bgr = cv2.imread(out_frame_path)

    if src_bgr is None:
        raise ValueError(f"Không thể đọc ảnh nguồn: {src_frame_path}")
    if out_bgr is None:
        raise ValueError(f"Không thể đọc ảnh đầu ra: {out_frame_path}")

    # Thu nhỏ về chuẩn 480x270
    target_size = (480, 270)
    src_small = cv2.resize(src_bgr, target_size, interpolation=cv2.INTER_AREA)
    out_small = cv2.resize(out_bgr, target_size, interpolation=cv2.INTER_AREA)

    # Chuyển đổi luma BT.709 (OpenCV đọc BGR: B=0, G=1, R=2)
    y_src = 0.2126 * src_small[:, :, 2] + 0.7152 * src_small[:, :, 1] + 0.0722 * src_small[:, :, 0]
    y_out = 0.2126 * out_small[:, :, 2] + 0.7152 * out_small[:, :, 1] + 0.0722 * out_small[:, :, 0]

    mean_y_src = float(y_src.mean())
    mean_y_out = float(y_out.mean())
    delta_y = abs(mean_y_out - mean_y_src)

    # Tính tương quan Pearson
    y_src_flat = y_src.flatten()
    y_out_flat = y_out.flatten()
    std_src = np.std(y_src_flat)
    std_out = np.std(y_out_flat)

    if std_src < 1e-5 and std_out < 1e-5:
        # Cả hai đều là ảnh thuần 1 màu phẳng
        corr = 1.0 if delta_y <= max_delta_y else 0.0
    elif std_src < 1e-5 or std_out < 1e-5:
        corr = 0.0
    else:
        corr_mat = np.corrcoef(y_src_flat, y_out_flat)
        corr = float(corr_mat[0, 1])
        if np.isnan(corr) or np.isinf(corr):
            corr = 0.0

    passed = bool((delta_y <= max_delta_y) and (corr >= min_corr))
    elapsed_ms = (time.time() - t0) * 1000.0

    return {
        "passed": passed,
        "delta_y": round(delta_y, 4),
        "mean_y_src": round(mean_y_src, 2),
        "mean_y_out": round(mean_y_out, 2),
        "corr": round(corr, 6),
        "max_delta_y": max_delta_y,
        "min_corr": min_corr,
        "elapsed_ms": round(elapsed_ms, 2)
    }


def write_failed_sanity_audit(
    audit_file_path: str,
    source_file: str,
    output_file: str,
    model_info: dict,
    engine: str,
    precision: str,
    chunk_idx: int,
    metrics: dict,
    all_checks: list = None
) -> None:
    """
    Ghi file audit.json với trạng thái FAIL (OUTPUT_SANITY) khi chốt chặn phát hiện lỗi.
    Tuyệt đối không mux và không fallback sang model khác.
    """
    audit_data = {
        "status": "FAIL (OUTPUT_SANITY)",
        "failure_reason": f"Chốt chặn đầu ra thất bại tại phân đoạn {chunk_idx+1}: |ΔY|={metrics.get('delta_y')} (ngưỡng <={metrics.get('max_delta_y')}), Corr={metrics.get('corr')} (ngưỡng >={metrics.get('min_corr')})",
        "source_file": os.path.abspath(source_file),
        "output_file": os.path.abspath(output_file),
        "model_id": model_info.get("id"),
        "model_name": model_info.get("name"),
        "model_sha256": model_info.get("sha256"),
        "engine": engine,
        "precision": precision,
        "failed_chunk": chunk_idx + 1,
        "failed_metrics": metrics,
        "output_sanity": {
            "status": "FAIL",
            "total_checks": len(all_checks) if all_checks else 1,
            "checks": all_checks or [metrics]
        }
    }
    with open(audit_file_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
