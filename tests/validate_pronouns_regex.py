#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/validate_pronouns_regex.py
Bộ kiểm tra tính nhất quán đại từ độc lập (Rule-based Regex Validator) trên toàn bộ 43 cue của sample_3min:
- Xác lập bảng xưng hô kỳ vọng theo từng cặp (Người nói -> Người nghe)
- Kiểm tra danh sách cấm (Forbidden Patterns) và danh sách hợp lệ (Allowed Patterns)
- Kiểm định công bằng trên tất cả các bản dịch:
  1. out/compare_raw/gemini-3.8-flash.srt
  2. out/compare_raw/gemini-3.5-flash-lite.srt
  3. out/compare_raw/translategemma_12b.srt
  4. out/local_fair/sample_3min_jpn.vi.srt (hoặc out/local/sample_3min_jpn.vi.srt)
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# BẢNG XÁC LẬP QUY TẮC ĐẠI TỪ CHO 43 CUES (Mushoku Tensei S02E24 Thư Rudeus & Eris rời Thánh Địa)
PRONOUN_RULES = {
    # Nhóm 1: Isolde & Nina & Eris trao đổi thư từ (Cues 0 - 6)
    2: {
        "speaker": "Eris", "listener": "Nina",
        "context": "Eris nhờ Nina đọc thư",
        "forbidden": [r"\banh\b", r"\bchị\b"],
        "note": "Eris và Nina là bạn đồng môn tập kiếm, không dùng anh/chị"
    },
    
    # Nhóm 2: Thư của Rudeus gửi Eris (Cues 7 - 22)
    # Speaker: Rudeus (nam), Listener: Eris (nữ, người yêu/vợ sắp cưới, CHƯA phải vợ chồng chính thức)
    9: { # Cue 10 trong phụ đề: 僕たちが初めて結ばれた夜のこと
        "speaker": "Rudeus", "listener": "Eris",
        "context": "Rudeus nhắc lại đêm đầu tiên bên nhau",
        # Lỗi chí tử của local model: gọi Eris là 'Anh' (ví dụ: 'Anh có còn nhớ...')
        "forbidden": [r"^[Aa]nh có", r"\bchị\b", r"\btớ\b", r"\bcậu\b"],
        "required_any": [r"\bem\b", r"\bchúng ta\b", r"\btôi\b", r"\banh\b"],
        "note": "CẤM gọi Eris là 'Anh' ('Anh có còn nhớ...'). Rudeus xưng tôi/anh, gọi Eris là em."
    },
    10: { # Cue 11: あの時、僕は生涯あなたを愛し続けると誓いました
        "speaker": "Rudeus", "listener": "Eris",
        "context": "Rudeus thề yêu Eris trọn đời",
        "forbidden": [r"\bchị\b", r"\btớ\b", r"\bcậu\b"],
        "required_any": [r"\bem\b", r"\bnàng\b", r"\bngười\b"],
        "note": "Đối tượng yêu là Eris (nữ), phải dùng 'em'/'nàng', cấm 'chị', cấm 'tớ-cậu'"
    },
    11: { # Cue 12: しかし、朝目覚めた時、あなたの姿はありませんでした
        "speaker": "Rudeus", "listener": "Eris",
        "context": "Rudeus nói sáng hôm sau Eris biến mất",
        "forbidden": [r"\banh lại\b", r"\bbóng dáng anh\b", r"\bchị\b"],
        "note": "Người biến mất là Eris (nữ), cấm gọi Eris là 'anh'"
    },
    15: { # Cue 16: あなたの心がずっと僕にあったと知らされたからです
        "speaker": "Rudeus", "listener": "Eris",
        "context": "Rudeus biết trái tim Eris vẫn hướng về mình",
        "forbidden": [r"\bchị\b", r"\btớ\b", r"\bcậu\b"],
        "note": "Tuyệt đối không đảo lộn vai vế hoặc dùng chị-em"
    },
    18: { # Cue 19: 現在僕には二人の妻がおり、あなたは三人目となります
        "speaker": "Rudeus", "listener": "Eris",
        "context": "Rudeus nói có 2 vợ và Eris sẽ là người thứ ba",
        "forbidden": [r"\banh sẽ là\b", r"\bchị sẽ là\b"],
        "required_any": [r"\bem\b", r"\bnàng\b", r"\bcô\b"],
        "note": "Eris là người thứ ba, phải xưng em/nàng/cô, cấm gọi Eris là 'anh'"
    },
    20: { # Cue 21: あなたの拳を受ける覚悟はできています
        "speaker": "Rudeus", "listener": "Eris",
        "context": "Rudeus sẵn sàng nhận nắm đấm của Eris",
        "forbidden": [r"nắm đấm của anh\b", r"đòn của anh\b", r"\bchị\b"],
        "note": "Nắm đấm là của Eris, cấm dịch thành 'nắm đấm của anh'"
    },
    
    # Nhóm 3: Phản ứng của Isolde & Nina (Cues 23 - 30)
    23: {
        "speaker": "Isolde", "listener": "Nina/Eris",
        "context": "Isolde chửi Rudeus",
        "forbidden": [r"\banh yêu\b"],
        "note": "Isolde phẫn nộ: 'Đồ tồi!'"
    },
    24: {
        "speaker": "Isolde", "listener": "Nina/Eris",
        "context": "Isolde mắng Rudeus coi thường phụ nữ",
        "note": "Isolde nói về Rudeus (hắn/tên đó/cậu ta)"
    },
    
    # Nhóm 4: Tái bút (Cues 34 - 37)
    35: { # Cue 36: 僕はこれから、龍神オルステッドに戦いを挑みます
        "speaker": "Rudeus", "listener": "Eris",
        "context": "Rudeus chuẩn bị khiêu chiến Long Thần Orsted",
        "forbidden": [r"\btớ\b"],
        "note": "Bối cảnh sinh tử chiến với Orsted, xưng tôi hoặc anh"
    },
    
    # Nhóm 5: Eris lên ngựa xuất phát (Cues 38 - 42)
    39: { # Cue 40: 行くわよ、ギレーヌ！
        "speaker": "Eris", "listener": "Ghislaine",
        "context": "Eris gọi Ghislaine lên đường",
        "required_any": [r"[Gg]hislaine", r"[Gg]i-lên"],
        "note": "Eris nói với Ghislaine (hầu cận/sư phụ kiếm)"
    },
    41: { # Cue 42: 今行くわ、
        "speaker": "Eris", "listener": "Rudeus",
        "context": "Eris gào gọi Rudeus khi phi ngựa",
        # CẤM TUYỆT ĐỐI gán nhầm sang Gal Farion ('Con đi đây' / 'sư phụ')!
        "forbidden": [r"\bcon đi\b", r"\bcháu đi\b", r"\bsư phụ\b"],
        "required_any": [r"\b[Tt]a tới\b", r"\b[Ee]m tới\b", r"\b[Tt]ới đây\b", r"\b[Đđ]ến đây\b", r"\b[Đđ]i đây\b"],
        "note": "Eris hướng về Rudeus gào thét, CẤM dịch thành 'Con đi đây' (nói với Gal Farion)!"
    },
    42: { # Cue 43: ルーデウス！
        "speaker": "Eris", "listener": "Rudeus",
        "context": "Eris gào tên Rudeus",
        "required_any": [r"[Rr]udeus", r"[Rr]u-đê-út"],
        "note": "Tiếng gào gọi tên Rudeus!"
    }
}

def parse_srt(srt_path: str) -> List[Dict[str, Any]]:
    """Phân tích file SRT thành danh sách các cue."""
    if not os.path.exists(srt_path):
        return []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    blocks = re.split(r"\n\s*\n", content.strip())
    cues = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) >= 3:
            try:
                idx = int(lines[0].strip())
            except ValueError:
                continue
            time_str = lines[1].strip()
            text = " ".join([l.strip() for l in lines[2:]])
            cues.append({"index": idx, "time": time_str, "text": text})
    return cues

def evaluate_srt(srt_path: str) -> Dict[str, Any]:
    cues = parse_srt(srt_path)
    if not cues:
        return {"error": "File not found or empty"}
        
    cue_dict = {c["index"]: c["text"] for c in cues}
    # Cũng ánh xạ theo vị trí (0-based) đề phòng đánh số lệch 1
    cue_list = [c["text"] for c in cues]
    
    violations = []
    passed_rules = 0
    total_rules = len(PRONOUN_RULES)
    
    details = []
    
    for cue_idx, rule in PRONOUN_RULES.items():
        # Tìm text của cue tương ứng (thử index 1-based, 0-based, hoặc số nguyên)
        text = ""
        if (cue_idx + 1) in cue_dict:
            text = cue_dict[cue_idx + 1]
        elif cue_idx in cue_dict:
            text = cue_dict[cue_idx]
        elif cue_idx < len(cue_list):
            text = cue_list[cue_idx]
            
        rule_passed = True
        failed_reasons = []
        
        # 1. Kiểm tra Forbidden
        for pat in rule.get("forbidden", []):
            if re.search(pat, text, re.IGNORECASE):
                rule_passed = False
                failed_reasons.append(f"Chứa từ cấm: '{pat}'")
                
        # 2. Kiểm tra Required Any
        if rule.get("required_any"):
            req_matched = any(re.search(pat, text, re.IGNORECASE) for pat in rule["required_any"])
            if not req_matched:
                rule_passed = False
                failed_reasons.append(f"Thiếu mẫu hợp lệ kỳ vọng: {rule['required_any']}")
                
        if rule_passed:
            passed_rules += 1
        else:
            violations.append({
                "cue_idx": cue_idx,
                "speaker": rule["speaker"],
                "listener": rule["listener"],
                "text": text,
                "reasons": failed_reasons,
                "note": rule["note"]
            })
            
        details.append({
            "cue_idx": cue_idx,
            "passed": rule_passed,
            "text": text,
            "failed_reasons": failed_reasons
        })
        
    score = (passed_rules / total_rules) * 10.0
    return {
        "file": os.path.basename(srt_path),
        "total_cues": len(cues),
        "rules_checked": total_rules,
        "rules_passed": passed_rules,
        "consistency_score_out_of_10": round(score, 2),
        "violations": violations,
        "details": details
    }

def run_all_evaluations():
    candidates = [
        "out/compare_raw/gemini-3.8-flash.srt",
        "out/compare_raw/gemini-3.5-flash-lite.srt",
        "out/compare_raw/translategemma_12b.srt",
        "out/local_fair/sample_3min_jpn.vi.srt",
        "out/gemini/sample_3min_jpn.vi.srt"
    ]
    
    print("=" * 80)
    print("🔍 KIỂM ĐỊNH ĐẠI TỪ BẰNG BỘ QUY TẮC ĐỘC LẬP (REGEX VALIDATOR TRÊN 43 CUES)")
    print("=" * 80)
    
    results = {}
    for p in candidates:
        if not os.path.exists(p):
            continue
        eval_res = evaluate_srt(p)
        results[p] = eval_res
        print(f"\n📄 ĐÁNH GIÁ FILE: {p}")
        print(f"  • Điểm đại từ: {eval_res['consistency_score_out_of_10']}/10 ({eval_res['rules_passed']}/{eval_res['rules_checked']} rules pass)")
        if eval_res["violations"]:
            print(f"  • Vi phạm ({len(eval_res['violations'])} lỗi):")
            for v in eval_res["violations"]:
                print(f"    - Cue {v['cue_idx']} [{v['speaker']} -> {v['listener']}]: \"{v['text']}\"")
                print(f"      Lỗi: {', '.join(v['reasons'])} | Ghi chú: {v['note']}")
        else:
            print("  • Không phát hiện vi phạm đại từ nào!")
            
    out_json = "out/pronoun_validation_report.json"
    os.makedirs("out", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Đã lưu báo cáo chi tiết vào: {out_json}")

if __name__ == "__main__":
    run_all_evaluations()
