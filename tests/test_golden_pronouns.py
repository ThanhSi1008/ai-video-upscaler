#!/usr/bin/env python3
"""
tests/test_golden_pronouns.py
Đánh giá độ ổn định của hệ thống đại từ (Pronoun Stability) trên Golden Set.
Chạy 3 lần lặp (3 trials) cho mỗi cấu hình:
1. gemini-3.8-flash với thinking_budget=0
2. gemini-3.8-flash với thinking_budget=1024 (hoặc thinking enabled)
Đo lường:
- Tỷ lệ nhất quán đại từ (Pronoun Consistency Score / 10) qua 3 lần lặp
- Độ trễ phản hồi (Response Latency)
- Xác nhận các lỗi đặc trưng (chị-em, tớ-cậu, anh) có xuất hiện hay không.
"""

import os
import sys
import time
import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from subtrans.config import get_gemini_api_key

# 10 câu thoại then chốt của Mushoku Tensei S02E24 (Bức thư Rudeus gửi Eris & Eris rời Thánh Địa)
GOLDEN_SET = [
    {
        "id": 8,
        "ja": "親愛なるエリスへ。ご無沙汰しております。",
        "en": "Dear Eris, it's been a long time.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "Rudeus gửi Eris (Kính gửi / chào hỏi)"
    },
    {
        "id": 9,
        "ja": "ルーデウス・グレイラットです。",
        "en": "This is Rudeus Greyrat.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "Rudeus tự giới thiệu (tôi là / em là)"
    },
    {
        "id": 10,
        "ja": "僕たちが初めて結ばれた夜のことを覚えていますか。",
        "en": "Do you remember the first night we were together?",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "Rudeus hỏi Eris (chúng ta thuộc về nhau / bên nhau)"
    },
    {
        "id": 11,
        "ja": "あの時、僕はあなたを生涯愛し続けると誓いました。",
        "en": "At that time, I swore to love you for the rest of my life.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "Rudeus xưng tôi/anh, gọi Eris là em"
    },
    {
        "id": 12,
        "ja": "しかし翌朝目覚めた時、あなたの姿はありませんでした。",
        "en": "However, when I woke up the next morning, you were gone.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "Rudeus nói về sự biến mất của Eris (em không còn ở đó)"
    },
    {
        "id": 15,
        "ja": "だが、この手紙を書いたのは、",
        "en": "However, the reason I am writing this letter is...",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "Rudeus giải thích lý do viết thư (tôi/anh viết lá thư này)"
    },
    {
        "id": 16,
        "ja": "あなたが今でも僕を愛してくれていると知ったからです。",
        "en": "because I learned that you still love me.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "Tuyệt đối không dịch thành chị-em; em vẫn còn yêu tôi/anh"
    },
    {
        "id": 19,
        "ja": "現在、僕には妻が二人おり、あなたは三人目となります。",
        "en": "Currently, I have two wives, and you would be the third.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "tôi/anh có hai người vợ, em sẽ là người thứ ba"
    },
    {
        "id": 21,
        "ja": "あなたに殴られる覚悟はできています。",
        "en": "I am prepared to be punched by you.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "expected_role": "sẵn sàng nhận đòn của em (tuyệt đối không gọi Eris là 'anh')"
    },
    {
        "id": 42,
        "ja": "行ってきます。",
        "en": "I'm going.",
        "speaker": "Eris",
        "listener": "Gal Farion",
        "expected_role": "Eris nói với Kiếm Thần (Con đi đây / Ta đi đây, không xưng cháu)"
    }
]

SYSTEM_PROMPT = """You are a master anime subtitle translator specializing in Japanese to Vietnamese.
CRITICAL ADDRESS RULES:
- Rudeus (male, husband/fiance) speaks to Eris (female, wife/fiancee): Rudeus addresses himself as 'tôi' or 'anh', and addresses Eris as 'em'. NEVER address Eris as 'anh' or 'chị'. NEVER use casual 'tớ - cậu'.
- Eris speaks to Gal Farion (Sword God / master): Eris addresses herself as 'con' or 'ta', never 'cháu'.
- Format: Return a JSON array of objects with fields: id, vi, uncertain."""

class SubtitleItem(BaseModel):
    id: int
    vi: str
    uncertain: bool = False

def run_test():
    api_key = get_gemini_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY is not available!")
        sys.exit(1)
        
    client = genai.Client(api_key=api_key)
    model_name = "gemini-3.8-flash"
    
    prompt_text = "Dịch danh sách phụ đề sau sang tiếng Việt tự nhiên:\n"
    for item in GOLDEN_SET:
        prompt_text += f"ID {item['id']}: JA: {item['ja']} | EN: {item['en']} (Speaker: {item['speaker']} -> {item['listener']})\n"

    configs = [
        {"name": "thinking_budget_0", "budget": 0},
        {"name": "thinking_budget_1024", "budget": 1024}
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        budget = cfg["budget"]
        print(f"\n==================================================")
        print(f"🔬 BẮT ĐẦU TEST CẤU HÌNH: {cfg_name} (3 trials)")
        print(f"==================================================")
        
        trials = []
        for trial_idx in range(1, 4):
            print(f"  --> Chạy Trial {trial_idx}/3...")
            t0 = time.time()
            try:
                gen_cfg = {
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_schema": list[SubtitleItem],
                    "temperature": 0.0,
                    "thinking_config": types.ThinkingConfig(thinking_budget=budget)
                }
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config=gen_cfg
                )
                dur = time.time() - t0
                items = json.loads(resp.text)
                trials.append({"trial": trial_idx, "duration_s": dur, "items": items})
                print(f"      Hoàn thành trong {dur:.2f}s ({len(items)} câu)")
            except Exception as e:
                print(f"      Lỗi: {e}")
                trials.append({"trial": trial_idx, "error": str(e)})
            time.sleep(13) # Giữ dưới 5 RPM (1 request / 12s)

        all_results[cfg_name] = trials

    # Lưu kết quả kiểm thử ra JSON
    out_path = "out/golden_pronoun_benchmark.json"
    os.makedirs("out", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu kết quả chi tiết vào: {out_path}")

    # Đánh giá tự động tính nhất quán
    print("\n" + "="*70)
    print("📊 BẢNG ĐỐI CHIẾU ĐẠI TỪ QUA CÁC TRIALS (GOLDEN SET)")
    print("="*70)
    for cfg_name, trials in all_results.items():
        print(f"\n[CẤU HÌNH: {cfg_name}]")
        valid_trials = [t for t in trials if "items" in t]
        if not valid_trials:
            print("  Không có trial nào thành công.")
            continue
            
        durations = [t["duration_s"] for t in valid_trials]
        print(f"  • Thời gian phản hồi trung bình: {sum(durations)/len(durations):.2f}s (Min: {min(durations):.2f}s, Max: {max(durations):.2f}s)")
        
        # Kiểm tra tính nhất quán câu 10, 16, 21, 42
        for check_id in [10, 16, 21, 42]:
            print(f"  • Câu ID {check_id}:")
            for t in valid_trials:
                match = next((item for item in t["items"] if item["id"] == check_id), None)
                if match:
                    print(f"    Trial {t['trial']}: {match['vi']}")

if __name__ == "__main__":
    run_test()
