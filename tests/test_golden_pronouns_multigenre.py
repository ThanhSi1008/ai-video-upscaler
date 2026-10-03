#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_golden_pronouns_multigenre.py
Bộ quy chuẩn đánh giá tính nhất quán đại từ xưng hô trên Golden Set mở rộng:
Gồm 4 cảnh thuộc 4 thể loại/bối cảnh khác nhau (đáp ứng P3):
1. Thư từ lãng mạn (Epistolary Romance): Rudeus -> Eris & Eris -> Rudeus
2. Chiến đấu & Sinh tồn (High Fantasy Battle / Mentor): Ruijerd <-> Rudeus / Eris
3. Thầy trò & Tôn kính (Master & Disciple): Rudeus <-> Roxy Migurdia
4. Triều đình & Quân thần (Royal Court / Vassalage): Fitz (Sylphie) <-> Vương nữ Ariel

Được thiết kế để:
- Người dùng duyệt bảng quy chuẩn (Speaker -> Listener -> Pronoun Pair)
- Chạy đối chiếu độc lập trên các mô hình
- Không tự hạ cấp xuống Lite/local
"""

import os
import sys
import time
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types

from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from subtrans.config import get_gemini_api_key

EXPANDED_GOLDEN_SET = [
    # =========================================================================
    # CẢNH 1: THƯ TỪ LÃNG MẠN (Epistolary Romance - Rudeus & Eris)
    # =========================================================================
    {
        "scene": "1_Romance_Letter",
        "id": 101,
        "ja": "親愛なるエリスへ。ご無沙汰しております、ルーデウス・グレイラットです。",
        "en": "Dear Eris, it has been a long time. This is Rudeus Greyrat.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "pronoun_pair": "tôi/anh - em",
        "forbidden": ["chị - em", "tớ - cậu", "anh (gọi Eris)"],
        "expected_role": "Rudeus viết thư gửi người yêu/vị hôn thê Eris"
    },
    {
        "scene": "1_Romance_Letter",
        "id": 102,
        "ja": "僕たちが初めて結ばれた夜のことを、今でも鮮明に覚えています。",
        "en": "I still vividly remember the night we were first together.",
        "speaker": "Rudeus",
        "listener": "Eris",
        "pronoun_pair": "tôi/anh - em",
        "forbidden": ["chúng tớ", "hai đứa mình"],
        "expected_role": "Kỷ niệm đêm đầu tiên hai người bên nhau"
    },
    {
        "scene": "1_Romance_Letter",
        "id": 103,
        "ja": "あなたに殴られる覚悟はできています。三人目の妻になってくれませんか。",
        "en": "I am prepared to be punched by you. Will you become my third wife?",
        "speaker": "Rudeus",
        "listener": "Eris",
        "pronoun_pair": "tôi/anh - em",
        "forbidden": ["anh (gọi Eris)", "chị"],
        "expected_role": "Sẵn sàng chịu đòn của Eris và cầu hôn làm vợ ba"
    },
    {
        "scene": "1_Romance_Letter",
        "id": 104,
        "ja": "待ってなさい、ルーデウス！今すぐそこへ行ってやるわ！",
        "en": "Wait for me, Rudeus! I'm coming to you right now!",
        "speaker": "Eris",
        "listener": "Rudeus",
        "pronoun_pair": "ta/em - Rudeus/chàng",
        "forbidden": ["ông", "mày"],
        "expected_role": "Eris phi ngựa gào thét hướng về phía Rudeus"
    },

    # =========================================================================
    # CẢNH 2: CHIẾN ĐẤU & BẢO HỘ (Fantasy Battle & Mentor - Ruijerd & Rudeus)
    # =========================================================================
    {
        "scene": "2_Battle_Mentor",
        "id": 201,
        "ja": "ルーデウス、下がっていろ。奴の相手は俺がする。",
        "en": "Rudeus, step back. I will handle this one.",
        "speaker": "Ruijerd",
        "listener": "Rudeus",
        "pronoun_pair": "ta/tôi - cháu/ngươi/Rudeus",
        "forbidden": ["tớ - cậu", "mình - bạn", "anh - em"],
        "expected_role": "Chiến binh Superd Ruijerd bảo vệ, ra lệnh chiến đấu"
    },
    {
        "scene": "2_Battle_Mentor",
        "id": 202,
        "ja": "ルイジェルドさん、後ろから増援が来ます！牽制は僕に任せてください！",
        "en": "Ruijerd-san, reinforcements are coming from behind! Leave the suppression to me!",
        "speaker": "Rudeus",
        "listener": "Ruijerd",
        "pronoun_pair": "cháu/tôi - chú Ruijerd/bác Ruijerd",
        "forbidden": ["anh - em", "tớ - cậu", "mày - tao"],
        "expected_role": "Rudeus phối hợp chiến thuật, kính trọng tiền bối Ruijerd"
    },
    {
        "scene": "2_Battle_Mentor",
        "id": 203,
        "ja": "子供を守るのが大人の役目だ。お前たちが命を懸ける必要はない。",
        "en": "It is an adult's duty to protect children. There is no need for you to risk your lives.",
        "speaker": "Ruijerd",
        "listener": "Rudeus",
        "pronoun_pair": "ta - các cháu/ngươi",
        "forbidden": ["bọn mày", "mình"],
        "expected_role": "Nguyên tắc bảo vệ trẻ em thiêng liêng của tộc Superd"
    },

    # =========================================================================
    # CẢNH 3: THẦY TRÒ & TÔN KÍNH (Master & Disciple - Rudeus & Roxy)
    # =========================================================================
    {
        "scene": "3_Master_Disciple",
        "id": 301,
        "ja": "ロキシー師匠、お久しぶりです！ご無事で本当によかった……！",
        "en": "Roxy-shishou, it's been so long! I am so relieved you are safe...!",
        "speaker": "Rudeus",
        "listener": "Roxy",
        "pronoun_pair": "con/em/tôi - sư phụ Roxy",
        "forbidden": ["chị Roxy", "bạn Roxy", "tớ - cậu"],
        "expected_role": "Rudeus gặp lại sư phụ Roxy trong mê cung Labyrinth"
    },
    {
        "scene": "3_Master_Disciple",
        "id": 302,
        "ja": "えっ、あの……助けていただいたのは感謝しますが、どなたでしょうか？",
        "en": "Um... I appreciate you saving me, but who might you be?",
        "speaker": "Roxy",
        "listener": "Rudeus",
        "pronoun_pair": "tôi - cậu/ngài",
        "forbidden": ["tao - mày", "chị - em"],
        "expected_role": "Roxy ngơ ngác trước thanh niên cao lớn vừa giải cứu mình"
    },
    {
        "scene": "3_Master_Disciple",
        "id": 303,
        "ja": "師匠の教えがあったからこそ、僕はここまで強くなれました。",
        "en": "It is precisely because of your teachings, Master, that I was able to become this strong.",
        "speaker": "Rudeus",
        "listener": "Roxy",
        "pronoun_pair": "con/em - sư phụ/người",
        "forbidden": ["tớ - cậu", "bạn"],
        "expected_role": "Sự tri ân sâu sắc của đệ tử đối với sư phụ ma pháp"
    },

    # =========================================================================
    # CẢNH 4: TRIỀU ĐÌNH & QUÂN THẦN (Royal Court / Vassal - Fitz & Ariel)
    # =========================================================================
    {
        "scene": "4_Royal_Court",
        "id": 401,
        "ja": "アリエル様、本日の護衛任務、滞りなく完了いたしました。",
        "en": "Lady Ariel, today's guard duties have been completed without issue.",
        "speaker": "Fitz",
        "listener": "Ariel",
        "pronoun_pair": "thần/tôi - Vương nữ Ariel/Điện hạ",
        "forbidden": ["cậu - tớ", "chị - em", "bạn"],
        "expected_role": "Thị vệ Fitz (Sylphie) báo cáo chính sự lên Đệ Nhị Vương nữ Ariel"
    },
    {
        "scene": "4_Royal_Court",
        "id": 402,
        "ja": "ご苦労様でした、フィッツ。あなたにはいつも感謝していますよ。",
        "en": "Thank you for your hard work, Fitz. I am always grateful to you.",
        "speaker": "Ariel",
        "listener": "Fitz",
        "pronoun_pair": "ta/tôi - cậu/ngươi/Fitz",
        "forbidden": ["mày", "anh"],
        "expected_role": "Vương nữ Ariel ân cần khen ngợi cận vệ trung thành"
    },
    {
        "scene": "4_Royal_Court",
        "id": 403,
        "ja": "この命に代えても、アリエル様をアスラ王国の玉座へお導きいたします。",
        "en": "Even at the cost of my life, I will guide Lady Ariel to the throne of the Asura Kingdom.",
        "speaker": "Fitz",
        "listener": "Ariel",
        "pronoun_pair": "thần - Điện hạ/Người",
        "forbidden": ["tớ - cậu", "em - chị"],
        "expected_role": "Lời thề trung thành tuyệt đối với ngai vàng"
    }
]

SYSTEM_PROMPT = """You are a master anime subtitle translator specializing in Japanese to Vietnamese.
CRITICAL ADDRESS RULES:
1. Scene 1 (Romance): Rudeus writes to Eris (lover/fiancee) -> Rudeus: 'tôi'/'anh', Eris: 'em'. Eris shouting to Rudeus: 'Em tới đây, Rudeus!' or 'Ta tới đây, Rudeus!'. NEVER 'chị-em' or 'tớ-cậu'.
2. Scene 2 (Battle/Mentor): Ruijerd (warrior) to Rudeus -> 'ta' - 'cháu'/'ngươi'/'Rudeus'. Rudeus to Ruijerd -> 'cháu'/'tôi' - 'chú Ruijerd'/'Ruijerd'.
3. Scene 3 (Master/Disciple): Rudeus to Roxy (master) -> 'con'/'em' - 'sư phụ Roxy'/'sư phụ'. Roxy to Rudeus -> polite 'tôi' - 'cậu'.
4. Scene 4 (Royal Court): Fitz (bodyguard) to Princess Ariel -> deferential 'thần'/'tôi' - 'Điện hạ'/'Người'/'Vương nữ Ariel'. Ariel to Fitz -> 'ta'/'tôi' - 'cậu'/'Fitz'.

Return a JSON array of objects with fields: id, vi, uncertain."""

class SubtitleItem(BaseModel):
    id: int
    vi: str
    uncertain: bool = False

def main():
    print("=" * 80)
    print("📜 BẢNG QUY CHUẨN XƯNG HÔ ĐA THỂ LOẠI (P3 GOLDEN SET SPECIFICATION)")
    print("=" * 80)
    print(f"{'Cảnh':<18} | {'ID':<4} | {'Người nói -> Nghe':<20} | {'Đại từ chuẩn':<20} | Cấm kỵ")
    print("-" * 80)
    for item in EXPANDED_GOLDEN_SET:
        speaker_rel = f"{item['speaker']} -> {item['listener']}"
        forbidden_str = ", ".join(item['forbidden'])
        print(f"{item['scene']:<18} | {item['id']:<4} | {speaker_rel:<20} | {item['pronoun_pair']:<20} | {forbidden_str}")
    print("=" * 80)

    # Xuất ra file JSON tham chiếu để người dùng phê duyệt
    out_spec = "out/golden_pronouns_specification.json"
    os.makedirs("out", exist_ok=True)
    with open(out_spec, "w", encoding="utf-8") as f:
        json.dump(EXPANDED_GOLDEN_SET, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã xuất bảng đặc tả xưng hô ra: {out_spec}")

if __name__ == "__main__":
    main()
