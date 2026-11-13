# -*- coding: utf-8 -*-
"""
subtrans/dynamic_cast.py
Kiến trúc Quản lý Quan hệ Nhân vật Động Trễ (Hysteresis Dynamic Cast Manager):
- Phân tầng quan hệ: Quan hệ gốc (Baseline) vs. Thay đổi nhất thời (Situational Override)
  vs. Bước ngoặt cốt truyện vĩnh viễn (Permanent Arc Shift).
- Ngăn chặn bẫy "Naive Auto-Update Trap": Khi một nhân vật nổi cáu hoặc nói đùa nhất thời
  trong 1 cảnh (SITUATIONAL_SHIFT: 'temee/kisama' -> mày/tao), hệ thống KHÔNG ghi đè bảng
  xưng hô gốc cho toàn bộ phần còn lại của tập phim.
- Tự động bổ sung nhân vật mới xuất hiện (NEW_CHARACTER: confidence >= 0.75).
- Khóa chặt bước ngoặt cốt truyện (PERMANENT_ARC_SHIFT: confidence >= 0.90) vào nhánh arc_overrides.
"""

import re
from typing import Dict, Any, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field

_VI_1P_CANDIDATES = (
    "thần thiếp", "nô tỳ", "tại hạ", "lão phu", "bổn vương", "thần", "thiếp", "trẫm",
    "ta", "tôi", "tớ", "mình", "anh", "chị", "em", "con", "cháu", "tao"
)
_VI_2P_CANDIDATES = (
    "nương nương", "sư phụ", "tiểu thư", "thiếu gia", "đại nhân", "bệ hạ", "điện hạ",
    "ngài", "các hạ", "huynh", "đệ", "tỷ", "muội", "nàng", "chàng",
    "ngươi", "cậu", "anh", "chị", "em", "cô", "ông", "bà", "bác", "chú", "mày"
)
_HOSTILE_PRONOUNS = {"tao", "mày"}

# Chặn các cụm từ ghép tiếng Việt chứa từ đồng âm với đại từ nhân xưng (tránh tự học sai quan hệ)
_COMPOUND_FALSE_POSITIVES_REGEX = re.compile(
    r"\b(?:"
    r"con\s+(?:rồng|đường|người|dao|gươm|kiếm|quái|thú|vật|mèo|chó|sói|chim|cá|ngựa|thuyền|tàu|sông|suối|ố|số|mắt|tim)|"
    r"một\s+mình|tự\s+mình|bản\s+thân\s+mình|chính\s+mình|rùng\s+mình|giật\s+mình|chuyển\s+mình|ẩn\s+mình|liều\s+mình|"
    r"anh\s+(?:hùng|ấy|ta|chàng|trai|em)|"
    r"chị\s+(?:ấy|ta|gái|em)|"
    r"em\s+(?:ấy|trai|gái|bé)|"
    r"cô\s+(?:ấy|ta|gái|bé|nương|đơn|lập|độc)|"
    r"ông\s+(?:ấy|ta|lão|già|trời|hoàng|chủ)|"
    r"bà\s+(?:ấy|ta|lão|già|chủ)|"
    r"cậu\s+(?:ấy|ta|bé|chủ)|"
    r"chàng\s+trai|nàng\s+tiên|bọn\s+ta|chúng\s+ta|chúng\s+tôi|các\s+ngươi|bọn\s+mày"
    r")\b",
    re.IGNORECASE,
)


def _normalize_pronoun_token(tok: Optional[str], allowed: Tuple[str, ...]) -> Optional[str]:
    if not tok:
        return None
    clean = re.sub(r"[^\w\sÀ-ỹ]", "", tok.strip().lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if clean in allowed:
        return clean
    return None


def _extract_vi_pronoun_pair(vi_text: str) -> Optional[Tuple[str, str]]:
    """Trích xuất cặp đại từ (1P, 2P) tiếng Việt rõ ràng trong câu thoại sau khi đã lọc cụm danh từ ghép."""
    if not vi_text:
        return None
    low = _COMPOUND_FALSE_POSITIVES_REGEX.sub(" ", vi_text.lower())
    found_1p = None
    for p1 in _VI_1P_CANDIDATES:
        if re.search(r"(?<!\w)" + re.escape(p1) + r"(?!\w)", low):
            found_1p = p1
            break
    if not found_1p:
        return None
    found_2p = None
    for p2 in _VI_2P_CANDIDATES:
        if p2 == found_1p:
            continue
        if re.search(r"(?<!\w)" + re.escape(p2) + r"(?!\w)", low):
            found_2p = p2
            break
    if not found_2p:
        return None
    return found_1p, found_2p


class CastRelationDelta(BaseModel):
    scope: Literal["NEW_CHARACTER", "SITUATIONAL_SHIFT", "PERMANENT_ARC_SHIFT"]
    speaker: str
    listener: str
    pronoun_1p: str = Field(description="Đại từ ngôi 1 tiếng Việt (VD: anh, em, ta, tớ, tôi)")
    pronoun_2p: str = Field(description="Đại từ ngôi 2 tiếng Việt (VD: em, anh, ngươi, cậu, bạn)")
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class HysteresisCastManager:
    """
    Trình quản lý Cast Sheet phân tầng có trễ (Hysteresis):
    - relations: Bộ xưng hô baseline
    - arc_overrides: Bước ngoặt cốt truyện đã xác nhận
    - situational_history: Nhật ký ngữ cảnh theo cảnh (chỉ dùng tạm thời)
    """
    def __init__(self, initial_cast_data: Optional[Dict[str, Any]] = None):
        data = initial_cast_data or {}
        self.relations: Dict[str, Dict[str, Any]] = dict(data.get("relations", {}))
        self.arc_overrides: Dict[str, Dict[str, Any]] = dict(data.get("arc_overrides", {}))
        self.situational_log: List[Dict[str, Any]] = []
        self._observed_candidates: Dict[str, Dict[Tuple[str, str], int]] = {}

    def apply_deltas(self, deltas: List[CastRelationDelta]) -> Dict[str, int]:
        """
        Cập nhật Cast Sheet có chọn lọc dựa trên cấp độ scope và ngưỡng tin cậy.
        Returns:
            Thống kê số lượng cập nhật theo từng cấp độ.
        """
        stats = {"new_characters": 0, "situational_shifts": 0, "permanent_shifts": 0}

        for delta in deltas:
            pair_key = f"{delta.speaker.strip()} -> {delta.listener.strip()}"
            if delta.scope == "NEW_CHARACTER" and delta.confidence >= 0.75:
                self.relations[pair_key] = {
                    "1p": delta.pronoun_1p.strip(),
                    "2p": delta.pronoun_2p.strip(),
                    "reason": delta.reason,
                    "confidence": delta.confidence
                }
                stats["new_characters"] += 1

            elif delta.scope == "PERMANENT_ARC_SHIFT" and delta.confidence >= 0.90:
                self.arc_overrides[pair_key] = {
                    "1p": delta.pronoun_1p.strip(),
                    "2p": delta.pronoun_2p.strip(),
                    "reason": delta.reason,
                    "confidence": delta.confidence
                }
                stats["permanent_shifts"] += 1

            elif delta.scope == "SITUATIONAL_SHIFT":
                # Chỉ lưu vào nhật ký cảnh, TUYỆT ĐỐI không sửa relations toàn cục!
                self.situational_log.append({
                    "pair": pair_key,
                    "1p": delta.pronoun_1p,
                    "2p": delta.pronoun_2p,
                    "reason": delta.reason
                })
                stats["situational_shifts"] += 1

        return stats

    def resolve_pronouns(self, speaker: str, listener: str) -> Optional[Tuple[str, str]]:
        """
        Tra cứu đại từ xưng hô theo thứ tự ưu tiên:
        1. Arc Overrides (Bước ngoặt đã xác nhận)
        2. Relations Baseline (Quy tắc gốc)
        """
        pair_key = f"{speaker.strip()} -> {listener.strip()}"
        if pair_key in self.arc_overrides:
            rec = self.arc_overrides[pair_key]
            return rec.get("1p", ""), rec.get("2p", "")
        if pair_key in self.relations:
            rec = self.relations[pair_key]
            return rec.get("1p", ""), rec.get("2p", "")
        return None

    def to_prompt_context(self) -> str:
        """
        Sinh chuỗi ngữ cảnh quan hệ nhân vật để đưa vào prompt Gemini API.
        Ưu tiên arc_overrides lên đầu, sau đó là relations baseline.
        """
        lines = []
        if self.arc_overrides:
            lines.append("XƯNG HÔ BƯỚC NGOẶT (ARC OVERRIDES - BẮT BUỘC TUÂN THỦ):")
            for pair, rel in self.arc_overrides.items():
                lines.append(f"  • {pair}: {rel.get('1p', '')}/{rel.get('2p', '')} (Lý do: {rel.get('reason', '')})")
        if self.relations:
            lines.append("XƯNG HÔ CHUẨN CƠ BẢN (BASELINE RELATIONS):")
            for pair, rel in self.relations.items():
                if pair not in self.arc_overrides:
                    lines.append(f"  • {pair}: {rel.get('1p', '')}/{rel.get('2p', '')}")
        return "\n".join(lines)

    @classmethod
    def from_glossary(cls, glossary: Any) -> "HysteresisCastManager":
        """Khởi tạo Cast Manager từ đối tượng Glossary nếu có."""
        relations = {}
        if glossary and hasattr(glossary, "characters"):
            for char in glossary.characters:
                c_name = getattr(char, "name_vi", None) or getattr(char, "name_ja", "")
                for st in getattr(char, "speaks_to", []):
                    pair = f"{c_name} -> {st.listener}"
                    relations[pair] = {
                        "1p": st.self,
                        "2p": st.other,
                        "reason": getattr(st, "note", "") or "Glossary baseline",
                        "confidence": 1.0
                    }
        return cls({"relations": relations})

    def observe_batch_cues(self, cues: List[Any], min_consistent_hits: int = 2) -> Dict[str, int]:
        """
        Đóng kín vòng lặp phản hồi từ đầu ra CoT đã xác thực của LLM:
        - Chỉ quan sát các cue tin cậy cao (`not uncertain`, `mode in ('spoken', 'comm')`).
        - Nếu cặp `speaker -> addressee` đã tồn tại nhưng xuất hiện đại từ nổi nóng (`tao/mày`),
          ghi nhận `SITUATIONAL_SHIFT` mà KHÔNG ghi đè baseline.
        - Nếu cặp nhân vật mới xuất hiện nhất quán >= `min_consistent_hits` lần với cùng cặp đại từ
          không thù địch, tự động khóa vào `relations` (`NEW_CHARACTER`) cho các batch kế tiếp.
        """
        deltas: List[CastRelationDelta] = []
        invalid_names = {"", "unknown", "all", "none", "null", "self", "narrator"}
        for cue in cues or []:
            if getattr(cue, "uncertain", False):
                continue
            mode = (getattr(cue, "mode", None) or "spoken").strip().lower()
            if mode not in ("spoken", "comm"):
                continue
            spk = (getattr(cue, "speaker", None) or "").strip()
            lst = (getattr(cue, "addressee", None) or "").strip()
            if spk.lower() in invalid_names or lst.lower() in invalid_names or spk.lower() == lst.lower():
                continue
            # Ưu tiên tuyệt đối cặp đại từ tường minh (p1, p2) từ Schema CoT của LLM trước khi fallback regex
            explicit_p1 = _normalize_pronoun_token(getattr(cue, "p1", None), _VI_1P_CANDIDATES)
            explicit_p2 = _normalize_pronoun_token(getattr(cue, "p2", None), _VI_2P_CANDIDATES)
            if explicit_p1 and explicit_p2 and explicit_p1 != explicit_p2:
                pair: Optional[Tuple[str, str]] = (explicit_p1, explicit_p2)
            else:
                pair = _extract_vi_pronoun_pair(getattr(cue, "vi", "") or "")
            if not pair:
                continue
            p1, p2 = pair
            existing = self.resolve_pronouns(spk, lst)
            if existing is not None:
                if (p1 in _HOSTILE_PRONOUNS or p2 in _HOSTILE_PRONOUNS) and existing != (p1, p2):
                    deltas.append(
                        CastRelationDelta(
                            scope="SITUATIONAL_SHIFT",
                            speaker=spk,
                            listener=lst,
                            pronoun_1p=p1,
                            pronoun_2p=p2,
                            reason="Observed situational hostile shift in batch dialogue",
                            confidence=0.85,
                        )
                    )
                continue

            if p1 in _HOSTILE_PRONOUNS or p2 in _HOSTILE_PRONOUNS:
                continue
            pair_key = f"{spk} -> {lst}"
            bucket = self._observed_candidates.setdefault(pair_key, {})
            bucket[(p1, p2)] = bucket.get((p1, p2), 0) + 1
            if bucket[(p1, p2)] >= min_consistent_hits and self.resolve_pronouns(spk, lst) is None:
                deltas.append(
                    CastRelationDelta(
                        scope="NEW_CHARACTER",
                        speaker=spk,
                        listener=lst,
                        pronoun_1p=p1,
                        pronoun_2p=p2,
                        reason=f"Consistent verified CoT pronoun usage ({bucket[(p1, p2)]}x)",
                        confidence=0.80,
                    )
                )
        return self.apply_deltas(deltas) if deltas else {"new_characters": 0, "situational_shifts": 0, "permanent_shifts": 0}

    def export_state(self) -> Dict[str, Any]:
        """Xuất toàn bộ trạng thái quan hệ nhân vật hiện hành."""
        return {
            "relations": self.relations,
            "arc_overrides": self.arc_overrides,
            "situational_shifts_count": len(self.situational_log)
        }

