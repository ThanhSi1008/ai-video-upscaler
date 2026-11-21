"""Pure-stdlib shows.yaml glob and season-subfolder path matching helpers."""

from __future__ import annotations

import os
import re

SEASON_SUBFOLDER_RE = re.compile(
    r"^(?:Season\s*\d+|S\d+|Part\s*\d+|Cour\s*\d+|Vol\.?\s*\d+|Disc\s*\d+|BD\s*\d*|Specials?|OVA|Movies?)$",
    re.IGNORECASE,
)


def match_show_pattern(text: str, pattern: str) -> bool:
    """Match a shows.yaml pattern case-insensitively without treating '[' or ']' as charsets."""
    if not text or not pattern:
        return False
    t_low = str(text).lower()
    p_low = str(pattern).strip().lower()
    if not p_low:
        return False
    if "*" not in p_low and "?" not in p_low:
        return p_low in t_low
    regex_parts = ["^"]
    for ch in p_low:
        if ch == "*":
            regex_parts.append(".*")
        elif ch == "?":
            regex_parts.append(".")
        else:
            regex_parts.append(re.escape(ch))
    regex_parts.append("$")
    return re.match("".join(regex_parts), t_low, flags=re.DOTALL) is not None


def candidate_show_names(video_path: str | os.PathLike[str]) -> list[str]:
    """Return basename, parent folder, and grandparent folder (only when parent is a season/disc subfolder)."""
    vid_str = os.fspath(video_path).rstrip("\\/")
    if not vid_str:
        return []
    abs_path = os.path.abspath(vid_str)
    vid_name = os.path.basename(abs_path)
    parent_path = os.path.dirname(abs_path)
    parent_name = os.path.basename(parent_path)
    candidates = [c for c in (vid_name, parent_name) if c]
    if parent_name and SEASON_SUBFOLDER_RE.match(parent_name.strip()):
        grandparent_name = os.path.basename(os.path.dirname(parent_path))
        if grandparent_name:
            candidates.append(grandparent_name)
    return candidates


def match_show_rule(video_path: str | os.PathLike[str], pattern: str) -> bool:
    """Return True when pattern matches filename, parent folder, or season-parent grandparent folder."""
    if not pattern:
        return False
    return any(match_show_pattern(cand, pattern) for cand in candidate_show_names(video_path))
