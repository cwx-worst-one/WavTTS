"""
Functions that clean up "downloaded_lyrics"
"""

import re

import zhconv


def clean_karaoke_lyrics(text: str) -> str:
    """
    Handles English, Chinese, and Japanese lyrics.
    Uses filtering rules for colons, slashes, and dashes to remove metadata.

    Args:
        text (str): Raw karaoke lyrics text

    Returns:
        str: Cleaned lyrics text
    """

    # Step 1: Remove timestamps in brackets [00:43.980]
    text = re.sub(r"\[[\d:.,\s]+\]", "", text)

    # Step 2: Split into lines for processing
    lines = text.split("\n")

    # Step 3: Apply filtering rules
    cleaned_lines = []
    lyrics_started = False
    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Rule 2: Detect other metadata patterns
        is_metadata = _is_metadata_line(line)

        # Rule 3: If we haven't found lyrics yet, be more aggressive about filtering
        if not lyrics_started:
            # Check if this looks like actual lyrics content
            if _looks_like_lyrics(line) and not is_metadata:
                lyrics_started = True
            else:
                continue  # Skip potential metadata at the beginning

        # Skip metadata lines
        if is_metadata:
            continue

        # Step 4: Remove special characters, keep Chinese/Japanese characters, English letters, and basic punctuation
        # Keep: Chinese characters, Japanese (Hiragana, Katakana), English letters,
        #   apostrophes, spaces, parentheses, commas, hyphens
        line = re.sub(
            r"[^\u4e00-\u9fff\u3040-\u309F\u30A0-\u30FF\u3400-\u4dbf\u20000-\
                \u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\
                \u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1fa-zA-Z\'\s(),\-]",
            "",
            line,
        )

        # Clean up extra spaces
        line = re.sub(r"\s+", " ", line).strip()

        # Step 5: Add cleaned line if it's not empty
        if line:
            cleaned_lines.append(line)

    cleaned_lines = [l.strip() for l in cleaned_lines if l.strip()]
    cleaned_lines = _remove_chinese_translations(cleaned_lines)
    cleaned_lines = [zhconv.convert(l, "zh-cn") for l in cleaned_lines]
    return "\n".join(cleaned_lines)


def calculate_overlap_metrics(asr_lyrics: str, proc_lyrics: str) -> dict:
    """
    Fast calculation of overlap metrics
    """
    asr_words = _preprocess_lyrics(_remove_section_tags(asr_lyrics))
    proc_words = _preprocess_lyrics(_remove_section_tags(proc_lyrics))

    if not asr_words or not proc_words:
        return {
            "overlap_rate": 0.0,
            "missing_beginning": False,
            "missing_ending": False,
        }

    # Convert to sets for fast intersection
    asr_set = set(asr_words)
    proc_set = set(proc_words)

    # Basic overlap rate
    intersection = asr_set & proc_set
    overlap_rate = len(intersection) / len(asr_set) if asr_set else 0.0

    # Fast missing detection using simple heuristics
    asr_len = len(asr_words)
    proc_len = len(proc_words)

    # Check beginning: compare first 20% of words
    check_len = min(10, max(3, asr_len // 5))
    if asr_len >= check_len and proc_len >= check_len:
        asr_start = set(asr_words[:check_len])
        proc_start = set(proc_words[:check_len])
        start_overlap = len(asr_start & proc_start) / len(asr_start)
        missing_beginning = start_overlap < 0.3
    else:
        missing_beginning = asr_len > proc_len

    # Check ending: compare last 20% of words
    if asr_len >= check_len and proc_len >= check_len:
        asr_end = set(asr_words[-check_len:])
        proc_end = set(proc_words[-check_len:])
        end_overlap = len(asr_end & proc_end) / len(asr_end)
        missing_ending = end_overlap < 0.3
    else:
        missing_ending = asr_len > proc_len

    return {
        "overlap_rate": overlap_rate,
        "missing_beginning": missing_beginning,
        "missing_ending": missing_ending,
    }


def _is_metadata_line(line: str) -> bool:
    """
    Determine if a line is metadata using pattern-based rules.
    Works for English, Chinese, and Japanese content.
    Note: Lines with colons, slashes, and dashes are already filtered out in the main function.

    Args:
        line (str): Line to analyze

    Returns:
        bool: True if line appears to be metadata
    """

    # Rule 1: Lines containing typical metadata symbols
    metadata_symbols = [
        "@",
        "Studio",
        "STUDIO",
        "Limited",
        "LIMITED",
        "Records",
        "RECORDS",
        "Inc",
        "LLC",
        "Ltd",
        "Corp",
        "©",
        "℗",
        "Publishing",
        "レコード",
        "スタジオ",
    ]
    if any(symbol in line for symbol in metadata_symbols):
        return True

    # Rule 2: Lines containing artist collaboration indicators (already filtered by dash rule)
    if " feat" in line.lower() or " ft" in line.lower() or "featuring" in line.lower():
        return True

    # Rule 3: Lines that are mostly symbols/numbers (version numbers, catalog numbers, etc.)
    alphanumeric_chars = len(
        re.findall(r"[a-zA-Z\u4e00-\u9fff\u3040-\u309F\u30A0-\u30FF]", line)
    )
    total_chars = len(re.sub(r"\s", "", line))
    if total_chars > 0 and alphanumeric_chars / total_chars < 0.5:
        return True

    # Rule 4: Lines that contain these special characters
    excluded_chars = ["：", ":", "/", " - ", "「", "」", "《", "》", "【", "】"]
    if any(e in line for e in excluded_chars):
        return True

    # Rule 5: Lines that start with these keywords
    startswiths = [
        # Chinese production terms
        "制作人",
        "制作",
        "编曲",
        "词",
        "曲",
        "监制",
        "录音师",
        "录音",
        "混音师",
        "混音",
        "录音棚",
        "封面",
        "贝斯",
        "吉他",
        "钢琴",
        "鼓",
        "弦乐",
        "伴唱",
        "人声",
        "母带",
        "唱片公司",
        "发行公司",
        "音乐制作",
        "发行",
        "出品",
        "策划",
        "统筹",
        "摄影",
        "设计",
        "以下歌词翻译",
    ]
    if any(line.startswith(t + " ") for t in startswiths):
        return True

    # Rule 6: Skip lines with hyphens that are not connecting English words
    if "-" in line:
        if not re.search(r"[a-zA-Z]-[a-zA-Z]", line):
            return True

    # # Rule 7: Very short lines (likely labels or incomplete metadata)
    # if len(line.strip()) <= 2:
    #     return True

    return False


def _looks_like_lyrics(line: str) -> str:
    """
    Determine if a line looks like actual lyrics content.
    Works for English, Chinese, and Japanese content.
    Note: Lines with colons, slashes, and dashes are already filtered out in the main function.

    Args:
        line (str): Line to analyze

    Returns:
        bool: True if line appears to be lyrics
    """

    # Check if line has substantial lyrical content
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", line))
    japanese_chars = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF]", line))
    english_words = len(re.findall(r"[a-zA-Z]+", line))

    # Lyrics should have reasonable content (Chinese characters, Japanese characters, or English words)
    if chinese_chars >= 3 or japanese_chars >= 3 or english_words >= 2:
        # Shouldn't have metadata patterns
        if not _is_metadata_line(line):
            return True
    return False


def _remove_chinese_translations(
    lines: list[str], strict: bool = False, threshold: float = 0.8
) -> list[str]:
    """Remove Chinese translation lines if lyrics follow alternating pattern."""
    if len(lines) < 2 or not _is_alternating_pattern(lines, strict, threshold):
        return lines

    return [line for line in lines if not _has_chinese(line)]


def _is_alternating_pattern(lines: list[str], strict: bool, threshold: float) -> bool:
    """Check if lines follow alternating English-Chinese pattern."""
    if len(lines) < 2:
        return False

    alt_pattern_matches = [
        _has_chinese(lines[i]) != _has_chinese(lines[i + 1])
        for i in range(len(lines) - 1)
    ]
    if strict:
        return all(alt_pattern_matches)
    return len([p for p in alt_pattern_matches if p]) >= len(lines) * threshold


def _has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _preprocess_lyrics(lyrics: str) -> list[str]:
    """
    Fast preprocessing for lyrics - works for English, Chinese, Japanese
    """
    if not lyrics or not lyrics.strip():
        return []

    # Convert to lowercase and clean in one pass
    lyrics = str(lyrics).lower()
    lyrics = re.sub(r"[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", " ", lyrics)
    lyrics = re.sub(r"\s+", " ", lyrics).strip()

    if not lyrics:
        return []

    # Fast tokenization
    words = []
    parts = lyrics.split()

    for part in parts:
        # Check for CJK characters
        if re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", part):
            words.extend(list(part))
        else:
            words.append(part)

    return words


_section_tag_pattern = re.compile("\\[[^\\]]*\\]\\n?")


def _remove_section_tags(lyrics: str) -> str:
    return _section_tag_pattern.sub("", lyrics)
