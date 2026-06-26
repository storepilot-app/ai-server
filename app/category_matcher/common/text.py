import re


def last_category_name(full_path: str) -> str:
    parts = [part.strip() for part in (full_path or "").split(">") if part.strip()]
    return parts[-1] if parts else ""


def normalize_keyword_text(value: str) -> str:
    return re.sub(r"[\s_/(),\[\]{}|,-]+", "", value or "").upper()
