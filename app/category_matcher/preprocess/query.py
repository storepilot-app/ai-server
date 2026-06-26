import re

try:
    from kiwipiepy import Kiwi
except ImportError:
    Kiwi = None

from app.category_matcher.rules.keywords import (
    BODY_KEYWORDS_BY_TYPE,
    CONTEXT_NOUNS_WHEN_BODY_EXISTS,
    LOW_SIGNAL_KOREAN_ADVERBS,
    LOW_SIGNAL_KOREAN_MODIFIERS,
)
from app.category_matcher.common.text import normalize_keyword_text


_kiwi = None


def get_kiwi():
    global _kiwi
    if Kiwi is None:
        return None
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def preprocess_embedding_query(product_name: str) -> str:
    text = preprocess_product_name(product_name)
    body_keywords = detect_body_keywords(text)
    noun_focused = extract_noun_focused_terms(text)
    noun_focused = apply_body_keyword_focus(noun_focused, body_keywords)
    if not noun_focused:
        return text
    return apply_tail_token_weight(noun_focused)


def detect_body_keywords(text: str) -> list[str]:
    normalized = normalize_keyword_text(text)
    detected: list[str] = []
    for body_type, keywords in BODY_KEYWORDS_BY_TYPE.items():
        if any(normalize_keyword_text(keyword) in normalized for keyword in keywords):
            detected.append(body_type)
    return detected


def apply_body_keyword_focus(text: str, body_keywords: list[str]) -> str:
    if not body_keywords:
        return text

    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    focused_tokens = [
        token for token in tokens
        if token not in CONTEXT_NOUNS_WHEN_BODY_EXISTS
    ]
    for body_keyword in body_keywords:
        if body_keyword not in focused_tokens:
            focused_tokens.append(body_keyword)
        focused_tokens.extend([body_keyword] * 3)
    return " ".join(focused_tokens)


def extract_noun_focused_terms(text: str) -> str:
    kiwi = get_kiwi()
    if kiwi is None:
        return emphasize_noun_like_terms(text)

    selected_tokens: list[str] = []
    for raw_token in [token for token in re.split(r"\s+", text.strip()) if token]:
        if raw_token in LOW_SIGNAL_KOREAN_ADVERBS:
            continue
        if is_model_like_token(raw_token):
            selected_tokens.append(raw_token)
            continue

        try:
            analyzed_tokens = kiwi.tokenize(raw_token)
        except Exception:
            return emphasize_noun_like_terms(text)

        noun_forms = [
            token.form.strip()
            for token in analyzed_tokens
            if token.form.strip()
            and token.form.strip() not in LOW_SIGNAL_KOREAN_ADVERBS
            and is_noun_like_pos(token.tag)
        ]
        if not noun_forms:
            continue
        if "".join(noun_forms) == raw_token:
            selected_tokens.append(raw_token)
        else:
            selected_tokens.extend(noun_forms)

    return " ".join(selected_tokens) if selected_tokens else emphasize_noun_like_terms(text)


def is_noun_like_pos(pos: str) -> bool:
    return pos.startswith("N") or pos in {"SL", "SN"}


def is_model_like_token(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token)) or bool(re.fullmatch(r"[A-Za-z0-9_.+-]+", token))


def emphasize_noun_like_terms(text: str) -> str:
    tokens = re.split(r"\s+", text.strip())
    noun_like_tokens: list[str] = []
    for token in tokens:
        cleaned = remove_low_signal_korean_modifiers(token)
        if not cleaned or cleaned in LOW_SIGNAL_KOREAN_ADVERBS:
            continue
        noun_like_tokens.append(cleaned)
    return " ".join(noun_like_tokens)


def apply_tail_token_weight(text: str) -> str:
    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    if len(tokens) <= 1:
        return text.strip()

    weighted_tokens: list[str] = []
    last_index = len(tokens) - 1
    for index, token in enumerate(tokens):
        repeat_count = 1 + round((index / last_index) * 2)
        weighted_tokens.extend([token] * repeat_count)
    return " ".join(weighted_tokens)


def remove_low_signal_korean_modifiers(token: str) -> str:
    cleaned = token
    for modifier in LOW_SIGNAL_KOREAN_MODIFIERS:
        cleaned = cleaned.replace(modifier, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def preprocess_product_name(product_name: str) -> str:
    text = product_name or ""
    text = re.sub(r"\([^)]*\)|（[^）]*）", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[\[\](){}（）]", " ", text)
    text = re.sub(r"[_/|,]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
