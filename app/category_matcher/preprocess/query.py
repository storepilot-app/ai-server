import re
import unicodedata


BRACKET_PATTERN = re.compile(r"[\[\](){}（）]")
SEPARATOR_PATTERN = re.compile(r"[_/|,]+")
QUANTITY_PATTERN = re.compile(
    r"(?<![0-9A-Za-z가-힣])"
    r"\d+(?:\.\d+)?\s*"
    r"(?:mg|kg|g|ml|l|cm|m|개|매|세트|팩|박스|입|봉|정|캡슐)"
    r"(?![0-9A-Za-z가-힣])"
)


def preprocess_embedding_query(product_name: str) -> str:
    return preprocess_product_name(product_name)


def preprocess_product_name(product_name: str) -> str:
    text = unicodedata.normalize("NFKC", product_name or "")
    text = BRACKET_PATTERN.sub(" ", text)
    text = QUANTITY_PATTERN.sub(" ", text)
    text = SEPARATOR_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
