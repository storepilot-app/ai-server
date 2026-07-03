import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.category_matcher.schemas import PredictionCandidate


CATEGORY_ALIAS_FILE = Path(__file__).with_name("category_aliases.csv")


@dataclass(frozen=True)
class CategoryAlias:
    keyword: str
    category_code: str


def load_category_aliases(path: Path = CATEGORY_ALIAS_FILE) -> list[CategoryAlias]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8-sig", newline="") as alias_file:
        rows = csv.DictReader(alias_file)
        aliases = [
            CategoryAlias(
                keyword=unicodedata.normalize("NFKC", row["keyword"].strip()),
                category_code=row["category_code"].strip(),
            )
            for row in rows
            if row.get("keyword", "").strip() and row.get("category_code", "").strip()
        ]

    # A specific phrase should win over a shorter, broader keyword.
    return sorted(aliases, key=lambda alias: len(alias.keyword), reverse=True)


def resolve_category_alias(
    product_name: str,
    categories: list[dict],
    aliases: list[CategoryAlias],
) -> PredictionCandidate | None:
    normalized_name = unicodedata.normalize("NFKC", product_name or "")
    matched_alias = next(
        (alias for alias in aliases if alias.keyword in normalized_name),
        None,
    )
    if matched_alias is None:
        return None

    category = next(
        (
            item
            for item in categories
            if str(item.get("categoryCode", "")) == matched_alias.category_code
        ),
        None,
    )
    if category is None:
        return None
    return PredictionCandidate(
        categoryId=int(category["categoryId"]),
        categoryCode=str(category["categoryCode"]),
        fullPath=str(category["fullPath"]),
        score=1.0,
    )
