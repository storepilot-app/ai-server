from app.category_matcher.config.settings import (
    AUTO_ACCEPT_MIN_EXAMPLES,
    AUTO_ACCEPT_THRESHOLD,
    CATEGORY_MARGIN_THRESHOLD,
    CATEGORY_SUPPORT_THRESHOLD,
)
from app.category_matcher.schemas import CategoryDistributionItem


def auto_accepted_category(
    distribution: list[CategoryDistributionItem],
) -> CategoryDistributionItem | None:
    if not distribution:
        return None

    first = distribution[0]
    second_support = distribution[1].support if len(distribution) > 1 else 0.0
    if (
        first.maxSimilarity >= AUTO_ACCEPT_THRESHOLD
        and first.support >= CATEGORY_SUPPORT_THRESHOLD
        and first.support - second_support >= CATEGORY_MARGIN_THRESHOLD
        and first.exampleCount >= AUTO_ACCEPT_MIN_EXAMPLES
    ):
        return first
    return None
