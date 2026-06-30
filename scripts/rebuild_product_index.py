from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.main import load_project_env


def main() -> None:
    load_project_env()
    from app.category_matcher.product_memory.store import NaverCategoryLabel, rebuild_product_index

    parser = argparse.ArgumentParser(description="Build the shared FAISS index from historical product Excel files.")
    parser.add_argument(
        "--mapping-json",
        required=True,
        type=Path,
        help="JSON array with myCategoryCode, categoryId, categoryCode, and fullPath fields.",
    )
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    missing = [str(path) for path in [args.mapping_json, *args.files] if not path.is_file()]
    if missing:
        parser.error(f"Files not found: {', '.join(missing)}")

    mapping_items = json.loads(args.mapping_json.read_text(encoding="utf-8"))
    mappings = {
        item["myCategoryCode"]: NaverCategoryLabel(
            category_id=int(item["categoryId"]),
            category_code=item["categoryCode"],
            full_path=item["fullPath"],
        )
        for item in mapping_items
    }
    result = rebuild_product_index(args.files, mappings)
    print(f"source rows: {result.source_row_count:,}")
    print(f"valid rows: {result.valid_row_count:,}")
    print(f"unmapped rows: {result.unmapped_row_count:,}")
    print(f"indexed products: {result.indexed_product_count:,}")
    print(f"duplicate rows: {result.duplicate_row_count:,}")
    print(f"conflicting titles: {result.conflicting_title_count:,}")


if __name__ == "__main__":
    main()
