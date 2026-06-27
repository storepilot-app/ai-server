from __future__ import annotations

import argparse
from pathlib import Path

from app.main import load_project_env


def main() -> None:
    load_project_env()
    from app.category_matcher.product_memory.store import rebuild_product_index

    parser = argparse.ArgumentParser(description="Build a per-user FAISS index from historical product Excel files.")
    parser.add_argument("--user-key", required=True)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    missing = [str(path) for path in args.files if not path.is_file()]
    if missing:
        parser.error(f"Files not found: {', '.join(missing)}")

    result = rebuild_product_index(args.user_key, args.files)
    print(f"valid rows: {result.valid_row_count:,}")
    print(f"indexed products: {result.indexed_product_count:,}")
    print(f"duplicate rows: {result.duplicate_row_count:,}")
    print(f"conflicting titles: {result.conflicting_title_count:,}")


if __name__ == "__main__":
    main()
