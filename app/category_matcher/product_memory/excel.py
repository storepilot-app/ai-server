from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO
from xml.etree.ElementTree import iterparse


CELL_REFERENCE = re.compile(r"([A-Z]+)(\d+)")
PRODUCT_NAME_HEADERS = {"상품명"}
MY_CATEGORY_HEADERS = {"마이카테", "마이카테고리", "마이카테고리코드"}


def read_product_rows(source: str | Path | BinaryIO) -> Iterator[tuple[str, str]]:
    with zipfile.ZipFile(source) as archive:
        strings = _read_shared_strings(archive)
        sheets = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        if not sheets:
            raise ValueError("Excel workbook has no worksheets.")

        for sheet_path in sheets:
            yield from _read_sheet(archive, sheet_path, strings)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []

    values: list[str] = []
    parts: list[str] = []
    with archive.open(path) as stream:
        for event, element in iterparse(stream, events=("start", "end")):
            tag = element.tag.rsplit("}", 1)[-1]
            if event == "end" and tag == "t":
                parts.append(element.text or "")
            elif event == "end" and tag == "si":
                values.append("".join(parts))
                parts.clear()
                element.clear()
    return values


def _read_sheet(
    archive: zipfile.ZipFile,
    sheet_path: str,
    strings: list[str],
) -> Iterator[tuple[str, str]]:
    row_number = 0
    current: dict[str, str] = {}
    product_name_column: str | None = None
    my_category_column: str | None = None
    with archive.open(sheet_path) as stream:
        for _, element in iterparse(stream, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "c":
                match = CELL_REFERENCE.fullmatch(element.attrib.get("r", ""))
                if match:
                    column = match.group(1)
                    if row_number == 0 or column in {product_name_column, my_category_column}:
                        current[column] = _cell_value(element, strings).strip()
                element.clear()
            elif tag == "row":
                row_number += 1
                if row_number == 1:
                    product_name_column, my_category_column = _resolve_columns(current)
                else:
                    product_name = current.get(product_name_column, "")
                    my_category = current.get(my_category_column, "")
                    if product_name and my_category:
                        yield product_name, my_category
                current.clear()
                element.clear()


def _resolve_columns(header_values: dict[str, str]) -> tuple[str, str]:
    product_name_column = _find_header_column(header_values, PRODUCT_NAME_HEADERS)
    my_category_column = _find_header_column(header_values, MY_CATEGORY_HEADERS)
    if product_name_column is None:
        raise ValueError("Required product header is missing: 상품명")
    if my_category_column is None:
        raise ValueError("Required product header is missing: 마이카테고리")
    return product_name_column, my_category_column


def _find_header_column(header_values: dict[str, str], headers: set[str]) -> str | None:
    normalized_headers = {_normalize_header(header) for header in headers}
    for column, value in header_values.items():
        if _normalize_header(value) in normalized_headers:
            return column
    return None


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _cell_value(cell, strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = ""
    for child in cell:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "v":
            value = child.text or ""
        elif tag == "is":
            value = "".join(
                node.text or ""
                for node in child.iter()
                if node.tag.rsplit("}", 1)[-1] == "t"
            )
    if cell_type == "s" and value:
        return strings[int(value)]
    return value
