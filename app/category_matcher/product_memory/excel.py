from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO
from xml.etree.ElementTree import iterparse


CELL_REFERENCE = re.compile(r"([A-Z]+)(\d+)")
PRODUCT_NAME_COLUMN = "D"
MY_CATEGORY_COLUMN = "T"


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
    with archive.open(sheet_path) as stream:
        for _, element in iterparse(stream, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "c":
                match = CELL_REFERENCE.fullmatch(element.attrib.get("r", ""))
                if match and match.group(1) in {PRODUCT_NAME_COLUMN, MY_CATEGORY_COLUMN}:
                    current[match.group(1)] = _cell_value(element, strings).strip()
                element.clear()
            elif tag == "row":
                row_number += 1
                if row_number > 1:
                    product_name = current.get(PRODUCT_NAME_COLUMN, "")
                    my_category = current.get(MY_CATEGORY_COLUMN, "")
                    if product_name and my_category:
                        yield product_name, my_category
                current.clear()
                element.clear()


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
