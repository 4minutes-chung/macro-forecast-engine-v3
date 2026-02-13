#!/usr/bin/env python3
"""
Sync macro_engine_schema.md table into README_MACRO_ENGINE.md between markers.
"""

import argparse
import os


START_MARK = "<!-- SCHEMA:START -->"
END_MARK = "<!-- SCHEMA:END -->"


def parse_args():
    parser = argparse.ArgumentParser(description="Sync schema table into README.")
    parser.add_argument("--schema", default="macro_engine_schema.md", help="Schema markdown file.")
    parser.add_argument("--readme", default="README_MACRO_ENGINE.md", help="README to update.")
    return parser.parse_args()


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, body: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def extract_table(schema_md: str) -> str:
    lines = schema_md.splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|") and "---" in line:
            in_table = True
        if in_table and line.strip().startswith("|"):
            table_lines.append(line)
        elif in_table and not line.strip().startswith("|"):
            break
    if not table_lines:
        raise RuntimeError("No markdown table found in schema file.")
    return "\n".join(table_lines)


def sync_table(readme: str, table: str) -> str:
    if START_MARK not in readme or END_MARK not in readme:
        raise RuntimeError("README missing schema markers.")
    before, rest = readme.split(START_MARK, 1)
    _, after = rest.split(END_MARK, 1)
    payload = f"{START_MARK}\n\n{table}\n\n{END_MARK}"
    return before + payload + after


def main():
    args = parse_args()
    schema_path = args.schema
    readme_path = args.readme

    schema_md = read_file(schema_path)
    table = extract_table(schema_md)
    readme = read_file(readme_path)
    updated = sync_table(readme, table)
    write_file(readme_path, updated)


if __name__ == "__main__":
    main()

