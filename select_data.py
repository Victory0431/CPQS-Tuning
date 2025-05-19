#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort JSON entries by CPQS_score and optionally extract the top N records (keeping only instruction, input, and output fields)."
    )
    parser.add_argument(
        "-i", "--input_file",
        required=True,
        help="Path to the input JSON file"
    )
    parser.add_argument(
        "-o", "--output_file",
        required=True,
        help="Path where the sorted full JSON data will be saved"
    )
    parser.add_argument(
        "-n", "--top_n",
        type=int,
        default=0,
        help="If greater than 0, extract the top N records into a separate file"
    )
    return parser.parse_args()

def compute_cpqs_score(entry):
    """Return the CPQS score based on the second probability, or 0 if unavailable."""
    probs = entry.get("probabilities", [])
    return probs[1] if len(probs) > 1 else 0

def main():
    args = parse_args()

    # 1. Load the original JSON data
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Compute CPQS_score for each entry and sort descending
    for entry in data:
        entry["CPQS_score"] = compute_cpqs_score(entry)
    sorted_data = sorted(data, key=lambda x: x["CPQS_score"], reverse=True)

    # 3. Write the fully sorted data to the output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(sorted_data)} sorted records to: {args.output_file}")

    # 4. If requested, extract the top N records with only instruction/input/output
    if args.top_n > 0:
        top_n = args.top_n
        top_records = sorted_data[:top_n]
        converted = [
            {
                "instruction": rec.get("instruction", ""),
                "input": rec.get("input", ""),
                "output": rec.get("output", "")
            }
            for rec in top_records
        ]
        out_path = Path(args.output_file)
        top_file = out_path.with_name(f"{out_path.stem}_top_{top_n}{out_path.suffix}")
        with open(top_file, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=4)
        print(f"Extracted top {top_n} records to: {top_file}")

if __name__ == "__main__":
    main()
