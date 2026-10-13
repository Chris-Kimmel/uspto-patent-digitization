# Demo, step 2: turn the OCR text of each patent into a structured record with
# an LLM -- grant and application date, every inventor and assignee, their
# addresses split into fields, plus the model's own evidence and quality notes.
#
# Uses the production extraction prompt (pipeline/extraction_prompt.py) and
# model (gpt-oss-20b via Groq). The production run sent ~2M patents through
# Groq's Batch API; this demo calls the same model synchronously.
#
# Requires: GROQ_API_KEY in the environment (or in a .env file in the repo root)
#
# Run:    python demo/2_run_extraction.py      (after demo/1_run_ocr.py)
# Output: demo/output/<patent>/4_extracted_record.json
#         demo/output/extracted_records.csv    (all patents, one row each)

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from json_repair import repair_json

DEMO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DEMO_DIR / "output"
sys.path.insert(0, str(DEMO_DIR.parent / "pipeline"))

from extraction_prompt import GENERATION_PROMPT

load_dotenv(DEMO_DIR.parent / ".env")

MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 4096

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_last_fenced_block(text):
    matches = re.findall(r"```(?:\w+)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    opens = list(re.finditer(r"```(?:\w+)?\s*", text))
    if opens:
        return text[opens[-1].end():].strip()
    return None


def extract_last_top_level_object(text):
    in_str = escape = False
    depth = start_idx = None
    last_complete = None
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth is None or depth == 0:
                    start_idx = i
                depth = (depth or 0) + 1
            elif ch == '}' and depth:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    last_complete = (start_idx, i + 1)
                    start_idx = None
    if last_complete:
        return text[last_complete[0]:last_complete[1]]
    if depth and start_idx is not None:
        return text[start_idx:]
    return None


def parse_response(content):
    """The model answers with JSON; take the last JSON object, repairing it if needed."""
    for candidate in [extract_last_fenced_block(content), extract_last_top_level_object(content)]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        repaired = repair_json(content, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
    except Exception:
        pass
    return {}


def flatten(val):
    return "" if val is None else re.sub(r"[\r\n]+", " ", str(val)).strip()


def main():
    text_files = sorted(OUTPUT_DIR.glob("*/3_ocr_text.txt"))
    if not text_files:
        sys.exit("No OCR text found in demo/output/ -- run demo/1_run_ocr.py first.")
    print(f"{len(text_files)} patents | model {MODEL}\n")

    rows = []
    for txt in text_files:
        pn = txt.parent.name
        print(f"{pn} ...", end=" ", flush=True)
        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": GENERATION_PROMPT},
                      {"role": "user", "content": txt.read_text(encoding="utf-8")}],
            include_reasoning=False,  # reasoning traces off, to keep the output compact
        )
        record = parse_response(response.choices[0].message.content or "")
        (txt.parent / "4_extracted_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"done ({time.time() - t0:.1f}s)" if record else "could not parse the answer")

        rows.append({
            "patent_number": pn,
            "grant_date": record.get("grant_date", ""),
            "application_date": record.get("application_date", ""),
            "inventors_and_locations": flatten(json.dumps(record.get("inventors_and_locations"), ensure_ascii=False)),
            "assignees_and_locations": flatten(json.dumps(record.get("assignees_and_locations"), ensure_ascii=False)),
            "text_quality": flatten(record.get("text_quality", "")),
        })

    with open(OUTPUT_DIR / "extracted_records.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote demo/output/<patent>/4_extracted_record.json and demo/output/extracted_records.csv")


if __name__ == "__main__":
    main()
