#!/usr/bin/env python3
"""Enrich MCQ question metadata with qwen3.6-35b.

This script enriches only:
  - metadata.concepts

It intentionally keeps instruction, stem, options, answerSpec, hints, solution,
source, ids, and metadata fields other than concepts unchanged.

Usage:
  python3 data/cleaning/edit_mcq.py
  python3 data/cleaning/edit_mcq.py --limit 5 --dry-run
  python3 data/cleaning/edit_mcq.py --start 20 --limit 10

Environment variables:
  EDUMATE_API_TOKEN  default: intern_2026
  EDUMATE_API_URL    default: https://llm-playground.gpu.test.edumate.ai.vn/v1/chat/completions
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import requests
except ModuleNotFoundError:
    requests = None


WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR.parent
DEFAULT_INPUT = DATA_DIR / "edumate_v4.mcq.questions.hints_reviewed.json"
DEFAULT_OUTPUT = DATA_DIR / "edumate_v4.mcq.questions.hints_reviewed.json"
DEFAULT_LOG = WORK_DIR / "edit_mcq.review_log.jsonl"

API_URL = os.getenv("EDUMATE_API_URL", "https://llm-playground.gpu.test.edumate.ai.vn/v1/chat/completions")
API_TOKEN = os.getenv("EDUMATE_API_TOKEN", "intern_2026")
MODEL = "qwen3.6-35b"


COMMON_PROMPT = """
Bạn là chuyên gia phân loại câu hỏi trắc nghiệm Toán lớp 9.

Nhiệm vụ:
- Bổ sung concepts: một list các khái niệm/kỹ năng Toán chính cần dùng để giải bài toán.

Tuyệt đối không review, không sửa, không nhận xét các phần khác ngoài concepts:
- Không sửa instruction.
- Không sửa stem.
- Không sửa options/distractors.
- Không sửa answerSpec/correctOptionId, kể cả khi em nghĩ đáp án hiện tại sai.
- Không sửa hints.
- Không sửa solution.
- Không sửa metadata/source/id ngoài metadata.concepts. Giữ nguyên metadata.difficultyScore nếu đang có.
- Không tạo câu hỏi mới.

Ràng buộc bắt buộc:
- concepts phải là list 2 đến 3 string ngắn gọn bằng tiếng Việt, cụ thể theo bài toán, không quá chung chung.
- concepts nên nêu kỹ năng/khái niệm như "hệ phương trình bậc nhất hai ẩn", "phương pháp thế", "tỉ số lượng giác của góc nhọn".
- Giữ LaTeX nếu nội dung cần công thức.
- Không thêm markdown, không dùng ```json.
- Chỉ trả về JSON hợp lệ duy nhất theo schema:
{
  "concepts": [
    "khái niệm 1",
    "khái niệm 2"
  ],
  "reviewNote": "Một câu ngắn giải thích vì sao chọn concepts."
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use qwen3.6-35b to regenerate MCQ concepts only.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--start", type=int, default=0, help="Start question index, zero-based.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of questions to process.")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls.")
    parser.add_argument("--dry-run", action="store_true", help="Call model and log results, but do not write output JSON.")
    return parser.parse_args()


def block_text(blocks: list[dict[str, Any]] | None) -> str:
    return "\n".join(block.get("text", "") for block in blocks or [] if block.get("text"))


def compact_question_for_review(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": question.get("id"),
        "source": question.get("source"),
        "metadata": question.get("metadata", {}),
        "instruction": question.get("instruction", []),
        "stem": question.get("stem", []),
        "options": question.get("options", []),
        "correctOptionId": question.get("answerSpec", {}).get("expected", {}).get("correctOptionId"),
        "hints": question.get("hints", []),
        "solution": question.get("solution", {}),
    }


def build_user_prompt(question: dict[str, Any]) -> str:
    payload = compact_question_for_review(question)
    return (
        "Hãy chỉ tạo lại concepts cho câu MCQ sau.\n"
        "Không sửa, không trả về options, không sửa correctOptionId, không sửa bất cứ trường nào khác.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def call_qwen(question: dict[str, Any], timeout: int, temperature: float) -> str:
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": COMMON_PROMPT},
            {"role": "user", "content": build_user_prompt(question)},
        ],
    }
    data = post_chat_completion(headers, payload, timeout)
    return data["choices"][0]["message"]["content"]


def post_chat_completion(headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    if requests is not None:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return parse_model_json(cleaned)


def parse_model_json(text: str) -> dict[str, Any]:
    repaired_text = repair_json_latex_backslashes(text)
    try:
        return json.loads(repaired_text)
    except json.JSONDecodeError as original_error:
        match = re.search(r"\{.*\}", repaired_text, flags=re.S)
        candidate = match.group(0) if match else repaired_text
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            raise original_error


def repair_json_latex_backslashes(text: str) -> str:
    """Escape raw LaTeX backslashes inside JSON strings.

    LLMs often return text like "\\sqrt{2}" inside JSON. That is invalid JSON
    because "\\s" is not a legal escape sequence. Commands such as "\\frac"
    are worse: JSON accepts "\\f" as form-feed and silently corrupts the text.
    This scanner only changes backslashes while inside string values and keeps
    ordinary JSON escapes intact.
    """
    repaired: list[str] = []
    in_string = False
    i = 0
    valid_simple_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}

    while i < len(text):
        char = text[i]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            i += 1
            continue

        if char != "\\":
            repaired.append(char)
            i += 1
            continue

        next_char = text[i + 1] if i + 1 < len(text) else ""
        if next_char.isalpha():
            command_end = i + 2
            while command_end < len(text) and text[command_end].isalpha():
                command_end += 1
            if command_end - (i + 1) >= 2:
                repaired.append("\\\\")
                i += 1
                continue

        if next_char in valid_simple_escapes:
            repaired.append(char)
            repaired.append(next_char)
            i += 2
            continue

        if next_char == "u" and i + 5 < len(text):
            hex_part = text[i + 2 : i + 6]
            if re.fullmatch(r"[0-9a-fA-F]{4}", hex_part):
                repaired.append(text[i : i + 6])
                i += 6
                continue

        repaired.append("\\\\")
        i += 1

    return "".join(repaired)


def validate_review(review: dict[str, Any]) -> tuple[list[str], str]:
    concepts = review.get("concepts")
    review_note = str(review.get("reviewNote", "")).strip() or "Không có ghi chú."

    if not isinstance(concepts, list) or not 2 <= len(concepts) <= 3:
        raise ValueError("concepts must be a list containing 2 to 3 items.")

    normalized_concepts: list[str] = []
    for concept in concepts:
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("Every concept must be a non-empty string.")
        normalized_concepts.append(re.sub(r"\s+", " ", concept.strip()))

    return normalized_concepts, review_note


def apply_review(
    question: dict[str, Any],
    concepts: list[str],
) -> dict[str, Any]:
    updated = copy.deepcopy(question)
    metadata = updated.setdefault("metadata", {})
    metadata["concepts"] = concepts
    return updated


def write_log(log_path: Path, record: dict[str, Any]) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_output(output_path: Path, data: dict[str, Any]) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data.get("questions", [])
    if not isinstance(questions, list):
        print("Input file must contain a questions array.", file=sys.stderr)
        return 1

    end = len(questions) if args.limit is None else min(len(questions), args.start + args.limit)
    if args.start < 0 or args.start >= len(questions):
        print(f"Invalid --start. Valid range: 0..{len(questions) - 1}", file=sys.stderr)
        return 1

    reviewed_data = copy.deepcopy(data)
    processed = 0
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("", encoding="utf-8")

    for index in range(args.start, end):
        question = reviewed_data["questions"][index]
        print(f"[{index + 1}/{len(questions)}] Enriching {question.get('id')}...", flush=True)
        try:
            raw = call_qwen(question, timeout=args.timeout, temperature=args.temperature)
            review = extract_json_object(raw)
            concepts, review_note = validate_review(review)
            old_metadata = question.get("metadata", {})
            old_concepts = old_metadata.get("concepts")
            reviewed_data["questions"][index] = apply_review(
                question,
                concepts,
            )
            changed = old_concepts != concepts
            write_log(
                args.log,
                {
                    "index": index,
                    "id": question.get("id"),
                    "status": "ok",
                    "changed": changed,
                    "oldConcepts": old_concepts,
                    "newConcepts": concepts,
                    "reviewNote": review_note,
                },
            )
            if not args.dry_run:
                write_output(args.output, reviewed_data)
            processed += 1
        except Exception as exc:
            write_log(
                args.log,
                {
                    "index": index,
                    "id": question.get("id"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

        time.sleep(args.sleep)

    if args.dry_run:
        print(f"Dry run complete. Processed {processed} questions. Log: {args.log}")
        return 0

    write_output(args.output, reviewed_data)

    print(f"Done. Processed {processed} questions.")
    print(f"Output: {args.output}")
    print(f"Log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
