#!/usr/bin/env python3
"""Review and edit MCQ options with qwen3.6-35b.

This script reviews only:
  - option contents / distractors
  - answerSpec.expected.correctOptionId

It intentionally keeps instruction, stem, hints, solution, metadata, and source
unchanged.

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
from pathlib import Path
from typing import Any

import requests


WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR.parent
DEFAULT_INPUT = DATA_DIR / "edumate_v4.mcq.questions.json"
DEFAULT_OUTPUT = DATA_DIR / "edumate_v4.mcq.questions.reviewed.json"
DEFAULT_LOG = WORK_DIR / "edit_mcq.review_log.jsonl"

API_URL = os.getenv("EDUMATE_API_URL", "https://llm-playground.gpu.test.edumate.ai.vn/v1/chat/completions")
API_TOKEN = os.getenv("EDUMATE_API_TOKEN", "intern_2026")
MODEL = "qwen3.6-35b"


COMMON_PROMPT = """
Bạn là chuyên gia thẩm định câu hỏi trắc nghiệm Toán.

Nhiệm vụ duy nhất:
- Review các lựa chọn A/B/C/D, tức distractors và đáp án đúng.
- Nếu đáp án đúng đang sai, sửa correctOptionId.
- Nếu distractor sai vai trò, trùng lặp, quá vô lý, hoặc cũng đúng, hãy sửa text của distractor đó.
- Nếu lựa chọn đúng có text sai/thiếu, hãy sửa text của lựa chọn đúng.

Tuyệt đối không review, không sửa, không nhận xét các phần khác:
- Không sửa instruction.
- Không sửa stem.
- Không sửa hints.
- Không sửa solution.
- Không sửa metadata/source/id.
- Không tạo câu hỏi mới.

Ràng buộc bắt buộc:
- Giữ đúng 4 options với id A, B, C, D.
- Chỉ có 1 đáp án đúng duy nhất.
- correctOptionId phải là một trong A, B, C, D.
- Các distractors phải hợp lý, liên quan tới lỗi sai thường gặp, và không được đúng.
- Kiểm tra kỹ các công thức, số liệu, và logic toán học trong text của các distractors.
- Giữ LaTeX nếu nội dung cần công thức.
- Không thêm markdown, không dùng ```json.
- Chỉ trả về JSON hợp lệ duy nhất theo schema:
{
  "options": [
    {
      "id": "A",
      "content": [
        {
          "id": "string",
          "type": "text",
          "text": "string"
        }
      ]
    },
    {
      "id": "B",
      "content": [
        {
          "id": "string",
          "type": "text",
          "text": "string"
        }
      ]
    },
    {
      "id": "C",
      "content": [
        {
          "id": "string",
          "type": "text",
          "text": "string"
        }
      ]
    },
    {
      "id": "D",
      "content": [
        {
          "id": "string",
          "type": "text",
          "text": "string"
        }
      ]
    }
  ],
  "correctOptionId": "A",
  "reviewNote": "Một câu ngắn nói đã sửa gì ở options/answer, hoặc 'Không cần sửa'."
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use qwen3.6-35b to review MCQ options and answers.")
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
        "instruction": question.get("instruction", []),
        "stem": question.get("stem", []),
        "options": question.get("options", []),
        "correctOptionId": question.get("answerSpec", {}).get("expected", {}).get("correctOptionId"),
    }


def build_user_prompt(question: dict[str, Any]) -> str:
    payload = compact_question_for_review(question)
    return (
        "Hãy review CHỈ options/distractors và correctOptionId của câu MCQ sau.\n"
        "Không sửa bất cứ trường nào khác ngoài options và correctOptionId.\n\n"
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
    response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


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


def validate_review(review: dict[str, Any], original_question: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    options = review.get("options")
    correct_option_id = review.get("correctOptionId")
    review_note = str(review.get("reviewNote", "")).strip() or "Không có ghi chú."

    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("Model response must contain exactly 4 options.")

    ids = [option.get("id") for option in options]
    if ids != ["A", "B", "C", "D"]:
        raise ValueError(f"Option ids must be exactly ['A', 'B', 'C', 'D'], got {ids}.")

    if correct_option_id not in ids:
        raise ValueError(f"correctOptionId must be one of A/B/C/D, got {correct_option_id!r}.")

    normalized_options: list[dict[str, Any]] = []
    original_options_by_id = {option.get("id"): option for option in original_question.get("options", [])}
    for option in options:
        option_id = option["id"]
        content = option.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError(f"Option {option_id} must contain non-empty content array.")

        normalized_content: list[dict[str, Any]] = []
        for block_index, block in enumerate(content):
            text = block.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Option {option_id} content block {block_index} has empty text.")
            original_block = (original_options_by_id.get(option_id, {}).get("content") or [{}])[0]
            normalized_content.append(
                {
                    "id": block.get("id") or original_block.get("id") or f"{original_question['id']}_option_{option_id.lower()}_text",
                    "type": block.get("type") or "text",
                    "text": text,
                }
            )

        normalized_options.append({"id": option_id, "content": normalized_content})

    return normalized_options, correct_option_id, review_note


def apply_review(question: dict[str, Any], options: list[dict[str, Any]], correct_option_id: str) -> dict[str, Any]:
    updated = copy.deepcopy(question)
    updated["options"] = options
    updated.setdefault("answerSpec", {}).setdefault("expected", {})["correctOptionId"] = correct_option_id
    return updated


def write_log(log_path: Path, record: dict[str, Any]) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        print(f"[{index + 1}/{len(questions)}] Reviewing {question.get('id')}...", flush=True)
        try:
            raw = call_qwen(question, timeout=args.timeout, temperature=args.temperature)
            review = extract_json_object(raw)
            options, correct_option_id, review_note = validate_review(review, question)
            old_correct = question.get("answerSpec", {}).get("expected", {}).get("correctOptionId")
            old_options = question.get("options", [])
            reviewed_data["questions"][index] = apply_review(question, options, correct_option_id)
            changed = old_correct != correct_option_id or old_options != options
            write_log(
                args.log,
                {
                    "index": index,
                    "id": question.get("id"),
                    "status": "ok",
                    "changed": changed,
                    "oldCorrectOptionId": old_correct,
                    "newCorrectOptionId": correct_option_id,
                    "reviewNote": review_note,
                },
            )
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
        print(f"Dry run complete. Reviewed {processed} questions. Log: {args.log}")
        return 0

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(reviewed_data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Done. Reviewed {processed} questions.")
    print(f"Output: {args.output}")
    print(f"Log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
