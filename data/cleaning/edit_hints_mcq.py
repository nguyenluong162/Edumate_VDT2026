#!/usr/bin/env python3
"""Review and edit MCQ hints with gemma-4-12b-it.

This script reviews only:
  - hints[].content[].text

It intentionally keeps instruction, stem, options, answerSpec, solution,
metadata, source, ids, hint names, and content block types unchanged.

Usage:
  python3 data/cleaning/edit_hints_mcq.py
  python3 data/cleaning/edit_hints_mcq.py --limit 5 --dry-run
  python3 data/cleaning/edit_hints_mcq.py --start 20 --limit 10

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
DEFAULT_INPUT = DATA_DIR / "edumate_v4.mcq.questions.reviewed.json"
DEFAULT_OUTPUT = DATA_DIR / "edumate_v4.mcq.questions.hints_reviewed.json"
DEFAULT_LOG = WORK_DIR / "edit_hints_mcq.review_log.jsonl"

API_URL = os.getenv("EDUMATE_API_URL", "https://llm-playground.gpu.test.edumate.ai.vn/v1/chat/completions")
API_TOKEN = os.getenv("EDUMATE_API_TOKEN", "intern_2026")
MODEL = "qwen3.6-35b"


COMMON_PROMPT = """
Bạn là chuyên gia thẩm định gợi ý học tập cho câu hỏi trắc nghiệm Toán.

Nhiệm vụ duy nhất:
- Kiểm tra hints của từng câu MCQ có đúng, liên quan trực tiếp tới đề, và dẫn dắt được học sinh tới đáp án đúng hay không.
- Nếu hint sai toán, mâu thuẫn với đáp án đúng, quá chung chung, không liên quan tới stem/options, hoặc làm học sinh đi sai hướng, hãy sửa text của hint đó.
- Nếu hint đã đúng và hữu ích, giữ nguyên text.

Tuyệt đối không review, không sửa, không nhận xét các phần khác:
- Không sửa instruction.
- Không sửa stem.
- Không sửa options/distractors.
- Không sửa answerSpec/correctOptionId.
- Không sửa solution.
- Không sửa metadata/source/id.
- Không tạo câu hỏi mới.

Ràng buộc bắt buộc:
- Giữ nguyên số lượng hints.
- Giữ nguyên name của từng hint.
- Giữ nguyên id và type của từng content block trong hint.
- Chỉ được sửa trường text bên trong hints[].content[].
- Hints nên đi theo hướng gợi mở từng bước, không biến thành lời giải hoàn chỉnh.
- Hints không nên nói thẳng "đáp án là A/B/C/D" trừ khi đề gốc vốn yêu cầu nhận diện lựa chọn ở bước cuối.
- Hints phải nhất quán với correctOptionId và nội dung option đúng.
- Giữ LaTeX nếu nội dung cần công thức.
- Không thêm markdown, không dùng ```json.
- Chỉ trả về JSON hợp lệ duy nhất theo schema:
{
  "hintsStatus": "ok",
  "hints": [
    {
      "name": "Gợi ý 1",
      "content": [
        {
          "id": "string",
          "type": "text",
          "text": "string"
        }
      ]
    }
  ],
  "reviewNote": "Một câu ngắn nói hints đã đúng, hoặc đã sửa gì."
}

Giá trị hintsStatus chỉ được là:
- "ok" nếu không cần sửa hint nào.
- "edited" nếu đã sửa ít nhất một hint.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use gemma-4-12b-it to review MCQ hints.")
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


def compact_question_for_review(question: dict[str, Any]) -> dict[str, Any]:
    correct_option_id = question.get("answerSpec", {}).get("expected", {}).get("correctOptionId")
    correct_option = next((option for option in question.get("options", []) if option.get("id") == correct_option_id), None)
    return {
        "id": question.get("id"),
        "source": question.get("source"),
        "instruction": question.get("instruction", []),
        "stem": question.get("stem", []),
        "options": question.get("options", []),
        "correctOptionId": correct_option_id,
        "correctOption": correct_option,
        "hints": question.get("hints", []),
        "solution": question.get("solution", {}),
    }


def build_user_prompt(question: dict[str, Any]) -> str:
    payload = compact_question_for_review(question)
    return (
        "Hãy review CHỈ hints của câu MCQ sau.\n"
        "Không sửa bất cứ trường nào khác ngoài text trong hints.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def call_gemma(question: dict[str, Any], timeout: int, temperature: float) -> str:
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


def validate_hints(review: dict[str, Any], original_question: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    hints = review.get("hints")
    hints_status = str(review.get("hintsStatus", "")).strip()
    review_note = str(review.get("reviewNote", "")).strip() or "Không có ghi chú."
    original_hints = original_question.get("hints", [])

    if hints_status not in {"ok", "edited"}:
        raise ValueError(f"hintsStatus must be 'ok' or 'edited', got {hints_status!r}.")

    if not isinstance(hints, list):
        raise ValueError("Model response must contain hints array.")

    if len(hints) != len(original_hints):
        raise ValueError(f"Model must keep {len(original_hints)} hints, got {len(hints)}.")

    normalized_hints: list[dict[str, Any]] = []
    for hint_index, original_hint in enumerate(original_hints):
        hint = hints[hint_index]
        if not isinstance(hint, dict):
            raise ValueError(f"Hint {hint_index} must be an object.")

        original_content = original_hint.get("content", [])
        content = hint.get("content")
        if not isinstance(content, list) or len(content) != len(original_content):
            raise ValueError(
                f"Hint {hint_index} must keep {len(original_content)} content blocks, got "
                f"{len(content) if isinstance(content, list) else 'invalid'}."
            )

        normalized_content: list[dict[str, Any]] = []
        for block_index, original_block in enumerate(original_content):
            block = content[block_index]
            if not isinstance(block, dict):
                raise ValueError(f"Hint {hint_index} content block {block_index} must be an object.")

            text = block.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Hint {hint_index} content block {block_index} has empty text.")

            normalized_block = copy.deepcopy(original_block)
            normalized_block["text"] = text
            normalized_content.append(normalized_block)

        normalized_hint = copy.deepcopy(original_hint)
        normalized_hint["content"] = normalized_content
        normalized_hints.append(normalized_hint)

    return normalized_hints, hints_status, review_note


def apply_hints(question: dict[str, Any], hints: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(question)
    updated["hints"] = hints
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
    started_at = time.perf_counter()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("", encoding="utf-8")

    for index in range(args.start, end):
        question = reviewed_data["questions"][index]
        print(f"[{index + 1}/{len(questions)}] Reviewing hints for {question.get('id')}...", flush=True)
        question_started_at = time.perf_counter()
        try:
            raw = call_gemma(question, timeout=args.timeout, temperature=args.temperature)
            review = extract_json_object(raw)
            hints, hints_status, review_note = validate_hints(review, question)
            old_hints = question.get("hints", [])
            reviewed_data["questions"][index] = apply_hints(question, hints)
            changed = old_hints != hints
            elapsed = time.perf_counter() - question_started_at
            write_log(
                args.log,
                {
                    "index": index,
                    "id": question.get("id"),
                    "status": "ok",
                    "changed": changed,
                    "hintsStatus": hints_status,
                    "elapsedSeconds": round(elapsed, 3),
                    "reviewNote": review_note,
                },
            )
            print(f"  OK in {elapsed:.2f}s | changed={changed} | hintsStatus={hints_status}", flush=True)
            processed += 1
        except Exception as exc:
            elapsed = time.perf_counter() - question_started_at
            write_log(
                args.log,
                {
                    "index": index,
                    "id": question.get("id"),
                    "status": "error",
                    "elapsedSeconds": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"  ERROR in {elapsed:.2f}s: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

        time.sleep(args.sleep)

    if args.dry_run:
        total_elapsed = time.perf_counter() - started_at
        print(f"Dry run complete. Reviewed {processed} questions in {total_elapsed:.2f}s. Log: {args.log}")
        return 0

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(reviewed_data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    total_elapsed = time.perf_counter() - started_at
    print(f"Done. Reviewed {processed} questions in {total_elapsed:.2f}s.")
    print(f"Output: {args.output}")
    print(f"Log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
