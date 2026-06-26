#!/usr/bin/env python3
"""Build MCQ data from ../edumate_v4.questions.json.

Each original questionItem becomes exactly one single-choice MCQ.
The correction map below is based on checkans.txt and propose_review.txt in
this cleaning folder.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = DATA_DIR / "edumate_v4.questions.json"
OUTPUT_FILE = DATA_DIR / "edumate_v4.mcq.questions.json"


# Key: (questionSetIndex, questionItemIndex)
# Value: corrected full-answer text for the converted MCQ.
CORRECTED_ANSWERS: dict[tuple[int, int], str] = {
    (0, 0): "Bảng giá trị đúng là: x = -2, -1, 0, 1, 2 tương ứng y = -7, -5, -3, -1, 1.",
    (0, 1): "Có thể suy ra 5 nghiệm: $(-2; -7)$, $(-1; -5)$, $(0; -3)$, $(1; -1)$, $(2; 1)$.",
    (1, 1): "Với $m = 0$, nghiệm tổng quát là $\\{(x; -2) \\mid x \\in \\mathbb{R}\\}$.",
    (3, 0): "Phương án thứ nhất: 0 ống loại 3m và 13 ống loại 5m.",
    (3, 1): "Phương án thứ hai: 5 ống loại 3m và 10 ống loại 5m.",
    (6, 0): "$(x; y) = (1; -1{,}5)$",
    (7, 0): "$(x; y) = (22; -7)$",
    (8, 0): "Hệ phương trình vô nghiệm.",
    (8, 1): "Hệ phương trình vô nghiệm.",
    (9, 0): "Hệ phương trình vô nghiệm.",
    (11, 0): "$Y = 9000$, $r = 0{,}06$.",
    (12, 0): "Điểm cân bằng là $x = 3\\,000\\,000$ và $p = 120$.",
    (14, 0): "Vận tốc thực của ca nô là 6 km/h.",
    (14, 1): "Vận tốc dòng nước là 2 km/h.",
    (15, 0): "Cần 900 lít dung dịch cồn nồng độ 10%.",
    (15, 1): "Cần 100 lít dung dịch cồn nồng độ 70%.",
    (17, 0): "Phải bán 2500 đĩa CD để hòa vốn.",
    (18, 1): "Giá sách mới là 27 500 đồng, giá sách cũ là 7 500 đồng.",
    (18, 2): "Sách mới giá 27 500 đồng, sách cũ giá 7 500 đồng.",
    (19, 0): "Vận tốc riêng của máy bay là 450 dặm/giờ.",
    (19, 1): "Vận tốc gió là 50 dặm/giờ.",
    (20, 0): "Hai người chạy quãng đường bằng nhau ở tuần thứ 4.",
    (20, 1): "Khi bằng nhau, mỗi người chạy 7 km; tổng quãng đường của hai người là 14 km.",
    (22, 0): "$(x; y) = (2; -3)$",
    (23, 0): "Các điểm $M(1;2)$, $P(-1;-1)$ và $Q(5;8)$ đều nằm trên đường thẳng.",
    (27, 0): "$a = -3$, $b = -3$.",
    (28, 0): "$m = 1$.",
    (29, 0): "$a = 0{,}1$.",
    (29, 1): "$b = -0{,}7$.",
    (30, 0): "Hệ phương trình có vô số nghiệm.",
    (31, 0): "Khi $m=1$, nghiệm là $x = \\frac{7}{4}$, $y = \\frac{1}{4}$.",
    (31, 2): "Khi $m=3$, hệ phương trình có vô số nghiệm.",
    (32, 0): "$a = -3$.",
    (33, 0): "Cần sản xuất và bán 6000 sản phẩm để hòa vốn.",
    (33, 1): "Doanh thu khi hòa vốn là 102 000.",
    (34, 1): "Số vé loại II đã bán là 900.",
    (35, 0): "Cần 1,25 khẩu phần súp cà chua.",
    (35, 1): "Cần 1,5 lát bánh mì nguyên hạt.",
    (36, 0): "Xe đi 15 km trong thành phố và 150 km trên đường cao tốc.",
    (37, 0): "Tập nghiệm là $\\{-2; 1\\}$.",
    (38, 0): "Tập nghiệm là $\\left\\{-\\frac{1}{3}; \\frac{5}{2}\\right\\}$.",
    (39, 0): "Tập nghiệm là $\\left\\{\\frac{1}{2}; 1\\right\\}$.",
    (40, 0): "$x = \\frac{5}{12}$.",
    (41, 0): "$x = -\\frac{3}{2}$.",
    (49, 0): "$x \\ge -\\frac{2}{9}$.",
    (50, 0): "$x < -\\frac{4}{3}$.",
    (54, 0): "$x > -\\frac{1}{5}$.",
    (60, 0): "$-2$.",
    (63, 0): "$4x^2$.",
    (63, 1): "$a^3$.",
    (64, 0): "$5(2x - 1)^2$.",
    (64, 1): "$65 - 20\\sqrt{3}$, xấp xỉ $30{,}359$.",
    (65, 0): "$\\sqrt{5}\\cdot\\sqrt{11} < \\sqrt{56}$.",
    (65, 1): "$\\frac{\\sqrt{141}}{\\sqrt{3}} < 7$.",
    (68, 0): "$P = 2$.",
    (72, 0): "$P = 80$.",
    (74, 0): "Giá trị biểu thức là 2.",
    (75, 0): "$P = 2$.",
    (77, 0): "$P = 0$.",
    (78, 0): "Giá trị biểu thức là $-3$.",
    (79, 0): "Giá trị biểu thức là $\\frac{5}{2}$.",
    (81, 1): "$\\sqrt[3]{0{,}009} > 0{,}2$.",
    (82, 0): "Diện tích tôn cần dùng khoảng $179{,}32\\ \\text{dm}^2$, tức khoảng $1{,}79\\ \\text{m}^2$.",
    (83, 0): "$P \\approx 5{,}769$.",
    (87, 0): "$0{,}031$.",
    (89, 0): "$8\\ \\text{cm}$.",
    (90, 0): "Giá trị biểu thức là 11.",
    (91, 0): "$\\sqrt{\\sqrt{89 + 24\\sqrt{5}}} < \\sqrt{1 + \\sqrt{122}}$.",
    (94, 1): "$(2\\sqrt{3} - 3)^2 = 21 - 12\\sqrt{3}$.",
    (96, 0): "$\\sin(45^\\circ - \\alpha)$ và $\\cos(45^\\circ + \\alpha)$ đều bằng $\\frac{\\sqrt{2}}{2}(\\cos\\alpha - \\sin\\alpha)$.",
    (96, 1): "$\\cos(45^\\circ - \\alpha)$ và $\\sin(45^\\circ + \\alpha)$ đều bằng $\\frac{\\sqrt{2}}{2}(\\cos\\alpha + \\sin\\alpha)$.",
    (97, 3): "$\\sin 40^\\circ \\approx 0{,}643$.",
    (98, 0): "$\\alpha \\approx 14^\\circ$.",
    (98, 3): "$\\alpha \\approx 42^\\circ$.",
    (99, 0): "Giá trị biểu thức là $\\frac{5}{2}$.",
    (102, 0): "$\\alpha \\approx 69{,}6^\\circ$.",
    (103, 0): "$\\alpha \\approx 48{,}6^\\circ$.",
    (106, 0): "$\\tan\\alpha = \\frac{5}{4} = 1{,}25$.",
    (106, 1): "$\\cot\\alpha = \\frac{4}{5} = 0{,}8$.",
    (110, 0): "$\\cos C = \\frac{AC}{BC}$.",
    (110, 1): "$\\cos C = \\frac{HC}{AC}$.",
    (110, 2): "$\\cos C = \\frac{AC}{BC} = \\frac{HC}{AC}$, suy ra $AC^2 = BC\\cdot HC$.",
    (111, 0): "$AH \\approx 3{,}65\\ \\text{cm}$.",
    (112, 0): "$b = AC \\approx 3{,}830$.",
    (113, 0): "Chiều cao tòa nhà khoảng $28{,}5$ m.",
    (114, 0): "Khoảng cách từ tàu đến đường thẳng $AB$ khoảng $1087{,}4$ m.",
    (115, 0): "Khoảng cách từ pháo cao xạ đến máy bay khoảng 3137 m.",
    (116, 0): "Nếu hiểu là khoảng cách đường ngắm từ đài quan sát đến tàu thì xấp xỉ $709{,}9$ m.",
}


def block_text(blocks: list[dict[str, Any]] | None) -> str:
    return "\n".join(str(block.get("text", "")) for block in blocks or [] if block.get("text"))


def content_block(text: str, block_id: str) -> list[dict[str, str]]:
    return [{"id": block_id, "type": "text", "text": text}]


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "latex" in value:
            return f"${value['latex']}$"
        if "numerator" in value and "denominator" in value:
            return f"$\\frac{{{value['numerator']}}}{{{value['denominator']}}}$"
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return ", ".join(value_to_text(item) for item in value)
    if isinstance(value, float):
        if math.isclose(value, round(value)):
            return str(int(round(value)))
        return str(value).replace(".", ",")
    return str(value)


def expected_values_from_spec(spec: dict[str, Any]) -> list[str]:
    values: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "value" in node and isinstance(node["value"], dict) and "correctValue" in node["value"]:
                values.append(value_to_text(node["value"]["correctValue"]))
            else:
                for value in node.values():
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(spec.get("expected"))
    return [value for value in values if value]


def option_text(option: dict[str, Any]) -> str:
    return block_text(option.get("content"))


def original_answer_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    specs = item.get("answerSpecs") or []
    interactions = {interaction.get("id"): interaction for interaction in item.get("interactions") or []}
    for spec_index, spec in enumerate(specs, start=1):
        spec_type = spec.get("type")
        interaction_id = spec.get("interactionId")
        if spec_type == "single_choice":
            correct_id = (spec.get("expected") or {}).get("correctOptionId")
            interaction = interactions.get(interaction_id)
            option = None
            for candidate in (interaction or {}).get("config", {}).get("options", []) or []:
                if candidate.get("id") == correct_id:
                    option = candidate
                    break
            if option:
                parts.append(option_text(option))
            elif correct_id:
                parts.append(str(correct_id))
        else:
            values = expected_values_from_spec(spec)
            if not values:
                continue
            if len(specs) == 1 and len(values) == 1:
                parts.append(values[0])
            elif len(values) == 1:
                parts.append(f"{interaction_id or 'answer_' + str(spec_index)} = {values[0]}")
            else:
                parts.append("; ".join(values))
    return "; ".join(part for part in parts if part) or "Không có đáp án trong dữ liệu gốc."


def is_single_choice_item(item: dict[str, Any]) -> bool:
    interactions = item.get("interactions") or []
    specs = item.get("answerSpecs") or []
    return len(interactions) == 1 and len(specs) == 1 and interactions[0].get("type") == "single_choice"


def map_existing_single_choice(
    mcq_id: str,
    item: dict[str, Any],
    corrected_text: str | None,
) -> tuple[list[dict[str, Any]], str]:
    interaction = item["interactions"][0]
    spec = item["answerSpecs"][0]
    source_options = interaction.get("config", {}).get("options") or []
    letters = ["A", "B", "C", "D", "E", "F"]
    options: list[dict[str, Any]] = []
    original_to_letter: dict[str, str] = {}
    for index, option in enumerate(source_options):
        letter = letters[index]
        original_to_letter[option.get("id")] = letter
        options.append(
            {
                "id": letter,
                "content": deepcopy(option.get("content") or content_block(option_text(option), f"{mcq_id}_option_{letter.lower()}_text")),
            }
        )

    if corrected_text:
        for option in options:
            if normalize(option_text(option)) == normalize(corrected_text):
                return normalize_options(mcq_id, options, option["id"], corrected_text, original_answer_text(item))
        return generated_options(mcq_id, corrected_text, original_answer_text(item)), "A"

    correct_original = (spec.get("expected") or {}).get("correctOptionId")
    correct_letter = original_to_letter.get(correct_original, "A")
    correct_text = ""
    for option in options:
        if option["id"] == correct_letter:
            correct_text = option_text(option)
            break
    return normalize_options(mcq_id, options, correct_letter, correct_text, original_answer_text(item))


def normalize_options(
    mcq_id: str,
    options: list[dict[str, Any]],
    correct_option_id: str,
    correct_text: str,
    original_text: str,
) -> tuple[list[dict[str, Any]], str]:
    """Ensure every MCQ has exactly A-D options and still contains the correct option."""
    letters = ["A", "B", "C", "D"]
    if len(options) == 4 and correct_option_id in letters:
        return options, correct_option_id

    correct_option = next((option for option in options if option.get("id") == correct_option_id), None)
    selected: list[dict[str, Any]] = []
    if correct_option:
        selected.append(correct_option)
    for option in options:
        if len(selected) >= 4:
            break
        if option is not correct_option:
            selected.append(option)

    for distractor in build_distractors(correct_text, original_text):
        if len(selected) >= 4:
            break
        selected.append(
            {
                "id": "",
                "content": content_block(distractor, f"{mcq_id}_option_extra_{len(selected) + 1}_text"),
            }
        )

    normalized: list[dict[str, Any]] = []
    new_correct = "A"
    for index, option in enumerate(selected[:4]):
        new_id = letters[index]
        if option is correct_option:
            new_correct = new_id
        normalized.append({"id": new_id, "content": deepcopy(option.get("content") or [])})

    return normalized, new_correct


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("{,}", ",").split())


def generated_options(mcq_id: str, correct_text: str, original_text: str) -> list[dict[str, Any]]:
    distractors = build_distractors(correct_text, original_text)
    option_texts = [correct_text] + distractors[:3]
    letters = ["A", "B", "C", "D"]
    return [
        {
            "id": letter,
            "content": content_block(text, f"{mcq_id}_option_{letter.lower()}_text"),
        }
        for letter, text in zip(letters, option_texts)
    ]


def build_distractors(correct_text: str, original_text: str) -> list[str]:
    seen = {normalize(correct_text)}
    distractors: list[str] = []

    def add(text: str) -> None:
        key = normalize(text)
        if text and key not in seen:
            distractors.append(text)
            seen.add(key)

    if normalize(original_text) != normalize(correct_text):
        add(original_text)

    numbers = [float(match.replace(",", ".")) for match in __import__("re").findall(r"-?\d+(?:[,.]\d+)?", correct_text)]
    if numbers:
        first = numbers[0]
        add(correct_text.replace(str(int(first)) if first.is_integer() else str(first).replace(".", ","), value_to_text(first + 1), 1))
        add(correct_text.replace(str(int(first)) if first.is_integer() else str(first).replace(".", ","), value_to_text(first - 1), 1))
        add(correct_text.replace(str(int(first)) if first.is_integer() else str(first).replace(".", ","), value_to_text(-first), 1))

    add("Không đủ dữ kiện để xác định.")
    add("Không có đáp án đúng trong các lựa chọn còn lại.")
    add("Kết quả khác với các lựa chọn trên.")
    return distractors


def convert_item(q: dict[str, Any], q_index: int, item: dict[str, Any], item_index: int, mcq_number: int) -> dict[str, Any]:
    mcq_id = f"mcq_{mcq_number:06d}"
    corrected_text = CORRECTED_ANSWERS.get((q_index, item_index))
    original_text = original_answer_text(item)
    answer_text = corrected_text or original_text

    if is_single_choice_item(item) and not corrected_text:
        options, correct_option_id = map_existing_single_choice(mcq_id, item, None)
    else:
        options = generated_options(mcq_id, answer_text, original_text)
        correct_option_id = "A"

    return {
        "id": mcq_id,
        "type": "single_choice",
        "source": {
            "questionSetIndex": q_index,
            "questionItemIndex": item_index,
            "questionSetId": q.get("_id"),
            "questionItemId": item.get("id"),
            "originalInteractionTypes": sorted({interaction.get("type") for interaction in item.get("interactions") or []}),
        },
        "metadata": {
            "chapterName": q.get("chapterName"),
            "lessonName": q.get("lessonName"),
            "difficulty": q.get("difficulty"),
            "bloom": q.get("bloom"),
            "concepts": deepcopy(q.get("concepts") or []),
            "createdBy": q.get("createdBy"),
        },
        "instruction": deepcopy(q.get("instruction") or []),
        "stem": deepcopy(item.get("stem") or []),
        "options": options,
        "answerSpec": {
            "interactionId": f"{mcq_id}_choice",
            "type": "single_choice",
            "expected": {
                "correctOptionId": correct_option_id,
            },
        },
        "hints": deepcopy(item.get("hints") or []),
        "solution": deepcopy((q.get("solutions") or [{"solverName": "default", "solutionContent": []}])[0]),
    }


def main() -> int:
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        source_data = json.load(file)

    questions: list[dict[str, Any]] = []
    for q_index, question_set in enumerate(source_data):
        for item_index, item in enumerate(question_set.get("questionItems") or []):
            questions.append(convert_item(question_set, q_index, item, item_index, len(questions) + 1))
            if len(questions) % 10 == 0:
                print(f"Đã tạo {len(questions)} câu hỏi MCQ")

    output = {
        "schemaVersion": 1,
        "source": {
            "file": INPUT_FILE.name,
            "schemaVersion": 4,
            "questionSetCount": len(source_data),
            "questionItemCount": len(questions),
        },
        "questions": questions,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Hoàn tất: đã tạo {len(questions)} câu hỏi MCQ")
    print(f"Đã ghi file: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
