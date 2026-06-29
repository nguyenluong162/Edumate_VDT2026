#!/usr/bin/env python3
"""Local web viewer for Edumate MCQ questions.

Run:
  python3 question_viewer.py

Then open:
  http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import argparse
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DATA_FILE = Path(__file__).parent / "data" / "edumate_v4.mcq.questions.hints_reviewed.json"


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Edumate MCQ Viewer</title>
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true
      },
      svg: { fontCache: 'global' }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #64748b;
      --line: #d8dee8;
      --soft: #f6f8fb;
      --panel: #ffffff;
      --accent: #176b87;
      --accent-2: #7a4f01;
      --danger: #9b1c31;
      --ok: #166534;
      --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #eef2f7;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
    }

    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfe;
      padding: 20px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }

    main {
      min-width: 0;
      padding: 24px 28px 56px;
    }

    h1 {
      font-size: 20px;
      line-height: 1.2;
      margin: 0 0 6px;
      letter-spacing: 0;
    }

    .subtle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin: 0;
    }

    .control-group {
      margin-top: 18px;
      display: grid;
      gap: 10px;
    }

    label {
      font-size: 12px;
      font-weight: 700;
      color: #334155;
      text-transform: uppercase;
      letter-spacing: 0;
    }

    input, select, button {
      font: inherit;
      min-height: 40px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
    }

    input, select {
      width: 100%;
      padding: 0 10px;
    }

    button {
      cursor: pointer;
      padding: 0 12px;
      font-weight: 700;
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }

    button.secondary {
      background: #fff;
      color: var(--accent);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: .55;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .status {
      margin-top: 16px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 6px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.45;
    }

    .hero {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
      margin-bottom: 16px;
    }

    .title-line {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }

    .title-line h2 {
      margin: 0;
      font-size: 22px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #e8f3f6;
      color: #0f596f;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }

    .meta > div {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      min-width: 0;
    }

    .meta dt {
      font-size: 11px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 3px;
    }

    .meta dd {
      margin: 0;
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .concepts {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .concept {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border: 1px solid #c8d7df;
      border-radius: 999px;
      background: #f2f8fa;
      color: #164e63;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.2;
    }

    .band {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }

    .band h3 {
      margin: 0 0 12px;
      font-size: 17px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .content-text {
      line-height: 1.7;
      font-size: 15px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .item {
      border-top: 1px solid var(--line);
      padding-top: 16px;
      margin-top: 16px;
    }

    .item:first-child {
      border-top: 0;
      padding-top: 0;
      margin-top: 0;
    }

    .item-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }

    .item-head h4 {
      margin: 0;
      font-size: 15px;
      line-height: 1.35;
      letter-spacing: 0;
    }

    .id {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .grid-two {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      margin-top: 12px;
    }

    .box {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #fcfdff;
      min-width: 0;
    }

    .box h5 {
      margin: 0 0 10px;
      font-size: 13px;
      letter-spacing: 0;
      color: #334155;
    }

    .list {
      display: grid;
      gap: 10px;
    }

    .entry {
      border-left: 3px solid #c8d7df;
      padding-left: 10px;
      min-width: 0;
    }

    .entry-title {
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }

    .option-list {
      margin: 6px 0 0;
      padding-left: 18px;
      line-height: 1.65;
    }

    .answer {
      border-left-color: #d7af56;
    }

    .hint {
      border-left-color: #7aa789;
    }

    .warning {
      color: var(--danger);
      font-weight: 700;
    }

    .empty {
      color: var(--muted);
      font-style: italic;
    }

    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      aside {
        position: relative;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      main { padding: 18px 14px 40px; }
      .meta, .grid-two { grid-template-columns: 1fr; }
      .title-line { display: block; }
      .pill { margin-top: 10px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Edumate MCQ Viewer</h1>
      <p class="subtle">Hiển thị từng câu hỏi MCQ từ file edumate_v4.mcq.questions.hints_reviewed.json. LaTeX được render trực tiếp trên trang.</p>

      <div class="control-group">
        <label for="questionIndex">MCQ index</label>
        <input id="questionIndex" type="number" min="0" value="0">
        <div class="row">
          <button id="loadBtn">Tải câu hỏi</button>
          <button id="copyBtn" class="secondary">Copy text</button>
        </div>
      </div>

      <div class="control-group">
        <label for="quickSelect">Chọn nhanh</label>
        <select id="quickSelect"></select>
      </div>

      <div class="row control-group">
        <button id="prevBtn" class="secondary">Trước</button>
        <button id="nextBtn" class="secondary">Sau</button>
      </div>

      <div id="status" class="status">Đang tải dữ liệu...</div>
    </aside>

    <main id="output" aria-live="polite"></main>
  </div>

  <script>
    const state = {
      total: 0,
      index: 0,
      current: null
    };

    const el = {
      index: document.getElementById('questionIndex'),
      select: document.getElementById('quickSelect'),
      load: document.getElementById('loadBtn'),
      copy: document.getElementById('copyBtn'),
      prev: document.getElementById('prevBtn'),
      next: document.getElementById('nextBtn'),
      status: document.getElementById('status'),
      output: document.getElementById('output')
    };

    function setStatus(message, isError = false) {
      el.status.textContent = message;
      el.status.classList.toggle('warning', isError);
    }

    function asArray(value) {
      return Array.isArray(value) ? value : [];
    }

    function textFromBlocks(blocks) {
      const values = asArray(blocks).map(block => {
        if (block && typeof block.text === 'string') return block.text;
        return block ? `[${block.type || 'unknown'}] ${JSON.stringify(block)}` : '';
      }).filter(Boolean);
      return values.length ? values.join('\n') : '(Không có nội dung)';
    }

    function make(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function addText(parent, text, className = 'content-text') {
      const node = make('div', className);
      node.textContent = text || '(Không có nội dung)';
      parent.appendChild(node);
      return node;
    }

    function valueToText(value) {
      if (value === null || value === undefined) return '(trống)';
      if (typeof value === 'object' && !Array.isArray(value)) {
        if ('latex' in value) return `$${value.latex}$`;
        if ('numerator' in value && 'denominator' in value) return `${value.numerator}/${value.denominator}`;
        return JSON.stringify(value);
      }
      if (Array.isArray(value)) return value.map(valueToText).join(', ');
      return String(value);
    }

    function optionText(option) {
      return textFromBlocks(option.content).replace(/\n+/g, ' ');
    }

    function scoreValue(metadata) {
      if (!metadata) return undefined;
      return metadata.difficultyScore ?? metadata.difficulty_score ?? metadata.score;
    }

    function scoreText(metadata) {
      const score = scoreValue(metadata);
      if (score === null || score === undefined || score === '') return '(trống)';
      const numeric = Number(score);
      return Number.isFinite(numeric) ? numeric.toFixed(2) : String(score);
    }

    function appendMetaValue(wrap, label, value) {
      wrap.appendChild(make('dt', null, label));
      wrap.appendChild(make('dd', null, value ?? '(trống)'));
    }

    function appendConcepts(wrap, concepts) {
      wrap.appendChild(make('dt', null, 'Concepts'));
      const dd = make('dd');
      const list = make('div', 'concepts');
      const values = asArray(concepts).filter(concept => typeof concept === 'string' && concept.trim());
      if (!values.length) {
        dd.textContent = '(trống)';
      } else {
        values.forEach(concept => list.appendChild(make('span', 'concept', concept.trim())));
        dd.appendChild(list);
      }
      wrap.appendChild(dd);
    }

    function renderMeta(q, index) {
      const hero = make('section', 'hero');
      const titleLine = make('div', 'title-line');
      const title = make('h2', null, q.metadata?.lessonName || 'MCQ question');
      const pill = make('span', 'pill', `Index ${index} / 0..${state.total - 1}`);
      titleLine.append(title, pill);
      hero.appendChild(titleLine);

      const meta = make('dl', 'meta');
      [
        ['ID', q.id],
        ['Type', q.type],
        ['Chapter', q.metadata?.chapterName],
        ['Difficulty', q.metadata?.difficulty],
        ['Score', scoreText(q.metadata)],
        ['Bloom', q.metadata?.bloom],
        ['Source set', q.source ? `${q.source.questionSetIndex} / ${q.source.questionSetId}` : '(trống)'],
        ['Source item', q.source ? `${q.source.questionItemIndex} / ${q.source.questionItemId}` : '(trống)'],
        ['Options', asArray(q.options).length]
      ].forEach(([label, value]) => {
        const wrap = document.createElement('div');
        appendMetaValue(wrap, label, value);
        meta.appendChild(wrap);
      });
      const conceptsWrap = document.createElement('div');
      appendConcepts(conceptsWrap, q.metadata?.concepts);
      meta.appendChild(conceptsWrap);
      hero.appendChild(meta);
      return hero;
    }

    function renderOptions(question) {
      const box = make('div', 'box');
      box.appendChild(make('h5', null, `Lựa chọn (${asArray(question.options).length})`));
      const list = make('div', 'list');
      asArray(question.options).forEach(option => {
        const entry = make('div', 'entry');
        entry.appendChild(make('div', 'entry-title', `${option.id}`));
        addText(entry, textFromBlocks(option.content));
        list.appendChild(entry);
      });
      box.appendChild(list);
      return box;
    }

    function renderInteraction(interaction, number) {
      const entry = make('div', 'entry');
      entry.appendChild(make('div', 'entry-title', `[${number}] ${interaction.id || '(không id)'} (${interaction.type || 'unknown'})`));
      const config = interaction.config || {};
      const display = interaction.display || {};

      if (interaction.type === 'single_choice') {
        const list = make('ol', 'option-list');
        asArray(config.options).forEach(option => {
          const li = document.createElement('li');
          li.textContent = `${option.id}: ${optionText(option)}`;
          list.appendChild(li);
        });
        entry.appendChild(list);
      } else if (interaction.type === 'fill_blank') {
        if (config.template) addText(entry, `Mẫu: ${config.template}`, 'content-text');
        const blanks = asArray(config.blanks);
        if (blanks.length) {
          addText(entry, 'Blanks: ' + blanks.map(blank => `${blank.id} (${blank.inputMode || 'unknown'})`).join(', '), 'content-text');
        } else {
          addText(entry, 'Blanks: không khai báo trong config', 'content-text warning');
        }
      } else if (interaction.type === 'short_answer') {
        addText(entry, `Kiểu nhập: ${(config.inputMode || config.input?.inputMode || 'không ghi rõ')}`, 'content-text');
      } else {
        addText(entry, JSON.stringify(config), 'content-text');
      }

      if (Object.keys(display).length) {
        addText(entry, `Hiển thị: layout=${display.layout ?? 'null'}, columns=${display.columns ?? 'null'}, width=${display.width ?? 'null'}`, 'content-text');
      }
      return entry;
    }

    function findInteraction(item, id) {
      return asArray(item.interactions).find(interaction => interaction.id === id);
    }

    function renderAnswer(spec, item, number) {
      const entry = make('div', 'entry answer');
      entry.appendChild(make('div', 'entry-title', `[${number}] ${spec.interactionId || '(không interactionId)'} (${spec.type || 'unknown'})`));
      const expected = spec.expected || {};

      if (spec.type === 'single_choice') {
        const optionId = expected.correctOptionId;
        const option = asArray(item.options).find(opt => opt.id === optionId);
        addText(entry, `Đáp án đúng: ${optionId}${option ? ' - ' + optionText(option) : ''}`, 'content-text');
      } else if (spec.type === 'fill_blank') {
        const blanks = asArray(expected.blanks);
        if (!blanks.length) addText(entry, 'Không có expected.blanks', 'content-text warning');
        blanks.forEach(blank => {
          const correct = blank.value?.correctValue;
          const acceptable = asArray(blank.value?.acceptableValues);
          const line = [
            `${blank.blankId}: ${valueToText(correct)}`,
            acceptable.length ? `chấp nhận thêm: ${acceptable.map(valueToText).join(', ')}` : '',
            `so khớp: ${blank.equivalence?.type || 'không ghi rõ'}`
          ].filter(Boolean).join(' | ');
          addText(entry, line, 'content-text');
        });
      } else if (spec.type === 'short_answer') {
        const answers = Array.isArray(expected) ? expected : [expected];
        answers.forEach(answer => {
          const correct = answer?.value?.correctValue;
          const acceptable = asArray(answer?.value?.acceptableValues);
          const line = [
            `Đáp án đúng: ${valueToText(correct)}`,
            acceptable.length ? `chấp nhận thêm: ${acceptable.map(valueToText).join(', ')}` : '',
            `so khớp: ${answer?.equivalence?.type || 'không ghi rõ'}`
          ].filter(Boolean).join(' | ');
          addText(entry, line, 'content-text');
        });
      } else {
        addText(entry, JSON.stringify(expected), 'content-text');
      }
      return entry;
    }

    function renderHints(hints) {
      const list = make('div', 'list');
      if (!asArray(hints).length) {
        list.appendChild(make('div', 'empty', '(Không có gợi ý)'));
        return list;
      }
      asArray(hints).forEach((hint, index) => {
        const entry = make('div', 'entry hint');
        entry.appendChild(make('div', 'entry-title', `[${index + 1}] ${hint.name || 'Gợi ý'}`));
        addText(entry, textFromBlocks(hint.content));
        list.appendChild(entry);
      });
      return list;
    }

    function renderSolutions(solutions) {
      const band = make('section', 'band');
      band.appendChild(make('h3', null, 'Lời giải tổng quát'));
      const list = make('div', 'list');
      if (!asArray(solutions).length) {
        list.appendChild(make('div', 'empty', '(Không có lời giải)'));
      } else {
        asArray(solutions).forEach((solution, index) => {
          const entry = make('div', 'entry');
          entry.appendChild(make('div', 'entry-title', `[${index + 1}] Solver: ${solution.solverName || 'default'}`));
          addText(entry, textFromBlocks(solution.solutionContent));
          list.appendChild(entry);
        });
      }
      band.appendChild(list);
      return band;
    }

    function renderQuestion(q, index) {
      state.current = q;
      state.index = index;
      el.output.replaceChildren();
      el.output.appendChild(renderMeta(q, index));

      const instruction = make('section', 'band');
      instruction.appendChild(make('h3', null, 'Lời dẫn chung'));
      addText(instruction, textFromBlocks(q.instruction));
      el.output.appendChild(instruction);

      const questionBand = make('section', 'band');
      questionBand.appendChild(make('h3', null, 'Câu hỏi MCQ'));

      const stemBox = make('div', 'box');
      stemBox.appendChild(make('h5', null, 'Đề bài'));
      addText(stemBox, textFromBlocks(q.stem));
      questionBand.appendChild(stemBox);

      const grid = make('div', 'grid-two');
      grid.appendChild(renderOptions(q));

      const answerBox = make('div', 'box');
      answerBox.appendChild(make('h5', null, 'Đáp án / Quy tắc chấm'));
      const answerList = make('div', 'list');
      if (q.answerSpec) {
        answerList.appendChild(renderAnswer(q.answerSpec, q, 1));
      } else {
        answerList.appendChild(make('div', 'empty', '(Không có answerSpec)'));
      }
      answerBox.appendChild(answerList);
      grid.appendChild(answerBox);
      questionBand.appendChild(grid);

      const hintBox = make('div', 'box');
      hintBox.style.marginTop = '12px';
      hintBox.appendChild(make('h5', null, 'Gợi ý'));
      hintBox.appendChild(renderHints(q.hints));
      questionBand.appendChild(hintBox);

      el.output.appendChild(questionBand);
      el.output.appendChild(renderSolutions(q.solution ? [q.solution] : []));
      setStatus(`Đang xem MCQ ${index}. Tổng số: ${state.total}.`);
      el.index.value = index;
      el.select.value = String(index);
      el.prev.disabled = index <= 0;
      el.next.disabled = index >= state.total - 1;

      if (window.MathJax?.typesetPromise) {
        window.MathJax.typesetClear?.([el.output]);
        window.MathJax.typesetPromise([el.output]).catch(err => setStatus(`MathJax lỗi: ${err.message}`, true));
      }
    }

    async function loadIndex(index) {
      if (!Number.isInteger(index) || index < 0 || index >= state.total) {
        setStatus(`Index không hợp lệ. Giới hạn: 0..${state.total - 1}`, true);
        return;
      }
      setStatus(`Đang tải MCQ ${index}...`);
      const response = await fetch(`/api/questions/${index}`);
      if (!response.ok) {
        setStatus(`Không tải được MCQ ${index}`, true);
        return;
      }
      renderQuestion(await response.json(), index);
    }

    async function init() {
      const response = await fetch('/api/questions');
      const meta = await response.json();
      state.total = meta.total;
      el.index.max = Math.max(0, state.total - 1);
      el.select.replaceChildren();
      meta.items.forEach(item => {
        const option = document.createElement('option');
        option.value = item.index;
        option.textContent = `${item.index}. ${item.lessonName || 'MCQ'} | set ${item.questionSetIndex}, item ${item.questionItemIndex}`;
        el.select.appendChild(option);
      });
      await loadIndex(0);
    }

    function readableText() {
      return el.output.innerText || '';
    }

    el.load.addEventListener('click', () => loadIndex(Number(el.index.value)));
    el.index.addEventListener('keydown', event => {
      if (event.key === 'Enter') loadIndex(Number(el.index.value));
    });
    el.select.addEventListener('change', () => loadIndex(Number(el.select.value)));
    el.prev.addEventListener('click', () => loadIndex(state.index - 1));
    el.next.addEventListener('click', () => loadIndex(state.index + 1));
    el.copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(readableText());
        setStatus('Đã copy nội dung đang hiển thị.');
      } catch {
        setStatus('Không copy được từ trình duyệt này.', true);
      }
    });

    init().catch(error => setStatus(error.message, true));
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def load_data(self) -> list[dict]:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload.get("questions", [])

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self.send_html()
            return

        if path == "/api/questions":
            data = self.load_data()
            self.send_json(
                {
                    "total": len(data),
                    "items": [
                        {
                            "index": index,
                            "lessonName": item.get("metadata", {}).get("lessonName"),
                            "chapterName": item.get("metadata", {}).get("chapterName"),
                            "questionSetIndex": item.get("source", {}).get("questionSetIndex"),
                            "questionItemIndex": item.get("source", {}).get("questionItemIndex"),
                        }
                        for index, item in enumerate(data)
                    ],
                }
            )
            return

        if path.startswith("/api/questions/"):
            raw_index = path.removeprefix("/api/questions/")
            try:
                index = int(raw_index)
            except ValueError:
                self.send_json({"error": "Index must be an integer"}, 400)
                return
            data = self.load_data()
            if index < 0 or index >= len(data):
                self.send_json({"error": f"Index out of range: 0..{len(data) - 1}"}, 404)
                return
            self.send_json(data[index])
            return

        self.send_json({"error": "Not found"}, 404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Edumate question viewer.")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind. Default: {DEFAULT_PORT}. Use another port if this one is busy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not DATA_FILE.exists():
        print(f"Missing data file: {DATA_FILE}")
        return 1
    try:
        server = ThreadingHTTPServer((HOST, args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"Port {args.port} is already in use.")
            print(f"Try: python3 question_viewer.py --port {args.port + 1}")
            return 1
        raise

    print(f"Serving Edumate viewer at http://{HOST}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
