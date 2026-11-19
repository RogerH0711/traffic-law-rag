"""把公路局汽、機車筆試題庫 PDF（中文版）解析成 CSV。

用法：python3 parse_bank.py car_zh.pdf car  > car.csv
每列：bank, qid, answer, category, section, question, has_image_stem
"""
import csv
import re
import sys

import pdfplumber

pdf_path, bank = sys.argv[1], sys.argv[2]

items = []          # 依閱讀順序：("head", text) 或 ("row", [no, ans, text])
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.find_tables()
        boxes = [t.bbox for t in tables]
        events = []
        for t in tables:
            for row, cells in zip(t.rows, t.extract()):
                events.append((row.bbox[1], "row", cells))
        for ln in page.extract_text_lines():
            x0, top, x1, bottom = ln["x0"], ln["top"], ln["x1"], ln["bottom"]
            inside = any(b[0] - 2 <= x0 and x1 <= b[2] + 2 and b[1] - 2 <= top and bottom <= b[3] + 2 for b in boxes)
            if not inside:
                events.append((top, "head", ln["text"].strip()))
        events.sort(key=lambda e: e[0])
        items += [(k, v) for _, k, v in events]

HEAD_SKIP = re.compile(r"(題庫|第\s*\d+\s*頁|—\s*\d+\s*—|^分類$|題號|━|【|^答$|^案$|^\d+$)")
questions, section, category = [], "", ""
cur = None
for kind, val in items:
    if kind == "head":
        txt = val.replace("▎", "").strip()
        if not txt or HEAD_SKIP.search(txt):
            continue
        if txt.startswith("架構") or txt in {"正確觀念與態度", "主動停讓文化", "安全駕駛能力", "危險感知能力"}:
            section = re.sub(r"^架構.\s*", "", txt)
        else:
            category = re.sub(r"^分類\s*", "", txt)
        continue
    cells = [(c or "").strip() for c in val]
    if any("題號" in c or c in {"答", "案", "答案", "題目內容", "題 目"} for c in cells):
        continue
    no = cells[0] if cells and re.fullmatch(r"\d{1,4}", cells[0]) else ""
    ans, rest = "", []
    for i, c in enumerate(cells):
        if not c or (i == 0 and no):
            continue
        if not ans and re.fullmatch(r"[（(]?[123１２３][)）]?", c):
            ans = c
        else:
            rest.append(c)
    text = "\n".join(rest)
    m_no = re.fullmatch(r"\d{1,4}", no)
    if m_no:
        cur = {"bank": bank, "qid": int(no), "answer": "", "category": category,
               "section": section, "question": ""}
        questions.append(cur)
    if cur is None:
        continue
    a = re.search(r"[123１２３]", ans)
    if a and not cur["answer"]:
        cur["answer"] = "123123"["123１２３".index(a.group())]
    if text:
        cur["question"] = (cur["question"] + "\n" + text).strip()

out = csv.DictWriter(sys.stdout, fieldnames=["bank", "qid", "answer", "category", "section", "question", "has_image_stem"])
out.writeheader()
for q in questions:
    s = re.sub(r"\s*\n\s*", "", q["question"])
    s = s.replace("（１）", "(1)").replace("（２）", "(2)").replace("（３）", "(3)")
    s = re.sub(r"[（(]\s*([123１２３])\s*[)）]", lambda m: f"({'123123'['123１２３'.index(m.group(1))]})", s)
    head = s[:12]
    pos = s.find(head, 12) if len(head) == 12 else -1
    if pos > 0 and s.startswith(s[pos:pos + 40]):          # 跨頁表格偶爾讓同一題文字重複一次
        s = s[:pos]
    q["question"] = s
    stem = s.split("(1)")[0].strip()
    opts = re.split(r"\([123]\)", s)[1:]
    q["has_image_stem"] = int(
        len(stem) < 4                                        # 題幹是圖片（標誌題）
        or any(len(o.strip(" 。.、；;")) == 0 for o in opts)  # 選項是圖片
        or bool(re.search(r"如圖|下圖|左圖|右圖|上圖|圖中|此圖|圖示|圖片|圖\s*\d|上列標誌|影片", s))
    )
    out.writerow(q)
