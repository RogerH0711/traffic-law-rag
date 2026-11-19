"""從 NCKU-1A notebook 抽出系統核心，產生 lawrag_core.py，給 NCKU-1B 評估直接重用。

做法：取第 2～11 節的程式碼，用 ast 移除最上層的展示呼叫（print／display／show，
示範查詢都包在這些呼叫裡），保留所有函式、類別與設定。這樣 1B 測的就是 1A 的同一套系統，
不會因為複製貼上而有差異。

用法（在 repo 根目錄）：python3 tools/make_core.py
"""
import ast
import re

import jupytext

SRC = "NCKU1A_traffic_law_rag.py"
SHOW_CALLS = {"print", "display", "show"}
OUT = "lawrag_core.py"

nb = jupytext.read(SRC)
section = 0
chunks = []
for cell in nb.cells:
    if cell.cell_type == "markdown":
        m = re.search(r"^## (\d+)\.", cell.source, flags=re.M)
        if m:
            section = int(m.group(1))
        continue
    if not 2 <= section <= 11:
        continue
    src = "\n".join(l for l in cell.source.splitlines() if not l.strip().startswith(("%", "!")))
    tree = ast.parse(src)
    kept = []
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id in SHOW_CALLS):
            continue                                         # 展示用的呼叫，例如 display(...)、print(...)
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.startswith("_") for t in node.targets):
            continue                                         # 示範用的暫存變數，例如 _t = ...
        kept.append(node)
    tree.body = kept
    chunks.append(f"# ---- 來自 1A 第 {section} 節 ----\n" + ast.unparse(tree))

header = '''"""lawrag_core：NCKU-1A 交通法規 RAG 系統核心（由 tools/make_core.py 自動產生，請勿手動修改）。

來源：NCKU1A_traffic_law_rag.py 第 2～11 節，移除了展示用的 print／display 與示範查詢。
"""
'''
open(OUT, "w", encoding="utf-8").write(header + "\n\n".join(chunks) + "\n")
print(f"wrote {OUT}: {len(chunks)} cells")
