# 🚦 交通法規智慧問答系統（交通部公路局 監理服務原型）

TAICA「生成式AI：文字與圖像生成的原理與實務」成大加分題 **Project NCKU-1A / 1B**
F74146115 資訊118 謝承祐

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RogerH0711/traffic-law-rag/blob/main/NCKU1A_traffic_law_rag.ipynb)

用 RAG 把交通違規問答綁在全國法規資料庫的最新條文上，回答會附上可以點開的條文出處，金額也會由程式自動核對。

## 特色

1. **條／項級切塊**，每段都附全國法規資料庫的單條連結
2. **LLM 法律用語改寫 ＋ 混合檢索**（條號直查＋BM25＋EmbeddingGemma 2 向量，RRF 融合）
3. **罰則保證名額**：問「會被怎麼罰」時，保證取回處罰條例的條文
4. **施行日期感知**：由程式把民國施行日期換算成西元，並依今天日期標註「已施行／尚未施行」
5. **自動核對**：引用編號是否存在、回答中的金額是否出現在引用條文

## 執行

1. 在 Colab Secrets 新增名稱為 `Groq` 的 API key，並開啟存取權
2. 執行階段選 T4 GPU
3. 「執行階段 → 全部執行」，最後會出現 Gradio 介面

法規資料會從 [kong0107/mojLawSplitJSON](https://github.com/kong0107/mojLawSplitJSON) 的固定 commit 自動下載（全國法規資料庫 2026-10-30 版）。

## 檔案

| 檔案 | 說明 |
|---|---|
| `NCKU1A_traffic_law_rag.ipynb` | 繳交用的 Colab notebook |
| `NCKU1A_traffic_law_rag.py` | 同一份 notebook 的 jupytext 原始檔（方便看 diff） |
| `tools/build.py` | `.py` → `.ipynb` |
| `tools/test_harness.py` | 本機測試：用假的 embedding／LLM 跑完整流程 |

> ⚖️ 課程原型，回答僅供參考；實際裁罰以處罰機關裁決為準。
