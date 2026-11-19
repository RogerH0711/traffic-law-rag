"""把四份題庫 PDF 解析、整理成 data/question_bank.csv。

用法（在 repo 根目錄）：python3 tools/build_bank.py
- 中文版是主資料；英文版只用來交叉核對答案。
- 分類統一成公路局題庫的 10 類（汽、機車題庫的分類名稱寫法略有不同）。
- text_only = 不需要看圖就能作答（排除題幹或選項是圖片的標誌題，以及提到「圖中」「如圖」「圖片」「此圖示」「影片」的題目）。
- key_mismatch = 中英文版答案不一致的題目（公路局題庫本身的出入，評估時排除）。
"""
import io
import subprocess

import pandas as pd

CATS = [
    ("注意大型車", "注意大型車"), ("路口安全", "路口安全"), ("轉彎", "轉彎"),
    ("行駛中應注意事項", "行駛中應注意事項"), ("正確使用燈光", "正確使用燈光"), ("貨物裝載", "貨物裝載"),
    ("事故預防", "事故預防及處理"), ("禁止不當行為", "禁止不當行為"), ("行車檢查", "行車檢查"),
    ("平交道", "其他"), ("其他", "其他"),
]


def parse(pdf, bank):
    out = subprocess.run(["python3", "tools/parse_bank.py", pdf, bank], capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(out))


def canon(c):
    for key, name in CATS:  # 注意大型車要排在「轉彎」前面（它的分類名稱也含「轉彎」）
        if key in str(c):
            return name
    return "其他"


frames = []
for bank, label in [("car", "汽車"), ("moto", "機車")]:
    zh = parse(f"data/raw/{bank}_zh.pdf", bank)
    en = parse(f"data/raw/{bank}_en.pdf", bank)[["qid", "answer"]].rename(columns={"answer": "answer_en"})
    df = zh.merge(en, on="qid", how="left")
    df["bank_label"] = label
    df["category10"] = df["category"].map(canon)
    df["text_only"] = (df["has_image_stem"] == 0).astype(int)
    df["key_mismatch"] = (df["answer"] != df["answer_en"]).astype(int)
    frames.append(df)

bank = pd.concat(frames, ignore_index=True)
bank = bank[["bank", "bank_label", "qid", "answer", "answer_en", "key_mismatch", "category10", "category", "section", "text_only", "question"]]
bank.to_csv("data/question_bank.csv", index=False, encoding="utf-8")
print(bank.groupby("bank_label").agg(題數=("qid", "size"), 文字題=("text_only", "sum"), 中英答案不一致=("key_mismatch", "sum")))
print(bank[bank.text_only == 1].groupby(["bank_label", "category10"]).size().unstack(0))
