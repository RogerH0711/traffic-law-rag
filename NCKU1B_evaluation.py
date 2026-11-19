# %% [markdown]
# # 📊 NCKU-1B：交通法規智慧問答系統的效能分析
#
# **F74146115 資訊118 謝承祐**｜TAICA「生成式AI：文字與圖像生成的原理與實務」成大加分題 **Project NCKU-1B**
#
# 接續 [Project NCKU-1A](https://github.com/RogerH0711/traffic-law-rag)（給交通部公路局監理服務用的交通法規 RAG 問答系統），
# 這份 Notebook 用**數據**回答一個問題：
#
# > **把法規條文檢索進來（RAG），真的讓 LLM 答得比較好嗎？哪些設計有用？代價是多少？**
#
# ## 測試資料：公路局汽、機車駕照筆試題庫
#
# 使用交通部公路局公告的 115 年新版題庫（汽車 1,090 題、機車 806 題，皆為 3 選 1），有公定答案、有分類，
# 是現成的「交通法規知識」標準測驗。前處理（`tools/build_bank.py`）：
# - 排除需要看圖的題目：題幹或選項是圖片的標誌題，以及提到「圖中」「如圖」「圖片」「影片」的題目（文字 RAG 無法處理）；
# - 中英文版交叉核對答案，排除兩版答案不一致的 5 題；
# - 剩下 **1,599 題文字題**。
#
# ## 兩組實驗
#
# | 實驗 | 方法 | 規模 | 要不要呼叫 LLM |
# |---|---|---|---|
# | **一、答題正確率** | 同一個 LLM、三種設定作答，比較答對率，並做統計檢定 | 分層抽樣 200 題 × 3 設定 | 要（受 Groq 免費額度限制） |
# | **二、檢索品質與語意相似度** | 參考 NCKU 學長範例的 Q→C／A→C 相似度指標，加上「條文支持正確選項的比例」 | 全部 1,599 題 | 不用 |
#
# 實驗一的三種設定：
#
# | 代號 | 設定 | 說明 |
# |---|---|---|
# | **A** | 純 LLM | 不給任何條文，直接作答（對照組） |
# | **B** | 向量 RAG | 一般教科書式 RAG：只用 EmbeddingGemma 向量檢索取前 K 段條文 |
# | **C** | 完整系統 | 1A 的完整流程：LLM 法律用語改寫＋條號直查＋BM25＋向量（RRF）＋罰則保證名額＋數字與施行日期預先換算 |
#
# **執行方式**：Colab Secrets 要有 `Groq`，選 T4 GPU，「全部執行」。
# 實驗一的結果會存到 Google Drive，**免費額度用完會自動停下，隔天再按一次「全部執行」就會從中斷處繼續**；
# 分析的格子只用已經跑完的題目，所以任何時候都可以看目前的結果。

# %% [markdown]
# ## 0. 安裝套件、下載 1A 系統與題庫

# %%
# %pip install -q "sentence-transformers==6.1.0" "transformers==5.20.1" "faiss-cpu==1.15.1" "aisuite[groq]==0.2.0" "groq==0.37.1"
# !apt-get -qq install -y fonts-noto-cjk > /dev/null
# !git clone -q --depth 1 https://github.com/RogerH0711/traffic-law-rag.git /content/traffic-law-rag 2>/dev/null || git -C /content/traffic-law-rag pull -q

# %%
import json
import math
import os
import random
import re
import time
from collections import Counter, deque
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy import stats

REPO = "/content/traffic-law-rag" if os.path.isdir("/content/traffic-law-rag") else "."
os.chdir(REPO)

# 中文字型（Colab 用 apt 安裝的 Noto Sans CJK）
for path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(path):
        matplotlib.font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = matplotlib.font_manager.FontProperties(fname=path).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ### 載入 1A 系統（同一套程式碼）
#
# `lawrag_core.py` 是用 `tools/make_core.py` 從 1A Notebook 第 2～11 節**自動抽出**的（只移除展示用的 print／display），
# 所以這裡評估的就是 1A 的同一套系統：同樣的法規資料版本、切塊、混合檢索、改寫提示詞與條文換算。
# 這一步會下載法規、建立向量索引（T4 約 1 分鐘）並連線 Groq。

# %%
t0 = time.time()
exec(open("lawrag_core.py", encoding="utf-8").read())
print(f"1A 系統載入完成：{len(documents)} 段條文，{time.time() - t0:.0f} 秒")

# %% [markdown]
# ## 1. 題庫與抽樣
#
# `data/question_bank.csv` 是四份題庫 PDF 解析後的結果（解析程式 `tools/parse_bank.py`、`tools/build_bank.py`，原始 PDF 在 `data/raw/`）。

# %%
bank = pd.read_csv("data/question_bank.csv")
eligible = bank[(bank.text_only == 1) & (bank.key_mismatch == 0)].copy()
eligible["key"] = eligible["bank"] + "-" + eligible["qid"].astype(str)

overview = bank.groupby("bank_label").agg(
    原始題數=("qid", "size"), 文字題=("text_only", "sum"), 中英答案不一致=("key_mismatch", "sum"))
overview["納入分析"] = eligible.groupby("bank_label").size()
display(overview)
display(eligible.groupby(["category10", "bank_label"]).size().unstack().fillna(0).astype(int))
print("答案分布（檢查有沒有偏向某個選項）：", eligible["answer"].value_counts(normalize=True).round(3).sort_index().to_dict())

# %% [markdown]
# ### 分層抽樣（實驗一用）
#
# Groq 免費方案的 `openai/gpt-oss-120b` 每天上限 1,000 次請求、20 萬 token，每分鐘 8,000 token；
# 一題跑三種設定約 4 次呼叫、數千 token，所以無法每題都測。做法：
# - 汽車、機車各抽 `N_PER_BANK` 題，依 10 個分類**按比例分層抽樣**（每類至少 2 題），亂數種子固定為 115；
# - 抽出的題目**隨機排序後逐題執行**，所以任何時候中斷，已完成的部分仍是隨機子樣本，不會偏向某一類。

# %%
N_PER_BANK = 100
SEED = 115


def stratified_sample(df, n, seed):
    counts = df.groupby("category10").size()
    quota = (counts / counts.sum() * n)
    alloc = quota.apply(math.floor).clip(lower=2)
    for cat in (quota - quota.apply(math.floor)).sort_values(ascending=False).index:
        if alloc.sum() >= n:
            break
        alloc[cat] += 1
    alloc = alloc.clip(upper=counts)
    parts = [g.sample(int(alloc[c]), random_state=seed) for c, g in df.groupby("category10")]
    return pd.concat(parts)


sample = pd.concat([stratified_sample(eligible[eligible.bank == b], N_PER_BANK, SEED) for b in ["car", "moto"]])
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)      # 執行順序也隨機
display(sample.groupby(["category10", "bank_label"]).size().unstack().fillna(0).astype(int))
print("抽樣題數：", len(sample))

# %% [markdown]
# ## 2. 實驗一：三種設定作答
#
# ### 作答提示詞與記錄方式
#
# - 三種設定用**同一個模型**（`openai/gpt-oss-120b`）、溫度 0、推理強度 low（省 token），唯一差別是有沒有條文、條文怎麼來。
# - RAG 的提示詞明確說：條文有寫到就以條文為準，沒寫到就依交通安全常識判斷。題庫有不少「觀念題」法規沒有直接規定，這樣才公平。
# - 每次呼叫都記錄：預測、token 用量、延遲、取回的條文編號，以及完整系統的改寫查詢與範圍判斷。
# - 題目送給檢索時用「題幹＋三個選項」全文。

# %%
TOP_K = 4                    # 取回的條文段數（比 1A 的 6 少，節省 token；三種設定一致）
REASONING = "low"
TPM_LIMIT = int(os.environ.get("NCKU1B_TPM", 8000))   # Groq 免費方案每分鐘 token 上限

MCQ_RULE = "這是一題 3 選 1 的選擇題。只輸出一個阿拉伯數字（1、2 或 3）代表答案，不要輸出任何其他文字。"
PURE_SYSTEM = "你是熟悉台灣交通法規與安全駕駛的應考者，正在作答交通部公路局的汽機車駕照筆試題。" + MCQ_RULE
RAG_SYSTEM = (
    "你是熟悉台灣交通法規與安全駕駛的應考者，正在作答交通部公路局的汽機車駕照筆試題。"
    "使用者訊息中的〈參考條文〉是檢索系統找到的法規，可能相關也可能無關："
    "條文有明確規定的，以條文為準；條文沒寫到的，依交通安全常識判斷。"
    "條文中數字後面括號內的阿拉伯數字是系統換算好的，可直接採用。" + MCQ_RULE
)


class QuotaExhausted(Exception):
    pass


_recent = deque()            # (時間, token 數)，用來控制每分鐘 token


def _wait_for_tpm(expected):
    while True:
        now = time.time()
        while _recent and now - _recent[0][0] > 60:
            _recent.popleft()
        if sum(t for _, t in _recent) + expected <= TPM_LIMIT * 0.9:
            return
        time.sleep(2)


def _retry_seconds(msg):
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
    return (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else 20


def tracked_call(messages, max_tokens=1024, expected_tokens=2000):
    """呼叫 LLM 並回傳 (文字, 用量 dict)。遇到每分鐘限制會等待重試；每日額度用完則丟出 QuotaExhausted。"""
    for attempt in range(6):
        _wait_for_tpm(expected_tokens)
        t0 = time.time()
        try:
            extra = {"reasoning_effort": REASONING} if REASONING else {}
            resp = client.chat.completions.create(
                model=LLM_MODEL, messages=messages, temperature=0, max_tokens=max_tokens, **extra,
            )
        except Exception as exc:
            msg = str(exc)
            if "reasoning_effort" in msg and REASONING:      # 舊版 SDK 不支援時退回預設推理強度
                globals()["REASONING"] = None
                print("⚠️ 這個 SDK 版本不支援 reasoning_effort，改用預設推理強度。")
                continue
            if re.search(r"per day|\(TPD\)|\(RPD\)", msg):
                raise QuotaExhausted(msg[:200]) from None
            if "429" in msg or "rate" in msg.lower() or "timeout" in msg.lower():
                time.sleep(_retry_seconds(msg))
                continue
            if attempt < 2:
                time.sleep(5)
                continue
            raise RuntimeError("Groq 呼叫失敗：" + msg[:200]) from None
        latency = time.time() - t0
        usage = resp.usage
        total = getattr(usage, "total_tokens", None) or expected_tokens
        _recent.append((time.time(), total))
        text = resp.choices[0].message.content or ""
        return text.strip(), {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": total, "latency": round(latency, 2),
        }
    raise RuntimeError("重試多次仍失敗")


def parse_choice(text):
    m = re.search(r"答案\s*[:：]?\s*[（(]?([123])", text) or re.search(r"(?<![\d.])([123])(?![\d.])", text)
    return int(m.group(1)) if m else None


def tracked_rewrite(question):
    """與 1A 的 rewrite_query() 相同提示詞與解析方式，但改用 tracked_call 以記錄 token。"""
    raw, usage = tracked_call(
        [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": question}],
        max_tokens=1024, expected_tokens=1200,
    )
    m = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        obj = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        obj = {}
    rewrite = {
        "in_scope": obj.get("in_scope", True) is not False,
        "asks_penalty": obj.get("asks_penalty", True) is not False,
        "queries": [q.strip() for q in obj.get("queries", []) if isinstance(q, str) and q.strip()][:3],
    }
    return rewrite, usage


def answer_with_context(question, results):
    context, _ = build_context(results)
    msgs = [{"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"〈參考條文〉\n{context}\n\n〈題目〉\n{question}"}]
    return tracked_call(msgs, max_tokens=512, expected_tokens=1800)


def run_config(row, config):
    q = row["question"]
    rec = {"key": row["key"], "bank": row["bank"], "qid": int(row["qid"]), "config": config,
           "answer": int(row["answer"]), "category10": row["category10"], "time": datetime.now(TZ).isoformat()}
    if config == "A":
        text, usage = tracked_call([{"role": "system", "content": PURE_SYSTEM}, {"role": "user", "content": q}],
                                   max_tokens=512, expected_tokens=600)
        results = []
    elif config == "B":
        results = hybrid_search([q], top_k=TOP_K, use_bm25=False, use_direct=False)
        text, usage = answer_with_context(q, results)
    else:
        rewrite, u1 = tracked_rewrite(q)
        results, _ = retrieve(q, top_k=TOP_K, rewrite=rewrite)
        text, u2 = answer_with_context(q, results)
        usage = {k: (u1.get(k) or 0) + (u2.get(k) or 0) for k in ["prompt_tokens", "completion_tokens", "total_tokens", "latency"]}
        rec.update({"rewrite_queries": rewrite["queries"], "in_scope": rewrite["in_scope"], "asks_penalty": rewrite["asks_penalty"]})
    pred = parse_choice(text)
    rec.update({"pred": pred, "correct": int(pred == int(row["answer"])), "raw": text[:200], **usage,
                "retrieved": [r["doc_id"] for r in results],
                "retrieved_titles": [r["title"] for r in results],
                "context_chars": sum(len(r["text"]) for r in results)})
    return rec

# %% [markdown]
# ### 結果存放位置
#
# 結果以 JSON Lines 存到 Google Drive（`MyDrive/ncku1b/results.jsonl`），Colab 斷線或隔天重開都不會遺失。
# 若 repo 裡已有 `results/results.jsonl`（之前跑完後上傳的），會先合併進來。

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    RESULT_DIR = "/content/drive/MyDrive/ncku1b"
except Exception:
    RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(RESULT_DIR, "results.jsonl")


def load_results():
    recs = {}
    for path in ["results/results.jsonl", RESULT_PATH]:
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    recs[(r["key"], r["config"])] = r
    return recs


done = load_results()
print("結果檔：", RESULT_PATH, "／已完成", len(done), "筆（題目×設定）")

# %% [markdown]
# ### 執行（可中斷、可續跑）
#
# - 逐題執行 A → B → C，三種設定都完成才換下一題（確保每題都是成對比較）。
# - 免費額度用完會印出提示並停止，**不會出錯中斷後面的分析格子**。
# - `MAX_NEW_QUESTIONS` 可以限制這次最多跑幾題（例如錄影時設 3 題示範）。

# %%
MAX_NEW_QUESTIONS = None      # None = 跑到額度用完或全部完成
CONFIGS = ["A", "B", "C"]
CONFIG_NAMES = {"A": "A 純 LLM", "B": "B 向量 RAG", "C": "C 完整系統"}

started, new_q, stop_reason = time.time(), 0, "全部完成"
with open(RESULT_PATH, "a", encoding="utf-8") as fout:
    try:
        for _, row in sample.iterrows():
            todo = [c for c in CONFIGS if (row["key"], c) not in done]
            if not todo:
                continue
            if MAX_NEW_QUESTIONS is not None and new_q >= MAX_NEW_QUESTIONS:
                stop_reason = f"已達 MAX_NEW_QUESTIONS={MAX_NEW_QUESTIONS}"
                break
            for c in todo:
                rec = run_config(row, c)
                done[(row["key"], c)] = rec
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            new_q += 1
            if new_q % 5 == 0:
                n_full = sum(all((k, c) in done for c in CONFIGS) for k in sample["key"])
                print(f"  進度 {n_full}/{len(sample)} 題，本次新增 {new_q} 題，經過 {time.time() - started:.0f} 秒")
    except QuotaExhausted as exc:
        stop_reason = "Groq 今日額度已用完，明天再執行一次即可續跑（" + str(exc)[:80] + "…）"

n_full = sum(all((k, c) in done for c in CONFIGS) for k in sample["key"])
print(f"本次新增 {new_q} 題；目前三種設定都完成的題目：{n_full}/{len(sample)}。停止原因：{stop_reason}")

# %% [markdown]
# ## 3. 實驗一結果
#
# 以下分析只使用**三種設定都完成**的題目（成對資料）。

# %%
res = pd.DataFrame(done.values())
res = res[res["key"].isin(sample["key"])]
complete_keys = res.groupby("key")["config"].nunique().loc[lambda s: s == len(CONFIGS)].index
res = res[res["key"].isin(complete_keys)].copy()
res["bank_label"] = res["bank"].map({"car": "汽車", "moto": "機車"})
wide = res.pivot(index="key", columns="config", values="correct")
N = len(wide)
print(f"分析題數 N = {N}（汽車 {res[res.bank == 'car'].key.nunique()}、機車 {res[res.bank == 'moto'].key.nunique()}）")
print("無法解析答案的回覆數：", int(res["pred"].isna().sum()))

# %% [markdown]
# ### 3.1 答對率與 95% 信賴區間
#
# 信賴區間用 **Wilson score interval**（樣本數不大、比例接近 0 或 1 時比常態近似穩定；Wilson, 1927）。

# %%
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = []
for scope, part in [("全部", res), ("汽車", res[res.bank == "car"]), ("機車", res[res.bank == "moto"])]:
    for c in CONFIGS:
        s = part[part.config == c]["correct"]
        lo, hi = wilson(s.sum(), len(s))
        rows.append({"範圍": scope, "設定": CONFIG_NAMES[c], "題數": len(s), "答對": int(s.sum()),
                     "答對率": s.mean(), "95% CI 下界": lo, "95% CI 上界": hi})
acc = pd.DataFrame(rows)
display(acc.style.format({"答對率": "{:.1%}", "95% CI 下界": "{:.1%}", "95% CI 上界": "{:.1%}"}).hide(axis="index"))

fig, ax = plt.subplots(figsize=(7, 3.6))
colors = ["#9aa0a6", "#7baaf7", "#1a73e8"]
for i, scope in enumerate(["全部", "汽車", "機車"]):
    part = acc[acc["範圍"] == scope]
    x = np.arange(len(CONFIGS)) + i * 4
    ax.bar(x, part["答對率"], color=colors, width=0.8)
    ax.errorbar(x, part["答對率"], yerr=[part["答對率"] - part["95% CI 下界"], part["95% CI 上界"] - part["答對率"]],
                fmt="none", ecolor="black", capsize=3, lw=1)
    for xi, v in zip(x, part["答對率"]):
        ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
ax.set_xticks([1, 5, 9], ["全部", "汽車", "機車"])
ax.set_ylim(0, 1.08)
ax.set_ylabel("答對率")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], list(CONFIG_NAMES.values()), loc="lower right", fontsize=8)
ax.set_title("三種設定的駕照筆試答對率（誤差線為 Wilson 95% CI）")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 差異是否顯著：Cochran's Q 與 McNemar 檢定
#
# 同一題被三種設定作答，資料是**成對的二元結果**（答對／答錯），所以不能用獨立樣本的卡方檢定：
# - 三組整體比較：**Cochran's Q 檢定**（Cochran, 1950）；
# - 兩兩比較：**McNemar 精確檢定**（McNemar, 1947）——只看「一個對、一個錯」的不一致題目；
# - 做了三次兩兩比較，用 **Holm 法**校正多重比較（Holm, 1979）。
#
# 這也是 LLM 評估文獻中比較兩個系統在同一組題目上表現的標準做法（Dietterich, 1998）。

# %%
def cochrans_q(x):
    x = np.asarray(x)
    k = x.shape[1]
    col, row, total = x.sum(0), x.sum(1), x.sum()
    denom = k * total - (row ** 2).sum()
    q = (k - 1) * (k * (col ** 2).sum() - total ** 2) / denom if denom else 0.0
    return q, stats.chi2.sf(q, k - 1)


def mcnemar_exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n01 = int(((a == 0) & (b == 1)).sum())     # a 錯 b 對
    n10 = int(((a == 1) & (b == 0)).sum())     # a 對 b 錯
    n = n01 + n10
    p = stats.binomtest(min(n01, n10), n, 0.5).pvalue if n else 1.0
    return n01, n10, p


q, p_q = cochrans_q(wide[CONFIGS].values)
print(f"Cochran's Q = {q:.2f}，自由度 2，p = {p_q:.4f}")

pairs = [("A", "B"), ("A", "C"), ("B", "C")]
tests = []
for x, y in pairs:
    n01, n10, p = mcnemar_exact(wide[x], wide[y])
    tests.append({"比較": f"{CONFIG_NAMES[x]} → {CONFIG_NAMES[y]}", f"只有後者答對": n01, f"只有前者答對": n10,
                  "答對率差": wide[y].mean() - wide[x].mean(), "McNemar p": p})
tests = pd.DataFrame(tests)
order = tests["McNemar p"].argsort().values
adj, running = np.empty(len(tests)), 0.0
for rank, i in enumerate(order):                       # Holm 逐步校正
    running = max(running, min(1.0, (len(tests) - rank) * tests.loc[i, "McNemar p"]))
    adj[i] = running
tests["Holm 校正 p"] = adj
tests["α=0.05 顯著"] = np.where(tests["Holm 校正 p"] < 0.05, "是", "否")
display(tests.style.format({"答對率差": "{:+.1%}", "McNemar p": "{:.4f}", "Holm 校正 p": "{:.4f}"}).hide(axis="index"))

# %% [markdown]
# ### 3.3 差異有多大：成對 bootstrap 信賴區間
#
# p 值只說「是不是偶然」，不說「差多少」。這裡對題目做 10,000 次重抽樣（Efron & Tibshirani, 1993），
# 估計答對率差的 95% 信賴區間；區間不含 0 代表有穩定的差異。

# %%
rng = np.random.default_rng(SEED)
idx = rng.integers(0, N, size=(10000, N)) if N else np.zeros((0, 0), dtype=int)
boot = []
for x, y in pairs:
    d = (wide[y].values - wide[x].values)
    samples = d[idx].mean(1) if N else np.array([np.nan])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    boot.append({"比較": f"{CONFIG_NAMES[y]} − {CONFIG_NAMES[x]}", "答對率差": d.mean() if N else np.nan,
                 "95% CI 下界": lo, "95% CI 上界": hi})
display(pd.DataFrame(boot).style.format({"答對率差": "{:+.1%}", "95% CI 下界": "{:+.1%}", "95% CI 上界": "{:+.1%}"}).hide(axis="index"))

# %% [markdown]
# ### 3.4 哪一類題目受益最多？
#
# 題庫的 10 個分類中，有些是法規明文規定（例如「禁止不當行為」裡的酒駕罰則），有些是安全觀念（例如「其他」裡的駕駛道德）。
# RAG 的效益應該集中在前者。各類樣本數不多，下表只作為描述，不做顯著性檢定。

# %%
cat = res.pivot_table(index="category10", columns="config", values="correct", aggfunc="mean")
cat.columns = [CONFIG_NAMES[c] for c in cat.columns]
cat["題數"] = res[res.config == "A"].groupby("category10").size()
cat["C − A"] = cat[CONFIG_NAMES["C"]] - cat[CONFIG_NAMES["A"]]
cat = cat.sort_values("C − A", ascending=False)
display(cat.style.format({c: "{:.0%}" for c in CONFIG_NAMES.values()} | {"C − A": "{:+.0%}"}))

fig, ax = plt.subplots(figsize=(7, 0.35 * len(cat) + 1))
y = np.arange(len(cat))
for i, c in enumerate(CONFIGS):
    ax.barh(y + (i - 1) * 0.26, cat[CONFIG_NAMES[c]], height=0.26, color=colors[i], label=CONFIG_NAMES[c])
ax.set_yticks(y, [f"{n}（{int(k)}題）" for n, k in zip(cat.index, cat["題數"])])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("答對率")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("各分類答對率（依 C − A 由大到小排序）")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.5 成本：token 與延遲
#
# 免費額度的瓶頸是 token。完整系統多一次改寫呼叫、又帶入條文，成本一定比較高；問題是**每多答對一題要多花多少**。

# %%
cost = res.groupby("config").agg(平均token=("total_tokens", "mean"), 中位數token=("total_tokens", "median"),
                                 平均延遲秒=("latency", "mean"), 平均條文字數=("context_chars", "mean"),
                                 答對率=("correct", "mean"))
cost.index = [CONFIG_NAMES[c] for c in cost.index]
base = cost.iloc[0]
cost["每多答對 1 題的額外 token"] = [(r["平均token"] - base["平均token"]) / (r["答對率"] - base["答對率"])
                                    if r["答對率"] > base["答對率"] else np.nan for _, r in cost.iterrows()]
display(cost.style.format({"平均token": "{:,.0f}", "中位數token": "{:,.0f}", "平均延遲秒": "{:.1f}",
                           "平均條文字數": "{:,.0f}", "答對率": "{:.1%}", "每多答對 1 題的額外 token": "{:,.0f}"}))
per_q = res.groupby("key")["total_tokens"].sum().mean() if N else 0
print(f"一題跑完三種設定平均 {per_q:,.0f} token → 免費方案每天約可測 {200000 / per_q:.0f} 題" if per_q else "")

# %% [markdown]
# ### 3.6 改寫步驟的範圍判斷
#
# 1A 的第一步會判斷問題是否屬於交通法規（範圍外就婉拒）。題庫每一題都屬於交通範圍，
# 所以「被判為範圍內」的比例就是這個分類器的**召回率**：太低代表會誤拒民眾的正常問題。

# %%
c_rows = res[res.config == "C"]
if len(c_rows):
    print(f"範圍內判斷召回率：{c_rows['in_scope'].mean():.1%}（{int(c_rows['in_scope'].sum())}/{len(c_rows)}）")
    print(f"被判為「在問罰則」的比例：{c_rows['asks_penalty'].mean():.1%}（觸發罰則保證名額）")
    display(c_rows[~c_rows["in_scope"].astype(bool)][["key", "category10"]].head(10))

# %% [markdown]
# ### 3.7 為什麼完整系統沒有贏過向量 RAG？（只用已存的結果，不呼叫 LLM）
#
# 前 44 題的結果顯示完整系統（C）的答對率低於單純向量 RAG（B）。假設：1A 的「罰則保證名額」是為「罰多少」這類問題設計的，
# 但筆試題多在問「應該怎麼做」，硬塞進的處罰條例條文可能把真正相關的安全規則擠出前 K 名。
# 以下檢查：(1) 兩種設定取回的條文來自哪部法規；(2) 兩者取回的條文重疊多少；(3) 在 B、C 結果不同的題目上，罰則名額是否被觸發。

# %%
def law_mix(config):
    laws = Counter(docs_by_id[d]["law_short"] for ids in res[res.config == config]["retrieved"] for d in ids)
    total = sum(laws.values())
    return {k: v / total for k, v in laws.items()}


mix = pd.DataFrame({CONFIG_NAMES[c]: law_mix(c) for c in ["B", "C"]}).fillna(0)
display(mix.style.format("{:.0%}"))

rb = res[res.config == "B"].set_index("key")
rc = res[res.config == "C"].set_index("key")
both = rb.index.intersection(rc.index)
jac = [len(set(rb.loc[k, "retrieved"]) & set(rc.loc[k, "retrieved"])) / len(set(rb.loc[k, "retrieved"]) | set(rc.loc[k, "retrieved"]))
       for k in both]
print(f"B、C 取回條文的平均重疊（Jaccard）：{np.mean(jac):.2f}")


def penalty_share(k):
    return np.mean([docs_by_id[d]["law_short"] == "處罰條例" for d in rc.loc[k, "retrieved"]])


groups = {"B 對 C 錯": wide[(wide.B == 1) & (wide.C == 0)].index,
          "B 錯 C 對": wide[(wide.B == 0) & (wide.C == 1)].index,
          "兩者都對": wide[(wide.B == 1) & (wide.C == 1)].index}
diag = pd.DataFrame([{
    "情況": g, "題數": len(ks),
    "罰則名額觸發率": rc.loc[ks, "asks_penalty"].astype(bool).mean() if len(ks) else np.nan,
    "C 條文中處罰條例占比": np.mean([penalty_share(k) for k in ks]) if len(ks) else np.nan,
    "B、C 條文重疊": np.mean([jac[list(both).index(k)] for k in ks]) if len(ks) else np.nan,
} for g, ks in groups.items()])
display(diag.style.format({"罰則名額觸發率": "{:.0%}", "C 條文中處罰條例占比": "{:.0%}", "B、C 條文重疊": "{:.2f}"}).hide(axis="index"))

cases = []
for k in groups["B 對 C 錯"][:8]:
    cases.append({"題目": eligible.set_index("key").loc[k, "question"][:70] + "…",
                  "C 改寫查詢": "；".join(rc.loc[k, "rewrite_queries"])[:60],
                  "B 取回": "；".join(t.split(" ", 1)[-1] for t in rb.loc[k, "retrieved_titles"][:3]),
                  "C 取回": "；".join(t.split(" ", 1)[-1] for t in rc.loc[k, "retrieved_titles"][:3])})
display(pd.DataFrame(cases))

# %% [markdown]
# ## 4. 實驗二：檢索品質與語意相似度（全部 1,599 題，不呼叫 LLM）
#
# NCKU 學長的範例用 **Question-to-Context（Q→C）** 與 **Answer-to-Context（A→C）** 兩個相似度指標，
# 觀察檢索內容是否切題、回答是否與參考資料語意一致。本節沿用這個想法，並針對選擇題多做一個更直接的指標：
#
# | 指標 | 定義 | 回答的問題 |
# |---|---|---|
# | **Q→C** | 題目向量與取回條文向量的最高餘弦相似度 | 取回的條文和題目切不切題？ |
# | **條文支持度** | 三個選項各自與取回條文的最高相似度，最高者是否就是正確選項 | 取回的條文是否「指向」正確答案？（隨機猜 = 33%） |
# | **領域區辨力（AUC）** | 題庫題目 vs. 30 題非交通問題的 Q→C 分布能否分開 | 能不能用相似度門檻判斷「範圍外」？ |
# | **A→C**（實驗一的樣本） | 模型選的選項與條文的相似度，答對 vs. 答錯 | 回答越貼近條文，是否越容易答對？ |
#
# 這一節只用 embedding，所以可以跑全部題目；比較的是「只用向量」與「向量＋BM25＋條號直查（RRF）」兩種檢索。
# 完整系統的 LLM 改寫需要呼叫 API，不在本節。

# %%
def split_options(q):
    parts = re.split(r"\(([123])\)", q)
    stem = parts[0].strip()
    opts = {int(parts[i]): parts[i + 1].strip(" 。．.;；") for i in range(1, len(parts) - 1, 2)}
    return stem, opts


qs = eligible.reset_index(drop=True)
qs[["stem", "options"]] = qs["question"].apply(lambda s: pd.Series(split_options(s)))
qs = qs[qs["options"].apply(lambda o: set(o) == {1, 2, 3})].reset_index(drop=True)
print("可拆出 3 個選項的題目：", len(qs))

t0 = time.time()
q_vecs = embed_queries(qs["question"].tolist())
opt_texts = [qs.loc[i, "options"][k] for i in range(len(qs)) for k in (1, 2, 3)]
opt_vecs = embed_queries(opt_texts).reshape(len(qs), 3, -1)
print(f"題目與選項向量化完成：{time.time() - t0:.0f} 秒")

doc_index = {d["doc_id"]: i for i, d in enumerate(documents)}


def retrieval_metrics(method):
    rows = []
    for i, row in qs.iterrows():
        if method == "向量":
            results = hybrid_search([row["question"]], top_k=TOP_K, use_bm25=False, use_direct=False)
        else:
            results = hybrid_search([row["question"]], top_k=TOP_K)
        ids = [doc_index[r["doc_id"]] for r in results]
        ctx = doc_vectors[ids]
        qc = float((ctx @ q_vecs[i]).max())
        support = (opt_vecs[i] @ ctx.T).max(1)              # 每個選項對條文的最高相似度
        rows.append({"key": row["key"], "方法": method, "Q→C": qc, "Q→C 平均": float((ctx @ q_vecs[i]).mean()),
                     "條文支持選項": int(support.argmax()) + 1, "正解": int(row["answer"]),
                     "支持度差": float(support[int(row["answer"]) - 1] - np.delete(support, int(row["answer"]) - 1).max()),
                     "category10": row["category10"], "bank_label": row["bank_label"],
                     "law_short": Counter(documents[j]["law_short"] for j in ids).most_common(1)[0][0]})
    return rows


t0 = time.time()
ret = pd.DataFrame(retrieval_metrics("向量") + retrieval_metrics("混合"))
ret["支持正解"] = (ret["條文支持選項"] == ret["正解"]).astype(int)
print(f"檢索評估完成：{time.time() - t0:.0f} 秒")

summary = ret.groupby("方法").agg(Q_C=("Q→C", "mean"), Q_C_平均=("Q→C 平均", "mean"), 條文支持正解率=("支持正解", "mean"))
display(summary.style.format({"Q_C": "{:.3f}", "Q_C_平均": "{:.3f}", "條文支持正解率": "{:.1%}"}))
for m in ["向量", "混合"]:
    k = ret[ret["方法"] == m]["支持正解"]
    lo, hi = wilson(k.sum(), len(k))
    p = stats.binomtest(int(k.sum()), len(k), 1 / 3, alternative="greater").pvalue
    print(f"{m}：條文支持正解率 {k.mean():.1%}（95% CI {lo:.1%}～{hi:.1%}），相對隨機 33.3% 的單尾二項檢定 p = {p:.2e}")
pv = ret.pivot(index="key", columns="方法", values="支持正解")
n01, n10, p = mcnemar_exact(pv["向量"], pv["混合"])
print(f"向量 vs 混合（McNemar）：只有混合支持正解 {n01} 題、只有向量支持正解 {n10} 題，p = {p:.4f}")

# %% [markdown]
# ### 4.1 哪些分類的題目，條文最能指向正確答案？

# %%
by_cat = ret[ret["方法"] == "混合"].groupby("category10").agg(題數=("key", "size"), Q_C=("Q→C", "mean"),
                                                              條文支持正解率=("支持正解", "mean"))
by_cat = by_cat.sort_values("條文支持正解率", ascending=False)
display(by_cat.style.format({"Q_C": "{:.3f}", "條文支持正解率": "{:.1%}"}))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for m, color in [("向量", colors[1]), ("混合", colors[2])]:
    axes[0].hist(ret[ret["方法"] == m]["Q→C"], bins=30, alpha=0.6, color=color, label=m)
axes[0].set_xlabel("Q→C（題目與條文的最高餘弦相似度）")
axes[0].set_ylabel("題數")
axes[0].legend()
axes[0].set_title("Q→C 分布")
axes[1].barh(by_cat.index, by_cat["條文支持正解率"], color=colors[2])
axes[1].axvline(1 / 3, color="gray", ls="--", lw=1)
axes[1].text(1 / 3 + 0.01, -0.6, "隨機 33%", fontsize=8, color="gray")
axes[1].invert_yaxis()
axes[1].set_xlim(0, 1)
axes[1].set_title("混合檢索：條文支持正解率（依分類）")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 領域區辨力：相似度能不能分辨「範圍外」的問題？
#
# 拿 30 題與交通無關的問題（天氣、烹飪、程式、理財…）做對照組，與題庫題目比較 Q→C 分布，
# 計算 ROC AUC（1.0 = 完全分得開，0.5 = 跟猜的一樣）。
# 這可以支持 1A 的設計選擇：範圍判斷該交給相似度門檻，還是交給 LLM。

# %%
OFF_TOPIC = [
    "今天台南會下雨嗎？", "番茄炒蛋要先炒蛋還是先炒番茄？", "台積電明年的股利大概多少？", "Python 的 list 和 tuple 差在哪？",
    "怎麼準備多益考試比較有效率？", "貓咪一天要吃幾餐？", "推薦幾部好看的日劇", "電腦藍屏要怎麼處理？",
    "如何申請就學貸款？", "黑膠唱盤的唱針多久要換？", "感冒可以喝咖啡嗎？", "怎麼寫一封請假信給教授？",
    "台北哪裡的夜景最漂亮？", "離散數學的鴿籠原理是什麼？", "要怎麼煮出好吃的白飯？", "手機電池怎麼保養比較久？",
    "週末去墾丁有什麼行程推薦？", "健身房新手要先練哪些動作？", "如何用 Excel 做樞紐分析？", "租屋押金最多可以收幾個月？",
    "期末報告的參考文獻格式怎麼寫？", "如何挑選一台筆電？", "颱風天公司要不要上班？", "怎麼種多肉植物？",
    "MOSFET 的三個工作區是什麼？", "牙齒敏感要用什麼牙膏？", "如何申請護照？", "咖啡豆要怎麼保存？",
    "什麼是區塊鏈？", "大學社團幹部要做哪些事？",
]
off_vecs = embed_queries(OFF_TOPIC)
off_scores = []
for i, q in enumerate(OFF_TOPIC):
    results = hybrid_search([q], top_k=TOP_K, use_bm25=False, use_direct=False)
    ctx = doc_vectors[[doc_index[r["doc_id"]] for r in results]]
    off_scores.append(float((ctx @ off_vecs[i]).max()))
in_scores = ret[ret["方法"] == "向量"]["Q→C"].values
auc = stats.mannwhitneyu(in_scores, off_scores, alternative="greater").statistic / (len(in_scores) * len(off_scores))
print(f"題庫題目 Q→C 平均 {np.mean(in_scores):.3f}；非交通問題 Q→C 平均 {np.mean(off_scores):.3f}；ROC AUC = {auc:.3f}")
thr = np.percentile(in_scores, 5)
print(f"若以題庫題目的第 5 百分位 {thr:.3f} 當門檻：題庫題被誤拒 5%，非交通問題被擋下 {np.mean(np.array(off_scores) < thr):.0%}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(in_scores, bins=30, alpha=0.6, density=True, color=colors[2], label=f"題庫題目（{len(in_scores)} 題）")
ax.hist(off_scores, bins=12, alpha=0.6, density=True, color="#ea4335", label=f"非交通問題（{len(off_scores)} 題）")
ax.axvline(thr, color="black", ls="--", lw=1)
ax.set_xlabel("Q→C")
ax.set_ylabel("密度")
ax.set_title(f"領域區辨力：ROC AUC = {auc:.3f}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 A→C：回答越貼近條文，越容易答對嗎？
#
# 用實驗一中 B、C 兩種 RAG 設定的作答：計算模型所選選項與取回條文的最高相似度（A→C），
# 比較答對與答錯兩組的分布（Mann–Whitney U 檢定，不假設常態）。

# %%
rows = []
key_to_i = {k: i for i, k in enumerate(qs["key"])}
for r in res[res.config.isin(["B", "C"])].itertuples():
    if r.key not in key_to_i or pd.isna(r.pred) or not r.retrieved:
        continue
    i = key_to_i[r.key]
    ctx = doc_vectors[[doc_index[d] for d in r.retrieved]]
    rows.append({"config": r.config, "A→C": float((ctx @ opt_vecs[i][int(r.pred) - 1]).max()), "correct": r.correct})
ac = pd.DataFrame(rows)
if len(ac) and ac["correct"].nunique() == 2:
    for c in ["B", "C"]:
        part = ac[ac.config == c]
        good, bad = part[part.correct == 1]["A→C"], part[part.correct == 0]["A→C"]
        if len(bad) and len(good):
            u = stats.mannwhitneyu(good, bad, alternative="greater")
            print(f"{CONFIG_NAMES[c]}：答對 A→C 中位數 {good.median():.3f}（{len(good)} 題） vs 答錯 {bad.median():.3f}（{len(bad)} 題），"
                  f"Mann–Whitney p = {u.pvalue:.4f}")
else:
    print("實驗一的資料還不夠（需要有答對也有答錯的 RAG 作答）。")

# %% [markdown]
# ## 5. 錯誤分析
#
# 看數字之外，實際讀幾題：完整系統**答錯但純 LLM 答對**的題目（RAG 帶來的傷害），以及反過來的題目（RAG 的貢獻）。

# %%
def show_cases(win, lose, n=5):
    keys = wide[(wide[win] == 1) & (wide[lose] == 0)].index[:n]
    out = []
    for k in keys:
        q = eligible.set_index("key").loc[k]
        rc = res[(res.key == k) & (res.config == "C")].iloc[0]
        ra = res[(res.key == k) & (res.config == "A")].iloc[0]
        out.append({"題目": q["question"][:90] + "…", "正解": int(q["answer"]), "A 選": ra["pred"], "C 選": rc["pred"],
                    "分類": q["category10"], "C 取回的條文": "；".join(rc["retrieved_titles"][:3])})
    return pd.DataFrame(out)


print("■ 完整系統答對、純 LLM 答錯（RAG 的貢獻）")
display(show_cases("C", "A"))
print("■ 完整系統答錯、純 LLM 答對（RAG 的傷害）")
display(show_cases("A", "C"))

# %% [markdown]
# ## 6. 結論（依目前資料自動產生）

# %%
def pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"


if N:
    a, b, c = (wide[x].mean() for x in CONFIGS)
    sig = tests.set_index("比較")["Holm 校正 p"]
    lines = [
        f"- 實驗一（{N} 題成對資料）：純 LLM {pct(a)}、向量 RAG {pct(b)}、完整系統 {pct(c)}；Cochran's Q p = {p_q:.4f}。",
        f"- 完整系統相對純 LLM 的差異 {c - a:+.1%}（Holm 校正 p = {sig.iloc[1]:.4f}），相對向量 RAG {c - b:+.1%}（p = {sig.iloc[2]:.4f}）。",
        f"- 受益最多的分類：{cat.index[0]}（C − A = {cat['C − A'].iloc[0]:+.0%}）；最少：{cat.index[-1]}（{cat['C − A'].iloc[-1]:+.0%}）。",
    ]
else:
    lines = ["- 實驗一尚未有完成的題目。"]
mix = summary.loc["混合"] if "混合" in summary.index else None
if mix is not None:
    lines += [
        f"- 實驗二（{len(qs)} 題、不呼叫 LLM）：混合檢索的條文支持正解率 {pct(mix['條文支持正解率'])}（隨機 33.3%），"
        f"向量檢索 {pct(summary.loc['向量', '條文支持正解率'])}。",
        f"- 題庫題目與非交通問題的 Q→C 區辨力 ROC AUC = {auc:.3f}。",
    ]
display(Markdown("\n".join(lines)))

# %% [markdown]
# ## 限制
#
# - 實驗一受 Groq 免費額度限制只抽樣 200 題；分類別的結果樣本小，只能當描述。
# - 題庫測的是「駕照考試知識」，包含不少安全觀念題，和 1A 主打的「違規罰則查詢」不完全相同；
#   所以本實驗是 RAG 效益的**保守估計**（法規明文題才是 RAG 的主場）。
# - 推理強度設為 low、TOP_K = 4 是為了省 token，和 1A 介面的預設（TOP_K = 6、預設推理強度）不同。
# - 「條文支持度」用 embedding 相似度近似「條文是否指向正解」，不等於法律上的正確推論。
#
# ## Reference
# 1. 交通部公路局，汽車駕照筆試題庫（115 年新版）、機車駕照筆試題庫（115.1 版），中、英文版。
# 2. 全國法規資料庫；kong0107, *mojLawSplitJSON*, https://github.com/kong0107/mojLawSplitJSON
# 3. 謝承祐，Project NCKU-1A：交通法規智慧問答系統，https://github.com/RogerH0711/traffic-law-rag
# 4. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
# 5. Cochran, W. G. (1950). The comparison of percentages in matched samples. *Biometrika*, 37(3/4), 256–266.
# 6. Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
# 7. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209–212.
# 8. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
# 9. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
# 10. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EACL System Demonstrations*.
# 11. 課程範例：「應用 RAG＋ChatGPT 於行政院核能安全委員會智慧問答系統」之 Q→C／A→C 雙相似度分析指標（NCKU-1B 題目說明）。
