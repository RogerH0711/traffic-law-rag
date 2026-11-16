"""本機測試：用假的 embedding / LLM / colab 模組跑完整 notebook（gradio 不 launch）。"""
import sys, types, re, json, hashlib
import numpy as np
import jupytext

# ---- fake torch ----
import gradio  # import real gradio before faking torch
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
torch.bfloat16, torch.float32 = "bf16", "fp32"
sys.modules["torch"] = torch

# ---- fake huggingface_hub ----
import huggingface_hub as hh
hh.HfApi.model_info = lambda self, *a, **k: types.SimpleNamespace(sha="0" * 40)

# ---- fake sentence_transformers: hashed char-bigram bag ----
st = types.ModuleType("sentence_transformers")
class SentenceTransformer:
    def __init__(self, *a, **k): self.max_seq_length = 2048
    def encode(self, texts, **k):
        D = 256
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, t in enumerate(texts):
            t = t.split("text:")[-1].split("query:")[-1]
            for j in range(len(t) - 1):
                out[i, int(hashlib.md5(t[j:j+2].encode()).hexdigest(), 16) % D] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out
st.SentenceTransformer = SentenceTransformer
sys.modules["sentence_transformers"] = st

# ---- fake aisuite ----
LOG = []
ai = types.ModuleType("aisuite")
class _Comp:
    def create(self, model, messages, **k):
        sys_msg = messages[0]["content"]
        if "法規檢索助理" in sys_msg:
            q = messages[1]["content"]
            if "下雨" in q:
                text = '{"in_scope": false, "queries": []}'
            else:
                text = json.dumps({"in_scope": True, "asks_penalty": True, "queries": ["機車駕駛人 以手持方式使用行動電話 罰鍰", "汽車駕駛人 處罰鍰", q]}, ensure_ascii=False)
        elif "參考條文" in messages[-1]["content"]:
            ids = re.findall(r"\[(L\d{4})\]", messages[-1]["content"])
            text = f"**一句話結論**：處新臺幣 1,200 元罰鍰【{ids[0]}、{ids[1]}】，另外亂寫 9,999 元 [L9999]。"
        else:
            text = "純 LLM：罰 3,000 元。"
        LOG.append((sys_msg[:20], text[:60]))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason="stop")])
class Client:
    def __init__(self, *a, **k): self.chat = types.SimpleNamespace(completions=_Comp())
ai.Client = Client
sys.modules["aisuite"] = ai

# ---- fake google.colab ----
g = types.ModuleType("google"); gc = types.ModuleType("google.colab")
gc.userdata = types.SimpleNamespace(get=lambda name: "fake-key")
g.colab = gc
sys.modules["google"] = g; sys.modules["google.colab"] = gc

# ---- IPython display -> print ----
import IPython.display as ipd
ipd.display = lambda *a, **k: [print(x.data if hasattr(x, "data") else x) for x in a]

# gradio: real, but no launch
import gradio as gr
gr.Blocks.launch = lambda self, *a, **k: print("[launch skipped]")

nb = jupytext.read(sys.argv[1])
ns = {"__name__": "__main__"}
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = "\n".join(l for l in cell.source.splitlines() if not l.strip().startswith("%"))
    print(f"\n######## cell {i} ########")
    exec(compile(src, f"cell{i}", "exec"), ns)

# extra checks
out = ns["ask"]("騎機車邊騎邊講手機會怎樣？")
print(ns["verification_markdown"](out["verification"]))
print(ns["link_citations"](out["answer"]))
print(ns["gradio_answer"]("喝酒騎車罰多少", "承辦人員版（精確）", 6, True, True)[:3])
