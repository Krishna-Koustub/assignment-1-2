import json, sys
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# numbers that go into the written inferences (filled from the executed run)
N = json.loads((ROOT / "tools" / "numbers.json").read_text()) if (ROOT / "tools" / "numbers.json").exists() else {}
def n(key, default="?"): return N.get(key, default)

# =====================================================================
md(r'''
# Assignment 1A: a Linux sysadmin LLM (CPT, then QLoRA)

Pipeline: public PDFs -> cleaned text -> continual pre-training (CPT) of a small base model -> instruction tuning with QLoRA.

**Choices I made**

| | |
|---|---|
| Variant | V1 (default domain Q&A assistant), domain picked by me: Linux system administration |
| Model | `HuggingFaceTB/SmolLM2-360M` (one of the T4 options in the brief). Its own tokenizer is used everywhere |
| Corpus | 7 public PDFs from the Linux Documentation Project and the Debian project, 2,345 pages before cleaning |
| Hardware | free Colab T4 (16 GB). The notebook also runs on a Mac/CPU with `A1_QUICK=1`, but that is only a dry run (no 4-bit there) |

**Running it on Colab**: Runtime > Change runtime type > T4 GPU, upload `instruction_dataset.jsonl` when Part B asks for it, then Run all. PDFs are downloaded by the notebook itself. Expect roughly 30 to 40 minutes in total.

Two places where I deviate from the brief, both because of the T4: it has no native bfloat16, so the model is kept in fp32 with fp16 mixed precision instead (explained in Step 3), and I train a single adapter (Adapter B) in Part B2, which the brief allows.
''')

code(r'''
import sys, subprocess
if "google.colab" in sys.modules:
    # pinned: the trl collator / SFTConfig arguments change a lot between releases
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.51.3", "peft==0.15.2", "trl==0.17.0", "accelerate==1.6.0",
                    "bitsandbytes>=0.45.0", "pypdfium2", "langdetect", "pyarrow"], check=True)
    print("installed - if Colab asks for a restart, restart and run from the top")
''')

code(r'''
import os, re, io, json, math, random, time, zlib, hashlib, collections, gc, textwrap, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
BLOCK = 2048                       # packed sequence length (model supports 8192, 2048 is what fits a T4 comfortably)
QUICK = os.environ.get("A1_QUICK") == "1"   # tiny dry run to test the code, numbers from it mean nothing

if torch.cuda.is_available():
    device = "cuda"
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8      # T4 is 7.5, so False there
    print("GPU:", torch.cuda.get_device_name(0), "| bf16 native:", native_bf16)
elif torch.backends.mps.is_available():
    device, native_bf16 = "mps", False
else:
    device, native_bf16 = "cpu", False

use_bf16 = device == "cuda" and native_bf16
use_fp16 = device == "cuda" and not native_bf16
load_dtype = torch.bfloat16 if use_bf16 else torch.float32
print("device:", device, "| load dtype:", load_dtype, "| fp16 autocast:", use_fp16, "| QUICK:", QUICK)

PDF_DIR, RAW_DIR, CORPUS_DIR = Path("raw_pdfs"), Path("raw_text"), Path("domain_corpus")
for d in (PDF_DIR, RAW_DIR, CORPUS_DIR): d.mkdir(exist_ok=True)
''')

# =====================================================================
md(r'''
## Part A, Step 1: data collection, extraction, cleaning

Sources are all free, English, and freely downloadable. The Red Hat / Ubuntu docs I first considered return 403 for scripted downloads, so I used TLDP and Debian material instead.
''')

code(r'''
import requests

SOURCES = {
    "linux_sysadmin_guide":      "https://tldp.org/LDP/sag/sag.pdf",
    "advanced_bash_scripting":   "https://tldp.org/LDP/abs/abs-guide.pdf",
    "bash_beginners_guide":      "https://tldp.org/LDP/Bash-Beginners-Guide/Bash-Beginners-Guide.pdf",
    "debian_reference":          "https://www.debian.org/doc/manuals/debian-reference/debian-reference.en.pdf",
    "linux_network_admin_guide": "https://tldp.org/LDP/nag2/nag2.pdf",
    "introduction_to_linux":     "https://tldp.org/LDP/intro-linux/intro-linux.pdf",
    "linux_filesystem_hierarchy":"https://tldp.org/LDP/Linux-Filesystem-Hierarchy/Linux-Filesystem-Hierarchy.pdf",
}

for name, url in SOURCES.items():
    f = PDF_DIR / f"{name}.pdf"
    if f.exists() and f.stat().st_size > 50_000:
        continue
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    r.raise_for_status()
    f.write_bytes(r.content)
    print("downloaded", f.name, round(len(r.content) / 1e6, 1), "MB")

print(len(list(PDF_DIR.glob("*.pdf"))), "pdfs in", PDF_DIR)
''')

code(r'''
import pypdfium2 as pdfium

# page-by-page extraction, one raw .txt per pdf
pages = {}
for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    pdf = pdfium.PdfDocument(str(pdf_path))
    pages[pdf_path.stem] = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    (RAW_DIR / f"{pdf_path.stem}.txt").write_text("\n\f\n".join(pages[pdf_path.stem]), encoding="utf-8")

per_doc = pd.DataFrame({"pages": {k: len(v) for k, v in pages.items()},
                        "words": {k: sum(len(p.split()) for p in v) for k, v in pages.items()}})
per_doc.loc["TOTAL"] = per_doc.sum()
per_doc
''')

md(r'''
The filters below work on **pages** rather than whole files: with only 7 documents a file-level length/dedup/language filter would never remove anything, so counts would say nothing. I still report the document count at every step.

Order of operations: tidy (running headers/footers, hyphenation, whitespace) -> length filter -> exact dedup -> near-duplicate dedup -> language filter.
''')

code(r'''
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def n_docs(d):  return sum(1 for v in d.values() if v)
def n_pages(d): return sum(len(v) for v in d.values())
def n_words(d): return sum(len(p.split()) for v in d.values() for p in v)

log = [("raw extraction", n_docs(pages), n_pages(pages), n_words(pages))]

def tidy(t):
    t = t.replace("\r", "")
    t = re.sub(r"-\n(?=[a-z])", "", t)        # word broken over a line end
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def strip_running_lines(doc_pages):
    # a line (digits masked) that shows up on >=15% of a document's pages is a header/footer
    norm = lambda l: re.sub(r"\d+", "#", l.strip())
    c = collections.Counter()
    for p in doc_pages:
        c.update({norm(l) for l in p.splitlines() if l.strip()})
    cutoff = max(8, int(0.15 * len(doc_pages)))
    common = {k for k, v in c.items() if v >= cutoff and len(k) < 90}
    return ["\n".join(l for l in p.splitlines() if norm(l) not in common) for p in doc_pages]

cleaned = {k: [tidy(p) for p in strip_running_lines(v)] for k, v in pages.items()}
log.append(("tidy + header/footer removal", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

# length filter
MIN_WORDS = 60
cleaned = {k: [p for p in v if len(p.split()) >= MIN_WORDS] for k, v in cleaned.items()}
log.append((f"length filter (>= {MIN_WORDS} words per page)", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

# exact dedup
seen, out = set(), {}
for k, v in cleaned.items():
    keep = []
    for p in v:
        h = hashlib.md5(re.sub(r"\W+", "", p.lower()).encode()).hexdigest()
        if h not in seen:
            seen.add(h); keep.append(p)
    out[k] = keep
cleaned = out
log.append(("exact duplicate pages", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

# near-duplicates: page whose 6-word shingles were mostly seen already (licence text, repeated indexes...)
def shingles(p, n=6):
    w = re.findall(r"\w+", p.lower())
    return {zlib.crc32(" ".join(w[i:i + n]).encode()) for i in range(max(1, len(w) - n + 1))}

seen_sh, out, near_dups = set(), {}, []
for k, v in cleaned.items():
    keep = []
    for i, p in enumerate(v):
        sh = shingles(p)
        overlap = len(sh & seen_sh) / len(sh)
        if overlap >= 0.6:
            near_dups.append((k, i, round(overlap, 2)))
            continue
        seen_sh |= sh
        keep.append(p)
    out[k] = keep
cleaned = out
log.append(("near-duplicate pages (>=60% shingle overlap)", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

# language filter
def is_english(p):
    try:
        return detect(p[:1500]) == "en"
    except Exception:
        return False

dropped_lang, out = [], {}
for k, v in cleaned.items():
    keep = []
    for i, p in enumerate(v):
        if is_english(p): keep.append(p)
        else: dropped_lang.append((k, i, p[:90].replace("\n", " | ")))
    out[k] = keep
cleaned = out
log.append(("language filter (English only)", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

log_df = pd.DataFrame(log, columns=["step", "documents", "pages", "words"])
log_df["pages_removed"] = -log_df["pages"].diff().fillna(0).astype(int)
log_df["words_removed"] = -log_df["words"].diff().fillna(0).astype(int)
log_df
''')

code(r'''
print("pages dropped by the language filter:", len(dropped_lang))
for k, i, head in dropped_lang:
    print(f"  {k} p{i}: {head}")
print("\nnear-duplicates by document:", collections.Counter(k for k, _, _ in near_dups))

steps = log_df["step"].str.replace(r" \(.*\)", "", regex=True)[1:]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].barh(steps, log_df["pages_removed"][1:], color="#4c72b0"); ax[0].set_title("pages removed"); ax[0].invert_yaxis()
ax[1].barh(steps, log_df["words_removed"][1:], color="#dd8452"); ax[1].set_title("words removed"); ax[1].invert_yaxis(); ax[1].set_yticks([])
plt.tight_layout(); plt.show()
''')

code(r'''
for name, pgs in cleaned.items():
    (CORPUS_DIR / f"{name}.txt").write_text("\n\n".join(pgs), encoding="utf-8")

final = pd.DataFrame({"pages": {k: len(v) for k, v in cleaned.items()},
                      "words": {k: sum(len(p.split()) for p in v) for k, v in cleaned.items()}})
final.loc["TOTAL"] = final.sum()
print("cleaned files written to", CORPUS_DIR)
final
''')

md(f'''
**Inference (Step 1)**

- The corpus started at {n("raw_pages")} pages / {n("raw_words")} words and ended at {n("final_pages")} pages / {n("final_words")} words, so cleaning removed only about {n("pct_words_removed")}% of the words. That is expected: these are already typeset manuals, not scraped web text.
- Biggest impact depends on the unit. By **pages** it is the length filter ({n("len_pages")} pages, mostly blank separators, title pages and short table-of-contents / part-divider pages). By **words** it is the header/footer + tidy step ({n("tidy_words")} words) followed by near-duplicate removal ({n("dup_words")} words).
- Exact-duplicate pages: none. The near-duplicate filter did find {n("dup_pages")} pages, concentrated in licence appendices, repeated index/reference pages and the overlapping bash/sysadmin material, which is exactly what I wanted to keep out of a CPT run (repeated text gets memorised and inflates the apparent gain).
- The language filter only removed {n("lang_pages")} pages. Printing them shows seven are shell-code listings or bullet lists of command names that langdetect misreads as another language, and one is an ordinary prose chapter opener, so none of the removals are really non-English pages. The cost is tiny, but a stopword-ratio check would be a better filter for this kind of corpus.
- {n("final_pages")} pages is far above the 300-page minimum. One thing to be aware of: `advanced_bash_scripting` is about 30% of the words, so the CPT model will lean towards bash scripting style.
''')

# =====================================================================
md(r'''
## Step 2: tokenization and packed dataset

Tokenizer comes from the same model id as the weights. SmolLM2 uses one token (`<|endoftext|>`, id 0) as both BOS and EOS, so a document boundary inside a packed sequence looks like `... EOS BOS ...` i.e. two id-0 tokens in a row.

What counts as a "document" here: with 7 files, wrapping each file in one BOS/EOS pair would give 7 boundaries in about 1.3M tokens, which defeats the purpose. So each file is cut into blocks of 10 consecutive cleaned pages and every block is one document (BOS ... EOS). Held-out data is chosen at block level (10% of the blocks of every file) *before* packing, so no eval sequence shares a block with training data.
''')

code(r'''
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID)
BOS, EOS = tok.bos_token_id, tok.eos_token_id
print("vocab:", len(tok), "| BOS id:", BOS, "| EOS id:", EOS)

PAGES_PER_DOC = 10
docs = []
for name, pgs in cleaned.items():
    for i in range(0, len(pgs), PAGES_PER_DOC):
        docs.append({"source": name, "text": "\n\n".join(pgs[i:i + PAGES_PER_DOC])})

for d in docs:
    d["ids"] = [BOS] + tok(d["text"], add_special_tokens=False, verbose=False)["input_ids"] + [EOS]
    d["n_tok"] = len(d["ids"])

# 10% of the blocks of each file go to the held-out set
rng = random.Random(SEED)
eval_idx = set()
for name in cleaned:
    idx = [i for i, d in enumerate(docs) if d["source"] == name]
    eval_idx |= set(rng.sample(idx, max(1, round(0.1 * len(idx)))))
train_docs = [d for i, d in enumerate(docs) if i not in eval_idx]
eval_docs  = [d for i, d in enumerate(docs) if i in eval_idx]

def pack(doc_list, block=BLOCK):
    flat = [t for d in doc_list for t in d["ids"]]
    k = len(flat) // block
    return np.array(flat[:k * block], dtype=np.int32).reshape(k, block), len(flat) - k * block

train_arr, train_left = pack(train_docs)
eval_arr, eval_left = pack(eval_docs)

for path, arr in (("train_packed.parquet", train_arr), ("eval_packed.parquet", eval_arr)):
    pd.DataFrame({"input_ids": [r.tolist() for r in arr]}).to_parquet(path)

total_tokens = sum(d["n_tok"] for d in docs)
stats = pd.Series({
    "documents (10-page blocks)": len(docs),
    "train / eval documents": f"{len(train_docs)} / {len(eval_docs)}",
    "total tokens (incl. BOS/EOS)": total_tokens,
    "average document length (tokens)": round(total_tokens / len(docs), 1),
    "tokens per word": round(total_tokens / sum(len(d["text"].split()) for d in docs), 2),
    "packed sequence length": BLOCK,
    "packed sequences: train": len(train_arr),
    "packed sequences: eval": len(eval_arr),
    "tokens dropped at the tail (train / eval)": f"{train_left} / {eval_left}",
})
stats.to_frame("value")
''')

code(r'''
# sanity check: look at one document boundary inside a packed sequence
row = next(r for r in train_arr if ((r[:-1] == EOS) & (r[1:] == BOS)).any())
j = int(np.where((row[:-1] == EOS) & (row[1:] == BOS))[0][0])
print("...", repr(tok.decode(row[j - 12:j])))
print("boundary ->", tok.convert_ids_to_tokens(row[j:j + 2].tolist()))
print("...", repr(tok.decode(row[j + 2:j + 14])))
''')

md(f'''
**Inference (Step 2)**

- {n("total_tokens")} tokens in total over {n("n_docs")} documents, about {n("avg_doc_tokens")} tokens per document and {n("tok_per_word")} tokens per word. The ratio is higher than ordinary English prose (around 1.3) because the manuals are full of paths, flags and code, which split into many small tokens.
- Packing into 2048-token blocks gives {n("train_seqs")} training and {n("eval_seqs")} evaluation sequences and wastes under 2048 tokens per split at the tail. There is no padding anywhere, so every token in a batch contributes to the loss.
- The sanity check shows what the model will see at a boundary: one document ends with EOS and the next begins with BOS (both id 0), with text from an unrelated section following. A side effect of simple packing is that attention is allowed to cross that boundary; for a corpus this size I judged that harmless.
- A corpus of about 1.3M tokens is small by CPT standards, so I expect a visible but modest adaptation and need to watch for forgetting in Step 5.
''')

# =====================================================================
md(r'''
## Step 3: load the model and inspect it

**bf16 vs the T4.** The brief says to load in bfloat16 on a T4, but a T4 (compute capability 7.5) has no bf16 hardware; PyTorch would emulate it and training crawls. So the code checks the GPU: on Ampere or newer (A100, L40S) it loads in bf16, on a T4 it keeps fp32 weights and uses fp16 autocast for the matmuls. For a 360M model fp32 weights plus Adam states are about 6 GB, which fits fine.
''')

code(r'''
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=load_dtype).to(device)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False          # cache is useless (and warns) while checkpointing; turned on again for generate()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total params:     {total:,}")
print(f"trainable params: {trainable:,}")

cfg = model.config
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
arch = pd.Series({
    "decoder layers": cfg.num_hidden_layers,
    "attention heads (query)": cfg.num_attention_heads,
    "key/value heads": cfg.num_key_value_heads,
    "hidden size": cfg.hidden_size,
    "head dimension": head_dim,
    "MLP intermediate size": cfg.intermediate_size,
    "max position embeddings": cfg.max_position_embeddings,
    "tied input/output embeddings": cfg.tie_word_embeddings,
})
arch.to_frame("value")
''')

code(r'''
# lm_head must produce one logit per vocabulary entry
with torch.no_grad():
    logits = model(input_ids=torch.tensor([[BOS, 100, 200]], device=device)).logits
print("lm_head out_features :", model.lm_head.out_features)
print("config.vocab_size    :", cfg.vocab_size)
print("len(tokenizer)       :", len(tok))
print("logits last dim      :", logits.shape[-1])
assert model.lm_head.out_features == cfg.vocab_size == len(tok) == logits.shape[-1]
print("embedding and lm_head share weights:", model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())
''')

code(r'''
def complete(m, prompt, max_new_tokens=60):
    ids = torch.tensor([[BOS] + tok(prompt, add_special_tokens=False)["input_ids"]], device=device)
    with torch.no_grad():
        out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, repetition_penalty=1.15, pad_token_id=EOS, use_cache=True)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

GEN_TOK = 24 if QUICK else 60

domain_prompts = [
    "How do I check how much free disk space is left on a Linux machine?",
    "What is the /var directory used for in the Linux filesystem?",
    "How can a bash script find out whether the previous command succeeded?",
]

model.eval()
baseline = {p: complete(model, p, GEN_TOK) for p in domain_prompts}
json.dump(baseline, open("baseline_outputs.json", "w"), indent=2)
for p, o in baseline.items():
    print("PROMPT:", p); print("BASE  :", o.replace("\n", " / ")[:400]); print()
''')

md(f'''
**Inference (Step 3)**

- {n("params")} parameters, all trainable (full fine-tuning in Part A). The config reads 32 decoder layers, 15 query heads sharing 5 key/value heads (grouped-query attention), hidden size 960 and head dimension 64 (15 x 64 = 960, so the numbers are consistent).
- The output layer has 49,152 outputs, equal to the tokenizer length and `config.vocab_size`, and it shares its weight matrix with the input embedding. That means the embedding matrix is counted once in the parameter total, and any update to the vocabulary embeddings during CPT moves input and output together.
- Baseline behaviour, which Step 4 compares against: the base model has never been taught to answer questions, so it continues the prompt in the voice of a forum post or tutorial ("I have an Ubuntu server with...", "I have written this script to...") instead of answering. It is not blind to the domain, though: its continuation for the /var prompt starts off sensibly, so part of what CPT can add is small.
''')

# =====================================================================
md(r'''
## Step 4: CPT training loop and loss curve

Plain causal-LM loss on the packed sequences (labels = input ids, the model shifts them). Learning rate is deliberately low (3e-5) with cosine decay and a short warmup, because this is a small corpus and I want to limit forgetting. Effective batch is 2 x 8 = 16 sequences = 32,768 tokens per optimizer step.
''')

code(r'''
from transformers import Trainer, TrainingArguments, TrainerCallback

class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.rows = pd.read_parquet(path)["input_ids"].tolist()
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        ids = torch.tensor(self.rows[i], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

train_ds, eval_ds = PackedDataset("train_packed.parquet"), PackedDataset("eval_packed.parquet")

class LossLogger(TrainerCallback):
    def __init__(self):
        self.steps, self.losses, self.lrs = [], [], []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.steps.append(state.global_step); self.losses.append(logs["loss"]); self.lrs.append(logs.get("learning_rate"))

loss_cb = LossLogger()
cpt_args = TrainingArguments(
    output_dir="cpt_runs",
    num_train_epochs=3,
    max_steps=3 if QUICK else -1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1 if QUICK else 8,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=1 if QUICK else 2,
    save_strategy="no",
    bf16=use_bf16, fp16=use_fp16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=SEED,
    dataloader_pin_memory=False,
)
trainer = Trainer(model=model, args=cpt_args, train_dataset=train_ds, callbacks=[loss_cb])

model.train()
t0 = time.time()
train_result = trainer.train()
print(f"CPT finished in {(time.time() - t0) / 60:.1f} min, {trainer.state.global_step} optimizer steps")
''')

code(r'''
steps, losses = np.array(loss_cb.steps), np.array(loss_cb.losses)
win = min(5, len(losses))
smooth = np.convolve(losses, np.ones(win) / win, mode="valid")
sm_steps = steps[win - 1:]

plt.figure(figsize=(7, 3.6))
plt.plot(steps, losses, alpha=0.35, label="logged loss")
plt.plot(sm_steps, smooth, lw=2, label=f"moving average ({win})")
plt.xlabel("optimizer step"); plt.ylabel("training loss"); plt.title("CPT loss"); plt.legend(); plt.grid(alpha=0.3)
plt.show()

first, last = float(losses[0]), float(losses[-win:].mean())
plateau_step = int(sm_steps[np.argmax(smooth <= smooth[-1] * 1.02)])
pd.DataFrame({"loss_first_log": [first], "loss_last_avg": [last], "drop_%": [(1 - last / first) * 100],
              "plateau_step(within 2% of final)": [plateau_step], "total_steps": [int(steps[-1])]}).round(3)
''')

code(r'''
print(f"first logged loss {first:.2f}, last {last:.2f}")
if first > 8:
    print("starting loss is close to ln(49152) = 10.8 -> looks like random init, loading step is wrong")
else:
    print("starting loss is far below ln(49152) = 10.8, so the pretrained weights did load")
print(f"the curve is within 2% of its final value from about step {plateau_step} of {int(steps[-1])}")

model.config.use_cache = True
model.save_pretrained("cpt_checkpoint")
tok.save_pretrained("cpt_checkpoint")
pd.DataFrame({"step": steps, "loss": losses}).to_csv("cpt_loss_log.csv", index=False)
print("saved cpt_checkpoint/")

# same three domain prompts after CPT
model.eval()
after_cpt = {p: complete(model, p, GEN_TOK) for p in domain_prompts}
json.dump(after_cpt, open("cpt_outputs.json", "w"), indent=2)
cmp_df = pd.DataFrame({"prompt": domain_prompts,
                       "base model": [baseline[p] for p in domain_prompts],
                       "after CPT": [after_cpt[p] for p in domain_prompts]})
pd.set_option("display.max_colwidth", 400)
cmp_df
''')

md(r'''
**Inference (Step 4)**

- The brief's "~2 to 4" starting loss is a rule of thumb for a mid-sized model on general text. SmolLM2 is trained on a lot of code and technical web text, and this corpus is manual pages and shell listings, so a first-step loss near or below that range is normal. What matters is that it is nowhere near 10.8.
- The loss falls fastest in the first few optimizer steps (the model picks up the house style: option lists, `$` prompts, path names, RFC-style wording) and then flattens. The flat part is the point where more passes over the same ~1.1M training tokens stop helping and start to memorise, which is the main reason to keep the learning rate low and the epoch count small.
- The before/after table is a qualitative check only. Read it for: do the continuations now stay on Linux topics, use real paths and commands, and avoid wandering off to unrelated web-page text. CPT does **not** teach the model to answer questions, so it may still continue the prompt instead of answering it; that is exactly what Part B is for.
''')

# =====================================================================
md(r'''
## Step 5: evaluation

### 5A: domain perplexity

`PPL = exp(mean negative log-likelihood per token)` on the held-out packed sequences (10% of the blocks, never used in training). Same code, same data, for the base weights and the CPT checkpoint. I reload both from disk so nothing is left over from training state.
''')

code(r'''
del trainer, model
gc.collect()
if device == "cuda": torch.cuda.empty_cache()

def load_lm(path):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=load_dtype).to(device)
    m.config.use_cache = True
    return m.eval()

@torch.no_grad()
def perplexity(m, parquet, bs=2, limit=None):
    seqs = pd.read_parquet(parquet)["input_ids"].tolist()[:limit]
    nll, count = 0.0, 0
    for i in range(0, len(seqs), bs):
        x = torch.tensor(seqs[i:i + bs], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            logits = m(input_ids=x).logits
        nll += F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum").item()
        count += x[:, 1:].numel()
    return math.exp(nll / count), count

LIMIT = 4 if QUICK else None
base_model = load_lm(MODEL_ID)
cpt_model = load_lm("cpt_checkpoint")

ppl_base, n_eval_tok = perplexity(base_model, "eval_packed.parquet", limit=LIMIT)
ppl_cpt, _ = perplexity(cpt_model, "eval_packed.parquet", limit=LIMIT)
reduction = (ppl_base - ppl_cpt) / ppl_base * 100

pd.DataFrame({"model": ["base (before CPT)", "after CPT"], "domain perplexity": [ppl_base, ppl_cpt]}).round(3).assign(
    held_out_tokens=n_eval_tok)
''')

code(r'''
print(f"base PPL {ppl_base:.2f} -> CPT PPL {ppl_cpt:.2f}  (reduction {reduction:.1f}%)")
if reduction <= 0:
    print("perplexity did not improve - training or checkpoint problem")
elif reduction < 10:
    print("reduction is below the 10-40% the brief mentions: the adaptation is small (lr / steps too low, or the base model already knew this text well)")
elif reduction <= 40:
    print("reduction is inside the 10-40% band the brief mentions")
else:
    print("reduction is larger than the 10-40% band. The corpus is narrow and repetitive, so a big drop is plausible, "
          "but it is also what overfitting to the training blocks would look like, so check 5B before calling it a win")
''')

md(r'''
### 5B: catastrophic forgetting check

Three prompts with nothing to do with Linux. Greedy decoding, same settings for both models. The verdict is a crude keyword test on the CPT output (does it still contain the right answer); I also print the base result so a "Retained" verdict is only meaningful where the base model got it right.
''')

code(r'''
general = [
    ("The capital of France is",            ["paris"]),
    ("Water boils at",                      ["100", "212"]),
    ("The speed of light is approximately", ["299", "300,000", "300000", "186,000", "3 x 10", "3x10", "3 × 10", "3.0 x 10"]),
]

rows = []
for prompt, keys in general:
    b = complete(base_model, prompt, 20 if QUICK else 30)
    c = complete(cpt_model, prompt, 20 if QUICK else 30)
    ok = lambda t: any(k in t.lower() for k in keys)
    rows.append({"prompt": prompt, "base output": b.replace("\n", " / "), "CPT output": c.replace("\n", " / "),
                 "base correct": ok(b), "verdict": "Retained" if ok(c) else "Degraded"})
forget_df = pd.DataFrame(rows)
forget_df
''')

code(r'''
print(forget_df["verdict"].value_counts().to_dict())
if (forget_df["verdict"] == "Degraded").any():
    print("at least one general fact was lost -> try lr / 10 or half the steps, as the brief suggests")
else:
    print("all three general facts still come out right after CPT")

del base_model, cpt_model
gc.collect()
if device == "cuda": torch.cuda.empty_cache()
''')

md(r'''
**Inference (Step 5)**

- Perplexity is the cleaner signal here: it is computed on text the model never trained on, so a drop means the model has learned how this kind of documentation reads (command names, flags, directory names, sentence patterns), not just memorised the training blocks. The size of the drop should be read together with the training-loss curve: if train loss kept falling while held-out PPL stopped improving, the extra epochs were only memorisation.
- The held-out blocks come from the same seven manuals as the training blocks, so this measures in-distribution adaptation. Text from a different Linux source (a man page, a distro wiki) would probably show a smaller gain. I did not test that.
- The forgetting test has only three prompts and a keyword match, so it can catch obvious damage but not subtle loss. A "Retained" verdict says the model still completes well-known facts; it says nothing about reasoning or longer generations. With this little data and a 3e-5 learning rate, I would expect general facts to survive, and the table above shows whether that held.
''')

# =====================================================================
md(r'''
## Part B1: instruction dataset

**How the pairs were made.** Written per source document, not generated in bulk and not filled from templates. Each of the 7 documents was read in about ten spread-out passages, and 14 question/answer pairs per document were written from those passages only (a mix of factual, procedural and comparative questions), paraphrasing instead of copying. An LLM assistant did the drafting; the instruction it was given is below (condensed, but every rule in it was in the original). Afterwards all pairs were read through, a sample was checked against the PDFs, four responses that reused too many words from the source were reworded, and two near-duplicate topics across documents (shadow passwords, single vs double quotes) were dropped, leaving 96 pairs.
''')

code(r'''
PROMPT_USED = """Read the text passages below (from <document>) and write 14 instruction/response pairs in JSON
(keys: instruction, response, type, source), based ONLY on what the passages say.
About 5 factual, 5 procedural (how-to / commands / troubleshooting scenario) and 4 comparative (X vs Y, when to pick A over B).
Each response must be a concise 1-3 sentence answer in your own words. Do not copy text verbatim.
Each response must be at least 30 words.
Instructions must be self-contained (never say "according to the text").
Vary the phrasing of the instructions: no two may share their first three words, at most 3 of 14 may start with "What".
Write them like real questions a sysadmin or student would ask, not textbook headings."""
print(PROMPT_USED)
''')

code(r'''
if not Path("instruction_dataset.jsonl").exists() and "google.colab" in sys.modules:
    from google.colab import files
    files.upload()          # pick instruction_dataset.jsonl

pairs = [json.loads(l) for l in open("instruction_dataset.jsonl", encoding="utf-8")]
df = pd.DataFrame(pairs)
df["resp_words"] = df["response"].str.split().str.len()
df["n_sentences"] = df["response"].apply(lambda r: len(re.split(r"(?<=[.!?])\s+(?=[A-Z`$/])", r.strip())))
print(len(df), "pairs loaded")

# minimum response length filter (>= 30 words)
MIN_RESP_WORDS = 30
too_short = df[df["resp_words"] < MIN_RESP_WORDS]
df = df[df["resp_words"] >= MIN_RESP_WORDS].reset_index(drop=True)
print(f"removed by the >= {MIN_RESP_WORDS} word filter: {len(too_short)} | kept: {len(df)}")
print("exact duplicate instructions:", df["instruction"].duplicated().sum())
print("responses with more than 3 sentences (rough splitter):", (df["n_sentences"] > 3).sum())
print(df["resp_words"].describe().round(1).to_dict())
''')

code(r'''
# coverage: question types and source documents
print(pd.crosstab(df["source"], df["type"], margins=True))
''')

code(r'''
# template check: how often does one phrasing dominate? (first three words of the instruction)
stem = df["instruction"].str.lower().str.replace(r"[^a-z0-9' ]", "", regex=True).str.split().str[:3].str.join(" ")
share = (stem.value_counts() / len(df) * 100).round(1)
print(share.head(8).to_string())
print("\ntemplates above 20%:", list(share[share > 20].index) or "none")
print("distinct 3-word openers:", stem.nunique(), "of", len(df), "pairs")
first_word = df["instruction"].str.split().str[0].str.lower().value_counts(normalize=True).mul(100).round(1)
print("\nfirst word:", first_word.head(6).to_dict())
''')

code(r'''
# is any response lifted from the source? longest shared run of words with its own document
def toks(t): return re.findall(r"[a-z0-9$/._-]+", t.lower())
src_cache, worst = {}, []
for r in df.itertuples():
    if r.source not in src_cache:
        words = toks((CORPUS_DIR / f"{r.source}.txt").read_text(encoding="utf-8"))
        src_cache[r.source] = {tuple(words[i:i + 7]) for i in range(len(words) - 6)}
    rw = toks(r.response)
    worst.append(sum(tuple(rw[i:i + 7]) in src_cache[r.source] for i in range(len(rw) - 6)))
df["shared_7grams"] = worst
print("responses sharing any 7-word run with their source:", (df["shared_7grams"] > 0).sum(), "of", len(df))
print("max shared 7-grams in one response:", df["shared_7grams"].max(), "(response is ~45 words = ~39 possible)")
''')

code(r'''
# spot check: 12 random pairs
for r in df.sample(12, random_state=7).itertuples():
    print(f"[{r.source} | {r.type}]\nQ: {r.instruction}\nA: {r.response}\n")
''')

code(r'''
# 80 / 20 split, done per source document so every manual shows up in the eval set
n_eval = len(df) - round(0.8 * len(df))
size = df["source"].value_counts()
quota = (size * n_eval / len(df)).apply(math.floor)
leftover = n_eval - quota.sum()
for src in (size * n_eval / len(df) - quota).sort_values(ascending=False).index[:leftover]:
    quota[src] += 1

rng = random.Random(SEED)
eval_ids = []
for src, k in quota.items():
    ids = df.index[df["source"] == src].tolist()
    rng.shuffle(ids)
    eval_ids += ids[:k]
cols = ["instruction", "response", "type", "source"]
eval_pairs  = df.loc[eval_ids, cols].to_dict("records")
train_pairs = df.drop(index=eval_ids)[cols].sample(frac=1, random_state=SEED).to_dict("records")
for name, part in (("instruction_train.jsonl", train_pairs), ("instruction_eval.jsonl", eval_pairs)):
    with open(name, "w", encoding="utf-8") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("train:", len(train_pairs), "| eval:", len(eval_pairs))
print("eval set covers", len({r["source"] for r in eval_pairs}), "of", df["source"].nunique(), "source documents")
print("eval types:", collections.Counter(r["type"] for r in eval_pairs))
''')

md(f'''
**Inference (B1)**

- All {n("n_pairs", 96)} pairs survive the 30-word filter (responses are {n("min_words", 31)} to {n("max_words", 59)} words, 1 to 3 sentences). Types are spread roughly evenly between factual, procedural and comparative, and each of the 7 documents contributes 13 or 14 pairs, so no single manual dominates the instruction data even though one of them dominates the CPT corpus.
- Template check: the most common three-word opening ("show how to") is used by about 4% of the pairs, and there are 86 distinct openers among 96 pairs. The most common single first word is "How" at about 18%, the only thing near the 20% line, and it is spread over many different continuations ("how can I", "how do I", "how does", ...). Scenario-style openers ("My script...", "A colleague just left...") add further variety, so the model cannot get away with learning one or two question shapes.
- Copy check: almost no response shares even a 7-word run with its source document, and the few that do are command fragments or short standard phrases. Responses are paraphrases, as the brief asks.
- Honest limitation: 96 pairs (77 for training) is tiny. It is enough to teach the *format* (answer the question, briefly, then stop) and a little of the content, but a 360M model will not reliably learn new facts from 77 examples. The 19 evaluation pairs are too few for fine-grained conclusions.
''')

# =====================================================================
md(r'''
## Part B2: QLoRA fine-tuning

The base model has no chat template, so I define a ChatML-style one (`<|im_start|>` / `<|im_end|>` already exist in SmolLM2's vocabulary, ids 1 and 2). Every example is rendered as system + user + assistant, and the loss is computed only on the assistant part (completion-only collator), so the model is not trained to reproduce the question.

The adapter I train is **Adapter B (balanced)**: r = 16, alpha = 32, target modules `q_proj`, `v_proj`. Switch `ADAPTER` to try A or C.
''')

code(r'''
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

ADAPTERS = {
    "A": dict(r=8,  lora_alpha=16, target_modules=["q_proj", "v_proj"]),
    "B": dict(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]),
    "C": dict(r=32, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj"]),
}
ADAPTER = "B"

tok = AutoTokenizer.from_pretrained("cpt_checkpoint")       # identical tokenizer, loaded from the CPT checkpoint
tok.pad_token = tok.eos_token
tok.padding_side = "right"
tok.chat_template = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
SYSTEM = "You are a helpful assistant for Linux system administration."

def render(ex):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}

sft_train = Dataset.from_list(train_pairs).map(render)
sft_eval  = Dataset.from_list(eval_pairs).map(render)
print(sft_train[0]["text"])
print("token lengths (max):", max(len(tok(t, add_special_tokens=False)["input_ids"]) for t in sft_train["text"]))
''')

code(r'''
if device == "cuda":
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16 if native_bf16 else torch.float16)
    qmodel = AutoModelForCausalLM.from_pretrained("cpt_checkpoint", quantization_config=bnb,
                                                  torch_dtype=torch.bfloat16 if native_bf16 else torch.float16,
                                                  device_map={"": 0})
    qmodel = prepare_model_for_kbit_training(qmodel, use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={"use_reentrant": False})
else:
    # no CUDA = no bitsandbytes 4-bit. This branch only exists so the rest of the code can be dry-run on a laptop.
    print("NO CUDA: loading in full precision, this is NOT QLoRA, dry run only")
    qmodel = AutoModelForCausalLM.from_pretrained("cpt_checkpoint", torch_dtype=torch.float32).to(device)
    qmodel.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    qmodel.enable_input_require_grads()

lora_cfg = LoraConfig(lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", **ADAPTERS[ADAPTER])
qmodel = get_peft_model(qmodel, lora_cfg)
qmodel.print_trainable_parameters()
if device == "cuda":
    print("quantised 4-bit layers:", sum(1 for m in qmodel.modules() if m.__class__.__name__ == "Linear4bit"))
    print("GPU memory after load: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))
''')

code(r'''
collator = DataCollatorForCompletionOnlyLM(
    response_template=tok.encode("<|im_start|>assistant\n", add_special_tokens=False), tokenizer=tok)

sft_args = SFTConfig(
    output_dir="sft_runs",
    num_train_epochs=4,
    max_steps=3 if QUICK else -1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.0,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=use_fp16, bf16=use_bf16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=512,
    dataset_text_field="text",
    packing=False,
    dataset_kwargs={"add_special_tokens": False},
    report_to="none",
    seed=SEED,
    dataloader_pin_memory=False,
)
sft = SFTTrainer(model=qmodel, args=sft_args, train_dataset=sft_train, eval_dataset=sft_eval,
                 data_collator=collator, processing_class=tok)

# check the collator really masks the prompt: only answer tokens should keep a label
ex0 = sft.train_dataset[0]
b = collator([{"input_ids": ex0["input_ids"], "attention_mask": ex0["attention_mask"]}])
kept = int((b["labels"][0] != -100).sum())
print(f"example 0: {b['input_ids'].shape[1]} tokens, {kept} of them carry loss (the assistant answer + <|im_end|>)")
assert 0 < kept < b["input_ids"].shape[1] - 10

t0 = time.time()
sft.train()
print(f"QLoRA ({ADAPTER}) finished in {(time.time() - t0) / 60:.1f} min")
''')

code(r'''
hist = pd.DataFrame(sft.state.log_history)
tr = hist[hist["loss"].notna()][["step", "loss"]] if "loss" in hist else pd.DataFrame()
ev = hist[hist["eval_loss"].notna()][["epoch", "eval_loss"]] if "eval_loss" in hist else pd.DataFrame()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(tr["step"], tr["loss"]); ax[0].set_title(f"SFT train loss (adapter {ADAPTER})"); ax[0].set_xlabel("step"); ax[0].grid(alpha=.3)
ax[1].plot(ev["epoch"], ev["eval_loss"], marker="o"); ax[1].set_title("eval loss per epoch"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
print(ev.round(3).to_string(index=False))

sft.model.save_pretrained(f"qlora_adapter_{ADAPTER}")
print("adapter saved to", f"qlora_adapter_{ADAPTER}")
if device == "cuda": print("peak GPU memory: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
''')

md(r'''
**Inference (B2)**

- Only the LoRA matrices are trained: that is a small fraction of a percent of the parameters (the count is printed above), on top of a frozen 4-bit copy of the CPT model. That is why the memory footprint is a lot lower than the full fine-tune in Part A.
- Read the two curves together. Training loss should keep dropping over the four epochs; the evaluation loss shows whether that is generalising. With only 77 training examples, if eval loss bottoms out after epoch 1 or 2 and climbs again, the later epochs are just memorising the training answers, and I would stop earlier or lower the rank/learning rate.
- Expected trade-off between the three adapters from the brief: A (r=8) is cheapest but can underfit; C (r=32, plus `o_proj`) has the most capacity and the highest VRAM use; B is the middle option, which is why I picked it. With a dataset this small, extra capacity is unlikely to help much, so I did not run the others.
''')

# =====================================================================
md(r'''
## Part B3: evaluation of the tuned model

Same three domain prompts as in Step 3, now asked through the chat template, next to what the base and the CPT model said.
''')

code(r'''
def chat(m, question, max_new_tokens=160):
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                                     tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(m.device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1,
                         eos_token_id=[IM_END, EOS], pad_token_id=EOS, use_cache=True)
    gen = out[0, enc["input_ids"].shape[1]:]
    stopped = bool((gen == IM_END).any() or (gen == EOS).any())
    return tok.decode(gen, skip_special_tokens=True).strip(), stopped

sft.model.eval()
sft.model.config.use_cache = True
MAXNEW = 40 if QUICK else 160
tuned = {p: chat(sft.model, p, MAXNEW) for p in domain_prompts}
b3 = pd.DataFrame({"prompt": domain_prompts,
                   "base model": [baseline[p].replace("\n", " / ") for p in domain_prompts],
                   "after CPT": [after_cpt[p].replace("\n", " / ") for p in domain_prompts],
                   "CPT + QLoRA": [tuned[p][0] for p in domain_prompts]})
pd.set_option("display.max_colwidth", 500)
b3
''')

code(r'''
# a bit more evidence than three prompts: token-level F1 against the reference answer on the held-out pairs
def f1(pred, ref):
    p, r = toks(pred), toks(ref)
    common = sum((collections.Counter(p) & collections.Counter(r)).values())
    if not common: return 0.0
    pr, rc = common / len(p), common / len(r)
    return 2 * pr * rc / (pr + rc)

rows = []
for ex in (eval_pairs[:3] if QUICK else eval_pairs):
    ans, stopped = chat(sft.model, ex["instruction"], MAXNEW)
    rows.append({"instruction": ex["instruction"], "model answer": ans, "F1 vs reference": f1(ans, ex["response"]),
                 "words": len(ans.split()), "ended cleanly": stopped})
ev_df = pd.DataFrame(rows)
print(f"mean token-F1 vs reference: {ev_df['F1 vs reference'].mean():.3f} | mean answer length: {ev_df['words'].mean():.0f} words "
      f"| ended on its own: {ev_df['ended cleanly'].mean() * 100:.0f}%")
ev_df.head(4)
''')

code(r'''
# where the tuned model answers in a sensible shape, and where it does not
print("average F1 of the 3 best vs 3 worst:",
      ev_df.nlargest(3, "F1 vs reference")["F1 vs reference"].mean().round(2), "/",
      ev_df.nsmallest(3, "F1 vs reference")["F1 vs reference"].mean().round(2))
for r in ev_df.nsmallest(2, "F1 vs reference").itertuples():
    print("\nweak example:", r.instruction, "\n  ->", r._2)
''')

md(r'''
**Observations (B3)**

The table above is the main evidence; what to look for, and how I read it:

- **Behaviour change.** The base and CPT models continue the prompt, the tuned model answers it and stops by itself (see the "ended on its own" share). That change of behaviour is the real result of instruction tuning; CPT alone cannot give it.
- **Content.** The answers should now use the right commands and paths for the three prompts (`df -h`, logs/spool/cache under `/var`, `$?`). Where the model gets a detail wrong, it is usually in the specifics (an option name or a directory) rather than the overall shape, which is what you would expect from a 360M model trained on 77 examples.
- **Overlap with the reference answers.** Token-F1 is a blunt tool: a correct paraphrase scores low, and an answer that reuses the question's words scores higher than it deserves. I only use it to compare runs, not as an accuracy number.
- **Limits.** Small model, small dataset, one adapter, greedy decoding, no human grading. The observations are about style and format much more than factual reliability. Running adapters A and C, more pairs, and a few more held-out prompts would be the next steps.
''')

code(r'''
# bundle the small deliverables so they are easy to download from Colab
import shutil
if "google.colab" in sys.modules:
    for folder in ("domain_corpus",):
        shutil.make_archive(folder, "zip", folder)
    print("zipped domain_corpus.zip - download it with instruction_dataset.jsonl and this notebook")
''')

# =====================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "accelerator": "GPU",
               "colab": {"provenance": [], "gpuType": "T4"}}
out = ROOT / "Assignment1A_Linux_SysAdmin_LLM.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
