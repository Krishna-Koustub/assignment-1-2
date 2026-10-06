import re, hashlib, json, zlib, collections
from pathlib import Path
import pypdfium2 as pdfium
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
ROOT = Path(__file__).resolve().parent.parent
PDF_DIR, RAW_DIR, OUT_DIR = ROOT / "raw_pdfs", ROOT / "raw_text", ROOT / "domain_corpus"
RAW_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

# ---------- extraction: page by page ----------
pages = {}   # doc name -> list of page strings
for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    pdf = pdfium.PdfDocument(str(pdf_path))
    pages[pdf_path.stem] = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    (RAW_DIR / f"{pdf_path.stem}.txt").write_text("\n\f\n".join(pages[pdf_path.stem]), encoding="utf-8")

def n_docs(d):  return sum(1 for v in d.values() if v)
def n_pages(d): return sum(len(v) for v in d.values())
def n_words(d): return sum(len(p.split()) for v in d.values() for p in v)

log = [("raw extraction", n_docs(pages), n_pages(pages), n_words(pages))]

# ---------- light normalisation (not a filter) ----------
def tidy(t):
    t = t.replace("\r", "")
    t = re.sub(r"-\n(?=[a-z])", "", t)          # hyphen at line break
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# running headers/footers: a line that shows up on a big share of a doc's pages
def strip_running_lines(doc_pages):
    def norm(l): return re.sub(r"\d+", "#", l.strip())
    c = collections.Counter()
    for p in doc_pages:
        c.update({norm(l) for l in p.splitlines() if l.strip()})
    cutoff = max(8, int(0.15 * len(doc_pages)))
    common = {k for k, v in c.items() if v >= cutoff and len(k) < 90}
    return ["\n".join(l for l in p.splitlines() if norm(l) not in common) for p in doc_pages]

cleaned = {k: [tidy(p) for p in strip_running_lines(v)] for k, v in pages.items()}
log.append(("header/footer + whitespace tidy", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

# ---------- 1. length filter ----------
MIN_WORDS = 60
cleaned = {k: [p for p in v if len(p.split()) >= MIN_WORDS] for k, v in cleaned.items()}
log.append((f"length filter (>= {MIN_WORDS} words/page)", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

# ---------- 2. dedup ----------
seen, deduped = set(), {}
for k, v in cleaned.items():
    keep = []
    for p in v:
        h = hashlib.md5(re.sub(r"\W+", "", p.lower()).encode()).hexdigest()
        if h not in seen:
            seen.add(h); keep.append(p)
    deduped[k] = keep
cleaned = deduped
log.append(("exact page dedup", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

# near-duplicate pages: most of the page's 6-word shingles were already seen in an earlier kept page
def shingles(p, n=6):
    w = re.findall(r"\w+", p.lower())
    return {zlib.crc32(" ".join(w[i:i+n]).encode()) for i in range(max(1, len(w) - n + 1))}

seen_sh, deduped, near_dups = set(), {}, []
for k, v in cleaned.items():
    keep = []
    for i, p in enumerate(v):
        sh = shingles(p)
        overlap = len(sh & seen_sh) / len(sh)
        if overlap >= 0.6:
            near_dups.append((k, i, round(overlap, 2)))
            continue
        seen_sh |= sh; keep.append(p)
    deduped[k] = keep
cleaned = deduped
log.append(("near-duplicate page dedup (>=60% shingle overlap)", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

# ---------- 3. language filter ----------
def is_english(p):
    try:
        return detect(p[:1500]) == "en"
    except Exception:
        return False
dropped_lang = []
new = {}
for k, v in cleaned.items():
    keep = []
    for i, p in enumerate(v):
        if is_english(p): keep.append(p)
        else: dropped_lang.append((k, i, p[:200]))
    new[k] = keep
cleaned = new
log.append(("language filter (English only)", n_docs(cleaned), n_pages(cleaned), n_words(cleaned)))

for name, pgs in cleaned.items():
    (OUT_DIR / f"{name}.txt").write_text("\n\n".join(pgs), encoding="utf-8")

for row in log: print(row)
print(len(dropped_lang))
for d in dropped_lang[:25]: print(d[0], d[1], repr(d[2][:120]))
