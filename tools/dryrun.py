import json, os, sys, copy
import nbformat as nbf
from nbclient import NotebookClient
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbf.read(os.path.join(ROOT, "Assignment1A_Linux_SysAdmin_LLM.ipynb"), as_version=4)
extra = r'''
import json
L = log_df
print("NUMBERS_JSON=" + json.dumps({
 "raw_pages": f"{int(L.pages[0]):,}", "raw_words": f"{int(L.words[0]):,}",
 "final_pages": f"{int(L.pages.iloc[-1]):,}", "final_words": f"{int(L.words.iloc[-1]):,}",
 "pct_words_removed": round((1 - L.words.iloc[-1] / L.words[0]) * 100, 1),
 "tidy_words": f"{int(L.words_removed[1]):,}", "len_pages": int(L.pages_removed[2]),
 "dup_pages": int(L.pages_removed[4]), "dup_words": f"{int(L.words_removed[4]):,}", "lang_pages": int(L.pages_removed[5]),
 "total_tokens": f"{total_tokens:,}", "n_docs": len(docs), "avg_doc_tokens": round(total_tokens / len(docs)),
 "tok_per_word": round(total_tokens / sum(len(d["text"].split()) for d in docs), 2),
 "train_seqs": len(train_arr), "eval_seqs": len(eval_arr), "params": f"{total:,}",
 "n_pairs": len(df), "min_words": int(df.resp_words.min()), "max_words": int(df.resp_words.max())}))
'''
nb.cells.append(nbf.v4.new_code_cell(extra))
os.environ["A1_QUICK"] = "1"
client = NotebookClient(nb, kernel_name=sys.argv[1] if len(sys.argv) > 1 else "python3", timeout=3600, allow_errors=True,
                        resources={"metadata": {"path": ROOT}})
client.execute()
out = os.path.join("/tmp", "dryrun_executed.ipynb")
nbf.write(nb, out)
bad = [(i, c.source[:60].replace("\n", " ")) for i, c in enumerate(nb.cells) if c.cell_type == "code" and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
print("cells with errors:", bad)
for c in nb.cells[-1:]:
    for o in c.get("outputs", []):
        t = o.get("text", "")
        if "NUMBERS_JSON=" in t:
            json.dump(json.loads(t.split("NUMBERS_JSON=")[1]), open(os.path.join(ROOT, "tools", "numbers.json"), "w"), indent=1)
            print("numbers saved")
