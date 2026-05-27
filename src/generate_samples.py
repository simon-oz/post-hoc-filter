import json
import ir_datasets
from collections import defaultdict

# -------------------------------
# Choose the correct Touché 2022 dataset ID
# -------------------------------
# Examples (uncomment one):
# ds_id = "argsme/2020-04-01/processed/touche-2022-task-1"           # Task 1
# ds_id = "clueweb12/touche-2022-task-2"                             # Task 2 (base)
# ds_id = "clueweb12/touche-2022-task-2/expanded-doc-t5-query"       # Task 2 (variant)
# ds_id = "touche-image/2022-06-13/touche-2022-task-3"               # Task 3

ds_id = "argsme/2020-04-01/processed/touche-2022-task-1"  # <-- pick your task

dataset = ir_datasets.load(ds_id)

# -------------------------------
# Collect qrels first (and the set of doc_ids we need)
# -------------------------------
relevance_map = {0: "I", 1: "P", 2: "R"}
query_doc_rel = defaultdict(list)
needed_doc_ids = set()

for qrel in dataset.qrels_iter():
    qid = qrel.query_id
    doc_id = qrel.doc_id
    rel = relevance_map.get(getattr(qrel, "relevance", 0), "I")
    query_doc_rel[qid].append({"doc_id": doc_id, "reference answer": rel})
    needed_doc_ids.add(doc_id)

print(f"📥 Loaded relevance judgments for {len(query_doc_rel)} queries")
print(f"📥 Unique judged documents: {len(needed_doc_ids)}")

# -------------------------------
# Map queries (robust field access)
# -------------------------------
query_text_map = {}
for q in dataset.queries_iter():
    text = getattr(q, "text", None) or getattr(q, "query", None) or getattr(q, "title", "")
    query_text_map[q.query_id] = text

print(f"📥 Loaded {len(query_text_map)} queries")

# -------------------------------
# Map only the documents referenced by qrels
# -------------------------------
doc_map = {}
for d in dataset.docs_iter():
    if d.doc_id in needed_doc_ids:
        title = getattr(d, "title", "")
        snippet = (
            getattr(d, "abstract", None)
            or getattr(d, "summary", None)
            or getattr(d, "snippet", None)  # some datasets provide this
            or getattr(d, "text", "")       # general fallback
        )
        if snippet and len(snippet) > 600:
            snippet = snippet[:600] + "…"

        # Common URL-ish fields across Touché datasets
        url = (
            getattr(d, "url", None)
            or getattr(d, "source_url", None)
            or getattr(d, "chatnoir_url", "")
        )
        doc_map[d.doc_id] = {"title": title, "snippet": snippet, "url": url}

missing_docs = needed_doc_ids - doc_map.keys()
if missing_docs:
    print(f"⚠️ Warning: {len(missing_docs)} judged doc_ids were not found in docs_iter().")

print(f"📥 Loaded {len(doc_map)} documents used in qrels")

# -------------------------------
# Generate final JSON
# -------------------------------
final_data = []

for qid, docs in sorted(query_doc_rel.items()):
    results = []
    for doc in docs:
        doc_id = doc["doc_id"]
        doc_info = doc_map.get(doc_id, {"title": "", "snippet": "", "url": ""})
        results.append({
            "title": doc_info["title"],
            "snippet": doc_info["snippet"],
            "reference answer": doc["reference answer"],
            "url": doc_info["url"],
        })
    final_data.append({
        "query": query_text_map.get(qid, ""),
        "results": results,
    })

# -------------------------------
# Save JSON
# -------------------------------
output_file = "touche2022_samples.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON output saved to {output_file}")