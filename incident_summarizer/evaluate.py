"""Evaluation: retrieval (recall@k, MRR) and summary quality (coverage, groundedness, citation validity)."""
import json, re, sys
from pathlib import Path
from .retrieval import BM25Retriever
from .summarizer import IncidentSummarizer
from .text import tokenize

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)

def reciprocal_rank(ranked, relevant):
    for i, r in enumerate(ranked, 1):
        if r in relevant:
            return 1 / i
    return 0.0

def coverage(summary, key_terms):
    s = summary.lower()
    return sum(t in s for t in key_terms) / len(key_terms)

def groundedness(summary, query, hits):
    """Share of summary content tokens that appear in the incident text or retrieved context."""
    ctx = set(tokenize(query))
    for h in hits:
        ctx |= set(tokenize(" ".join(str(v) for v in h.values())))
    toks = [t for t in tokenize(re.sub(r"\[INC-\d+\]", "", summary))
            if t not in {"summary", "likely", "cause", "recommended", "actions", "related", "incidents", "score"}]
    toks = [t for t in toks if not t.isdigit()]
    return sum(t in ctx for t in toks) / max(len(toks), 1)

def citations_valid(summary, hits):
    cited = set(re.findall(r"INC-\d+", summary))
    ids = {h["id"] for h in hits}
    return 1.0 if cited <= ids else 0.0

def run(data_dir="data", k=3, use_llm=False):
    data = Path(data_dir)
    retr = BM25Retriever.from_file(data / "incidents.jsonl")
    summ = IncidentSummarizer(retr, k=k, use_llm=use_llm)
    cases = [json.loads(l) for l in open(data / "eval_set.jsonl") if l.strip()]
    rows = []
    for c in cases:
        out = summ.summarize(c["query"])
        ranked = [h["id"] for h in out["retrieved"]]
        rows.append({
            "query": c["query"][:60],
            "recall@k": recall_at_k(ranked, c["relevant_ids"], k),
            "rr": reciprocal_rank(ranked, c["relevant_ids"]),
            "coverage": coverage(out["summary"], c["key_terms"]),
            "grounded": groundedness(out["summary"], c["query"], out["retrieved"]),
            "cite_ok": citations_valid(out["summary"], out["retrieved"]),
        })
    avg = lambda key: round(sum(r[key] for r in rows) / len(rows), 3)
    report = {"n": len(rows), "k": k, f"recall@{k}": avg("recall@k"), "MRR": avg("rr"),
              "coverage": avg("coverage"), "groundedness": avg("grounded"), "citation_validity": avg("cite_ok"),
              "per_case": rows}
    return report

if __name__ == "__main__":
    rep = run()
    for r in rep["per_case"]:
        print(f"{r['query']:<62} R@k={r['recall@k']:.2f} RR={r['rr']:.2f} cov={r['coverage']:.2f} gr={r['grounded']:.2f}")
    print({k: v for k, v in rep.items() if k != "per_case"})
