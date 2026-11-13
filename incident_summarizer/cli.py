import argparse, sys
from pathlib import Path
from .retrieval import BM25Retriever
from .summarizer import IncidentSummarizer
from . import evaluate

def main():
    p = argparse.ArgumentParser(prog="incident-summarizer")
    p.add_argument("--data", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize"); s.add_argument("text", nargs="?", help="incident text (or stdin)")
    s.add_argument("--k", type=int, default=3); s.add_argument("--service"); s.add_argument("--llm", action="store_true")
    sub.add_parser("eval")
    a = p.parse_args()
    if a.cmd == "eval":
        evaluate.main_report = evaluate.run(a.data)
        print({k: v for k, v in evaluate.main_report.items() if k != "per_case"})
        return
    text = a.text or sys.stdin.read()
    retr = BM25Retriever.from_file(Path(a.data) / "incidents.jsonl")
    out = IncidentSummarizer(retr, k=a.k, use_llm=a.llm or None).summarize(text, a.service)
    print(out["summary"])

if __name__ == "__main__":
    main()
