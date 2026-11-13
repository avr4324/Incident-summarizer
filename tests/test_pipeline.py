from incident_summarizer.retrieval import BM25Retriever
from incident_summarizer.summarizer import IncidentSummarizer
from incident_summarizer import evaluate

R = BM25Retriever.from_file("data/incidents.jsonl")

def test_retrieval_top1():
    assert R.search("SSL handshake failure certificate expired")[0]["id"] == "INC-004"

def test_service_filter():
    assert all(h["service"] == "k8s-prod" for h in R.search("pods crash", service="k8s-prod"))

def test_summary_cites_retrieved():
    out = IncidentSummarizer(R, use_llm=False).summarize("redis failover users logged out")
    assert "INC-007" in out["summary"]

def test_no_hits():
    out = IncidentSummarizer(R, use_llm=False).summarize("zzzz qqqq")
    assert "manual triage" in out["summary"]

def test_eval_thresholds():
    rep = evaluate.run()
    assert rep["recall@3"] >= 0.9 and rep["MRR"] >= 0.8 and rep["citation_validity"] == 1.0
