# Incident Summarizer with Retrieval & Evaluation

Given a new incident (alerts, logs, description), it retrieves similar past incidents (BM25),
writes a grounded summary with cited past incidents, and is scored by an evaluation harness.

## Step by step
1. **Setup**: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. **Data**: past incidents live in `data/incidents.jsonl` (fields: id, title, service, severity, symptoms, root_cause, resolution, impact). Replace/extend with your own postmortems.
3. **Retrieval**: `incident_summarizer/retrieval.py` builds a BM25 index over title, service, symptoms, root cause. Supports a `service` filter.
4. **Summarize**: `python -m incident_summarizer.cli summarize "checkout-api 500s after deploy, NullPointerException"`
   - Default is offline and rule-based (cites retrieved incidents).
   - For Claude: `pip install anthropic`, `export ANTHROPIC_API_KEY=...`, add `--llm`. It falls back to rule-based on any error.
5. **Evaluate**: `python -m incident_summarizer.evaluate` uses `data/eval_set.jsonl`. Metrics: recall@k, MRR, key-term coverage, groundedness, citation validity.
6. **Test**: `python -m pytest -q` (includes metric thresholds as a regression gate).
7. **Improve loop**: add failing queries to `eval_set.jsonl`, tune k / BM25 params / prompt, re-run step 5.
8. **Production**: swap BM25 for embeddings + a vector DB if needed (keep the `search()` interface), wrap `IncidentSummarizer.summarize` in FastAPI, run step 6 in CI.
