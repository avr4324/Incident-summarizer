"""Summarizer: grounded in the incoming incident text + retrieved past incidents.
Offline rule-based mode by default; optional Claude mode if ANTHROPIC_API_KEY is set."""
import os, re
from .retrieval import BM25Retriever

ERR_RE = re.compile(r"(error|exception|fail|timeout|timed out|oom|evict|crash|expired|lag|5\d\d)", re.I)

def key_signals(text: str, limit: int = 4) -> list[str]:
    parts = [p.strip() for p in re.split(r"[\n.;]+", text) if p.strip()]
    sig = [p for p in parts if ERR_RE.search(p)]
    return (sig or parts)[:limit]

def build_prompt(text: str, hits: list[dict]) -> str:
    ctx = "\n".join(
        f"[{h['id']}] {h['title']} | cause: {h['root_cause']} | fix: {h['resolution']}" for h in hits)
    return (
        "You are an SRE assistant. Summarize the NEW incident using ONLY the incident text and the "
        "similar past incidents below. Cite past incidents as [INC-xxx]. If evidence is weak, say so.\n"
        "Output sections: Summary, Likely cause, Recommended actions, Related incidents.\n\n"
        f"NEW INCIDENT:\n{text}\n\nSIMILAR PAST INCIDENTS:\n{ctx}")

def _llm_summary(text, hits):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
                                 max_tokens=700, messages=[{"role": "user", "content": build_prompt(text, hits)}])
    return msg.content[0].text

def _rule_summary(text, hits):
    sig = key_signals(text)
    lines = ["Summary: " + "; ".join(sig) + "."]
    if not hits:
        lines.append("Likely cause: no similar past incident found; manual triage needed.")
        return "\n".join(lines)
    top = hits[0]
    lines.append(f"Likely cause: similar to {top['id']} ({top['title']}): {top['root_cause']} [{top['id']}].")
    acts = []
    for h in hits[:2]:
        acts.append(f"- {h['resolution']} [{h['id']}]")
    lines.append("Recommended actions:\n" + "\n".join(acts))
    lines.append("Related incidents: " + ", ".join(f"{h['id']} (score {h['score']})" for h in hits))
    return "\n".join(lines)

class IncidentSummarizer:
    def __init__(self, retriever: BM25Retriever, k: int = 3, use_llm: bool | None = None):
        self.retriever, self.k = retriever, k
        self.use_llm = bool(os.getenv("ANTHROPIC_API_KEY")) if use_llm is None else use_llm

    def summarize(self, text: str, service: str | None = None) -> dict:
        hits = self.retriever.search(text, self.k, service)
        try:
            summary = _llm_summary(text, hits) if self.use_llm else _rule_summary(text, hits)
        except Exception as e:  # fall back safely
            summary = _rule_summary(text, hits) + f"\n(note: LLM unavailable, used rule-based mode: {type(e).__name__})"
        return {"summary": summary, "retrieved": hits}
