"""Pure-python BM25 retriever over past incidents (no external deps)."""
import json, math
from collections import Counter
from pathlib import Path
from .text import tokenize

def load_incidents(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def doc_text(inc: dict) -> str:
    # title and symptoms weighted by repetition
    return " ".join([inc["title"]] * 2 + [inc["service"], inc["symptoms"] , inc["symptoms"], inc["root_cause"]])

class BM25Retriever:
    def __init__(self, incidents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.incidents, self.k1, self.b = incidents, k1, b
        self.docs = [Counter(tokenize(doc_text(i))) for i in incidents]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @classmethod
    def from_file(cls, path):
        return cls(load_incidents(path))

    def score(self, q_tokens, idx) -> float:
        d, L, s = self.docs[idx], self.lens[idx], 0.0
        for t in q_tokens:
            if t in d:
                f = d[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
        return s

    def search(self, query: str, k: int = 3, service: str | None = None) -> list[dict]:
        q = tokenize(query)
        scored = []
        for i, inc in enumerate(self.incidents):
            if service and inc["service"] != service:
                continue
            sc = self.score(q, i)
            if sc > 0:
                scored.append((sc, inc))
        scored.sort(key=lambda x: -x[0])
        return [{**inc, "score": round(sc, 4)} for sc, inc in scored[:k]]
