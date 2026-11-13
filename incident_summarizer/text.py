import re

STOP = set("""a an the and or of to in on at for with by from is are was were be been it this that as after
before into over under again while during than then so if not no we you they he she its their our""".split())

def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9_.\-]*", text.lower())
    out = []
    for t in toks:
        t = t.strip(".-_")
        if t and t not in STOP:
            out.append(t)
            # also index sub-parts of hyphenated/dotted terms
            for p in re.split(r"[-_.]", t):
                if p and p != t and p not in STOP:
                    out.append(p)
    return out
