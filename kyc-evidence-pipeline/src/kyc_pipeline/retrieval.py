from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from .extraction import OpenAICompatible, ProviderError

STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "how", "in", "is", "it", "its", "much", "of", "on", "or", "report", "reported", "reports", "s", "the", "this", "to", "was", "were", "what", "which", "who", "with", "year", "end"}
NORMALIZE = {"ownership": "own", "owned": "own", "owns": "own", "owner": "own", "owners": "own", "owning": "own", "repurchased": "repurchase", "repurchases": "repurchase", "repurchasing": "repurchase", "revenues": "revenue", "proceedings": "proceeding", "investigations": "investigation"}
SYNONYMS = {"control": ["voting", "rights", "votes"], "own": ["stake", "equity", "shareholder", "shares", "holding", "interest"], "repurchase": ["buyback"], "lawsuit": ["litigation", "proceedings", "court"], "regulatory": ["investigation", "enforcement"]}


def _tokens(text: str) -> list[str]:
    result = []
    for t in re.findall(r"[^\W_]+", text.casefold(), re.UNICODE):
        t = NORMALIZE.get(t, t)
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        if t not in STOP:
            result.append(t)
    return result


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def search_chunks(chunks: list[dict], query: str, top_k: int = 5, *, provider: OpenAICompatible | None = None, parent_limit: int = 4000) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Query must not be blank")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    query_tokens = _tokens(query)
    for token in list(query_tokens):
        for expanded in SYNONYMS.get(token, []):
            query_tokens.extend(_tokens(expanded))
    tokenized = [_tokens((c.get("table_header") or "") + " " + c["text"] + " " + " ".join(ref.get("text", "") for ref in c.get("context_refs", []))) for c in chunks]
    n = len(chunks)
    avg_len = sum(len(t) for t in tokenized) / max(n, 1)
    df = Counter(t for toks in tokenized for t in set(toks))
    lex = []
    for toks in tokenized:
        tf = Counter(toks)
        score = 0.0
        for term in set(query_tokens):
            frequency = tf[term]
            if not frequency:
                continue
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = frequency + 1.2 * (1 - 0.75 + 0.75 * len(toks) / max(avg_len, 1))
            score += idf * frequency * 2.2 / denom
        if chunks[len(lex)].get("kind") == "heading":
            score *= 0.72
        lex.append(score)
    warnings: list[str] = []
    dense: list[float | None] = [None] * n
    if provider is not None:
        compatible = [i for i, c in enumerate(chunks) if c.get("embedding") and c.get("embedding_model") == provider.embedding_model]
        if not compatible:
            warnings.append("No stored embeddings match the requested model; lexical search was used.")
        else:
            try:
                qvec = provider.embed_query(query) if hasattr(provider, "embed_query") else provider.embed([query])[0]
                for i in compatible:
                    if len(chunks[i]["embedding"]) == len(qvec):
                        dense[i] = _cosine(qvec, chunks[i]["embedding"])
                    else:
                        warnings.append("At least one stored vector had a mismatched dimension and was skipped.")
            except ProviderError:
                warnings.append("Query embedding failed; lexical search was used.")
    max_lex = max(lex, default=0.0)
    scored = []
    for i, chunk in enumerate(chunks):
        normalized_lex = lex[i] / max_lex if max_lex else 0.0
        # Missing vectors get lexical-only scores, not fabricated dense values.
        if dense[i] is None:
            score = normalized_lex
        else:
            score = 0.55 * normalized_lex + 0.45 * max(0.0, dense[i])
        if score > 0:
            scored.append((score, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    siblings: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        siblings[c["section_id"]].append(c)
    results = []
    for score, i in scored[:top_k]:
        c = chunks[i]
        neighbors = siblings[c["section_id"]]
        center = next(pos for pos, other in enumerate(neighbors) if other["chunk_id"] == c["chunk_id"])
        chosen = {center}
        length = min(len(c["text"]), parent_limit)
        distance = 1
        while length < parent_limit and (center - distance >= 0 or center + distance < len(neighbors)):
            for pos in (center - distance, center + distance):
                if 0 <= pos < len(neighbors) and length + len(neighbors[pos]["text"]) + 2 <= parent_limit:
                    chosen.add(pos)
                    length += len(neighbors[pos]["text"]) + 2
            distance += 1
        parent_units = []
        for pos in sorted(chosen):
            other = neighbors[pos]
            visible = other["text"][:parent_limit]
            parent_units.append({**{k: other.get(k) for k in ("chunk_id", "document_id", "segment_id", "page", "locator", "start")}, "end": other["start"] + len(visible), "text": visible, "text_truncated": len(visible) < len(other["text"])})
        parts = [unit["text"] for unit in parent_units]
        truncated = len(chosen) != len(neighbors) or any(u["text_truncated"] for u in parent_units)
        results.append({
            "score": round(score, 6), "lexical_score": round(lex[i], 6), "dense_cosine": round(dense[i], 6) if dense[i] is not None else None,
            "child": {k: c.get(k) for k in ("chunk_id", "document_id", "segment_id", "section_id", "section_path", "page", "locator", "start", "end", "text", "table_header", "footnote_context", "context_refs", "section_context")},
            "parent_context": "\n\n".join(parts), "parent_context_units": parent_units, "parent_context_truncated": truncated,
            "note": "Discovery result, not a verified fact or evidence-completeness conclusion.",
        })
    return {"query": query, "method": "hybrid_bm25_embedding" if any(x is not None for x in dense) else "bm25_lexical", "warnings": sorted(set(warnings)), "results": results}
