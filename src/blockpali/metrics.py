from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data import read_json


def load_qrels(path):
    output = defaultdict(dict)
    for row in read_json(path):
        qid, cid = str(row["query_id"]), str(row["corpus_id"])
        relevance = int(row.get("relevance", row.get("score", 1)))
        if relevance < 0:
            raise ValueError("Relevance must be nonnegative")
        output[qid][cid] = max(relevance, output[qid].get(cid, 0))
    return dict(output)


def compute_metrics(rankings, qrels, k_values=(1, 3, 5, 10)):
    per_query = {}
    for qid, retrieved in rankings.items():
        relevance = qrels.get(qid, {})
        relevant_ids = {cid for cid, score in relevance.items() if score > 0}
        if not relevant_ids:
            continue
        row = {}
        for k in k_values:
            row[f"Hit@{k}"] = float(bool(set(retrieved[:k]) & relevant_ids))
            gains = [relevance.get(cid, 0) for cid in retrieved[:k]]
            ideal = sorted(relevance.values(), reverse=True)[:k]
            dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(gains))
            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))
            row[f"nDCG@{k}"] = float(dcg / idcg)
        per_query[qid] = row
    if not per_query:
        raise ValueError("No evaluated queries have positive relevance judgments")
    summary = {key: float(np.mean([row[key] for row in per_query.values()])) for key in next(iter(per_query.values()))}
    return summary, per_query
