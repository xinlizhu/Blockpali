from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from blockpali.metrics import compute_metrics, load_qrels
from blockpali.retrieval import MultiAggregationRetriever
from blockpali.storage import load_block_embeddings, load_query_embeddings


def main():
    parser = argparse.ArgumentParser(description="Evaluate exhaustive multi-vector page retrieval with three aggregations")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--block-batch-size", type=int, default=128)
    parser.add_argument("--soft-or-temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 10:
        parser.error("top-k must be at least 10 to report Hit@10 and nDCG@10")
    if min(args.query_batch_size, args.block_batch_size) <= 0 or args.soft_or_temperature <= 0:
        parser.error("Batch sizes and soft-or temperature must be positive")
    queries, blocks = load_query_embeddings(args.queries), load_block_embeddings(args.blocks)
    qrels = load_qrels(args.qrels)
    if not queries or not blocks:
        raise ValueError("Query and block archives must be nonempty")
    for qid, relevance in qrels.items():
        if any(score > 0 for score in relevance.values()) and qid not in queries:
            raise ValueError("A judged query is missing from the query archive")
        if any(score > 0 and cid not in blocks for cid, score in relevance.items()):
            raise ValueError("A relevant page is missing from the block archive")
    retriever = MultiAggregationRetriever(
        blocks, device=args.device, dtype=getattr(torch, args.dtype),
        block_batch_size=args.block_batch_size, soft_or_temperature=args.soft_or_temperature,
    )
    rankings = retriever.retrieve(queries, query_batch_size=args.query_batch_size, top_k=args.top_k)
    report = {"config": {
        "dtype": args.dtype, "top_k": args.top_k, "soft_or_temperature": args.soft_or_temperature,
        "queries": len(queries), "pages": len(blocks),
    }, "aggregations": {}}
    for name, results in rankings.items():
        metrics, per_query = compute_metrics(results, qrels)
        report["aggregations"][name] = {"metrics": metrics, "per_query_metrics": per_query, "rankings": results}
        print(name, " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
