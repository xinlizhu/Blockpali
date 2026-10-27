from __future__ import annotations

import argparse
import io
import json
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import version
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from blockpali.data import read_json
from blockpali.layout import PPLayoutCDLA, write_page_blocks


_DETECTOR = None


def init_worker(model_path, confidence, iou):
    global _DETECTOR
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    _DETECTOR = PPLayoutCDLA(model_path, confidence, iou)


def image_bytes(value, source_root):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return value["bytes"]
        value = value.get("path")
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return (Path(source_root) / value).read_bytes()
    raise ValueError("Image must contain encoded bytes or an image path")


def parquet_rows(paths, columns=None):
    import pyarrow.parquet as pq

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=columns):
            yield from batch.to_pylist()


def source_records(source, source_format, image_root=None):
    """Yield page/image/query records without loading an entire corpus into memory."""
    source = Path(source)
    if source_format == "pages":
        root = image_root or source.parent
        for row in read_json(source):
            yield {"corpus_id": row["corpus_id"], "image": image_bytes(row["page_image_path"], root)}
    elif source_format == "train-parquet":
        paths = [source] if source.is_file() else sorted(source.glob("train*.parquet"))
        if not paths:
            raise FileNotFoundError("No training parquet shards found")
        root = source.parent if source.is_file() else source
        for index, row in enumerate(parquet_rows(paths, ["image", "query"])):
            yield {"corpus_id": index, "image": image_bytes(row["image"], root), "query": row["query"]}
    else:
        paths = sorted((source / "corpus").glob("*.parquet"))
        if not paths:
            raise FileNotFoundError("No corpus parquet shards found")
        for row in parquet_rows(paths, ["corpus_id", "image"]):
            yield {"corpus_id": row["corpus_id"], "image": image_bytes(row["image"], source)}


def process_task(task):
    index, row, output, margin = task
    with Image.open(io.BytesIO(row["image"])) as image:
        result = write_page_blocks(image, row["corpus_id"], index, output, _DETECTOR, margin)
    if "query" in row:
        result["query"] = {"query_id": row["corpus_id"], "query": row["query"]}
        result["qrel"] = {"query_id": row["corpus_id"], "corpus_id": row["corpus_id"], "relevance": 1}
    return result


def prepare(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output directory must be empty")
    if args.workers < 1 or not 0 <= args.margin < 1:
        raise ValueError("workers must be positive and margin must be between 0 and 1")
    # Initialize once before starting workers; validate configuration and cache
    # the model download so workers do not race on the same model file.
    init_worker(args.layout_model, args.confidence, args.iou)
    for directory in ("page_images", "block_images"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    queries, qrels = [], []
    if args.format == "vidore-parquet":
        query_paths = sorted((args.input / "queries").glob("*.parquet"))
        qrel_paths = sorted((args.input / "qrels").glob("*.parquet"))
        if not query_paths or not qrel_paths:
            raise FileNotFoundError("Missing queries or qrels parquet files")
        queries = list(parquet_rows(query_paths, ["query_id", "query"]))
        for row in parquet_rows(qrel_paths):
            qrels.append({
                "query_id": row["query_id"], "corpus_id": row["corpus_id"],
                "relevance": int(row.get("score", row.get("relevance", 1))),
                "evidence_bboxes": row.get("bounding_boxes") or [],
            })
    elif args.format == "pages":
        if args.queries is None or args.qrels is None:
            raise ValueError("pages format requires --queries and --qrels")
        queries, qrels = read_json(args.queries), read_json(args.qrels)
    pages, blocks, whole = [], [], []
    seen = set()
    records = source_records(args.input, args.format, args.image_root)
    tasks = ((index, row, args.output, args.margin) for index, row in enumerate(records))

    def collect(result):
        cid = str(result["page"]["corpus_id"])
        if cid in seen:
            raise ValueError("Duplicate page ID")
        seen.add(cid)
        pages.append(result["page"])
        blocks.extend(result["blocks"])
        whole.append(result["whole_page_block"])
        if "query" in result:
            queries.append(result["query"])
            qrels.append(result["qrel"])

    if args.workers == 1:
        for task in tqdm(tasks, desc="PP-Layout CDLA"):
            collect(process_task(task))
    else:
        # Bound the queue; large datasets must not enqueue all encoded images.
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=init_worker,
            initargs=(args.layout_model, args.confidence, args.iou),
        ) as executor, tqdm(desc="PP-Layout CDLA", unit="page") as progress:
            pending = set()
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < args.workers * 2:
                    task = next(tasks, None)
                    if task is None:
                        exhausted = True
                    else:
                        pending.add(executor.submit(process_task, task))
                if pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        collect(future.result())
                        progress.update()
    if not pages:
        raise ValueError("No pages found")
    for rows in (pages, whole):
        rows.sort(key=lambda row: str(row["corpus_id"]))
    blocks.sort(key=lambda row: (str(row["corpus_id"]), row["block_idx"]))
    queries.sort(key=lambda row: str(row["query_id"]))
    qrels.sort(key=lambda row: (str(row["query_id"]), str(row["corpus_id"])))
    query_ids = {str(row["query_id"]) for row in queries}
    if any(str(row["query_id"]) not in query_ids or str(row["corpus_id"]) not in seen for row in qrels):
        raise ValueError("Qrels reference missing queries or pages")
    counts = Counter(str(row["corpus_id"]) for row in blocks)
    stats = {
        "pages": len(pages), "queries": len(queries), "qrels": len(qrels), "blocks": len(blocks),
        "pages_over_16_blocks": sum(count > 16 for count in counts.values()),
        "layout_model": "pp_layout_cdla", "layout_engine": "onnxruntime",
        "rapid_layout_version": version("rapid-layout"), "confidence": args.confidence,
        "iou": args.iou, "crop_margin": args.margin,
    }
    for name, rows in {
        "page_meta": pages, "block_meta": blocks, "whole_page_block_meta": whole,
        "query_meta": queries, "qrel_meta": qrels, "preparation_stats": stats,
    }.items():
        (args.output / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare page crops using PP-Layout CDLA")
    parser.add_argument("--format", choices=["train-parquet", "vidore-parquet", "pages"], required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--layout-model", type=Path, help="Optional local PP-Layout CDLA ONNX model")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
