from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def group_blocks(rows, image_root=None):
    grouped = defaultdict(list)
    for source in rows:
        row = dict(source)
        if image_root is not None and row.get("block_image_path"):
            row["block_image_path"] = str(Path(image_root) / row["block_image_path"])
        grouped[str(row["corpus_id"])].append(row)
    for blocks in grouped.values():
        blocks.sort(key=lambda row: int(row["block_idx"]))
    return dict(grouped)


class RetrievalDataset(Dataset):
    """Expand positive query/page pairs; retain every page regardless of block count."""

    def __init__(self, query_meta, qrel_meta, block_meta, image_root=None):
        queries = {str(row["query_id"]): row["query"] for row in read_json(query_meta)}
        self.documents = group_blocks(read_json(block_meta), image_root)
        positives = set()
        for row in read_json(qrel_meta):
            if int(row.get("relevance", row.get("score", 1))) > 0:
                positives.add((str(row["query_id"]), str(row["corpus_id"])))
        self.samples = []
        for qid, cid in sorted(positives):
            if qid not in queries or cid not in self.documents:
                raise ValueError("A positive qrel references a missing query or page")
            self.samples.append((queries[qid], cid))
        if not self.samples:
            raise ValueError("No positive query/page pairs found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        query, cid = self.samples[index]
        return {"query": query, "blocks": self.documents[cid]}


class BlockCollator:
    """Right pad document sequences so block positions stay unchanged."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):
        query = self.processor.process_texts([row["query"] for row in samples])
        packed = [self.processor.process_document_blocks(row["blocks"]) for row in samples]
        inputs, boundaries = zip(*packed)
        result = {"query_" + key: value for key, value in query.items()}
        result["doc_block_boundaries"] = list(boundaries)
        for key in inputs[0]:
            tensors = [item[key] for item in inputs]
            if key in {"input_ids", "attention_mask"}:
                length = max(value.shape[1] for value in tensors)
                fill = self.processor.tokenizer.pad_token_id if key == "input_ids" else 0
                tensors = [F.pad(value, (0, length - value.shape[1]), value=fill) for value in tensors]
            elif key not in {"pixel_values", "image_grid_thw"}:
                raise ValueError(f"Unexpected document input: {key}")
            result["doc_" + key] = torch.cat(tensors, dim=0)
        return result
