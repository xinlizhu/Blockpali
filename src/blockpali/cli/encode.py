from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from blockpali.data import group_blocks, read_json
from blockpali.runtime import load_retriever
from blockpali.storage import RaggedNPZWriter


@torch.inference_mode()
def encode(model, processor, rows, output, kind, shard_max_vectors=500_000):
    device = next(model.parameters()).device
    dim = model.get_base_model().dim if hasattr(model, "get_base_model") else model.dim
    with RaggedNPZWriter(output, kind=kind, dim=dim, max_vectors_per_shard=shard_max_vectors) as writer:
        if kind == "query":
            seen = set()
            for row in tqdm(rows, desc="Encoding queries"):
                qid = str(row["query_id"])
                if qid in seen:
                    raise ValueError("Duplicate query ID")
                seen.add(qid)
                inputs = processor.process_texts([row["query"]]).to(device)
                vectors = model(**inputs)[0]
                writer.add(qid, vectors[inputs["attention_mask"][0].bool()])
        else:
            for cid, blocks in tqdm(group_blocks(rows).items(), desc="Encoding pages"):
                inputs, boundaries = processor.process_document_blocks(blocks)
                vectors = model(**inputs.to(device))[0]
                for span in boundaries:
                    writer.add(cid, vectors[span["start"]:span["end"]], block_idx=span["source_block_idx"])


def main():
    parser = argparse.ArgumentParser(description="Encode queries or joint page blocks with a trained adapter")
    parser.add_argument("kind", choices=["query", "block"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", help="Override backbone location, using identical backbone weights")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--shard-max-vectors", type=int, default=500_000)
    parser.add_argument("--native-resolution", action="store_true", help="Pad images to a 28-pixel grid without resizing; for whole-page ablation")
    args = parser.parse_args()
    model, processor = load_retriever(args.adapter, args.device, args.dtype, args.base_model)
    processor.image_base_dir = args.image_root or args.input.parent
    processor.preserve_original_resolution = args.native_resolution
    encode(model, processor, read_json(args.input), args.output, args.kind, args.shard_max_vectors)
    # Record processing choices without retaining private machine paths.
    (args.output / "encoding_config.json").write_text(json.dumps({
        "kind": args.kind, "dtype": args.dtype, "native_resolution": args.native_resolution,
        "max_blocks_per_doc": processor.max_blocks_per_doc,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
