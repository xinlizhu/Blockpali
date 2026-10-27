"""Sharded ragged NPZ storage for blockpali multi-vector embeddings.

The format deliberately avoids NumPy object arrays and pickle.  Every shard
contains one flat 2-D embedding array plus offsets and record identifiers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch


FORMAT_NAME = "blockpali-ragged-npz-v1"


class RaggedNPZWriter:
    """Incrementally write variable-length embeddings to NPZ shards."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        kind: str,
        dim: int = 128,
        max_vectors_per_shard: int = 500_000,
        compressed: bool = True,
    ) -> None:
        if kind not in {"query", "block"}:
            raise ValueError(f"kind must be 'query' or 'block', got {kind!r}")
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty embedding directory: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.dim = int(dim)
        self.max_vectors_per_shard = int(max_vectors_per_shard)
        self.compressed = bool(compressed)
        self._ids: list[str] = []
        self._block_idxs: list[int] = []
        self._arrays: list[np.ndarray] = []
        self._buffered_vectors = 0
        self._num_records = 0
        self._num_vectors = 0
        self._shards: list[dict] = []
        self._closed = False

    def add(
        self,
        record_id: str | int,
        embedding: np.ndarray | torch.Tensor,
        *,
        block_idx: Optional[int] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot add records after closing the writer")
        if isinstance(embedding, torch.Tensor):
            array = embedding.detach().cpu().float().numpy()
        else:
            array = np.asarray(embedding)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.dim:
            raise ValueError(
                f"Expected embedding shape [tokens, {self.dim}], got {array.shape}"
            )

        array = np.ascontiguousarray(array, dtype=np.float16)
        self._ids.append(str(record_id))
        self._arrays.append(array)
        if self.kind == "block":
            self._block_idxs.append(int(-1 if block_idx is None else block_idx))
        self._buffered_vectors += int(array.shape[0])
        self._num_records += 1
        self._num_vectors += int(array.shape[0])

        if self._buffered_vectors >= self.max_vectors_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._arrays:
            return

        shard_index = len(self._shards)
        shard_name = f"shard-{shard_index:05d}.npz"
        shard_path = self.output_dir / shard_name
        lengths = np.asarray([array.shape[0] for array in self._arrays], dtype=np.int64)
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        values = np.concatenate(self._arrays, axis=0)
        max_id_len = max(1, max(len(item) for item in self._ids))
        payload = {
            "values": values,
            "offsets": offsets,
            "ids": np.asarray(self._ids, dtype=f"<U{max_id_len}"),
        }
        if self.kind == "block":
            payload["block_idxs"] = np.asarray(self._block_idxs, dtype=np.int32)

        save = np.savez_compressed if self.compressed else np.savez
        save(shard_path, **payload)
        self._shards.append(
            {
                "file": shard_name,
                "records": len(self._arrays),
                "vectors": int(values.shape[0]),
                "bytes": shard_path.stat().st_size,
            }
        )
        self._ids.clear()
        self._block_idxs.clear()
        self._arrays.clear()
        self._buffered_vectors = 0

    def close(self) -> Path:
        if self._closed:
            return self.output_dir / "manifest.json"
        self.flush()
        manifest = {
            "format": FORMAT_NAME,
            "kind": self.kind,
            "dim": self.dim,
            "dtype": "float16",
            "compressed": self.compressed,
            "num_records": self._num_records,
            "num_vectors": self._num_vectors,
            "shards": self._shards,
        }
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        self._closed = True
        return manifest_path

    def __enter__(self) -> "RaggedNPZWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()


def read_manifest(path: str | Path) -> tuple[Path, dict]:
    root = Path(path)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"Unsupported embedding format in {manifest_path}")
    return manifest_path.parent, manifest


def iter_ragged_npz(path: str | Path) -> Iterator[dict]:
    """Yield records as NumPy views, one shard at a time."""
    root, manifest = read_manifest(path)
    kind = manifest["kind"]
    for shard in manifest["shards"]:
        with np.load(root / shard["file"], allow_pickle=False) as data:
            values = data["values"]
            offsets = data["offsets"]
            ids = data["ids"]
            block_idxs = data["block_idxs"] if kind == "block" else None
            for index, record_id in enumerate(ids):
                start = int(offsets[index])
                end = int(offsets[index + 1])
                record = {
                    "id": str(record_id),
                    "embedding": np.array(values[start:end], copy=True),
                }
                if block_idxs is not None:
                    record["block_idx"] = int(block_idxs[index])
                yield record


def load_query_embeddings(path: str | Path) -> dict[str, torch.Tensor]:
    root, manifest = read_manifest(path)
    if manifest["kind"] != "query":
        raise ValueError(f"Expected query embeddings, found {manifest['kind']}")
    return {
        record["id"]: torch.from_numpy(record["embedding"])
        for record in iter_ragged_npz(root)
    }


def load_block_embeddings(path: str | Path) -> dict[str, list[dict]]:
    root, manifest = read_manifest(path)
    if manifest["kind"] != "block":
        raise ValueError(f"Expected block embeddings, found {manifest['kind']}")
    output: dict[str, list[dict]] = {}
    for record in iter_ragged_npz(root):
        output.setdefault(record["id"], []).append(
            {
                "block_idx": record["block_idx"],
                "embedding": torch.from_numpy(record["embedding"]),
            }
        )
    for blocks in output.values():
        blocks.sort(key=lambda item: item["block_idx"])
    return output
