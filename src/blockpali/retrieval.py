from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

AGGREGATIONS = ("token_cross_block", "block_hard_max", "block_soft_or")


class MultiAggregationRetriever:
    def __init__(
        self,
        corpus_blocks: dict[str, list[dict]],
        *,
        device: str,
        block_batch_size: int,
        dtype: torch.dtype,
        soft_or_temperature: float,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.soft_or_temperature = float(soft_or_temperature)
        self.corpus_ids = sorted(corpus_blocks, key=lambda item: (len(item), item))
        corpus_index = {corpus_id: index for index, corpus_id in enumerate(self.corpus_ids)}
        block_records = []
        for corpus_id, blocks in corpus_blocks.items():
            for block in blocks:
                block_records.append(
                    (
                        block["embedding"],
                        corpus_index[corpus_id],
                        int(block["block_idx"]),
                    )
                )
        block_records.sort(key=lambda item: item[0].shape[0])
        self.block_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(block_records), block_batch_size):
            chunk = block_records[start : start + block_batch_size]
            max_length = max(item[0].shape[0] for item in chunk)
            values = torch.zeros(
                (len(chunk), max_length, chunk[0][0].shape[1]),
                dtype=dtype,
                device=self.device,
            )
            for index, (embedding, _, _) in enumerate(chunk):
                length = embedding.shape[0]
                values[index, :length] = embedding.to(device=self.device, dtype=dtype)
            doc_indices = torch.tensor(
                [item[1] for item in chunk],
                dtype=torch.long,
                device=self.device,
            )
            lengths = torch.tensor(
                [item[0].shape[0] for item in chunk],
                dtype=torch.long,
                device=self.device,
            )
            self.block_batches.append((values, doc_indices, lengths))

    @torch.inference_mode()
    def retrieve(
        self,
        queries: dict[str, torch.Tensor],
        *,
        query_batch_size: int,
        top_k: int,
    ) -> dict[str, dict[str, list[str]]]:
        output = {name: {} for name in AGGREGATIONS}
        query_items = list(queries.items())
        num_docs = len(self.corpus_ids)

        for start in tqdm(
            range(0, len(query_items), query_batch_size),
            desc="Scoring query batches",
        ):
            batch = query_items[start : start + query_batch_size]
            max_query_length = max(item[1].shape[0] for item in batch)
            dim = batch[0][1].shape[1]
            query_values = torch.zeros(
                (len(batch), max_query_length, dim),
                dtype=self.dtype,
                device=self.device,
            )
            query_mask = torch.zeros(
                (len(batch), max_query_length),
                dtype=torch.bool,
                device=self.device,
            )
            for index, (_, embedding) in enumerate(batch):
                length = embedding.shape[0]
                query_values[index, :length] = embedding.to(
                    device=self.device,
                    dtype=self.dtype,
                )
                query_mask[index, :length] = True

            all_block_scores = []
            all_doc_indices = []
            token_doc_best = torch.full(
                (len(batch), max_query_length, num_docs),
                -torch.inf,
                dtype=torch.float32,
                device=self.device,
            )
            for block_values, doc_indices, block_lengths in self.block_batches:
                similarities = torch.einsum(
                    "qld,bpd->qblp",
                    query_values,
                    block_values,
                )
                patch_positions = torch.arange(
                    block_values.shape[1],
                    device=self.device,
                )
                patch_mask = patch_positions[None, :] < block_lengths[:, None]
                similarities = similarities.masked_fill(
                    ~patch_mask[None, :, None, :],
                    -torch.inf,
                )
                token_best = similarities.max(dim=-1).values.float()
                token_best = token_best.masked_fill(
                    ~query_mask[:, None, :],
                    0.0,
                )
                block_scores = token_best.sum(dim=-1)
                all_block_scores.append(block_scores)
                all_doc_indices.append(doc_indices)

                scatter_index = doc_indices.view(1, 1, -1).expand(
                    len(batch),
                    max_query_length,
                    -1,
                )
                token_doc_best.scatter_reduce_(
                    2,
                    scatter_index,
                    token_best.transpose(1, 2),
                    reduce="amax",
                    include_self=True,
                )

            scores = torch.cat(all_block_scores, dim=1)
            doc_indices = torch.cat(all_doc_indices)
            scatter_index = doc_indices.view(1, -1).expand(len(batch), -1)

            hard_scores = torch.full(
                (len(batch), num_docs),
                -torch.inf,
                dtype=torch.float32,
                device=self.device,
            )
            hard_scores.scatter_reduce_(
                1,
                scatter_index,
                scores,
                reduce="amax",
                include_self=True,
            )

            cross_scores = token_doc_best.masked_fill(
                ~query_mask[:, :, None],
                0.0,
            ).sum(dim=1)

            doc_max = hard_scores.gather(1, scatter_index)
            weights = torch.exp(
                (scores - doc_max) / self.soft_or_temperature
            )
            denominator = torch.zeros_like(hard_scores)
            numerator = torch.zeros_like(hard_scores)
            denominator.scatter_add_(1, scatter_index, weights)
            numerator.scatter_add_(1, scatter_index, weights * scores)
            soft_scores = numerator / denominator.clamp_min(1e-12)

            score_map = {
                "token_cross_block": cross_scores,
                "block_hard_max": hard_scores,
                "block_soft_or": soft_scores,
            }
            for aggregation, aggregation_scores in score_map.items():
                top_indices = aggregation_scores.topk(
                    min(top_k, num_docs),
                    dim=1,
                ).indices.cpu().tolist()
                for (query_id, _), indices in zip(batch, top_indices):
                    output[aggregation][query_id] = [
                        self.corpus_ids[index] for index in indices
                    ]
        return output
