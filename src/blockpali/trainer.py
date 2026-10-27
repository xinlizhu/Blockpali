from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from transformers import Trainer


class BlockTrainer(Trainer):
    """Gather document embeddings and boundaries for negatives across DDP ranks."""

    def __init__(self, *args, loss_func, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_inputs = {key[6:]: value for key, value in inputs.items() if key.startswith("query_")}
        doc_inputs = {
            key[4:]: value for key, value in inputs.items()
            if key.startswith("doc_") and key != "doc_block_boundaries"
        }
        queries = model(**query_inputs)
        documents = model(**doc_inputs)
        boundaries = inputs["doc_block_boundaries"]
        local_batch = documents.shape[0]
        offset = 0
        if dist.is_available() and dist.is_initialized():
            sizes = [None] * dist.get_world_size()
            dist.all_gather_object(sizes, local_batch)
            if len(set(sizes)) != 1:
                raise ValueError("DDP needs equal local batch sizes; set dataloader_drop_last=true")
            # Right padding preserves all local block boundary offsets.
            documents = self.accelerator.pad_across_processes(documents, dim=1, pad_index=0)
            documents = torch.cat(all_gather(documents), dim=0)
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, boundaries)
            boundaries = [spans for rank_spans in gathered for spans in rank_spans]
            offset = dist.get_rank() * local_batch
        loss = self.loss_func(queries, documents, boundaries, query_inputs["attention_mask"], offset)
        return (loss, {"query_embeddings": queries, "document_embeddings": documents}) if return_outputs else loss
