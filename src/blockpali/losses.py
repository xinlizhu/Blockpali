from __future__ import annotations

import torch
import torch.nn.functional as F


class DynamicBlockContrastiveLoss(torch.nn.Module):
    """Patch MaxSim, block aggregation and query-to-page in-batch cross entropy."""

    def __init__(self, temperature=0.02, block_temperature=0.1, block_aggregation="soft_or"):
        super().__init__()
        if temperature <= 0 or block_temperature <= 0:
            raise ValueError("Temperatures must be positive")
        if block_aggregation not in {"soft_or", "hard_max", "token_cross_block"}:
            raise ValueError("Unknown block aggregation")
        self.temperature = temperature
        self.block_temperature = block_temperature
        self.block_aggregation = block_aggregation

    def score(self, query, document, boundaries):
        token_scores = []
        for boundary in boundaries:
            patches = document[int(boundary["start"]):int(boundary["end"])]
            if len(patches):
                token_scores.append((query @ patches.T).max(dim=-1).values)
        if not token_scores:
            raise ValueError("Document has no visual tokens")
        tokens = torch.stack(token_scores)
        if self.block_aggregation == "token_cross_block":
            return tokens.max(dim=0).values.sum()
        blocks = tokens.sum(dim=-1)
        if self.block_aggregation == "hard_max":
            return blocks.max()
        weights = torch.softmax(blocks / self.block_temperature, dim=0)
        return (weights * blocks).sum()

    def forward(self, queries, documents, boundaries, query_mask, offset=0):
        scores = torch.stack([
            torch.stack([self.score(query, doc, spans) for doc, spans in zip(documents, boundaries)])
            for query in queries
        ])
        scores = scores / query_mask.sum(dim=-1).clamp_min(1).unsqueeze(1)
        labels = torch.arange(len(queries), device=queries.device) + offset
        return F.cross_entropy(scores / self.temperature, labels)
