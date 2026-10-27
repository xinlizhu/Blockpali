import math

import torch

from blockpali.metrics import compute_metrics
from blockpali.retrieval import MultiAggregationRetriever
from blockpali.storage import RaggedNPZWriter, load_block_embeddings, load_query_embeddings


def test_cross_block_combines_distributed_evidence():
    # Each of two blocks contains exactly one part of the query evidence.
    queries = {"q": torch.eye(2)}
    blocks = {
        "split": [{"block_idx": 0, "embedding": torch.tensor([[1., 0.]])},
                  {"block_idx": 1, "embedding": torch.tensor([[0., 1.]])}],
        "joint": [{"block_idx": 0, "embedding": torch.tensor([[0.8, 0.6]])}],
    }
    retriever = MultiAggregationRetriever(blocks, device="cpu", dtype=torch.float32,
                                         block_batch_size=1, soft_or_temperature=0.1)
    rankings = retriever.retrieve(queries, query_batch_size=1, top_k=2)
    assert rankings["token_cross_block"]["q"] == ["split", "joint"]
    assert rankings["block_hard_max"]["q"] == ["joint", "split"]
    assert rankings["block_soft_or"]["q"] == ["joint", "split"]


def test_padding_cannot_replace_negative_similarity():
    queries = {"q": torch.tensor([[1., 0.]])}
    blocks = {
        "short": [{"block_idx": 0, "embedding": torch.tensor([[-1., 0.]])}],
        "long": [{"block_idx": 0, "embedding": torch.tensor([[-0.8, 0.6], [-0.8, -0.6]])}],
    }
    retriever = MultiAggregationRetriever(blocks, device="cpu", dtype=torch.float32,
                                         block_batch_size=2, soft_or_temperature=0.1)
    for rankings in retriever.retrieve(queries, query_batch_size=1, top_k=2).values():
        assert rankings["q"] == ["long", "short"]


def test_graded_ndcg_and_hit():
    metrics, rows = compute_metrics({"q": ["b", "a"], "unjudged": ["a"]}, {"q": {"a": 2, "b": 1}}, (1, 2))
    assert metrics["Hit@1"] == 1
    assert metrics["nDCG@1"] == 1 / 3
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert abs(metrics["nDCG@2"] - expected) < 1e-8
    assert len(rows) == 1


def test_sharded_archives_round_trip(tmp_path):
    query_path, block_path = tmp_path / "queries", tmp_path / "blocks"
    with RaggedNPZWriter(query_path, kind="query", dim=2, max_vectors_per_shard=1) as writer:
        writer.add("q0", torch.eye(2))
        writer.add("q1", torch.tensor([[0.5, -0.5]]))
    with RaggedNPZWriter(block_path, kind="block", dim=2, max_vectors_per_shard=1) as writer:
        writer.add("p0", torch.eye(2), block_idx=3)
        writer.add("p0", torch.tensor([[0.5, -0.5]]), block_idx=1)
    assert torch.equal(load_query_embeddings(query_path)["q0"], torch.eye(2).half())
    assert [row["block_idx"] for row in load_block_embeddings(block_path)["p0"]] == [1, 3]
    assert len(list(query_path.glob("shard-*.npz"))) == 2
