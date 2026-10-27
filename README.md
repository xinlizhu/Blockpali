# blockpali

Training and evaluation code for dynamic block multi-vector document retrieval
with Qwen2.5-VL. This repository contains source code and configuration only.

## Method

1. Detect page layout with **PP-Layout CDLA**, using RapidLayout and ONNX Runtime.
   Crop each detected region with a 5% margin. If no valid region is detected,
   retain the complete page as one block. Keep detector order.
2. Select the first **16 blocks** in block-index order. Pack them into one
   multi-image input, using the original Qwen2.5-VL causal attention.
3. Project final hidden states to **128 dimensions** and L2 normalize them.
   Retain visual tokens for pages and valid text tokens for queries.
4. Train with patch MaxSim, block soft-or and in-batch cross entropy.
5. Encode pages offline and rank them with exhaustive MaxSim. Evaluate all
   three aggregation rules; `token_cross_block` is the primary evaluation rule.

For query token vectors q_i and block patch vectors b_kj:

```text
block score:        s_k = sum_i max_j dot(q_i, b_kj)
training score:     sum_k softmax(s_k / 0.1)_k * s_k
training objective: cross_entropy((query/page scores / query length) / 0.02)

token_cross_block:  sum_i max_k max_j dot(q_i, b_kj)
block_hard_max:     max_k s_k
block_soft_or:      sum_k softmax(s_k / 0.1)_k * s_k
```

Training and primary evaluation deliberately use different block aggregations.
Each query/page positive pair becomes one training sample; other pages in the
batch are treated as negatives. This single-positive objective does not mask
other known positives that happen to occur in the same batch.

Blocks first undergo aspect-preserving `smart_resize` with factor 28,
`min_pixels=784` and `max_pixels=589824`. The image processor then applies its
saved limits (`min_pixels=3136`, `max_pixels=602112`). These are per-image limits;
there is no fixed total visual-token budget per page.

## Install

Use Python 3.10 or newer and a PyTorch installation suitable for your accelerator:

```bash
python -m pip install -e '.[layout,test]'
```

The backbone is `Qwen/Qwen2.5-VL-3B-Instruct`, loaded directly through Transformers.
Change `model.base_model` to a local directory for offline use. Model weights
and datasets are downloaded or supplied separately. PP-Layout CDLA can be
downloaded by RapidLayout or supplied with `--layout-model model.onnx`.

## Prepare data

For a ViDoRe V3 dataset exported as local parquet files:

```text
raw_dataset/
  corpus/*.parquet    # corpus_id, image
  queries/*.parquet   # query_id, query
  qrels/*.parquet     # query_id, corpus_id, score, optional bounding_boxes
```

```bash
blockpali-prepare --format vidore-parquet \
  --input data/raw_dataset --output data/test --workers 4
```

For training shards containing `image` and `query` columns:

```bash
blockpali-prepare --format train-parquet \
  --input data/raw_train --output data/train --workers 4
```

Each training row creates one query/page pair. For an existing page dataset with
separate query and relevance metadata, use the same layout pipeline:

```bash
blockpali-prepare --format pages \
  --input data/raw_train/page_meta.json \
  --queries data/raw_train/query_meta.json \
  --qrels data/raw_train/qrel_meta.json \
  --output data/train --workers 4
```

`page_meta.json` records contain `corpus_id` and `page_image_path`. Input image
paths are relative to that JSON's directory unless `--image-root` is supplied.
Prepared output contains relative paths and the following files:

```text
data/train/
  page_images/             block_images/
  page_meta.json           block_meta.json
  query_meta.json          qrel_meta.json
  whole_page_block_meta.json
  preparation_stats.json
```

Metadata examples:

```json
{"query_id": "q0", "query": "What is the reported revenue?"}
{"query_id": "q0", "corpus_id": "p0", "relevance": 1}
{"corpus_id": "p0", "block_idx": 0, "block_image_path": "block_images/0_0.jpg"}
```

These are individual records; each metadata file stores a JSON array. Relevance
can also be named `score`; missing relevance defaults to 1. Bounding boxes are
retained as evaluation metadata and are not used to select training evidence.
Image paths, class labels, boxes and confidence scores are saved for blocks.
Preparation stops on detector or image errors instead of silently changing the
corpus. A populated output directory is never overwritten.

## Train

Edit `configs/train.yaml` to select prepared training data and the backbone.
Run commands from this repository's root; configuration paths are relative to
the current working directory.

```bash
blockpali-train configs/train.yaml
```

For distributed training:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m blockpali.cli.train configs/train.yaml
```

The supplied configuration uses 2 epochs, batch size 16 per device, learning
rate 2e-4, linear decay, no warmup, and LoRA rank/alpha 32 with dropout 0.1.
LoRA targets the language decoder projections and the retrieval projection;
the visual encoder is frozen. Multiple GPUs provide cross-rank batch negatives,
so changing the GPU count changes the negative pool. Adjust batch size for
available memory. Gradient accumulation does not combine microbatches into a
larger contrastive negative pool.

```bash
blockpali-train configs/train.yaml \
  --resume-from-checkpoint outputs/adapter/checkpoint-200
```

Every checkpoint includes adapter weights, processor files,
`retrieval_config.json`, and `projection_init.safetensors`. The last file stores
the frozen initial retrieval projection, which does not exist in the raw
backbone and must be restored together with its LoRA weights. The final output
also includes the complete training configuration and trainer state.

## Encode and evaluate

Encoding loads preprocessing settings and the projection initialization from
the trained adapter directory:

```bash
blockpali-encode query --input data/test/query_meta.json \
  --adapter outputs/adapter --output embeddings/test/queries

blockpali-encode block --input data/test/block_meta.json \
  --adapter outputs/adapter --output embeddings/test/blocks

blockpali-evaluate --queries embeddings/test/queries \
  --blocks embeddings/test/blocks --qrels data/test/qrel_meta.json \
  --output results/test.json
```

`--image-root` overrides the prepared metadata directory for relative image
paths. `--base-model` overrides the backbone location when moving an adapter
between machines; it must point to identical backbone weights. CUDA is the
default device. Use `--device cpu --dtype float32` for CPU smoke checks.
Processing or encoding failures stop the run; joint encoding never silently
falls back to independent block encoding.

Embeddings are variable-length 128-d vectors saved as sharded float16 NPZ
archives, with offsets and record IDs. ZIP/DEFLATE compression is lossless
relative to the stored float16 arrays. This archive is not an ANN index.

Evaluation returns rankings and per-query metrics for all three aggregations.
`Hit@K` means at least one positively judged page occurs in the top K;
`nDCG@K` uses graded relevance and gain `2**relevance - 1`. Queries without a
positive judgment are excluded from metric averaging. The evaluator checks
that all positively judged queries and pages have embeddings. Use separate
training and evaluation datasets.

For the native-resolution whole-page representation ablation, encode
`whole_page_block_meta.json` with the same adapter and queries:

```bash
blockpali-encode block --input data/test/whole_page_block_meta.json \
  --native-resolution --adapter outputs/adapter \
  --output embeddings/test/whole_pages

blockpali-evaluate --queries embeddings/test/queries \
  --blocks embeddings/test/whole_pages --qrels data/test/qrel_meta.json \
  --output results/test_whole_page.json
```

Native mode pads the right/bottom edges to a 28-pixel grid and disables image
resizing. It may require more memory. Vector counts are recorded in each
archive's `manifest.json`; count ratios are theoretical scoring-cost proxies,
not measured wall-clock speedups.

## Validation

```bash
python -m pytest -q
```

Tests use synthetic data and a tiny randomly initialized Qwen2.5-VL model.
They check multimodal training, checkpoint restoration, block boundaries,
aggregation semantics, metrics and archive round trips without downloading
the 3B backbone. Real PP-Layout inference additionally requires its ONNX
weights and optional dependencies.

Changing backbone initialization requires training new adapters. This source
release does not include pretrained retrieval weights or claim reproduction
of a numerical result from a different initialization.

## Dependencies

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [RapidLayout: PP-Layout CDLA configuration](https://github.com/RapidAI/RapidLayout/blob/main/docs/quickstart.md)

Third-party code, models and datasets retain their respective licenses. This
repository does not redistribute their weights or datasets.

## License

Original blockpali code and modifications are licensed under the
[Apache License, Version 2.0](LICENSE). Copyright is attributed to
`blockpali contributors` for this anonymous release; no author names,
affiliations or contact details are included for the project contributors.

Adapted third-party portions retain their MIT copyright and permission notice
in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). The distribution's license
expression is `Apache-2.0 AND MIT` to account for both. See [NOTICE](NOTICE)
for the project notice. The licenses of separately obtained dependencies,
model weights and datasets remain applicable to those artifacts.
