import json

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2VLVideoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from blockpali.cli.encode import encode
from blockpali.cli.train import train
from blockpali.data import BlockCollator
from blockpali.losses import DynamicBlockContrastiveLoss
from blockpali.models import DynamicBlockQwen25, DynamicBlockQwen25Processor
from blockpali.runtime import load_retriever
from blockpali.storage import load_query_embeddings


@pytest.fixture
def tiny_backbone(tmp_path):
    torch.set_num_threads(1)
    specials = ["[PAD]", "[UNK]", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
                "<|video_pad|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    vocab = {token: index for index, token in enumerate(specials + ["user", "red", "blue", "report"])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                                       additional_special_tokens=specials[2:])
    processor = DynamicBlockQwen25Processor(
        image_processor=Qwen2VLImageProcessor(min_pixels=3136, max_pixels=602112),
        video_processor=Qwen2VLVideoProcessor(), tokenizer=tokenizer,
    )
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]}, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 32,
                       "num_heads": 4, "fullatt_block_indexes": [0]},
        image_token_id=4, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3,
    )
    model = DynamicBlockQwen25(config)
    directory = tmp_path / "backbone"
    Qwen2_5_VLForConditionalGeneration(config).save_pretrained(directory)
    processor.save_pretrained(directory)
    return directory, model, processor


def test_raw_generation_backbone_weights_are_loaded(tiny_backbone):
    directory, _, _ = tiny_backbone
    original = Qwen2_5_VLForConditionalGeneration.from_pretrained(directory)
    loaded, info = DynamicBlockQwen25.from_pretrained(directory, output_loading_info=True)
    assert set(info["missing_keys"]) == {"custom_text_proj.weight", "custom_text_proj.bias"}
    assert torch.equal(loaded.visual.patch_embed.proj.weight, original.model.visual.patch_embed.proj.weight)
    assert torch.equal(loaded.language_model.layers[0].self_attn.q_proj.weight,
                       original.model.language_model.layers[0].self_attn.q_proj.weight)


def test_joint_blocks_boundaries_and_backward(tiny_backbone):
    _, model, processor = tiny_backbone
    samples = [{"query": "red report", "blocks": [{"image": Image.new("RGB", (56, 56), "red"), "block_idx": 0}]},
               {"query": "blue", "blocks": [{"image": Image.new("RGB", (56, 84), "blue"), "block_idx": 0},
                                              {"image": Image.new("RGB", (56, 56), "white"), "block_idx": 1}]}]
    batch = BlockCollator(processor)(samples)
    queries = model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"])
    docs = model(**{key[4:]: value for key, value in batch.items() if key.startswith("doc_") and key != "doc_block_boundaries"})
    assert queries.shape[-1] == docs.shape[-1] == 128
    assert [len(spans) for spans in batch["doc_block_boundaries"]] == [1, 2]
    loss = DynamicBlockContrastiveLoss()(queries, docs, batch["doc_block_boundaries"], batch["query_attention_mask"])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.custom_text_proj.weight.grad.abs().sum() > 0


def test_train_save_reload_and_encode(tiny_backbone, tmp_path):
    backbone, _, _ = tiny_backbone
    data = tmp_path / "data"
    data.mkdir()
    blocks = []
    for index, color in enumerate(("red", "blue")):
        Image.new("RGB", (56, 56), color).save(data / f"{index}.png")
        blocks.append({"corpus_id": index, "block_idx": 0, "block_image_path": f"{index}.png"})
    queries = [{"query_id": 0, "query": "red report"}, {"query_id": 1, "query": "blue report"}]
    for name, rows in {"block_meta": blocks, "query_meta": queries,
                       "qrel_meta": [{"query_id": i, "corpus_id": i} for i in range(2)]}.items():
        (data / f"{name}.json").write_text(json.dumps(rows))
    config = {
        "model": {"base_model": str(backbone), "embedding_dim": 128, "dtype": "float32", "attn_implementation": "sdpa"},
        "processor": {"max_blocks_per_doc": 16},
        "data": {"query_meta": str(data / "query_meta.json"), "qrel_meta": str(data / "qrel_meta.json"),
                 "block_meta": str(data / "block_meta.json"), "image_root": str(data)},
        "loss": {"temperature": 0.02, "block_temperature": 0.1, "block_aggregation": "soft_or"},
        "lora": {"r": 2, "lora_alpha": 2, "lora_dropout": 0.0,
                 "target_modules": "(language_model\\.layers\\.\\d+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))|custom_text_proj)"},
        "training": {"output_dir": str(tmp_path / "adapter"), "max_steps": 1, "per_device_train_batch_size": 2,
                     "learning_rate": 0.0002, "save_steps": 1, "logging_steps": 1, "use_cpu": True, "seed": 42},
    }
    result = train(config)
    assert result.global_step == 1
    adapter = tmp_path / "adapter"
    model, processor = load_retriever(adapter, "cpu", "float32")
    saved, saved_processor = load_retriever(adapter / "checkpoint-1", "cpu", "float32")
    inputs = processor.process_texts(["red report"])
    with torch.no_grad():
        assert torch.allclose(model(**inputs), saved(**saved_processor.process_texts(["red report"])), atol=1e-6)
    # Changing random seed cannot change the restored retrieval projection.
    torch.manual_seed(987)
    restored, _ = load_retriever(adapter, "cpu", "float32")
    with torch.no_grad():
        assert torch.allclose(model(**inputs), restored(**inputs), atol=1e-6)
    encode(model, processor, queries, tmp_path / "queries", "query", 1)
    assert set(load_query_embeddings(tmp_path / "queries")) == {"0", "1"}
