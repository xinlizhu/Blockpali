from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from transformers import Qwen2VLProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


class DynamicBlockQwen25(Qwen2_5_VLModel):
    """Joint block input, original causal attention, normalized 128-d token output."""

    main_input_name = "input_ids"

    def __init__(self, config, embedding_dim=128):
        super().__init__(config)
        self.dim = embedding_dim
        self.config.retrieval_embedding_dim = embedding_dim
        self.custom_text_proj = nn.Linear(config.hidden_size, embedding_dim)
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("key_mapping", cls._checkpoint_conversion_mapping)
        return super().from_pretrained(*args, **kwargs)

    def forward(self, *args, **kwargs):
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        outputs = super().forward(
            *args, use_cache=False, output_hidden_states=True, return_dict=True, **kwargs
        )
        vectors = F.normalize(self.custom_text_proj(outputs.last_hidden_state), dim=-1)
        if kwargs.get("attention_mask") is not None:
            vectors = vectors * kwargs["attention_mask"].unsqueeze(-1)
        if kwargs.get("pixel_values") is not None:
            vectors = vectors * kwargs["input_ids"].eq(self.config.image_token_id).unsqueeze(-1)
        return vectors


class DynamicBlockQwen25Processor(Qwen2VLProcessor):
    """Pack up to 16 blocks in their supplied order, preserving visual boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"
        self.max_blocks_per_doc = 16
        self.image_base_dir = None
        self.preserve_original_resolution = False
        self.block_min_pixels = 28 * 28
        self.block_max_pixels = 768 * 768

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        settings = {key: kwargs.pop(key) for key in (
            "max_blocks_per_doc", "image_base_dir", "preserve_original_resolution",
            "block_min_pixels", "block_max_pixels",
        ) if key in kwargs}
        # Preserve the two-stage preprocessing used by the final implementation.
        kwargs.setdefault("min_pixels", 56 * 56)
        kwargs.setdefault("max_pixels", 602112)
        kwargs.setdefault("use_fast", True)
        instance = super().from_pretrained(*args, **kwargs)
        for key, value in settings.items():
            setattr(instance, key, value)
        if instance.max_blocks_per_doc < 1:
            raise ValueError("max_blocks_per_doc must be positive")
        return instance

    def process_texts(self, texts):
        return self(text=list(texts), return_tensors="pt", padding="longest")

    def process_document_blocks(self, blocks, return_boundaries=True):
        selected = list(blocks[:self.max_blocks_per_doc])
        if not selected:
            raise ValueError("Document has no blocks")
        images = []
        for block in selected:
            source = block.get("image", block.get("block_image_path"))
            if isinstance(source, Image.Image):
                image = source.convert("RGB")
            else:
                if source is None:
                    raise ValueError("Block requires image or block_image_path")
                path = Path(source)
                if self.image_base_dir is not None:
                    path = Path(self.image_base_dir) / path
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
            if self.preserve_original_resolution:
                width, height = image.size
                target_width = max(28, ((width + 27) // 28) * 28)
                target_height = max(28, ((height + 27) // 28) * 28)
                image = ImageOps.expand(image, (0, 0, target_width - width, target_height - height), "white")
            else:
                height, width = smart_resize(
                    height=image.height, width=image.width, factor=28,
                    min_pixels=self.block_min_pixels, max_pixels=self.block_max_pixels,
                )
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
            images.append(image)
        prompt = (
            "<|im_start|>user\n"
            + "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
            + "<|im_end|><|endoftext|>"
        )
        options = {"do_resize": False} if self.preserve_original_resolution else {}
        inputs = self(text=[prompt], images=images, padding="longest", return_tensors="pt", **options)
        if not return_boundaries:
            return inputs
        ids = inputs["input_ids"][0].tolist()
        start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        starts = [i for i, token in enumerate(ids) if token == start_id]
        ends = [i for i, token in enumerate(ids) if token == end_id]
        if len(starts) != len(images) or len(ends) != len(images):
            raise ValueError("Visual spans do not match the number of input blocks")
        boundaries = []
        merge_size = self.image_processor.merge_size
        for index, (start, end) in enumerate(zip(starts, ends)):
            expected = int(inputs["image_grid_thw"][index].prod()) // (merge_size ** 2)
            if end - start - 1 != expected:
                raise ValueError("Visual span length does not match the image grid")
            boundaries.append({
                "block_idx": index, "source_block_idx": selected[index].get("block_idx", index),
                "start": start + 1, "end": end, "n_tokens": expected,
            })
        return inputs, boundaries
