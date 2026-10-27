from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file

from .models import DynamicBlockQwen25, DynamicBlockQwen25Processor


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def save_runtime(output_dir, config, projection):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "retrieval_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # The raw backbone has no retrieval projection. Its frozen initialization
    # must accompany the projection's LoRA weights for exact inference reload.
    save_file(projection, str(directory / "projection_init.safetensors"))


def load_retriever(adapter, device="cuda", dtype="bfloat16", base_model=None):
    directory = Path(adapter)
    config = json.loads((directory / "retrieval_config.json").read_text(encoding="utf-8"))
    model_config = config["model"]
    backbone = base_model or model_config["base_model"]
    model = DynamicBlockQwen25.from_pretrained(
        backbone, embedding_dim=model_config["embedding_dim"],
        dtype=DTYPES[dtype], attn_implementation=model_config["attn_implementation"],
    )
    model.custom_text_proj.load_state_dict(load_file(str(directory / "projection_init.safetensors")))
    model = PeftModel.from_pretrained(model, directory).to(device).eval()
    processor = DynamicBlockQwen25Processor.from_pretrained(directory, **config["processor"])
    return model, processor
