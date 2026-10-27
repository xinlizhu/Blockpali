from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback, TrainingArguments, set_seed

from blockpali.data import BlockCollator, RetrievalDataset
from blockpali.losses import DynamicBlockContrastiveLoss
from blockpali.models import DynamicBlockQwen25, DynamicBlockQwen25Processor
from blockpali.runtime import DTYPES, save_runtime
from blockpali.trainer import BlockTrainer


class SaveRetrievalState(TrainerCallback):
    def __init__(self, config, projection, processor):
        self.config, self.projection, self.processor = config, projection, processor

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            save_runtime(directory, self.config, self.projection)
            self.processor.save_pretrained(directory)


def train(config, resume_from_checkpoint=None):
    config = copy.deepcopy(config)
    arguments = dict(config["training"])
    set_seed(arguments.get("seed", 42))
    model_config = {"embedding_dim": 128, "attn_implementation": "sdpa", "dtype": "bfloat16", **config["model"]}
    config["model"] = model_config
    dtype = model_config.get("dtype", "bfloat16")
    model = DynamicBlockQwen25.from_pretrained(
        model_config["base_model"], embedding_dim=model_config.get("embedding_dim", 128),
        dtype=DTYPES[dtype], attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    projection = {key: value.detach().cpu().clone().contiguous() for key, value in model.custom_text_proj.state_dict().items()}
    runtime_config = {"model": model_config, "processor": config["processor"]}
    if resume_from_checkpoint:
        from safetensors.torch import load_file
        projection = load_file(str(Path(resume_from_checkpoint) / "projection_init.safetensors"))
        model.custom_text_proj.load_state_dict(projection)
    model = get_peft_model(model, LoraConfig(
        **config["lora"], task_type="FEATURE_EXTRACTION", bias="none", init_lora_weights="gaussian"
    ))
    model.print_trainable_parameters()
    processor = DynamicBlockQwen25Processor.from_pretrained(model_config["base_model"], **config["processor"])
    dataset = RetrievalDataset(**config["data"])
    arguments.update(remove_unused_columns=False, report_to="none", bf16=dtype == "bfloat16", fp16=dtype == "float16")
    trainer = BlockTrainer(
        model=model, args=TrainingArguments(**arguments), train_dataset=dataset,
        data_collator=BlockCollator(processor), loss_func=DynamicBlockContrastiveLoss(**config["loss"]),
        callbacks=[SaveRetrievalState(runtime_config, projection, processor)],
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    if trainer.is_world_process_zero():
        directory = Path(arguments["output_dir"])
        processor.save_pretrained(directory)
        save_runtime(directory, runtime_config, projection)
        (directory / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
    return result


def main():
    parser = argparse.ArgumentParser(description="Train the Qwen2.5-VL dynamic block retriever")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    train(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
