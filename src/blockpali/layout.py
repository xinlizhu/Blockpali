from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class PPLayoutCDLA:
    """PP-Layout CDLA detection through RapidLayout's CPU ONNX backend."""

    def __init__(self, model_path=None, confidence=0.5, iou=0.5):
        from rapid_layout import EngineType, ModelType, RapidLayout

        options = {
            "model_type": ModelType.PP_LAYOUT_CDLA,
            "engine_type": EngineType.ONNXRUNTIME,
            "conf_thresh": confidence,
            "iou_thresh": iou,
            "engine_cfg": {"intra_op_num_threads": 1, "inter_op_num_threads": 1},
        }
        if model_path is not None:
            options["model_dir_or_path"] = str(model_path)
        self.engine = RapidLayout(**options)

    def __call__(self, image):
        return self.engine(np.asarray(image.convert("RGB")))


def crop_with_margin(image, bbox, margin=0.05):
    x1, y1, x2, y2 = map(float, bbox)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    bounds = (
        max(0, int(x1 - margin * width)), max(0, int(y1 - margin * height)),
        min(image.width, int(x2 + margin * width)), min(image.height, int(y2 + margin * height)),
    )
    if bounds[2] - bounds[0] < 5 or bounds[3] - bounds[1] < 5:
        return None
    return image.crop(bounds)


def write_page_blocks(image, corpus_id, filename, output_dir, detector, margin=0.05):
    """Write relative image paths; use the whole page only when no valid boxes exist."""
    output_dir = Path(output_dir)
    image = image.convert("RGB")
    page_path = Path("page_images") / f"{filename}.jpg"
    image.save(output_dir / page_path, "JPEG", quality=95)
    result = detector(image)
    boxes = list(result.boxes) if result is not None and result.boxes is not None else []
    labels = list(result.class_names) if result is not None and result.class_names is not None else []
    scores = list(result.scores) if result is not None and getattr(result, "scores", None) is not None else []
    records = []
    for index, box in enumerate(boxes):
        bbox = [float(value) for value in box[:4]]
        cropped = crop_with_margin(image, bbox, margin)
        if cropped is None:
            continue
        path = Path("block_images") / f"{filename}_{index}.jpg"
        cropped.save(output_dir / path, "JPEG", quality=95)
        cropped.close()
        records.append({
            "block_id": f"{corpus_id}_{index}", "corpus_id": corpus_id, "block_idx": index,
            "label": str(labels[index]) if index < len(labels) else "unknown",
            "bbox": bbox, "confidence": float(scores[index]) if index < len(scores) else None,
            "block_image_path": path.as_posix(),
        })
    if not records:
        path = Path("block_images") / f"{filename}_0.jpg"
        image.save(output_dir / path, "JPEG", quality=95)
        records = [{
            "block_id": f"{corpus_id}_0", "corpus_id": corpus_id, "block_idx": 0,
            "label": "full_page", "bbox": [0, 0, image.width, image.height],
            "confidence": None, "block_image_path": path.as_posix(),
        }]
    return {
        "page": {"corpus_id": corpus_id, "page_image_path": page_path.as_posix()},
        "blocks": records,
        "whole_page_block": {
            "block_id": f"{corpus_id}_whole", "corpus_id": corpus_id, "block_idx": 0,
            "label": "whole_page", "bbox": None, "block_image_path": page_path.as_posix(),
        },
    }
