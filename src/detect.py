import os
from typing import Dict, List

import torch

from .any_model import get_model_info
from .utils import load_model_and_tokenizer
from .visualize import save_score_distribution, render_prompt_heatmap_for_model


def _load_artifact(path: str) -> tuple[List[torch.Tensor], Dict]:
    saved = torch.load(path, map_location="cpu")
    vectors = saved.get("vectors")
    meta = saved.get("meta", {})
    return vectors, meta


def mode_detect(args):
    print(f"[detect] Loading artifact: {args.vector_file}")
    vectors, meta = _load_artifact(args.vector_file)
    model_name = meta["model"]
    best_layer = int(meta["chosen_layer"])
    pos_means = meta.get("pos_means", [])

    if args.layer is not None:
        best_layer = int(args.layer)

    print(f"[detect] Loading model {model_name} (device={args.device}, dtype={args.dtype})")
    model, tokenizer = load_model_and_tokenizer(model_name, args.device, args.dtype)
    layer_names = [layer.name for layer in get_model_info(model).mlp_layers]

    if not (0 <= best_layer < len(layer_names)):
        raise ValueError(f"--layer must be within 0..{len(layer_names)-1}")

    layer_name = layer_names[best_layer]
    color_max = pos_means[best_layer] if 0 <= best_layer < len(pos_means) else None
    trigger_name = meta.get("trigger_name")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[detect] Scoring with layer {best_layer} ({layer_name}) [dot product]")
    vec = vectors[best_layer]

    scores, out_path = render_prompt_heatmap_for_model(
        model=model,
        tokenizer=tokenizer,
        layer_name=layer_name,
        vector=vec,
        text=args.prompt,
        output_path=os.path.join(args.output_dir, "prompt_heatmap.png"),
        color_max=color_max,
    )
    dist_path = save_score_distribution(
        scores,
        os.path.join(args.output_dir, "score_distribution.png"),
        title_suffix=trigger_name,
    )
    print(f"[detect] Saved -> {out_path}")
    if dist_path:
        print(f"[detect] Saved -> {dist_path}")
