import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
from tqdm import tqdm

from .any_model import ModelInfo, get_model_info
from .nethook import TraceDict
from .utils import (
    ensure_fp32,
    extract_tagged_trigger_token_spans,
    load_model_and_tokenizer,
    model_device,
    move_batch_to_device,
    display_tokens_from_encoding,
)
from .visualize import save_token_heatmap, save_auroc_curve_lineplot


@dataclass
class FitConfig:
    model: str                                      # model HF id or local path
    prompts: List[str]                              # list of training prompts (with <T|...|T> tags)
    device: str = "auto"                            # device for model, "cpu", "cuda", or "auto" 
    dtype: str = "auto"                             # "fp32", "fp16", "bf16", or "auto"
    vectors_out: str = "outputs/vectors.pt"         # path to save vectors + metadata (torch .pt)
    plot_out: str = "outputs/auroc_per_layer.png"   # path to save AUROC per-layer plot
    trigger_name: Optional[str] = None              # optional human-readable name to insert in plots


@dataclass
class LayerStats:
    layer: int      # layer index
    auroc: float    # AUROC score
    pos_mean: float # mean positive score (for trigger tokens)
    neg_mean: float # mean background score


@dataclass
class FitArtifacts:
    model: str                          # model HF id or local path
    chosen_layer: int                   # The layer index chosen by AUROC
    vectors: List[torch.Tensor]         # one per layer
    stats: List[LayerStats]             # one per layer
    pos_means: List[float]              # positive score mean per layer, used for visualization color scaling
    trigger_name: Optional[str] = None  # optional human-readable name to insert in plots

    def to_torch(self) -> Dict:
        return {
            "meta": {
                "model": self.model,
                "chosen_layer": self.chosen_layer,
                "pos_means": self.pos_means,
                "stats": [s.__dict__ for s in self.stats],
                "scoring": "dot_product",
                "trigger_name": self.trigger_name,
            },
            "vectors": self.vectors,
        }


def _normalize_activation_shape(act: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Return [T, H] tensor for a single example."""
    act = ensure_fp32(act)
    if act.dim() == 3:  # [B, T, H]
        act = act[0]
    if act.dim() == 2 and act.size(1) == seq_len and act.size(0) != seq_len:
        act = act.transpose(0, 1)  # [H,T] -> [T,H]
    return act  # [T, H]


def _collect_activations_per_prompt(
    model,
    tokenizer,
    prompts: Sequence[str],
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]], ModelInfo, List[dict]]:
    """Returns per-layer lists of trigger/background vectors and per-prompt caches for viz."""
    model_info = get_model_info(model)
    trace_layers = [layer.name for layer in model_info.mlp_layers if layer.name]

    per_layer_pos: Dict[int, List[torch.Tensor]] = {i: [] for i in range(model_info.layer_count)}
    per_layer_neg: Dict[int, List[torch.Tensor]] = {i: [] for i in range(model_info.layer_count)}
    per_prompt_cached: List[dict] = []

    for p in tqdm(prompts, desc="[fit] Collecting activations"):
        cleaned, trig_span_lists = extract_tagged_trigger_token_spans(p, tokenizer)
        # captured positions is the LAST token of each trigger span
        trig_positions = [max(span) for span in trig_span_lists if span]
        if not trig_positions:
            raise ValueError(f"Prompt missing <T|...|T> tag: {p!r}")

        enc = tokenizer(cleaned, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        toks = display_tokens_from_encoding(cleaned, tokenizer, enc)

        seq_len = int(enc["input_ids"].shape[-1])
        # background tokens are all tokens not part of any tagged trigger tokens
        all_span_positions = sorted({i for span in trig_span_lists for i in span})
        if len(set(range(seq_len)) - set(all_span_positions)) == 0:
            raise ValueError("Prompt has no background tokens; all text is inside <T|...|T>.")

        inputs = move_batch_to_device(enc, model_device(model))

        with TraceDict(model, trace_layers, retain_input=True, retain_output=False) as traces:
            with torch.no_grad():
                model(**inputs)

        cached_seq_acts = {}
        for layer in model_info.mlp_layers:
            name = layer.name
            if name not in traces:
                continue
            act = _normalize_activation_shape(traces[name].input[0], seq_len)  # [T,H]
            Hcpu = act.detach().cpu()

            for i in trig_positions:
                if 0 <= i < Hcpu.size(0):
                    per_layer_pos[layer.index].append(ensure_fp32(Hcpu[i]))
            pos_set = set(all_span_positions)
            for j in range(Hcpu.size(0)):
                if j not in pos_set:
                    per_layer_neg[layer.index].append(ensure_fp32(Hcpu[j]))

            cached_seq_acts[layer.index] = Hcpu.half()

        per_prompt_cached.append({
            "tokens": toks,
            "trig_positions": trig_positions,
            "trig_span_positions": all_span_positions,
            "layer_acts": cached_seq_acts,  # [T,H] half
        })

    return per_layer_pos, per_layer_neg, model_info, per_prompt_cached


def _compute_vector_for_layer(
    pos_list: List[torch.Tensor],
    neg_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Concept reading vector r_l: mean of positive activations (optionally L2-normalized).
    Scores use raw activations (not normalized): s = a · r.
    Returns:
      r (float32, CPU), pos_scores (np), neg_scores (np)
    """
    eps = 1e-8
    if not pos_list or not neg_list:
        H = pos_list[0].numel() if pos_list else (neg_list[0].numel() if neg_list else 1)
        r = torch.zeros(H, dtype=torch.float32)
        return r, np.array([]), np.array([])

    T = torch.stack(pos_list, dim=0).float()  # [P,H] (P = number of positive token activations)
    B = torch.stack(neg_list, dim=0).float()  # [B,H] (B = number of background token activations)

    m = T.mean(dim=0)
    denom = m.norm(p=2).clamp_min(eps)
    m = m / denom

    r = m.to(torch.float32).cpu()

    # score with raw activations dot product
    pos = (T.double() @ r.double()).cpu().numpy()
    neg = (B.double() @ r.double()).cpu().numpy()
    return r, pos, neg


def _evaluate_layers(
    per_layer_pos: Dict[int, List[torch.Tensor]],
    per_layer_neg: Dict[int, List[torch.Tensor]],
    model_info: ModelInfo,
) -> Tuple[List[torch.Tensor], List[LayerStats], int]:
    vectors: List[torch.Tensor] = []
    stats: List[LayerStats] = []

    for layer_idx in range(model_info.layer_count):
        pos_vecs = per_layer_pos[layer_idx]
        neg_vecs = per_layer_neg[layer_idx]

        r, pos, neg = _compute_vector_for_layer(pos_vecs, neg_vecs)

        if pos.size and neg.size and pos.mean() < neg.mean():
            r.mul_(-1.0)
            pos, neg = -pos, -neg

        if pos.size and neg.size:
            y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
            s = np.concatenate([pos, neg])
            auc = float(roc_auc_score(y, s))
            pm, nm = float(pos.mean()), float(neg.mean())
        else:
            auc, pm, nm = float("nan"), 0.0, 0.0

        vectors.append(r)
        stats.append(LayerStats(layer=layer_idx, auroc=auc, pos_mean=pm, neg_mean=nm))

    # choose best using AUROC
    best_layer = 0
    best_auc = -1.0
    for s in stats:
        a = s.auroc
        if a is not None and np.isfinite(a) and a > best_auc:
            best_auc = a
            best_layer = s.layer

    return vectors, stats, best_layer


def _print_chosen_layer_stats(*, vectors: List[torch.Tensor], stats: List[LayerStats],
                              best_layer: int):
    vec = vectors[best_layer].float()
    chosen = next((s for s in stats if s.layer == best_layer), None)

    print("[fit] =================== Chosen layer stats ===================")
    print(f"[fit] layer: {best_layer}")
    if chosen:
        delta = chosen.pos_mean - chosen.neg_mean
        print(f"[fit] AUROC: {chosen.auroc:.6f}")
        print(f"[fit] pos_mean: {chosen.pos_mean:.6f} | neg_mean: {chosen.neg_mean:.6f} | delta: {delta:.6f}")
    print("[fit] ==========================================================")


def _load_fit_config(path: str) -> FitConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for key in ("model", "prompts", "device", "dtype", "vectors_out", "plot_out"):
        if key not in raw:
            raise ValueError(f"Missing required config key: '{key}'")
    if not isinstance(raw["prompts"], list) or not all(isinstance(x, str) for x in raw["prompts"]):
        raise ValueError("config['prompts'] must be a list of strings")

    return FitConfig(
        model=raw["model"],
        prompts=raw["prompts"],
        device=raw.get("device", "auto"),
        dtype=raw.get("dtype", "auto"),
        vectors_out=raw.get("vectors_out", "vectors.pt"),
        plot_out=raw.get("plot_out", "outputs/auroc_per_layer.png"),
        trigger_name=raw.get("trigger_name"),
    )


def _save_artifacts(art: FitArtifacts, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(art.to_torch(), out_path)
    return out_path


def _render_training_heatmaps(
    per_prompt_cached: List[dict],
    best_layer: int,
    vec: torch.Tensor,
    out_dir: str,
    color_max: Optional[float] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    for i, info in enumerate(per_prompt_cached):
        toks = info["tokens"]
        highlight = info.get("trig_span_positions") or info.get("trig_positions") or []
        act_seq = info["layer_acts"].get(best_layer)
        if act_seq is None:
            continue
        # raw dot product in double precision
        scores = (ensure_fp32(act_seq).double() @ vec.double()).cpu().tolist()
        save_token_heatmap(
            toks,
            scores,
            os.path.join(out_dir, f"prompt{i+1}_heatmap.png"),
            color_max=color_max,
            highlight_token_indices=highlight,
        )


def mode_fit(args):
    cfg = _load_fit_config(args.config)
    print(f"[fit] Loading model: {cfg.model} (device={cfg.device}, dtype={cfg.dtype})")
    model, tokenizer = load_model_and_tokenizer(cfg.model, cfg.device, cfg.dtype)

    per_layer_pos, per_layer_neg, model_info, per_prompt_cached = _collect_activations_per_prompt(
        model, tokenizer, cfg.prompts
    )

    print("[fit] Computing concept vectors (dot product) + AUROC per layer")
    vectors, stats, best_layer = _evaluate_layers(
        per_layer_pos, per_layer_neg, model_info
    )

    pos_means = [s.pos_mean for s in stats]
    art = FitArtifacts(
        model=cfg.model,
        chosen_layer=best_layer,
        vectors=vectors,
        stats=stats,
        pos_means=pos_means,
        trigger_name=cfg.trigger_name,
    )

    out_path = _save_artifacts(art, cfg.vectors_out)
    print(f"[fit] Saved vectors + metadata -> {out_path}")
    chosen = next((s for s in stats if s.layer == best_layer), None)
    if chosen:
        print(f"[fit] Best layer = {best_layer} | AUROC={chosen.auroc:.4f}")

    _print_chosen_layer_stats(vectors=vectors, stats=stats, best_layer=best_layer)

    os.makedirs(os.path.dirname(cfg.plot_out) or ".", exist_ok=True)
    save_auroc_curve_lineplot(
        layers=[s.layer for s in stats],
        aucs=[s.auroc for s in stats],
        best_layer=best_layer,
        out_path=cfg.plot_out,
        title_suffix=cfg.trigger_name,
    )
    print(f"[fit] Saved AUROC plot -> {cfg.plot_out}")

    try:
        out_dir = os.path.dirname(cfg.plot_out) or "."
        vec = vectors[best_layer]
        color_max = chosen.pos_mean if chosen else None
        _render_training_heatmaps(per_prompt_cached, best_layer, vec, out_dir, color_max=color_max)
        print(f"[fit] Saved training example heatmaps -> {out_dir}/prompt*_heatmap.png")
    except Exception as e:
        print(f"[fit] Warning: failed to render training heatmaps: {e}")
