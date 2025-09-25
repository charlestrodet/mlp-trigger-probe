import os
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import torch

from .nethook import TraceDict
from .utils import (
    display_tokens_from_encoding,
    ensure_fp32,
    model_device,
    move_batch_to_device,
)


def save_auroc_curve_lineplot(
    layers: Sequence[int],
    aucs: Sequence[float],
    best_layer: int,
    out_path: str,
    *,
    title_suffix: Optional[str] = None,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 3.2))
    plt.plot(layers, aucs, marker="o", label="AUROC")
    ax = plt.gca()
    ax.set_xlim(min(layers), max(layers))
    xticks = sorted(set(int(x) for x in layers))
    ax.set_xticks(xticks)
    labels = []
    for x in xticks:
        if int(x) == int(best_layer):
            labels.append(f"$\\mathbf{{{x}}}$")
        else:
            labels.append(str(x))
    ax.set_xticklabels(labels)
    plt.axvline(best_layer, linestyle="--", linewidth=2.2, alpha=0.8, label="Best")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    title = "Per-layer AUROC"
    if title_suffix:
        title = f"{title} for \"{title_suffix}\""
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def save_score_distribution(
    all_scores: Sequence[float],
    output_path: str,
    *,
    vline: Optional[float] = None,
    title_suffix: Optional[str] = None,
) -> Optional[str]:
    if not all_scores or len(all_scores) <= 1:
        return None
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(all_scores, bins=50)
    if vline is not None:
        xv = float(vline)
        plt.axvline(x=xv, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
        ymax = plt.ylim()[1]
        plt.text(xv, ymax * 0.97, "threshold", color="#d62728", rotation=90,
                 va="top", ha="right", fontsize=9)
    plt.xlabel("Concept score")
    plt.ylabel("Count")
    title = "Token-level score distribution"
    if title_suffix:
        title = f"{title} for \"{title_suffix}\""
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _make_figure(n_rows: int, height_per_row: float, fig_width: float = 12.0):
    fig, ax = plt.subplots(figsize=(fig_width, max(0.6, height_per_row * n_rows)))
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    return fig, ax


def save_token_heatmap(
    tokens: List[str],
    scores: Sequence[float],
    output_path: str,
    *,
    color_max: Optional[float] = None,
    highlight_token_indices: Optional[Iterable[int]] = None,
    fig_width: float = 12.0,
) -> str:
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams.setdefault("font.monospace", ["DejaVu Sans Mono", "Liberation Mono", "Monospace"])
    plt.rcParams["text.usetex"] = False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # negatives clamped to 0
    clamped = [max(float(s), 0.0) for s in (scores or [])]
    vmin = 0.0
    vmax = float(color_max) if color_max is not None else (float(max(clamped)) if clamped else 1.0)
    # uadratic color scaling to emphasize higher scores
    norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("bright_red", ["#ffffff", "#ff0000"])

    height_per_row = 0.20
    fig, ax = _make_figure(1, height_per_row, fig_width=fig_width)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    axes_w_px = axes_bbox.width if axes_bbox is not None else 1.0
    width_cache: dict[str, float] = {}

    rows: List[List[Tuple[str, float, float, int]]] = [[]]
    x_pos = 0.0

    fp = FontProperties(family="DejaVu Sans Mono", size=12)
    NL_SENTINEL = "\u2063"

    for idx, (tok, s_raw) in enumerate(zip(tokens, scores)):
        s = max(float(s_raw), 0.0)

        break_count = 0
        while tok.endswith(NL_SENTINEL):
            break_count += 1
            tok = tok[:-1]

        if tok == "":
            for _ in range(break_count):
                rows.append([]); x_pos = 0.0
            continue

        w_axes = width_cache.get(tok)
        if w_axes is None:
            w_px, _, _ = renderer.get_text_width_height_descent(tok, fp, ismath=False)
            w_axes = (w_px + 1.0) / max(1.0, axes_w_px)  # +1 px epsilon
            width_cache[tok] = w_axes

        if x_pos + w_axes > 0.995 and rows[-1]:
            rows.append([]); x_pos = 0.0

        rows[-1].append((tok, s, w_axes, idx))
        x_pos += w_axes

        for _ in range(break_count):
            rows.append([]); x_pos = 0.0

    if rows and not rows[-1]:
        rows.pop()

    n_rows = max(1, len(rows))
    fig.set_size_inches(fig.get_figwidth(), max(0.6, height_per_row * n_rows))

    idx_pos: dict[int, Tuple[int, float, float]] = {}
    for r, line in enumerate(rows):
        x = 0.0
        for tok, s, w_axes, idx in line:
            color = cmap(norm(s))
            y1 = 1.0 - (r / n_rows)
            y0 = 1.0 - ((r + 1) / n_rows)
            y = (y0 + y1) / 2.0
            ax.text(
                x, y, tok, transform=ax.transAxes, va="center_baseline", ha="left",
                fontsize=12, color="black", fontfamily="DejaVu Sans Mono",
                bbox={"facecolor": color, "edgecolor": "none", "boxstyle": "square,pad=0.0"},
                zorder=2,
            )
            idx_pos[idx] = (r, x, w_axes)
            x += w_axes

    # draw boxes around highlighted spans
    if highlight_token_indices:
        hi = sorted(set(int(i) for i in highlight_token_indices if i in idx_pos))
        if hi:
            span_start = hi[0]
            prev = hi[0]
            for cur in hi[1:] + [None]:
                flush = (cur is None) or (cur != prev + 1) or (idx_pos[cur][0] != idx_pos[prev][0])
                if flush:
                    r, x0, w0 = idx_pos[span_start]
                    r_end, x1, w1 = idx_pos[prev]
                    assert r == r_end
                    y1 = 1.0 - (r / n_rows)
                    y0 = 1.0 - ((r + 1) / n_rows)
                    total_w = (x1 + w1) - x0
                    rect = Rectangle(
                        (x0, y0), total_w, (y1 - y0),
                        transform=ax.transAxes, fill=False, edgecolor="#c217cf",
                        linewidth=2.2, zorder=10,
                    )
                    ax.add_patch(rect)
                    if cur is not None:
                        span_start = cur
                prev = cur if cur is not None else prev

    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return output_path



def _display_tokens_from_encoding(text: str, tokenizer, enc):
    toks = display_tokens_from_encoding(text, tokenizer, enc)
    token_ids = enc["input_ids"][0].tolist()
    return toks, token_ids


def render_prompt_heatmap_for_model(
    model,
    tokenizer,
    layer_name: str,
    vector: torch.Tensor, 
    text: str,
    output_path: str,
    *,
    color_max: Optional[float] = None,
    highlight_token_indices: Optional[Iterable[int]] = None,
    fig_width: float = 12.0,
):
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    inputs = move_batch_to_device(enc, model_device(model))
    toks, _ = _display_tokens_from_encoding(text, tokenizer, enc)

    with TraceDict(model, [layer_name], retain_input=True, retain_output=False) as traces:
        with torch.no_grad():
            model(**inputs)

    act = traces[layer_name].input[0]
    act = ensure_fp32(act)
    if act.dim() == 3:
        act = act[0]

    # raw dot product, computed in double for stability
    scores = (act.double() @ vector.double().to(act.device)).detach().cpu().tolist()

    saved = save_token_heatmap(
        toks, scores, output_path,
        color_max=color_max, highlight_token_indices=highlight_token_indices, fig_width=fig_width,
    )
    return scores, saved
