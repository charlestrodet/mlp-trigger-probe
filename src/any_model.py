"""Locate MLP modules and model info."""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class LayerInfo:
    """Transformer MLP layer info."""

    index: int                 # zero-based layer index
    name: Optional[str]        # fully-qualified name, if available
    mlp_down: torch.nn.Module  # mlp-down module


@dataclass(frozen=True)
class ModelInfo:
    """Model metadata."""

    mlp_layers: List[LayerInfo]
    hidden_dim: int
    layer_count: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "layer_count", len(self.mlp_layers))


def get_model_info(
    model: torch.nn.Module,
    blocks_attr_candidates: Optional[Sequence[Tuple[Optional[str], str]]] = None,
    mlp_attr_candidates: Optional[Sequence[str]] = None,
) -> ModelInfo:
    """Return MLP layers and hidden dim."""
    blocks = _get_blocks(model, blocks_attr_candidates)
    module_names = {m: name for name, m in model.named_modules()}

    mlp_layers = []
    for i, block in enumerate(blocks):
        mlp_module = _find_mlp_down_proj(block, mlp_attr_candidates)
        if mlp_module is not None:
            mlp_layers.append(
                LayerInfo(index=i, name=module_names.get(mlp_module), mlp_down=mlp_module)
            )

    hidden_dim = _infer_hidden_dim(mlp_layers[0].mlp_down) if mlp_layers else 0
    if not hidden_dim:
        raise RuntimeError("Could not infer hidden dimension from model.")

    return ModelInfo(mlp_layers=mlp_layers, hidden_dim=hidden_dim)


def _get_blocks(
    model: torch.nn.Module,
    blocks_attr_candidates: Optional[Sequence[Tuple[Optional[str], str]]] = None,
) -> List[torch.nn.Module]:
    """Find transformer blocks."""
    candidates = blocks_attr_candidates or [
        ("model", "layers"),
        ("transformer", "h"),
        (None, "layers"),
        (None, "h"),
        ("encoder", "layer"),
        ("decoder", "layer"),
        (None, "blocks"),
    ]
    for obj_attr, attr in candidates:
        base = getattr(model, obj_attr) if obj_attr else model
        seq = getattr(base, attr, None)
        if seq is not None:
            return list(seq)
    raise RuntimeError("Could not find transformer blocks.")


def _find_mlp_down_proj(
    block: torch.nn.Module, mlp_attr_candidates: Optional[Sequence[str]] = None
) -> Optional[torch.nn.Module]:
    """Return down-projection module or None."""
    candidates = mlp_attr_candidates or [
        "mlp.down_proj",
        "mlp.c_proj",
        "mlp.fc_in",
        "mlp.fc1",
        "down_proj",
        "c_proj",
        "fc_in",
        "fc1",
        "dense_4h_to_h",
    ]
    for name in candidates:
        attrs = name.split(".")
        mod = block
        ok = True
        for attr in attrs:
            mod = getattr(mod, attr, None)
            if mod is None:
                ok = False
                break
        if ok:
            return mod

    for _, mod in block.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if hasattr(mod, "out_features") and hasattr(mod, "in_features"):
                if mod.out_features < mod.in_features:
                    return mod
    return None


def _infer_hidden_dim(module: Optional[torch.nn.Module]) -> Optional[int]:
    """Infer hidden dim from module."""
    if module is None:
        return None
    if hasattr(module, "in_features"):
        return module.in_features
    if hasattr(module, "weight"):
        w = module.weight
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return w.size(1)
    return None
