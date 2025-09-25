from typing import List, Tuple
import re
import unicodedata2

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_fp32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.float()


def model_device(model) -> torch.device:
    params = model.parameters()
    first = next(params, None)
    return first.device if first is not None else torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    if batch is None:
        return batch
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _charspan_to_token_indices_with_offsets(
    offsets: List[Tuple[int, int]], cstart: int, cend: int
) -> List[int]:
    hits = []
    for ti, (s, e) in enumerate(offsets):
        if s is None or e is None or (s == 0 and e == 0):
            continue
        if not (e <= cstart or s >= cend):
            hits.append(ti)
    return hits


def extract_tagged_trigger_token_spans(prompt: str, tokenizer) -> Tuple[str, List[List[int]]]:
    """
    Replace <T|...|T> with content and return per-span token indices.

    Returns:
      cleaned: prompt with tags removed
      spans_token_indices: list of index lists for each <T|...|T> span
    """
    tag_re = re.compile(r"<T\|(.*?)\|T>")
    matches = list(tag_re.finditer(prompt))
    if not matches:
        raise ValueError(f"Prompt missing <T|...|T> tag: {prompt!r}")

    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    last = 0
    for m in matches:
        parts.append(prompt[last:m.start()])
        inner = m.group(1)
        start = sum(len(p) for p in parts)
        parts.append(inner)
        end = start + len(inner)
        spans.append((start, end))
        last = m.end()
    parts.append(prompt[last:])
    cleaned = "".join(parts)

    enc = tokenizer(cleaned, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    if not offsets:
        raise ValueError("Tokenizer did not return offsets; cannot locate triggers.")
    if not isinstance(offsets[0], (list, tuple)):
        offsets = offsets[0]

    spans_token_indices: List[List[int]] = []
    for cs, ce in spans:
        spans_token_indices.append(_charspan_to_token_indices_with_offsets(offsets, cs, ce))
    return cleaned, spans_token_indices


def load_model_and_tokenizer(model_name: str, device: str, dtype_arg: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "auto": None,
    }
    torch_dtype = dtype_map.get(dtype_arg, None)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=("auto" if device == "auto" else None),
        torch_dtype=torch_dtype,
    )
    model.eval()

    if device != "auto":
        model.to(torch.device("cpu" if device == "cpu" else device))

    return model, tokenizer


def clean_token_display(tok: str) -> str:
    if not isinstance(tok, str):
        return str(tok)
    t = unicodedata2.normalize("NFKC", tok)
    t = t.replace("Ċ", "\n").replace("ĉ", "\n").replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("▁\n", "\n").replace("\t", "    ")
    t = t.replace("Ġ", " ").replace("▁", " ")
    t = t.replace("\u00A0", " ").replace("\u200b", "")
    if t == "\n":
        return "\n"
    if "$" in t:
        t = t.replace("$", r"\$")
    return t or " "


def display_tokens_from_encoding(text: str, tokenizer, enc) -> List[str]:
    offsets = enc["offset_mapping"][0]
    toks = []
    NL_SENTINEL = "\u2063"

    for (cs, ce) in offsets:
        if cs is None or ce is None or (cs == 0 and ce == 0):
            raw = " "
        else:
            raw = text[cs:ce]

        if "\n" in raw:
            parts = raw.split("\n")
            visible = parts[0]
            n_breaks = len(parts) - 1
            tok = (visible + (NL_SENTINEL * n_breaks)) if visible != "" else (NL_SENTINEL * n_breaks)
            toks.append(tok)
        else:
            toks.append(raw)

    return [clean_token_display(t) for t in toks]

