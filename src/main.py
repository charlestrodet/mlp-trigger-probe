#!/usr/bin/env python3
"""CLI entrypoint."""

import argparse
import os

from . import detect, fit


def _read_prompt_arg(maybe_path: str) -> str:
    if os.path.isfile(maybe_path):
        with open(maybe_path, "r", encoding="utf-8") as f:
            return f.read()
    return maybe_path


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trigger-vector fit + detection in pre-down MLP activations."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # fit
    p_fit = sub.add_parser("fit", help="Fit trigger vectors from tagged prompts.")
    p_fit.add_argument("--config", required=True, type=str, help="Path to JSON fit configuration.")

    # detect
    p_det = sub.add_parser("detect", help="Score a single prompt and visualize.")
    p_det.add_argument("--prompt", required=True, type=str, help="The prompt text, or a path to a file containing it.")
    p_det.add_argument("--vector-file", required=True, type=str, help="Path to artifact from `fit` (torch .pt).")
    p_det.add_argument("--layer", type=int, help="Override the layer index (0-based).")
    p_det.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p_det.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "auto"], default="auto", help="Model weights dtype.")
    p_det.add_argument("--output-dir", type=str, default="outputs", help="Directory to write visualization images.")
    
    return parser


def main():
    args = build_cli().parse_args()
    if args.mode == "detect":
        args.prompt = _read_prompt_arg(args.prompt)
        detect.mode_detect(args)
    else:  # fit
        fit.mode_fit(args)


if __name__ == "__main__":
    main()
