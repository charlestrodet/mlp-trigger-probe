"""Minimal tracing helpers: Trace one or many layers (TraceDict)."""

import contextlib
from collections import OrderedDict

import torch


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Return the submodule with the exact dotted name."""
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


class Trace(contextlib.AbstractContextManager):
    """Attach a forward hook to a named submodule and retain input/output."""

    def __init__(self, module: torch.nn.Module, layer: str, *, retain_output: bool = True, retain_input: bool = False):
        self.layer = layer
        target = get_module(module, layer)

        def hook_fn(_m, inputs, output):
            if retain_input:
                # If single tensor arg, keep that tensor; else keep the tuple
                self.input = inputs[0] if len(inputs) == 1 else inputs
            if retain_output:
                self.output = output
            return output

        self._hook = target.register_forward_hook(hook_fn)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        self._hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """Retain inputs/outputs for multiple named layers during a forward pass."""

    def __init__(self, module: torch.nn.Module, layers, *, retain_output: bool = True, retain_input: bool = False):
        super().__init__()
        for layer in layers or []:
            self[layer] = Trace(module, layer, retain_output=retain_output, retain_input=retain_input)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        for tr in self.values():
            tr.close()
