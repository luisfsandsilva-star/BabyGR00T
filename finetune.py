#!/usr/bin/env python3
"""
Thin wrapper around `pretrain.py`.

Historically this repo had two near-identical entrypoints (`pretrain.py` and `finetune.py`).
To reduce maintenance overhead and avoid subtle divergence, `finetune.py` now delegates
directly to `pretrain.launch()` and therefore supports the same Hydra config/overrides.
"""

from __future__ import annotations

from pretrain import launch


if __name__ == "__main__":
    launch()



