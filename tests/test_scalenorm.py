import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.layers import ScaleNorm
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1Config,
)


def test_scale_norm_initialization_sets_expected_g():
    norm = ScaleNorm(dim=16)
    assert math.isclose(norm.g.detach().item(), math.sqrt(16.0))


def test_scale_norm_forward_produces_expected_norm():
    norm = ScaleNorm(dim=8)
    inputs = torch.randn(2, 3, 8)
    outputs = norm(inputs)
    expected = torch.full(outputs.shape[:-1], norm.g.detach().item(), dtype=outputs.dtype)
    actual = torch.linalg.norm(outputs, dim=-1)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_trm_block_uses_scale_norm_pairs():
    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=1,
        seq_len=4,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=16,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=1,
        hidden_size=8,
        expansion=2.0,
        num_heads=2,
        pos_encodings="rope",
        rope_theta=10000.0,
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
        use_cross_attn=False,
    )
    block = TinyRecursiveReasoningModel_ACTV1Block(config)

    assert isinstance(block.self_attn_in_norm, ScaleNorm)
    assert isinstance(block.self_attn_out_norm, ScaleNorm)
    assert isinstance(block.mlp_in_norm, ScaleNorm)
    assert isinstance(block.mlp_out_norm, ScaleNorm)
    assert math.isclose(
        block.self_attn_in_norm.g.detach().item(),
        math.sqrt(config.hidden_size),
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
