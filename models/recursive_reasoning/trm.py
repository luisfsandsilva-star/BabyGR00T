from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import ScaleNorm, LinearSwish, SwiGLU, Attention, CrossAttention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.revin import RevIN

# Optional sparse embedding (only needed for puzzle_emb_ndim > 0)
try:
    from models.sparse_embedding import CastedSparseEmbedding
except ImportError:
    CastedSparseEmbedding = None  # type: ignore

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Task selector (default: language modeling)
    task: str = "language_modeling"  # "language_modeling" | "regression"

    # Regression knobs (used when task=="regression")
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    use_cls_token: bool = True
    pooling: Optional[str] = None  # "first"|"mean"|"last"|"cls" -> point regression; None -> sequence
    # RevIN (flattened for simplicity)
    revin_enabled: bool = False
    revin_eps: float = 1e-5
    revin_affine: bool = True
    revin_apply_on_outputs: bool = False

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # Attention causal mask along sequence (H) dimension
    causal_attn: bool = False
    cross_attn_enabled: bool = False
    cross_attn_num_heads: int = 0  # 0 -> use num_heads
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=config.causal_attn
            )
            # Optional cross-attention after self-attn, before MLP
            if self.config.cross_attn_enabled:
                num_heads_ca = (self.config.cross_attn_num_heads if self.config.cross_attn_num_heads > 0 else self.config.num_heads)
                self.cross_attn = CrossAttention(
                    hidden_size=config.hidden_size,
                    head_dim=config.hidden_size // num_heads_ca,
                    num_heads=num_heads_ca,
                    num_key_value_heads=num_heads_ca,
                    causal=False
                )
                # Learnable single-token context (placeholder until external context is wired)
                self.cross_context = nn.Parameter(trunc_normal_init_(torch.empty(1, 1, config.hidden_size), std=1.0 / (config.hidden_size ** 0.5)))
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps
        if self.config.mlp_t:
            # Peri-ScaleNorm for MLP-T over length dimension
            self.pre_mlp_t = ScaleNorm(self.config.seq_len + self.puzzle_emb_len, eps=self.norm_eps)
            self.post_mlp_t = ScaleNorm(self.config.seq_len + self.puzzle_emb_len, eps=self.norm_eps)
        else:
            # Peri-ScaleNorm pairs for sublayers
            self.pre_attn = ScaleNorm(config.hidden_size, eps=self.norm_eps)
            self.post_attn = ScaleNorm(config.hidden_size, eps=self.norm_eps)
            if self.config.cross_attn_enabled:
                self.pre_cross = ScaleNorm(config.hidden_size, eps=self.norm_eps)
                self.post_cross = ScaleNorm(config.hidden_size, eps=self.norm_eps)
        self.pre_mlp = ScaleNorm(config.hidden_size, eps=self.norm_eps)
        self.post_mlp = ScaleNorm(config.hidden_size, eps=self.norm_eps)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Peri-ScaleNorm: pre-norm before sublayer, post-norm after residual
        if self.config.mlp_t:
            hs = hidden_states.transpose(1,2)
            mlp_in = self.pre_mlp_t(hs)
            out = self.mlp_t(mlp_in)
            hs = hs + out
            hs = self.post_mlp_t(hs)
            hidden_states = hs.transpose(1,2)
        else:
            # Self Attention
            attn_in = self.pre_attn(hidden_states)
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=attn_in)
            hidden_states = hidden_states + attn_out
            hidden_states = self.post_attn(hidden_states)
            # Cross Attention (optional)
            if getattr(self, 'cross_attn', None) is not None:
                # Prefer external context if provided, else fallback to learnable single-token
                context = kwargs.get('cross_context', None)
                if context is None:
                    B = hidden_states.size(0)
                    context = self.cross_context.to(hidden_states.dtype).expand(B, 1, -1)
                cross_in = self.pre_cross(hidden_states)
                cross_out = self.cross_attn(hidden_states=cross_in, context=context)
                hidden_states = hidden_states + cross_out
                hidden_states = self.post_cross(hidden_states)
        # Fully Connected
        mlp_in = self.pre_mlp(hidden_states)
        out = self.mlp(mlp_in)
        hidden_states = hidden_states + out
        hidden_states = self.post_mlp(hidden_states)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Determine task
        self.is_regression = (self.config.task == "regression")

        # Heads shared/gated
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        if not self.is_regression:
            # LM path modules
            self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
            if self.config.puzzle_emb_ndim > 0:
                # Zero init puzzle embeddings
                if CastedSparseEmbedding is None:
                    raise RuntimeError("CastedSparseEmbedding not available. Set puzzle_emb_ndim=0 for regression tasks.")
                self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                        batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
            prefix_len = self.puzzle_emb_len
        else:
            # Regression path modules
            Fin = self.config.input_dim if self.config.input_dim is not None else self.config.hidden_size
            Fout = self.config.output_dim if self.config.output_dim is not None else self.config.hidden_size

            self.in_proj = CastedLinear(Fin, self.config.hidden_size, bias=True)
            if self.config.use_cls_token:
                self.cls = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size, dtype=self.forward_dtype))
                self.reg_prefix_len = 1
            else:
                self.reg_prefix_len = 0
            self.reg_head = CastedLinear(self.config.hidden_size, Fout, bias=True)

            # RevIN
            self.revin = None
            if self.config.revin_enabled:
                self.revin = RevIN(num_features=Fin, eps=self.config.revin_eps, affine=self.config.revin_affine)
            prefix_len = self.reg_prefix_len

        # LM Blocks
        pos_len = self.config.seq_len + (prefix_len if 'prefix_len' in locals() else 0)
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=pos_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(pos_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Cross-attention context projection (lazy init when external context is provided)
        self.cross_ctx_proj: Optional[CastedLinear] = None

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore
        # IO Peri-ScaleNorm
        self.io_norm = ScaleNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale and optional IO norm
        out = self.embed_scale * embedding
        return self.io_norm(out) if getattr(self, "io_norm", None) is not None else out

    def _input_embeddings_regression(self, inputs: torch.Tensor):
        # inputs: [B, L, Fin]
        x = inputs.to(torch.float32)
        ctx = None
        if self.revin is not None:
            x, ctx = self.revin.normalize(x)
        emb = self.in_proj(x.to(self.forward_dtype))  # [B, L, D]
        if getattr(self, 'reg_prefix_len', 0) == 1:
            emb = torch.cat([self.cls.expand(emb.size(0), 1, -1), emb], dim=1)
        if self.config.pos_encodings == "learned":
            emb = 0.707106781 * (emb + self.embed_pos.embedding_weight.to(self.forward_dtype))
        out = self.embed_scale * emb
        out = self.io_norm(out) if getattr(self, "io_norm", None) is not None else out
        return out, ctx

    def empty_carry(self, batch_size: int):
        if self.is_regression:
            total_len = self.config.seq_len + getattr(self, 'reg_prefix_len', 0)
        else:
            total_len = self.config.seq_len + self.puzzle_emb_len
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, total_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, total_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        if self.is_regression:
            input_embeddings, revin_ctx = self._input_embeddings_regression(batch["inputs"])
        else:
            revin_ctx = None
            input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Prepare cross-attention context if enabled and provided (project to hidden_size)
        cross_ctx = None
        if getattr(self.config, 'cross_attn_enabled', False) and ("cross_context_raw" in batch):
            raw = batch["cross_context_raw"].to(self.forward_dtype)  # [B, Lc, Fc]
            B, Lc, Fc = raw.shape
            D = self.config.hidden_size
            if Fc == D:
                cross_ctx = raw
            else:
                # Lazy-create or replace projection if feature size changed.
                # IMPORTANT: ensure the projection lives on the same device as `raw`
                # so we don't end up with CPU weights and CUDA inputs during inference.
                if (self.cross_ctx_proj is None) or (self.cross_ctx_proj.weight.shape[1] != Fc):
                    self.cross_ctx_proj = CastedLinear(Fc, D, bias=False).to(raw.device)
                cross_ctx = self.cross_ctx_proj(raw)

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info, cross_context=cross_ctx)
                z_H = self.L_level(z_H, z_L, **seq_info, cross_context=cross_ctx)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info, cross_context=cross_ctx)
        z_H = self.L_level(z_H, z_L, **seq_info, cross_context=cross_ctx)

        # Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        if self.is_regression:
            # q-head reads CLS (position 0)
            q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
            pooling = self.config.pooling
            if pooling is None:
                # sequence regression
                y_pred = self.reg_head(z_H)[:, getattr(self, 'reg_prefix_len', 0):]
            else:
                # point regression
                if pooling == "cls":
                    pooled = z_H[:, 0]
                elif pooling == "first":
                    pooled = z_H[:, getattr(self, 'reg_prefix_len', 0)]
                elif pooling == "last":
                    pooled = z_H[:, -1]
                elif pooling == "mean":
                    pooled = z_H[:, getattr(self, 'reg_prefix_len', 0):].mean(dim=1)
                else:
                    pooled = z_H[:, getattr(self, 'reg_prefix_len', 0):].mean(dim=1)
                y_pred = self.reg_head(pooled)

            if (self.revin is not None) and (not self.config.revin_apply_on_outputs) and (revin_ctx is not None) and (y_pred.shape[-1] == revin_ctx["mu"].shape[-1]):
                y_out = self.revin.denormalize_out(y_pred, revin_ctx)
            else:
                # Optional IO norm on final hidden before reg_head already applied; here y_pred is after head
                y_out = y_pred
            return new_carry, y_out, (q_logits[..., 0], q_logits[..., 1]), revin_ctx
        else:
            h_out = self.io_norm(z_H) if getattr(self, "io_norm", None) is not None else z_H
            output = self.lm_head(h_out)[:, self.puzzle_emb_len:]
            q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
            return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), None


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), revin_ctx = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        if revin_ctx is not None:
            outputs["revin_ctx"] = revin_ctx

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step
            # Always allow ACT halting based on Q-head logits (both train and eval),
            # but keep Q-continue targets only for training (no exploration).
            if self.config.halt_max_steps > 1:
                # Halt signal
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Only during training do we compute Q-continue bootstrapping targets.
                if self.training and (not self.config.no_ACT_continue):
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
