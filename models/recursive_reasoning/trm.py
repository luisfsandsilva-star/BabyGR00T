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
from models.layers import (
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
    CrossAttention,
    make_inbound_scalenorm,
    make_outbound_scalenorm,
)
from models.sparse_embedding import CastedSparseEmbedding


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

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    latent_dim: Optional[int] = None
    output_dim: Optional[int] = None
    expansion: float
    num_heads: int
    pos_encodings: str

    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    use_cross_attn: bool = False
    keep_act_halting_head: bool = True
    use_constant_cross_attn: bool = False
    cross_attn_constant_dim: int = 0

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
            mlp_t_dim = self.config.seq_len + self.puzzle_emb_len
            self.mlp_t_in_norm = make_inbound_scalenorm(mlp_t_dim)
            self.mlp_t_out_norm = make_outbound_scalenorm(mlp_t_dim)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
            self.self_attn_in_norm = make_inbound_scalenorm(config.hidden_size)
            self.self_attn_out_norm = make_outbound_scalenorm(config.hidden_size)
            if getattr(self.config, "use_cross_attn", False):
                self.cross_attn = CrossAttention(
                    hidden_size=config.hidden_size,
                    head_dim=config.hidden_size // config.num_heads,
                    num_heads=config.num_heads,
                    num_key_value_heads=config.num_heads,
                    causal=False
                )
                self.cross_attn_in_norm = make_inbound_scalenorm(config.hidden_size)
                self.cross_attn_out_norm = make_outbound_scalenorm(config.hidden_size)
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.mlp_in_norm = make_inbound_scalenorm(config.hidden_size)
        self.mlp_out_norm = make_outbound_scalenorm(config.hidden_size)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, encoder_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            residual = hidden_states
            branch = self.mlp_t_out_norm(self.mlp_t(self.mlp_t_in_norm(hidden_states)))
            hidden_states = residual + branch
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            residual = hidden_states
            branch = self.self_attn(
                cos_sin=cos_sin,
                hidden_states=self.self_attn_in_norm(hidden_states)
            )
            hidden_states = residual + self.self_attn_out_norm(branch)
            # Cross Attention (optional)
            if getattr(self.config, "use_cross_attn", False) and (encoder_states is not None):
                # no rope on cross attention
                residual = hidden_states
                branch = self.cross_attn(
                    cos_sin=None,
                    hidden_states=self.cross_attn_in_norm(hidden_states),
                    encoder_states=encoder_states,
                )
                hidden_states = residual + self.cross_attn_out_norm(branch)
        # Fully Connected
        residual = hidden_states
        branch = self.mlp(self.mlp_in_norm(hidden_states))
        hidden_states = residual + self.mlp_out_norm(branch)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        if len(layers) > 0:
            hidden_size = layers[0].config.hidden_size
            self.input_in_norm = make_inbound_scalenorm(hidden_size)
            self.input_out_norm = make_outbound_scalenorm(hidden_size)
        else:
            self.input_in_norm = None
            self.input_out_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cross_attn_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_injection is not None and self.input_in_norm is not None:
            input_injection = self.input_out_norm(self.input_in_norm(input_injection))
            hidden_states = hidden_states + input_injection
        elif input_injection is not None:
            hidden_states = hidden_states + input_injection
        encoder_states = cross_attn_states if cross_attn_states is not None else input_injection
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_states=encoder_states,
                **kwargs,
            )
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.latent_dim = self.config.latent_dim or self.config.hidden_size
        self.output_dim = self.config.output_dim or self.config.hidden_size
        self.config.latent_dim = self.latent_dim
        self.config.output_dim = self.output_dim

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.input_projector = CastedLinear(self.latent_dim, self.config.hidden_size, bias=False)
        self.regression_head = CastedLinear(self.config.hidden_size, self.output_dim, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True) if self.config.keep_act_halting_head else None
        self.embedding_in_norm = make_inbound_scalenorm(self.config.hidden_size)
        self.embedding_out_norm = make_outbound_scalenorm(self.config.hidden_size)
        self.output_in_norm = make_inbound_scalenorm(self.config.hidden_size)
        self.output_out_norm = make_outbound_scalenorm(self.config.hidden_size)

        if (
            getattr(self.config, "use_cross_attn", False)
            and getattr(self.config, "use_constant_cross_attn", False)
            and self.config.cross_attn_constant_dim > 0
        ):
            constant = trunc_normal_init_(
                torch.empty(self.config.cross_attn_constant_dim, dtype=self.forward_dtype),
                std=1.0,
            )
            self.cross_attn_constant = nn.Parameter(constant)
            self.cross_attn_constant_projector = CastedLinear(
                self.config.cross_attn_constant_dim,
                self.config.hidden_size,
                bias=False,
            )
        else:
            self.register_parameter("cross_attn_constant", None)
            self.cross_attn_constant_projector = None

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        if self.q_head is not None:
            with torch.no_grad():
                self.q_head.weight.zero_()
                self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor]):
        # Token embedding
        embedding = self.input_projector(input.to(self.forward_dtype))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0 and puzzle_identifiers is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        elif self.puzzle_emb_len > 0:
            zeros = torch.zeros(
                embedding.shape[0],
                self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=embedding.device,
            )
            embedding = torch.cat((zeros, embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale + Normalize
        embedding = self.embed_scale * embedding
        embedding = self.embedding_out_norm(self.embedding_in_norm(embedding))

        cross_attn_context = None
        if self.cross_attn_constant_projector is not None and self.cross_attn_constant is not None:
            projected = self.cross_attn_constant_projector(self.cross_attn_constant)
            cross_attn_context = projected.unsqueeze(0)

        return embedding, cross_attn_context

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings, cross_attn_context = self._input_embeddings(
            batch["inputs"], batch.get("puzzle_identifiers")
        )
        cross_attn_states = None
        if cross_attn_context is not None:
            cross_attn_states = cross_attn_context.unsqueeze(0).expand(
                input_embeddings.shape[0], -1, -1
            )

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(
                        z_L,
                        z_H + input_embeddings,
                        cross_attn_states=cross_attn_states,
                        **seq_info,
                    )
                z_H = self.L_level(
                    z_H,
                    z_L,
                    cross_attn_states=cross_attn_states,
                    **seq_info,
                )
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(
                z_L,
                z_H + input_embeddings,
                cross_attn_states=cross_attn_states,
                **seq_info,
            )
        z_H = self.L_level(
            z_H,
            z_L,
            cross_attn_states=cross_attn_states,
            **seq_info,
        )

        # LM Outputs
        z_proj = self.output_out_norm(self.output_in_norm(z_H))
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.regression_head(z_proj)[:, self.puzzle_emb_len:]
        if self.q_head is None:
            return new_carry, output, None

        q_logits = self.q_head(z_proj[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


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
        new_inner_carry, logits, q_logits = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
        }
        if q_logits is not None:
            q_halt_logits, q_continue_logits = q_logits
            outputs.update({
                "q_halt_logits": q_halt_logits,
                "q_continue_logits": q_continue_logits
            })
        else:
            q_halt_logits = q_continue_logits = None

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1) and (q_halt_logits is not None):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes

                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, next_q_logits = self.inner(new_inner_carry, new_current_data)
                    if next_q_logits is not None:
                        next_q_halt_logits, next_q_continue_logits = next_q_logits
                        outputs["target_q_continue"] = torch.sigmoid(
                            torch.where(
                                is_last_step,
                                next_q_halt_logits,
                                torch.maximum(next_q_halt_logits, next_q_continue_logits),
                            )
                        )

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
