"""Vision pipeline.

Two stages:
  1. InternVL3Vision (offline, used by scripts/cache_vision.py): runs the full
     InternVL3-1B (8-bit, frozen) on per-chunk PIL frames + the task prompt and
     returns all 25 LLM-layer hidden states.
  2. LayerAggregator + PerceiverResampler (online, trainable): aggregates the
     25 layers into one tensor per-token and compresses N_tok ≈ hundreds of
     tokens down to NUM_RESAMPLER_LATENTS=128 latents that the policy reads
     via cross-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Module-level constants used by external callers (vision cache + policy) ──
INTERNVL3_MODEL_ID    = "OpenGVLab/InternVL3-1B-hf"
TILE_SIZE             = 448
VIS_HIDDEN_DIM        = 896      # InternVL3-1B LLM hidden size
NUM_RESAMPLER_LATENTS = 128
NUM_FRAMES            = 4
LAYER_SCALE_INIT      = 0.1


class ScaleNorm(nn.Module):
    """Single-scalar L2-normalized rescale (Nguyen & Salazar 2019)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=self.eps) * self.g


# ════════════════════════════════════════════════════════════
#  LayerAggregator — per-token softmax over the 25 LLM layers
# ════════════════════════════════════════════════════════════

class LayerAggregator(nn.Module):
    """Per-(token, channel) softmax gating over the LLM-layer stack.
    Lets the resampler see a learned mixture of low/mid/high LLM features
    instead of just the final layer.
    """
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias_pre  = nn.Parameter(torch.zeros(n_layers, hidden_dim))
        self.bias_post = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, hidden_states_list):
        # list of N tensors, each (B, T, D) → (N, B, T, D)
        stacked = torch.stack(hidden_states_list, dim=0)
        N, B, L, D = stacked.shape
        gate = self.gate_proj(stacked) + self.bias_pre.view(N, 1, 1, D)
        alpha = F.softmax(gate, dim=0)
        return (alpha * stacked).sum(0) + self.bias_post


# ════════════════════════════════════════════════════════════
#  PerceiverResampler — Flamingo-style cross-attention to N_latent tokens
# ════════════════════════════════════════════════════════════

class PerceiverResampler(nn.Module):
    """Flamingo-style resampler (Alayrac et al. 2022).
    KV = concat(latents, vis_proj) so latents cross-attend to visual tokens
    AND to each other within a single attention pass.
    """
    def __init__(self, input_dim=VIS_HIDDEN_DIM, dim=512,
                 num_latents=NUM_RESAMPLER_LATENTS, depth=6, heads=8,
                 max_vis_tokens=2048, layer_scale_init=LAYER_SCALE_INIT):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.input_proj = nn.Linear(input_dim, dim, bias=False)
        self.vis_pos_emb = nn.Parameter(torch.randn(max_vis_tokens, dim))
        self.max_vis_tokens = max_vis_tokens
        self.layers = nn.ModuleList()
        self.ls_attn = nn.ParameterList()
        self.ls_ff = nn.ParameterList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'ln_q':  ScaleNorm(dim),
                'ln_kv': ScaleNorm(dim),
                'wq': nn.Linear(dim, dim, bias=False),
                'wk': nn.Linear(dim, dim, bias=False),
                'wv': nn.Linear(dim, dim, bias=False),
                'wo': nn.Linear(dim, dim, bias=False),
                'ln_ff':   ScaleNorm(dim),
                'ff_gate': nn.Linear(dim, dim * 4, bias=False),
                'ff_val':  nn.Linear(dim, dim * 4, bias=False),
                'ff_out':  nn.Linear(dim * 4, dim, bias=False),
            }))
            self.ls_attn.append(nn.Parameter(torch.full((dim,), layer_scale_init)))
            self.ls_ff.append(nn.Parameter(torch.full((dim,), layer_scale_init)))
        self.out_norm = ScaleNorm(dim)

    def forward(self, vis_features):
        B, N_vis = vis_features.shape[0], vis_features.shape[1]
        H, Hd = self.heads, self.hd
        assert N_vis <= self.max_vis_tokens, \
            f"vis_features has {N_vis} tokens > max {self.max_vis_tokens}"
        vis_proj = self.input_proj(vis_features)
        vis_proj = vis_proj + self.vis_pos_emb[:N_vis].unsqueeze(0).to(vis_proj.dtype)
        x = self.latents.unsqueeze(0).expand(B, -1, -1)
        for i, layer in enumerate(self.layers):
            kv = torch.cat([x, vis_proj], dim=1)
            q  = layer['wq'](layer['ln_q'](x)).view(B, -1, H, Hd).transpose(1, 2)
            k_ = layer['wk'](layer['ln_kv'](kv)).view(B, -1, H, Hd).transpose(1, 2)
            v_ = layer['wv'](layer['ln_kv'](kv)).view(B, -1, H, Hd).transpose(1, 2)
            o  = F.scaled_dot_product_attention(q, k_, v_)
            x  = x + self.ls_attn[i] * layer['wo'](o.transpose(1, 2).reshape(B, -1, H * Hd))
            h  = layer['ln_ff'](x)
            ff = layer['ff_out'](F.silu(layer['ff_gate'](h)) * layer['ff_val'](h))
            x  = x + self.ls_ff[i] * ff
        return self.out_norm(x)


# ════════════════════════════════════════════════════════════
#  Vision encoder (offline only — used by scripts/cache_vision.py)
# ════════════════════════════════════════════════════════════

def _pixel_unshuffle(x, downsample_ratio):
    """Spatial-to-channel pixel-unshuffle, matching InternVL3's projector."""
    B, T, D = x.shape
    side = int(T ** 0.5)
    x = x.view(B, side, side, D)
    side_ds = int(side * downsample_ratio)
    factor = side // side_ds
    x = x.view(B, side_ds, factor, side_ds, factor, D)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, side_ds * side_ds, D * factor * factor)
    return x


class InternVL3Vision(nn.Module):
    """InternVL3-1B (8-bit, frozen). Returns all 25 LLM hidden states.
    Used offline by scripts/cache_vision.py — never run inside the train loop.
    """
    def __init__(self, model_id=INTERNVL3_MODEL_ID, device=None):
        super().__init__()
        from PIL import Image  # noqa: F401  (used by callers via the same alias)
        from transformers import (AutoProcessor, AutoModelForImageTextToText,
                                  BitsAndBytesConfig)
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading {model_id} (8-bit) ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        ).eval()
        self.img_token_id = self.model.config.image_token_id

        vis_cfg = self.model.config.vision_config
        img_sz = (vis_cfg.image_size if isinstance(vis_cfg.image_size, int)
                  else vis_cfg.image_size[0])
        patch_sz = (vis_cfg.patch_size if isinstance(vis_cfg.patch_size, int)
                    else vis_cfg.patch_size[0])
        self.dr = self.model.config.downsample_ratio
        ds = round(1 / self.dr)
        self.n_orig = (img_sz // patch_sz) ** 2 // (ds * ds)

        inner = self.model.model
        self._vision_tower = self._projector = None
        for name, mod in inner.named_children():
            n = name.lower()
            if any(k in n for k in ("vision_tower", "vision_model", "visual")):
                self._vision_tower = mod
                print(f"  vision tower: model.model.{name}")
            if any(k in n for k in ("projector", "mlp", "connector")):
                self._projector = mod
                print(f"  projector:    model.model.{name}")
        assert self._vision_tower and self._projector
        self.d_llm = self.model.config.text_config.hidden_size
        self.n_llm_layers = self.model.config.text_config.num_hidden_layers + 1
        print(f"  D_llm={self.d_llm}  N_orig={self.n_orig}  layers={self.n_llm_layers}")
        for p in self.model.parameters():
            p.requires_grad = False

    def _get_pixel_values(self, pil_frames):
        from PIL import Image
        pvs = [self.processor.image_processor(
                images=f.resize((TILE_SIZE, TILE_SIZE), Image.BICUBIC),
                return_tensors="pt")["pixel_values"]
               for f in pil_frames]
        return torch.cat(pvs, 0).to(self._device, dtype=torch.float16)

    @torch.no_grad()
    def _extract_projector_features(self, pil_frames):
        pv = self._get_pixel_values(pil_frames)
        vis = self._vision_tower(pixel_values=pv).last_hidden_state[:, 1:, :]
        return self._projector(_pixel_unshuffle(vis, self.dr))

    def _build_prompt(self, pil_frames, prompt):
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": pil_frames},
            {"type": "text",  "text": prompt},
        ]}]
        return self.processor.apply_chat_template(
            msgs, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt").to(self._device)

    def _build_inputs_embeds(self, input_ids, img_feats):
        embed_fn = self.model.model.language_model.get_input_embeddings()
        embeds = embed_fn(input_ids).clone()
        mask = (input_ids[0] == self.img_token_id)
        embeds[0, mask] = img_feats.to(embeds.dtype)
        return embeds

    @torch.no_grad()
    def forward(self, pil_frames_batch, prompt="Observe the scene and the robot."):
        from PIL import Image
        if isinstance(pil_frames_batch[0], Image.Image):
            pil_frames_batch = [pil_frames_batch]
        per_sample = []
        for pil_frames in pil_frames_batch:
            proj_feats = self._extract_projector_features(pil_frames)
            T = len(pil_frames); N = proj_feats.shape[1]
            all_feats = proj_feats.reshape(T * N, -1)
            inp = self._build_prompt(pil_frames, prompt)
            embeds = self._build_inputs_embeds(inp["input_ids"], all_feats)
            # Match the model's actual parameter dtype — older transformers
            # used fp16, recent versions (with bitsandbytes 8-bit) use bf16.
            model_dtype = next(self.model.parameters()).dtype
            out = self.model(inputs_embeds=embeds.to(model_dtype),
                             attention_mask=torch.ones_like(inp["input_ids"]),
                             output_hidden_states=True, return_dict=True)
            per_sample.append(out.hidden_states)

        # Pad per-layer to common length and stack the batch dim.
        max_len = max(h[0].shape[1] for h in per_sample)
        N_layers = len(per_sample[0])
        result = []
        for li in range(N_layers):
            padded = []
            for sh in per_sample:
                h = sh[li]
                if h.shape[1] < max_len:
                    h = F.pad(h, (0, 0, 0, max_len - h.shape[1]))
                padded.append(h)
            result.append(torch.cat(padded, 0))
        return result
