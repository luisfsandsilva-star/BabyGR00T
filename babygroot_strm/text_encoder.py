"""Frozen T5 encoder for task-language conditioning.

Replaces feeding prompts through InternVL3. The T5 encoder is frozen (no grad,
eval) — we only train a thin projection (in the policy) from T5's hidden size
to the policy width. Returns per-token text embeddings + an attention mask so
the policy can cross-attend to the instruction.
"""
import torch
import torch.nn as nn


class T5TextEncoder(nn.Module):
    """Frozen T5 encoder. `model_id` default = google/flan-t5-base (d=768).
    Use t5-small (d=512) for a lighter option."""
    def __init__(self, model_id='google/flan-t5-base', dtype=torch.bfloat16,
                 device='cuda', max_len=64):
        # NB: T5 activations overflow fp16 → use bf16/fp32 only.
        super().__init__()
        from transformers import T5EncoderModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.encoder = T5EncoderModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.dim = self.encoder.config.d_model
        self.max_len = max_len
        self.device = device
        self.model_id = model_id

    @torch.no_grad()
    def forward(self, texts, all_layers=False):
        """texts: list[str].
        all_layers=False → (last_hidden (B,T,dim) float, attn_mask (B,T)).
        all_layers=True  → (stacked (n_layers, B, T, dim) float, attn_mask) — for
                           the LayerAggregator (embeddings + each encoder block)."""
        b = self.tokenizer(list(texts), return_tensors='pt', padding=True,
                           truncation=True, max_length=self.max_len)
        b = {k: v.to(self.device) for k, v in b.items()}
        out = self.encoder(**b, output_hidden_states=all_layers)
        if all_layers:
            return torch.stack(out.hidden_states, dim=0).float(), b['attention_mask']
        return out.last_hidden_state.float(), b['attention_mask']
