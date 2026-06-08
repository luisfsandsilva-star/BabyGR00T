"""babygroot_strm — S-TRM v3 over a 3-level CQ-VAE for SO-101 manipulation."""
from .cqvae import (RevIN, ActionRQUNet1d, VQ1d_EMA, cosine_snce_tau,
                    SEQ_LENS_1D)
from .vqvae import ActionVQVAE1d
from .vision import (LayerAggregator, PerceiverResampler, ScaleNorm,
                     InternVL3Vision, NUM_RESAMPLER_LATENTS,
                     VIS_HIDDEN_DIM, NUM_FRAMES, TILE_SIZE)
from .policy import STRMPolicy, STRMPolicyVAE
from .optimizer import MuSGD_LARS
from .data import (load_so101_episodes, load_lerobot_episodes,
                   SO101Streamer, ChunkDataset,
                   make_loader, chunk_collate, OUNoise,
                   load_vision_cache, TASK_PROMPTS, SUPPLEMENT_PROMPTS)
from .augment import (visual_augment_chunk, sample_paraphrases,
                      build_paraphrase_pool, build_task_paraphrase_pool,
                      template_paraphrases, PARAPHRASE_BANK)

__all__ = [
    "RevIN", "ActionRQUNet1d", "ActionVQVAE1d",
    "VQ1d_EMA", "cosine_snce_tau", "SEQ_LENS_1D",
    "LayerAggregator", "PerceiverResampler", "ScaleNorm", "InternVL3Vision",
    "NUM_RESAMPLER_LATENTS", "VIS_HIDDEN_DIM", "NUM_FRAMES", "TILE_SIZE",
    "STRMPolicy", "STRMPolicyVAE", "MuSGD_LARS",
    "load_so101_episodes", "load_lerobot_episodes",
    "SO101Streamer", "ChunkDataset", "make_loader",
    "chunk_collate", "OUNoise", "load_vision_cache",
    "TASK_PROMPTS", "SUPPLEMENT_PROMPTS",
    "visual_augment_chunk", "sample_paraphrases",
    "build_paraphrase_pool", "build_task_paraphrase_pool",
    "template_paraphrases", "PARAPHRASE_BANK",
]
