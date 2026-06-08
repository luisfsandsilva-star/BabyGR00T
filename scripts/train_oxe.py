#!/usr/bin/env python3
"""Multi-embodiment OXE trainer (GR00T-style).

What's shared:                What's per-embodiment:
  CNN (per-image normalized)    VQ-VAE (own codebook + decoder)
  T5 text encoder (cached)      state encoder MLP (8 → dim)
  policy transformer (STRM)     [logit head shared — codebook integers are routed to the right VAE]
  ScaleNorm on fused KV
  embodiment-id Embedding (prepended as an extra KV token)

The flat sample stream is `MultiOXEDataset`: every chunk is (frame, state, action,
prev_action, task_str, embodiment_id, dataset_idx). At train time we route by
embodiment_id to the matching (state encoder, VQ-VAE for target codes), then
feed the shared backbone, and supervise with cross-entropy on those codes.

Usage (after downloads + per-emb VAEs + T5 cache + image var_global are built):
  python -m scripts.train_oxe \\
      --oxe-root data/oxe --vae-dir data/ckpts \\
      --t5-cache data/cache/t5_text_cache.pt \\
      --image-var data/cache/image_var_global.pt \\
      --steps 12000 --batch-size 128 --lr 3.8e-3 --dim 512 \\
      --ckpt-path data/ckpts/oxe_policy_v3.pt
"""
import os, sys, glob, json, math, time, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from babygroot_strm import (RevIN, ActionVQVAE1d, VQ1d_EMA, STRMPolicy, STRMPolicyVAE,
                            LayerAggregator, ScaleNorm)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.optimizer import MuSGD_LARS
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       build_balanced_index, EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.streaming_oxe import (load_streaming_dataset_spec,
                                           StreamingMultiOXEDataset, OXE_REPOS)
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm import augment

# Per-embodiment observation.state dimension. OXE cartesian robots are 8-D; AgiBot
# arms-only is 16-D (joint14+effector2). Used to size each robot's state encoder and
# to validate datasets in the per-emb filter.
EMB_STATE_DIM = {'agibot': 16}
def _emb_state_dim(emb): return EMB_STATE_DIM.get(emb, 8)


class _NullCtx:
    """No-op context manager used to disable autocast when --no-amp is set."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument('--oxe-root', default='data/oxe')
    ap.add_argument('--vae-dir', default='data/ckpts')
    ap.add_argument('--t5-cache', required=True)
    ap.add_argument('--image-var', required=True)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--max-text', type=int, default=24)
    ap.add_argument('--samples-per-ds', type=int, default=0,
                    help="If >0, cap each dataset to this many chunks per epoch (balanced sampling).")
    # model — keep dim=512 small per user; data scale is the lever
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--layerscale-init', type=float, default=0.1,
                    help="Initial LayerScale gain on each g sub-layer output (CaiT). "
                         "Small => g starts near-identity/non-expansive; the gain is learnable, "
                         "so the model sets each block's Lipschitz contribution (fixed-point control).")
    ap.add_argument('--output-scalenorm', action='store_true',
                    help="Re-add ScaleNorm on the recursion OUTPUT (fixed-point z) before the readout. "
                         "Bounds the readout's input magnitude against ‖z*‖ drift (which long runs show "
                         "growing to ~70-117 + loosening the fixed point). Watch val acc/loss for any "
                         "expressiveness loss (it discards z magnitude).")
    ap.add_argument('--one-step-grad', action='store_true',
                    help="HRM/DEQ 1-step (Jacobian-free) gradient: reach the fixed point under "
                         "no_grad, backprop only the last step → O(1) recursion-depth memory, much "
                         "faster. Valid only because the damped map is contractive (damped mode only).")
    ap.add_argument('--heads', type=int, default=8,
                    help="Number of query attention heads (dim must be divisible by heads).")
    ap.add_argument('--kv-heads', type=int, default=None,
                    help="GQA: number of K/V heads (must divide heads). None = vanilla MHA.")
    ap.add_argument('--ff-hidden', type=int, default=None,
                    help="FF hidden size; None = (8/3)·dim rounded to 64 (Llama-style).")
    ap.add_argument('--L-inner', type=int, default=5)
    ap.add_argument('--H-outer', type=int, default=2)
    ap.add_argument('--h-max', type=int, default=3)
    # Sudoku-validated recursion upgrades (now in babygroot_strm.policy):
    ap.add_argument('--carry-zl', action='store_true',        # persist z_L across cycles (THE fix)
                    help="Carry the reasoning latent z_L across outer cycles (TRM-faithful). damped+one_step.")
    ap.add_argument('--nesterov', action='store_true',        # Nesterov-accelerated inner fixed point
                    help="Nesterov + adaptive restart on the inner loop.")
    ap.add_argument('--nesterov-beta', type=float, default=0.7)
    ap.add_argument('--inner-tol', type=float, default=0.0,   # inner early-stop (speed, no value change)
                    help="Inner no_grad warmup early-stops at this rel. fixed-point residual (0=off).")
    ap.add_argument('--compile', action='store_true',         # torch.compile the g-net (~2-3×)
                    help="torch.compile the policy's shared net g.")
    ap.add_argument('--g-scalenorm', action='store_true',     # ScaleNorm on g's output (bounds ‖z‖ runaway)
                    help="ScaleNorm on g's output inside the recurrence — bounds the iterated latent "
                         "so carry_zl can't inflate ‖z‖ (fixes the magnitude drift).")
    ap.add_argument('--cnn-dims', type=int, nargs='+', default=[24, 48, 96, 192])
    ap.add_argument('--cnn-depths', type=int, nargs='+', default=[1, 1, 1, 1])
    ap.add_argument('--cnn-expand', type=int, default=2)
    ap.add_argument('--cnn-out-dim', type=int, default=192)
    ap.add_argument('--cnn-norm', default='scalenorm')
    ap.add_argument('--cnn-pe', action='store_true')
    ap.add_argument('--cnn-dropout', type=float, default=0.1)
    ap.add_argument('--cnn-film-by-emb', action='store_true',
                    help="Add embodiment-conditioned FiLM (γ, β per stage) to the CNN. "
                         "Identity-initialized so baseline behavior is unchanged at step 0. "
                         "Same pattern as RT-1's FiLM-on-EfficientNet but conditioned on robot "
                         "identity (embodiment) instead of language. Complementary to the 16 "
                         "prefix tokens that condition the transformer policy.")
    ap.add_argument('--beta', type=float, default=1e-3)
    ap.add_argument('--free-bits', type=float, default=0.1)
    # train
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3.8e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--policy-dropout', type=float, default=0.0,
                    help="dropout inside STRMPolicy: post-softmax attention dropout (via SDPA dropout_p, "
                         "the std Attention-Is-All-You-Need formulation) + FFN dropout in GeGLU. Try 0.1.")
    ap.add_argument('--use-lars', action='store_true', default=True)
    ap.add_argument('--no-lars', dest='use_lars', action='store_false',
                    help="disable the LARS trust ratio in MuSGD_LARS → pure Newton-Schulz orthogonalized SGD+momentum")
    ap.add_argument('--dropout-prob', type=float, default=0.3)
    ap.add_argument('--state-noise', type=float, default=0.02)
    ap.add_argument('--action-noise', type=float, default=0.0,
                    help="σ of Gaussian noise on the precision-normalized actions before VAE encoding. "
                         "DATA AUGMENTATION ONLY — prevents memorizing exact action sequences, makes the "
                         "VAE see near-targets that hash to nearby codes. Acts like denoising-AE, NOT "
                         "Tikhonov on the policy (use --g-input-noise for that — separate, complementary).")
    ap.add_argument('--g-input-noise', type=float, default=0.0,
                    help="σ of Gaussian noise added to g's input at EVERY g-call inside the policy "
                         "iteration (training only). Bishop-1995 Tikhonov: equivalent to penalizing "
                         "σ²·E[‖∂g/∂x‖²], i.e. regularizes g's Jacobian / Lipschitz constant. "
                         "Helps damped-iteration contractivity. Sweep needed: too large hurts. "
                         "Typical: 0.01–0.05 in feature units.")
    ap.add_argument('--amp', action='store_true', default=True,
                    help="Mixed-precision training via bfloat16 autocast (default ON for A4000/A6000+). "
                         "Halves activation memory; ~1.5–2× speed up. Master weights stay fp32.")
    ap.add_argument('--no-amp', dest='amp', action='store_false',
                    help="Disable AMP (force fp32). Use if numerical issues suspected.")
    ap.add_argument('--chunk-stride', type=int, default=None,
                    help="Distance between chunk start indices. Default = chunk_len (no overlap). "
                         "Set to 4 with chunk_len=16 for 4× more chunks per episode (overlapping windows). "
                         "Free data multiplier when dataset is small.")
    ap.add_argument('--use-paraphrase-sampling', action='store_true',
                    help="If the t5 cache has a 'paraphrase_map' (built by scripts/build_paraphrased_t5_cache.py), "
                         "randomly sample one of {original, paraphrase_1, ...} per batch per sample. "
                         "Prevents the policy from memorizing exact instruction wording.")
    ap.add_argument('--strong-aug', action='store_true', default=True)
    # texture-invariance augs (combat CNN texture bias → sim render OOD). Both training-only.
    ap.add_argument('--apr-prob', type=float, default=0.0,
                    help="APR (Amplitude-Phase Recombination, ICCV2021) probability per sample. 0=off. "
                         "Keeps FFT phase (content/object positions → labels valid) and randomizes the "
                         "amplitude (texture/style) by blending with another batch sample. Try 0.5.")
    ap.add_argument('--apr-eta-max', type=float, default=1.0,
                    help="Max fraction of the other sample's amplitude to mix in (eta~U(0,eta_max)); "
                         "1.0 allows a full amplitude swap, lower = gentler.")
    ap.add_argument('--mixstyle-p', type=float, default=0.0,
                    help="MixStyle (ICLR2021) probability: parameter-free per-instance channel-stat "
                         "mixing in the early CNN stages (texture-invariance, training-only). 0=off. Try 0.5.")
    # vision backbone: 'cnn' = EfficientCNN (legacy); 'vit' = TwoFrameViT (prev+cur frames, transformer-first)
    ap.add_argument('--vision', choices=['cnn', 'vit', 'dinoagg'], default='cnn')
    ap.add_argument('--dino-name', default='facebook/dinov2-small',
                    help="HF id for frozen DINOv2 backbone (used by --vision dinoagg)")
    ap.add_argument('--dino-tok-side', type=int, default=16,
                    help="vision token grid side (16=native 256 patches for DINOv2-small@224; 7=49 pooled)")
    # val-acc early-stop on the disjoint val split (true generalization signal)
    ap.add_argument('--val-probe-every', type=int, default=0,
                    help="periodic masked-CE acc on --val-eps-file held-out episodes (0=off)")
    ap.add_argument('--val-n', type=int, default=192)
    ap.add_argument('--val-best-path', default='')
    ap.add_argument('--val-min-delta', type=float, default=0.003,
                    help="min val_acc improvement to count (fraction, e.g. 0.003 = 0.3 points)")
    ap.add_argument('--val-early-stop-patience', type=int, default=0,
                    help="stop after this many probes without ≥min-delta improvement (0=off)")
    ap.add_argument('--eval-only', action='store_true',
                    help="Load weights (with --resume), run the val/train probe ONCE, print, and exit — "
                         "no training step. Use to evaluate a checkpoint cleanly (e.g. H_outer sweep).")
    ap.add_argument('--eval-iter-schemes', action='store_true',
                    help="With --resume: drive the trained g through 3 fixed-point schemes (accumulator "
                         "z+w·g / damped (1-α)z+α·g / nesterov) on one batch, plot step_resid + TRUE "
                         "fp_resid ‖g(z)-z‖ + ‖z‖ + readout CE/acc, save data/logs/iter_schemes.png, exit.")
    ap.add_argument('--iter-beta', type=float, default=0.7, help="Nesterov momentum β for --eval-iter-schemes.")
    ap.add_argument('--iter-n', type=int, default=40, help="iterations for --eval-iter-schemes.")
    ap.add_argument('--eval-why-depth', action='store_true',
                    help="With --resume: per-iteration readout CE/acc + cosine(z_k,z_final) + ‖g(z)-z‖ "
                         "— shows WHEN the readout-relevant signal saturates (why more iters don't help).")
    ap.add_argument('--eval-maskgit-variants', action='store_true',
                    help="With --resume: MaskGIT generation acc for greedy/sampled × maxprob/entropy "
                         "confidence × n_steps — tests the faithful-sampling fix and entropy confidence.")
    ap.add_argument('--eval-spectrum', action='store_true',
                    help="With --resume: dominant eigenvalue |λ| + Re(λ) of ∂g/∂z vs the norm σ_g "
                         "(explains the α·σ_g ratio) + a MaskGIT n_steps generation sweep.")
    ap.add_argument('--eval-depth-sweep', action='store_true',
                    help="With --resume: sweep n_outer(H) × n_inner(L) recursion counts via the real "
                         "forward, report masked-CE/acc at each — does iteration COUNT move the metric "
                         "(unlike iteration SCHEME)? Exercises BOTH the L and H recurrences.")
    ap.add_argument('--eval-invariance', action='store_true',
                    help="With --resume: probe WHY the readout looked iteration-invariant — readout "
                         "CE/acc at z*/0/random/scaled/accum, with vs without out-norm, at all-masked "
                         "AND 50%% mask. Tells us if z is low-impact (all-masked) or readout ignores z.")
    ap.add_argument('--val-mask-ratio', type=float, default=0.0,
                    help="If >0, val-probe uses this FIXED mask ratio instead of cosine (for mask-ratio sweeps).")
    ap.add_argument('--fp-probe-every', type=int, default=0,
                    help="damped-mode fixed-point/Lipschitz probe every N steps (0=off): logs inner-loop "
                         "residual decay, empirical contraction, σ_max(∂g/∂z), Lipschitz(T) bound, "
                         "α, and LayerScale magnitudes on the val batch.")
    ap.add_argument('--vit-dim', type=int, default=384)
    ap.add_argument('--vit-depth', type=int, default=6)
    ap.add_argument('--vit-heads', type=int, default=6)
    ap.add_argument('--vit-kv-heads', type=int, default=2)
    ap.add_argument('--vit-ff', type=int, default=1536)
    ap.add_argument('--prev-frame-drop', type=float, default=0.1,
                    help="prob of zeroing the previous frame during training (robustness to rollout step 0)")
    # episode-level split files (JSON lists of episode ids)
    ap.add_argument('--train-eps-file', default=None, help="restrict TRAINING chunks to these episode ids")
    ap.add_argument('--val-eps-file', default=None, help="held-out val episodes for masked-CE early-stop/selection")
    # vision-only recovery finetune
    ap.add_argument('--freeze-except-cnn', action='store_true',
                    help="Finetune ONLY the EfficientCNN; freeze policy, projections, VAEs, state/text/emb "
                         "encoders. Texture-invariance recovery without disturbing the working policy.")
    ap.add_argument('--reset-step', action='store_true',
                    help="On --resume, reset the step counter to 0 (fresh LR schedule + step budget).")
    # OOD-recovery probe (objective-aligned early-stop: sim→real CNN-feature kNN ratio)
    ap.add_argument('--ood-probe-every', type=int, default=0,
                    help="If >0, every N steps measure sim→real CNN-feature kNN ratio (sim-OOD recovery "
                         "signal) and save the lowest-ratio ckpt to --ood-best-path.")
    ap.add_argument('--ood-sim', default='/tmp/sim_frames.npy')
    ap.add_argument('--ood-real', default='/tmp/real_frames.npy')
    ap.add_argument('--ood-best-path', default='')
    ap.add_argument('--ood-early-stop-patience', type=int, default=0,
                    help="Stop when the OOD ratio hasn't improved by ≥--ood-min-delta for this many probes.")
    ap.add_argument('--ood-min-delta', type=float, default=0.01)
    ap.add_argument('--ema-decay', type=float, default=0.999)
    ap.add_argument('--num-workers', type=int, default=4,
                    help="DataLoader workers. Each does av.open+seek+decode per __getitem__ — "
                         "scale with CPU count. Profiling shows ~2.5× speedup going 8→24 workers "
                         "on a 64-CPU box (per-step 1.84s → 0.74s at BS=512). Use 16-24 here.")
    ap.add_argument('--prefetch-factor', type=int, default=4,
                    help="DataLoader prefetch_factor (per worker). Bumped from 2 to 4 to smooth "
                         "occasional spikes when episodes vary in length.")
    ap.add_argument('--ckpt-path', default='data/ckpts/oxe_policy_v3.pt')
    ap.add_argument('--ckpt-every', type=int, default=500)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--reset-opt', action='store_true',
                    help="On --resume, do NOT load the saved optimizer state (momentum buffers). "
                         "Useful when changing lr — fresh momentum avoids the stale-momentum × new-lr overshoot. "
                         "Weights, EMA, and step counter still resume normally.")
    ap.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant',
                    help="'constant' = legacy (150-step warmup only, then base lr).  "
                         "'cosine' = linear warmup (--lr-warmup-frac of total steps) then cosine decay "
                         "from base lr down to --lr-min. Standard for transformer training (BERT 10%% warmup; "
                         "GPT-3, Llama decay to 10%% of peak).")
    ap.add_argument('--lr-warmup-frac', type=float, default=0.10,
                    help="(cosine schedule) fraction of total steps used for linear warmup. "
                         "0.10 = BERT-style 10%% warmup. GPT-3/Llama use <1%%.")
    ap.add_argument('--lr-min', type=float, default=None,
                    help="(cosine schedule) floor lr at end of decay. Default = 0.1 × --lr "
                         "(GPT-3, Llama style: decay to 10%% of peak).")
    # ρ revival (same defaults as train_cnn_policy)
    ap.add_argument('--revive', action='store_true', default=True)
    ap.add_argument('--no-revive', dest='revive', action='store_false',
                    help="Disable ρ-gate revive. Use when you want to let ρ collapse "
                         "naturally — revive's bounce-up perturbs training (we've seen it "
                         "trigger loss spikes when ρ_H is dead but the model has adapted).")
    ap.add_argument('--revive-thresh', type=float, default=0.02)
    ap.add_argument('--revive-patience', type=int, default=200)
    ap.add_argument('--revive-decay', type=float, default=0.7)
    ap.add_argument('--revive-cooldown', type=int, default=400)
    ap.add_argument('--rho-fixed', type=float, default=0.0,
                    help="If >0, FREEZE both ρ_L and ρ_H at this value (non-trainable). "
                         "Tests whether the recursion is useful when it can't collapse to 0. "
                         "Overrides --revive.")
    ap.add_argument('--no-vae', action='store_true',
                    help="Use STRMPolicy (non-VAE) instead of STRMPolicyVAE. "
                         "Drops the latent bottleneck; just predicts codes from (vis, state).")
    ap.add_argument('--weighting', choices=['geometric', 'linear'], default='geometric',
                    help="Weighting scheme for inner/outer convex combination. "
                         "geometric: w_t = ρ^(t/(n-1)), single ρ ∈ (0,1) (clamp_direct). "
                         "linear:    w_t = clamp(1 + slope·(t-mid), eps, ∞), slope ∈ ℝ unbounded. "
                         "Use init 0.0 for linear (= uniform) and 0.1 for geometric.")
    ap.add_argument('--rho-init', type=float, default=None,
                    help="Override the initial value of rho_L_raw and rho_H_raw. "
                         "Defaults: 0.1 for geometric/clamp, 0.0 for linear (uniform), 0.0 for damped+sigmoid (α=0.5).")
    ap.add_argument('--update-mode', choices=['accumulator', 'damped', 'bayesian'], default='accumulator',
                    help="accumulator: z = z + w_t·g (Cauchy partial sum).  "
                         "damped:     z = (1-α)·z + α·g (damped Banach).  "
                         "bayesian:   precision-weighted Gaussian fusion (VAE only): "
                         "τ_post = τ_p + τ_g, μ_post = (τ_p·μ_p + τ_g·μ_g)/τ_post. "
                         "No α gate — evidence's own precision controls the update.")
    ap.add_argument('--alpha-parametrization', choices=['clamp', 'sigmoid', 'tilt'], default='sigmoid',
                    help="(damped only) parameterization of α. "
                         "sigmoid: α=σ(raw), raw=0→α=0.5.  "
                         "clamp: α=clamp(raw, eps, 1-eps).  "
                         "tilt: α=clamp(0.5+raw, eps, 1-eps), unit gradient inside, raw=0→α=0.5 (asymmetric parameterization).")
    ap.add_argument('--alpha-per-dim', action='store_true',
                    help="(damped only) make α a per-channel (D,) vector instead of scalar — "
                         "each feature gates independently. Tests if a single α is too coarse.")
    ap.add_argument('--n-emb-prefix', type=int, default=1,
                    help="Number of emb-id prefix tokens prepended to KV (Octo uses many, default 1).")
    ap.add_argument('--per-emb-head', action='store_true',
                    help="Use per-embodiment action head (one set of out_head Linears per emb). "
                         "Lets each robot have its own code-distribution mapping (shared codebook).")
    ap.add_argument('--label-smoothing', type=float, default=0.0,
                    help="Label smoothing for the masked-CE loss (e.g. 0.05).")
    ap.add_argument('--mask-sampler', choices=['cosine'], default='cosine',
                    help="MaskGIT cosine sampling (Chang+ 2022): r = cos(π/2·U(0,1)), floored at 1/T. "
                         "The only sampler — linear was removed (it under-masked vs the val-probe, "
                         "making train/val look like an overfit gap that was a measurement artifact).")
    ap.add_argument('--grad-clip-max', type=float, default=10.0,
                    help="Hard upper bound on ||grad||. Always applied. Catches exponential explosions "
                         "that the spike-vs-median smart clip would otherwise track upward. Set 0 to disable.")
    ap.add_argument('--streaming', action='store_true',
                    help="Stream OXE datasets from HuggingFace (lazy hf_hub_download per episode) "
                         "instead of reading --oxe-root local dir. Falls back to local cache after first touch.")
    ap.add_argument('--hf-repos', type=str, nargs='+', default=None,
                    help="(--streaming only) HF repo IDs to use. Default = all 36 OXE LeRobot repos.")
    ap.add_argument('--no-shared-vae', dest='shared_vae', action='store_false',
                    help="Use per-embodiment VQ-VAEs (data/ckpts/oxe_vqvae_<emb>.pt) instead of the "
                         "single shared conditional one. Set when per-emb VAEs are trained and you want "
                         "each robot routed through its own codebook.")
    ap.add_argument('--exclude-datasets', nargs='+', default=None,
                    help="List of dataset names to skip (matched against os.path.basename of the dataset dir). "
                         "Used to drop incompatible action distributions: for franka, drop iamlab_cmu_pickup_insert, "
                         "toto, cmu_play_fusion, nyu_franka_play, stanford_hydra. Also drop ucsd_kitchen (xarm broken).")
    ap.add_argument('--only-robots', nargs='+', default=None,
                    help="If set, only load datasets whose robot_type is in this list. "
                         "Useful for single-embodiment ablations: --only-robots widowx (bridge only).")
    ap.add_argument('--shared-vae', action='store_true', default=True,
                    help="Use the single shared conditional VQ-VAE (data/ckpts/oxe_shared_vae.pt) "
                         "instead of per-embodiment VQ-VAEs. Healthier codebook usage on data-poor embodiments.")
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── load datasets ──
    specs = []
    excl = set(args.exclude_datasets) if args.exclude_datasets else set()
    only_robots = set(args.only_robots) if args.only_robots else None
    if excl:
        print(f"  --exclude-datasets: skipping {sorted(excl)}")
    if only_robots:
        print(f"  --only-robots: keeping ONLY robots in {sorted(only_robots)}")
    if args.streaming:
        repos = args.hf_repos if args.hf_repos else OXE_REPOS
        print(f"loading dataset specs (STREAMING from HuggingFace, {len(repos)} repos)...")
        for repo in repos:
            ds_name = repo.split('/')[-1]
            if ds_name in excl:
                print(f"  EXCLUDED by request: {ds_name}"); continue
            try:
                sp = load_streaming_dataset_spec(repo, chunk_len=16, lookback=16,
                                                  chunk_stride=args.chunk_stride, verbose=True)
                if sp is not None and sp.chunk_index:
                    specs.append(sp)
            except Exception as e:
                print(f"  skip {repo}: {e}")
    else:
        print("loading dataset specs (local)...")
        for ds_dir in sorted(glob.glob(os.path.join(args.oxe_root, '*'))):
            if not os.path.isfile(os.path.join(ds_dir, 'meta', 'info.json')): continue
            ds_name = os.path.basename(ds_dir)
            if ds_name in excl:
                print(f"  EXCLUDED by request: {ds_name}"); continue
            try:
                sp = load_dataset_spec(ds_dir, chunk_len=16, lookback=16,
                                        chunk_stride=args.chunk_stride)
                if only_robots is not None and sp.robot not in only_robots:
                    print(f"  SKIP (robot '{sp.robot}' not in --only-robots)")
                    continue
                if sp.chunk_index:
                    specs.append(sp)
                    print(f"  {sp.name:<50s} robot={sp.robot:<14s} emb_id={sp.embodiment_id} chunks={len(sp.chunk_index)}")
            except Exception as e:
                print(f"  skip {ds_dir}: {e}")
    if not specs:
        print("no usable datasets — abort"); return
    present_emb = sorted({sp.robot for sp in specs})
    print(f"  → {len(specs)} datasets, embodiments: {present_emb}")

    # ── single shared conditional VQ-VAE (new) — falls back to per-embodiment VAEs (old) ──
    shared_vae_path = os.path.join(args.vae_dir, 'oxe_shared_vae.pt')
    if args.shared_vae and os.path.isfile(shared_vae_path):
        from babygroot_strm.cond_vae import CondActionVQVAE1d
        sc = torch.load(shared_vae_path, map_location=dev, weights_only=False)
        shared_vae = CondActionVQVAE1d(action_dim=sc['action_dim'], n_embodiments=sc['n_embodiments'],
                                       k=sc['k']).to(dev)
        shared_vae.load_state_dict(sc['vae']); shared_vae.eval()
        for p in shared_vae.parameters(): p.requires_grad_(False)
        # API parity with the per-emb path
        vaes = {emb: shared_vae for emb in present_emb}        # all robots → same VAE module
        var_globals = {emb: sc['action_var_globals'][EMBODIMENT_ID.get(emb, -1)].to(dev).view(1, 1, -1)
                       for emb in present_emb if EMBODIMENT_ID.get(emb, -1) in sc['action_var_globals']}
        seq_lens = tuple(shared_vae.seq_lens); K = shared_vae.vqs[0].K
        print(f"  SHARED conditional VAE loaded: K={K}, seq_lens={seq_lens}, n_emb_cond={sc['n_embodiments']}")
        # Filter specs to those compatible with the shared VAE & action stats:
        #   - robot must be in var_globals (i.e. shared VAE saw it during VAE training)
        #   - dataset action_dim must equal the VAE's expected action_dim
        # Anything else would NaN or shape-mismatch at the encode_targets call.
        _vae_action_dim = sc['action_dim']
        def _spec_ok(sp):
            if sp.robot not in var_globals: return (False, f"robot '{sp.robot}' not in shared-VAE var_globals")
            # Read action_dim / state_dim from meta/info.json (no download needed — already fetched).
            try:
                _info = json.load(open(os.path.join(sp.root, 'meta', 'info.json')))
                _feats = _info.get('features', {})
                _ad = _feats.get('action', {}).get('shape', [None])[0]
                _sd = _feats.get('observation.state', {}).get('shape', [None])[0]
                if _ad != _vae_action_dim:
                    return (False, f"action_dim={_ad} != VAE {_vae_action_dim}")
                if _sd != 8:
                    return (False, f"state_dim={_sd} != expected 8")
            except Exception as e:
                return (False, f"meta probe failed: {type(e).__name__}: {e}")
            return (True, '')
        kept, dropped = [], []
        for sp in specs:
            ok, why = _spec_ok(sp)
            (kept if ok else dropped).append((sp, why))
        if dropped:
            print(f"  [filter] dropped {len(dropped)} incompatible datasets:")
            for sp, why in dropped[:10]:
                print(f"    - {sp.name:<55s} ({why})")
            if len(dropped) > 10: print(f"    ... +{len(dropped)-10} more")
        specs = [sp for sp, _ in kept]
        if not specs:
            print("  ERROR: no compatible datasets remain after filtering — abort"); return
        present_emb = sorted({sp.robot for sp in specs})
        # Build VAE/var_globals dicts for ALL embodiments the shared VAE knows about
        # (not just present_emb). This keeps the model shape STABLE across restarts
        # — as the parallel downloader adds new datasets that bring new embodiments,
        # those state_encoders/var_globals already exist and just start receiving gradient.
        all_known_embs = sorted([e for e in EMBODIMENTS
                                 if EMBODIMENT_ID.get(e, -1) in sc['action_var_globals']])
        vaes = {emb: shared_vae for emb in all_known_embs}
        var_globals = {emb: sc['action_var_globals'][EMBODIMENT_ID.get(emb, -1)].to(dev).view(1, 1, -1)
                       for emb in all_known_embs}
        total_chunks = sum(len(sp.chunk_index) for sp in specs)
        print(f"  → after filtering: {len(specs)} datasets ({total_chunks} chunks), "
              f"present embodiments: {present_emb}")
        print(f"  → state_encoders/var_globals pre-allocated for ALL {len(all_known_embs)} known "
              f"embodiments (shape-stable across restarts): {all_known_embs}")
        # use all_known_embs (not present_emb) for state_encoders construction below
        present_emb = all_known_embs
    else:
        # PER-EMBODIMENT VAE path. Loads one VQ-VAE per embodiment from
        # data/ckpts/oxe_vqvae_<emb>.pt. Each VAE has its own K, seq_lens, action_dim,
        # and var_global. Filters specs to those whose (embodiment, action_dim) match
        # an available VAE — same shape-stability story as the shared path.
        vaes, var_globals, per_emb_ad = {}, {}, {}
        all_known_embs = []
        for emb_glob in EMBODIMENTS:
            ck_path = os.path.join(args.vae_dir, f'oxe_vqvae_{emb_glob}.pt')
            if not os.path.isfile(ck_path): continue
            c = torch.load(ck_path, map_location=dev, weights_only=False)
            adim = c.get('action_dim', 7)
            k = c.get('k', 128)
            vae = ActionVQVAE1d(action_dim=adim, vq_cls=VQ1d_EMA, k=k,
                                binary_last=c.get('binary_gripper', False)).to(dev)
            vae.load_state_dict(c['vae']); vae.eval()
            for p in vae.parameters(): p.requires_grad_(False)
            vaes[emb_glob] = vae
            var_globals[emb_glob] = c['action_var_global'].to(dev).view(1, 1, -1)
            per_emb_ad[emb_glob] = adim
            all_known_embs.append(emb_glob)
            print(f"  per-emb VAE [{emb_glob:<14s}]: K={k}, action_dim={adim}, "
                  f"convention={c.get('convention', 'cartesian')}, "
                  f"trained on {c.get('n_train_pairs', '?')} pairs")
        if not vaes:
            print("  ERROR: no per-emb VAE ckpts found in data/ckpts/ — abort"); return
        # filter loaded specs to those whose emb has a VAE AND whose action_dim matches
        def _spec_ok_peremb(sp):
            if sp.robot not in vaes: return (False, f"robot '{sp.robot}' has no per-emb VAE")
            want_adim = per_emb_ad[sp.robot]
            try:
                _info = json.load(open(os.path.join(sp.root, 'meta', 'info.json')))
                ad = _info.get('features', {}).get('action', {}).get('shape', [None])[0]
                sd = _info.get('features', {}).get('observation.state', {}).get('shape', [None])[0]
                if ad != want_adim:
                    return (False, f"action_dim={ad} != VAE's {want_adim}")
                if sd != _emb_state_dim(sp.robot):
                    return (False, f"state_dim={sd} != expected {_emb_state_dim(sp.robot)}")
            except Exception as e:
                return (False, f"meta probe failed: {type(e).__name__}: {e}")
            return (True, '')
        kept, dropped = [], []
        for sp in specs:
            ok, why = _spec_ok_peremb(sp)
            (kept if ok else dropped).append((sp, why))
        if dropped:
            print(f"  [filter] dropped {len(dropped)} incompatible datasets (no matching VAE / dim):")
            for sp, why in dropped[:10]:
                print(f"    - {sp.name:<55s} ({why})")
            if len(dropped) > 10: print(f"    ... +{len(dropped)-10} more")
        specs = [sp for sp, _ in kept]
        if not specs:
            print("  ERROR: no compatible datasets remain after per-emb filtering — abort"); return
        present_emb = sorted({sp.robot for sp in specs})
        seq_lens = tuple(next(iter(vaes.values())).seq_lens)
        K = next(iter(vaes.values())).vq.K
        total_chunks = sum(len(sp.chunk_index) for sp in specs)
        print(f"  per-emb VAEs loaded: {len(vaes)} ({sorted(vaes.keys())}). "
              f"seq_lens={seq_lens}, first-VAE K={K}.")
        print(f"  → after filtering: {len(specs)} datasets ({total_chunks} chunks), "
              f"present embodiments: {present_emb}")
        print(f"  → state_encoders pre-allocated for ALL {len(all_known_embs)} VAE-supported "
              f"embodiments (shape-stable across restarts): {all_known_embs}")
        present_emb = all_known_embs                                          # for state_encoders below

    # ── T5 cache + image var ──
    t5 = torch.load(args.t5_cache, map_location='cpu', weights_only=False)
    t5_emb, t5_dim, t5_layers = t5['embeddings'], t5['dim'], t5['n_layers']
    # optional paraphrase_map: {original_task_str -> [paraphrased_str, ...]}.
    # If present and --use-paraphrase-sampling, training samples one variant per call.
    t5_paraphrase_map = t5.get('paraphrase_map', {})
    if t5_paraphrase_map and args.use_paraphrase_sampling:
        print(f"  paraphrase sampling enabled: {len(t5_paraphrase_map)} tasks have paraphrases "
              f"(avg {sum(len(v) for v in t5_paraphrase_map.values())/len(t5_paraphrase_map):.1f}/task)")
    img_var = torch.load(args.image_var, map_location='cpu', weights_only=False)
    var_global_img = img_var['var_global'].to(dev)
    print(f"  T5 dim={t5_dim} L={t5_layers} n={len(t5_emb)}  | image var={var_global_img.tolist()}")

    # ── modules: shared backbone + per-embodiment state encoders ──
    if args.vision == 'vit':
        from babygroot_strm.vision_vit import TwoFrameViT
        cnn = TwoFrameViT(dim=args.vit_dim, depth=args.vit_depth, heads=args.vit_heads,
                          kv_heads=args.vit_kv_heads, ff_hidden=args.vit_ff, out_dim=args.dim,
                          img_size=args.img_size, n_frames=2, dropout=args.cnn_dropout,
                          mixstyle_p=args.mixstyle_p).to(dev)
        cnn_proj = nn.Identity().to(dev)                          # ViT already outputs args.dim
        n_vis = 2 * (args.img_size // 32) ** 2                    # 2 frames × 7×7 = 98 tokens
        print(f"  vision: TwoFrameViT dim={args.vit_dim} depth={args.vit_depth} "
              f"heads={args.vit_heads}/{args.vit_kv_heads} → {n_vis} tokens", flush=True)
    elif args.vision == 'dinoagg':
        from babygroot_strm.vision_dino_agg import DinoLayerAggMLP
        cnn = DinoLayerAggMLP(out_dim=args.dim, hidden=args.vit_ff, dropout=args.cnn_dropout,
                              dino_name=args.dino_name, n_tok_side=args.dino_tok_side).to(dev)
        cnn_proj = nn.Identity().to(dev)                          # MLP already outputs args.dim
        n_vis = args.dino_tok_side ** 2                           # 256 for native, 49 for 7×7 pooled
        ntr = sum(p.numel() for p in cnn.parameters() if p.requires_grad) / 1e6
        print(f"  vision: DinoLayerAggMLP frozen {args.dino_name} + LayerAgg + MLP "
              f"→ {n_vis} tokens ({args.dino_tok_side}×{args.dino_tok_side}, trainable {ntr:.2f}M)", flush=True)
    else:
        cnn = EfficientCNN(dims=args.cnn_dims, depths=args.cnn_depths, expand=args.cnn_expand,
                           out_dim=args.cnn_out_dim, norm=args.cnn_norm, pos_emb=args.cnn_pe,
                           img_size=args.img_size, dropout=args.cnn_dropout,
                           mixstyle_p=args.mixstyle_p,
                           n_embodiments=(len(EMBODIMENTS) + 1) if args.cnn_film_by_emb else 0).to(dev)
        cnn_proj = nn.Linear(args.cnn_out_dim, args.dim).to(dev)
        n_vis = (args.img_size // 32) ** 2
    text_agg = LayerAggregator(hidden_dim=t5_dim, n_layers=t5_layers).to(dev)
    text_proj = nn.Linear(t5_dim, args.dim).to(dev)
    kv_norm = ScaleNorm(args.dim).to(dev)

    # per-embodiment state encoders (state_dim → dim) — own MLP per robot type
    state_encoders = nn.ModuleDict({
        emb: nn.Sequential(nn.Linear(_emb_state_dim(emb), args.dim), nn.GELU(), nn.Linear(args.dim, args.dim))
        for emb in present_emb}).to(dev)
    # embodiment-id token (prepended to KV so policy knows which robot)
    # N prefix tokens per embodiment (Octo-style). Each emb has n_emb_prefix learnable vectors;
    # they're prepended to KV at the start of every forward pass.
    emb_id_emb = nn.Embedding(len(EMBODIMENTS) + 1, args.dim * args.n_emb_prefix).to(dev)
    emb_id_to_idx = {emb: EMBODIMENT_ID.get(emb, len(EMBODIMENTS)) for emb in present_emb}

    # shared policy — state_dim=args.dim because state encoders already project up
    if args.rho_init is not None:
        rho_init = args.rho_init
    elif args.update_mode == 'damped' and args.alpha_parametrization == 'sigmoid':
        rho_init = 0.0                                                # σ(0) = α=0.5 (balanced)
    elif args.weighting == 'linear':
        rho_init = 0.0
    else:
        rho_init = 0.1
    # max_prefix budget: vision tokens + text tokens + 16 action positions + N emb prefix tokens
    max_prefix = n_vis + args.max_text + 16 + args.n_emb_prefix
    policy_common = dict(seq_lens=seq_lens, k_codebook=K, dim=args.dim, heads=args.heads,
                         kv_heads=args.kv_heads, ff_hidden=args.ff_hidden,
                         depth=args.depth, L_inner=args.L_inner, H_outer=args.H_outer,
                         state_dim=args.dim, max_prefix=max_prefix,
                         weighting=args.weighting, rho_L=rho_init, rho_H=rho_init,
                         update_mode=args.update_mode,
                         alpha_parametrization=args.alpha_parametrization,
                         alpha_per_dim=args.alpha_per_dim,
                         n_embodiments=len(EMBODIMENTS) + 1,
                         per_emb_head=args.per_emb_head,
                         dropout=args.policy_dropout,
                         g_input_noise=args.g_input_noise,
                         layerscale_init=args.layerscale_init,
                         one_step_grad=args.one_step_grad,
                         output_scalenorm=args.output_scalenorm,
                         carry_zl=args.carry_zl, nesterov=args.nesterov,
                         nesterov_beta=args.nesterov_beta, inner_tol=args.inner_tol,
                         g_scalenorm=args.g_scalenorm)
    if args.no_vae:
        policy = STRMPolicy(**policy_common).to(dev)
        print(f"  STRMPolicy (no-VAE); update={args.update_mode}; α-param={args.alpha_parametrization}; "
              f"init={rho_init}; n_emb_prefix={args.n_emb_prefix}; per_emb_head={args.per_emb_head}")
    else:
        policy = STRMPolicyVAE(beta=args.beta, free_bits=args.free_bits,
                               **policy_common).to(dev)
        print(f"  STRMPolicyVAE; update={args.update_mode}; α-param={args.alpha_parametrization}; "
              f"init={rho_init}; n_emb_prefix={args.n_emb_prefix}; per_emb_head={args.per_emb_head}")
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']

    # resume if asked
    _ck = None; _step = 0
    if args.resume and os.path.exists(args.ckpt_path):
        _ck = torch.load(args.ckpt_path, map_location=dev, weights_only=False)
        for k, m in zip(keys, mods): m.load_state_dict(_ck[k])
        _step = _ck.get('step', 0)
        print(f"resumed from {args.ckpt_path} @ step {_step}", flush=True)

    # Optionally pin ρ at a fixed value (frozen). With the clamp_direct parameterization
    # rho_*_raw IS ρ — no logit transform needed.
    if args.rho_fixed > 0.0:
        policy.rho_L_raw.data.fill_(args.rho_fixed); policy.rho_L_raw.requires_grad_(False)
        policy.rho_H_raw.data.fill_(args.rho_fixed); policy.rho_H_raw.requires_grad_(False)
        args.revive = False
        print(f"  [rho-fixed] ρ_L = ρ_H = {args.rho_fixed} (frozen, no revive)")

    # Freeze all non-CNN modules BEFORE collecting trainable params + EMA shadow, so both track cnn only.
    if args.freeze_except_cnn:
        nfz = 0
        for nm, m in zip(keys, mods):
            if nm == 'cnn': continue
            for p in m.parameters():
                if p.requires_grad: p.requires_grad_(False); nfz += 1
        print(f"  [freeze-except-cnn] froze {nfz} tensors across all modules except cnn — VISION ONLY", flush=True)
    if args.reset_step and _ck is not None:
        _step = 0
        print(f"  [reset-step] step counter → 0 (fresh {args.steps}-step budget + LR schedule)", flush=True)

    trainable = [p for m in mods for p in m.parameters() if p.requires_grad]
    opt = MuSGD_LARS(trainable, lr=args.lr, momentum=0.95, weight_decay=args.weight_decay,
                     use_lars=args.use_lars)
    n_train = sum(p.numel() for p in trainable) / 1e6
    n_total = sum(p.numel() for m in mods for p in m.parameters()) / 1e6
    print(f"params: TRAINABLE {n_train:.2f}M / {n_total:.2f}M  n_vis_tok={n_vis}")

    # EMA shadow
    ema_state = None
    if args.ema_decay > 0:
        ema_state = {}
        for nm, m in zip(keys, mods):
            for pname, p in m.named_parameters():
                if p.requires_grad:
                    ema_state[f"{nm}.{pname}"] = p.detach().clone()
        if _ck is not None and 'ema_params' in _ck:
            for k in list(ema_state.keys()):
                if k in _ck['ema_params']: ema_state[k] = _ck['ema_params'][k].to(dev)
            print(f"  [ema] resumed {sum(1 for k in ema_state if k in _ck.get('ema_params',{}))}/{len(ema_state)} EMA params")
        else:
            print(f"  [ema] initialized {len(ema_state)} EMA params @ decay={args.ema_decay}")

    # opt state resume (skipped if --reset-opt so fresh momentum at new lr)
    if _ck is not None and 'opt' in _ck and not args.reset_opt:
        try: opt.load_state_dict(_ck['opt'])
        except: pass
    elif args.reset_opt and _ck is not None:
        print(f"  [reset-opt] discarded saved optimizer state — fresh momentum at lr={args.lr}", flush=True)

    # ── dataloader ──
    _DSCls = StreamingMultiOXEDataset if args.streaming else MultiOXEDataset
    _ds_kw = dict(chunk_len=16, lookback=16)
    if not args.streaming: _ds_kw['two_frame'] = (args.vision == 'vit')
    # episode-level TRAIN split: restrict chunks to train episodes (no leakage into val/test)
    if args.train_eps_file:
        tr_set = set(json.load(open(args.train_eps_file)))
        for sp in specs:
            sp.chunk_index = [(ep, s) for (ep, s) in sp.chunk_index if ep in tr_set]
        print(f"  [train-split] restricted to {len(tr_set)} train episodes → "
              f"{sum(len(sp.chunk_index) for sp in specs)} chunks", flush=True)
    ds = _DSCls(specs, **_ds_kw)
    print(f"  dataset: {len(ds)} chunks total")

    _AUG_KW = dict(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4),
                   blur_sigma=(0.0, 1.5), crop_keep=(0.70, 1.0)) if args.strong_aug else {}
    def _aug_pil(pil, params):
        pil = pil.convert('RGB')
        if params: pil = augment._apply_visual_params(pil, params)
        pil = pil.resize((args.img_size, args.img_size))
        return torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.
    def frame_to_tensor(f):
        params = augment._sample_visual_params(random, **_AUG_KW) if _AUG_KW else None
        if isinstance(f, tuple):                              # (prev, cur) for ViT — SAME aug on both
            return torch.stack([_aug_pil(f[0], params), _aug_pil(f[1], params)])   # (2,3,H,W)
        return _aug_pil(f, params)

    def collate(batch):
        # batch: list of (pil_frame, state, action, prev_action, task_str, embodiment_id, ds_idx)
        frames = torch.stack([frame_to_tensor(b[0]) for b in batch])
        states = torch.stack([b[1] for b in batch])
        actions = torch.stack([b[2] for b in batch])
        prevs = torch.stack([b[3] for b in batch])
        tasks = [b[4] for b in batch]
        emb_ids_int = [b[5] for b in batch]
        emb_robots = [EMBODIMENTS[e] if e < len(EMBODIMENTS) else 'unknown' for e in emb_ids_int]
        ds_idxs = torch.tensor([b[6] for b in batch], dtype=torch.long)
        return frames, states, actions, prevs, tasks, emb_robots, ds_idxs

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, collate_fn=collate,
                                         drop_last=True, persistent_workers=args.num_workers > 0,
                                         pin_memory=True,
                                         prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    def t5_layers_batch(tasks):
        B, T = len(tasks), args.max_text
        out = torch.zeros(t5_layers, B, T, t5_dim)
        for b, tk in enumerate(tasks):
            # paraphrase sampling: with --use-paraphrase-sampling, randomly pick
            # original or one of the paraphrases. Same intent, different wording.
            if args.use_paraphrase_sampling and tk in t5_paraphrase_map:
                choices = [tk] + t5_paraphrase_map[tk]
                tk = random.choice(choices)
            e = t5_emb.get(tk)
            if e is None: continue
            h = e['hidden'].float(); t = min(h.shape[1], T)
            out[:, b, :t, :] = h[:, :t, :]
        return out.to(dev)

    def encode_modalities(frames, tasks, emb_robots, mod_drop=0.0):
        # embodiment-id tensor (used both for CNN-FiLM and for prefix tokens)
        idx = torch.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb_robots], device=dev)
        if args.vision in ('vit', 'dinoagg'):
            vtok, _ = cnn(frames)                            # vit (B,2,3,H,W) / dinoagg (B,3,H,W); both normalize internally
        else:
            x = normalize_image(frames, var_global_img)      # per-image RevIN
            vtok, _ = cnn(x, emb_id=idx if args.cnn_film_by_emb else None)
        vtok = cnn_proj(vtok)
        t5s = t5_layers_batch(tasks)
        tagg = text_agg([t5s[l] for l in range(t5s.shape[0])])
        ttok = text_proj(tagg)
        # embodiment-id prefix tokens (B, n_emb_prefix, dim)
        etok = emb_id_emb(idx).view(idx.shape[0], args.n_emb_prefix, args.dim)
        if mod_drop > 0.0:
            B = vtok.shape[0]
            vk = (torch.rand(B, 1, 1, device=dev) >= mod_drop).float()
            tk_ = (torch.rand(B, 1, 1, device=dev) >= mod_drop).float()
            vtok = vtok * vk; ttok = ttok * tk_
        return kv_norm(torch.cat([etok, vtok, ttok], dim=1))

    @torch.no_grad()
    def encode_state(state, emb_robots, noisy=False):
        # group by embodiment, run each through its own encoder, scatter back
        out = torch.zeros(state.shape[0], args.dim, device=dev)
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=dev)
            if mask.sum() == 0: continue
            s = state[mask]
            if noisy and args.state_noise > 0: s = s + torch.randn_like(s) * args.state_noise
            out[mask] = state_encoders[emb](s)
        return out

    # Whether the loaded VAE is the embodiment-conditioned shared one (needs emb_id arg).
    _is_shared_vae = isinstance(next(iter(vaes.values())), type(next(iter(vaes.values())))) and \
                     'CondActionVQVAE1d' in type(next(iter(vaes.values()))).__name__

    def encode_targets(actions, prevs, emb_robots):
        """Per-embodiment precision-normalize, then encode to GT codes.
        Routes via the shared cond VAE (one model, emb_id arg) or per-emb VAE dict."""
        all_codes = [torch.zeros(actions.shape[0], T_l, dtype=torch.long, device=dev) for T_l in seq_lens]
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=dev)
            if mask.sum() == 0: continue
            ac = actions[mask]; pv = prevs[mask]
            vg = var_globals[emb]
            nT = ac.shape[1]
            m = pv.mean(dim=1, keepdim=True)
            S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
            lam = nT / (S + nT * vg)
            xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
            # action-noise augmentation: perturb the precision-normalized actions
            # (i.e., in the same space the VAE encoder sees). Prevents the model from
            # memorizing exact action sequences when data is limited; analogous to state_noise.
            if args.action_noise > 0.0:
                xn = xn + torch.randn_like(xn) * args.action_noise
            vae = vaes[emb]
            if _is_shared_vae:
                eid = torch.full((ac.shape[0],), EMBODIMENT_ID.get(emb, len(EMBODIMENTS)),
                                  dtype=torch.long, device=dev)
                gt, _ = vae.encode_with_soft(xn, eid, tau=0.1)
            else:
                gt, _ = vae.encode_with_soft(xn, tau=0.1)
            for l in range(len(seq_lens)):
                all_codes[l][mask] = gt[l]
        return all_codes

    # ── OOD-recovery probe: sim→real CNN-feature kNN ratio (objective-aligned early-stop) ──
    _ood = None
    if args.ood_probe_every > 0:
        _sim_np = np.load(args.ood_sim); _real_np = np.load(args.ood_real)
        def _bank(arr, n=256):
            sel = np.random.RandomState(0).choice(len(arr), min(n, len(arr)), replace=False)
            return torch.from_numpy(arr[sel].astype(np.float32) / 255.).permute(0, 3, 1, 2)
        _ood = {'sim': _bank(_sim_np), 'real': _bank(_real_np), 'best': float('inf'),
                'noimp': 0, 'path': args.ood_best_path or (args.ckpt_path + '.best_ood')}
        print(f"  [ood-probe] sim {tuple(_ood['sim'].shape)} real {tuple(_ood['real'].shape)}; "
              f"every {args.ood_probe_every} steps → best ckpt to {_ood['path']}", flush=True)

    @torch.no_grad()
    def _ood_feats(imgs01):
        was = cnn.training
        cnn.eval()                                  # eval() → MixStyle is identity → clean features
        out = []
        for i in range(0, len(imgs01), 32):
            x = imgs01[i:i+32].to(dev)
            x = F.interpolate(x, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
            x = normalize_image(x, var_global_img)
            v, _ = cnn(x); v = cnn_proj(v)
            out.append(v.mean(1).float().cpu())
        if was: cnn.train()
        return torch.cat(out).numpy()

    def _ood_ratio():
        rf = _ood_feats(_ood['real']); sf = _ood_feats(_ood['sim'])
        def _knn(A, B, k):
            d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            return np.sort(d, axis=1)[:, :k].mean()
        rr = _knn(rf[:min(50, len(rf))], rf, 6)
        return _knn(sf, rf, 5) / rr

    # ── batched probe over the FULL set: FRESH batches + FRESH random masks every call
    #    (never a fixed batch/mask). val and train_eps are evaluated identically → clean overfit
    #    signal, and the probe-to-probe variance is honest eval noise, not a fixed artifact. ──
    def _collate_eval(batch):                                  # like collate but NO visual aug on eval frames
        def _ft(f):
            return (torch.stack([_aug_pil(f[0], None), _aug_pil(f[1], None)])
                    if isinstance(f, tuple) else _aug_pil(f, None))
        return (torch.stack([_ft(b[0]) for b in batch]),
                torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch]),
                torch.stack([b[3] for b in batch]), [b[4] for b in batch],
                [EMBODIMENTS[b[5]] if b[5] < len(EMBODIMENTS) else 'unknown' for b in batch])

    def _make_eval_loader(eps_file):
        # Episode ids are GLOBALLY unique across AgiBot tasks, so one eps_set filters every task.
        # Build a fresh (eval-stride) spec per task dir and keep only its in-set chunks — multi-task safe.
        eps_set = set(json.load(open(eps_file)))
        ev_specs, nw = [], 0
        for base in specs:
            sp = load_dataset_spec(base.root, chunk_len=16, lookback=16, chunk_stride=args.chunk_stride)
            sp.chunk_index = [(ep, s) for (ep, s) in sp.chunk_index if ep in eps_set]
            if sp.chunk_index:
                ev_specs.append(sp); nw += len(sp.chunk_index)
        if not ev_specs: return None, 0
        ds = MultiOXEDataset(ev_specs, chunk_len=16, lookback=16, two_frame=(args.vision == 'vit'))
        ld = torch.utils.data.DataLoader(ds, batch_size=min(96, args.batch_size), shuffle=True,
                                         num_workers=4, collate_fn=_collate_eval, drop_last=False)
        return ld, nw

    _eval_bs = min(96, args.batch_size)
    _probe_nb = max(1, args.val_n // _eval_bs)                  # batches per probe (≈ val_n windows)
    _val_loader = _train_loader = None
    _val_best = {'best': float('inf'), 'noimp': 0, 'path': args.val_best_path or (args.ckpt_path + '.best_val')}
    if args.val_probe_every > 0 and args.val_eps_file:
        _val_loader, _nvw = _make_eval_loader(args.val_eps_file)
        if _val_loader:
            print(f"  [val-probe] full val set: {_nvw} windows (stride {args.chunk_stride}); every "
                  f"{args.val_probe_every} steps, {_probe_nb}×{_eval_bs} FRESH windows+masks → best → {_val_best['path']}", flush=True)
        if args.train_eps_file:
            _train_loader, _ntw = _make_eval_loader(args.train_eps_file)
            if _train_loader:
                print(f"  [train-probe] full train set: {_ntw} windows — evaluated identically to val", flush=True)

    @torch.no_grad()
    def _probe(loader, gen=False, n_batches=None, gen_batches=1):
        """Teacher-forced masked-CE acc+loss over FRESH batches of the full set with FRESH cosine
        masks each call (no fixed batch/mask). gen=True adds MaskGIT-decode vs single-forward
        generation acc on the first gen_batches. Returns (acc, loss[, gen_acc, single_acc])."""
        nb = n_batches or _probe_nb
        was_t = [m.training for m in mods]
        for m in mods: m.eval()
        try:
            tc = tm = 0; lsum = ln = 0.0; gc = sc = gt_tot = 0
            for bi, (frames, states, actions, prevs, tasks, emb) in enumerate(loader):
                if bi >= nb: break
                frames = frames.to(dev); states = states.to(dev)
                actions = actions.to(dev); prevs = prevs.to(dev)
                vis = encode_modalities(frames, tasks, emb, mod_drop=0.0)
                s_enc = encode_state(states, emb, noisy=False)
                eid_t = torch.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb],
                                     dtype=torch.long, device=dev)
                gt = encode_targets(actions, prevs, emb)
                T_l = gt[0].shape[1]; B = gt[0].shape[0]
                u = torch.rand(B, device=dev)                    # FRESH cosine masks (no fixed seed)
                r = (torch.full((B,), float(args.val_mask_ratio), device=dev) if args.val_mask_ratio > 0
                     else torch.cos(math.pi * u / 2).clamp(min=1.0 / T_l))
                ns = torch.rand(B, T_l, device=dev); mk = ns < r.unsqueeze(1)
                mk[torch.arange(B, device=dev), ns.argmin(1)] = True
                logits = policy(gt, vis, s_enc, mask_list=[mk],
                                n_outer=args.H_outer, n_inner=args.L_inner, emb_id=eid_t)[-1][0]
                tc += ((logits.argmax(-1) == gt[0]) & mk).float().sum().item(); tm += mk.float().sum().item()
                lsum += F.cross_entropy(logits[mk], gt[0][mk], reduction='sum').item(); ln += mk.float().sum().item()
                if gen and bi < gen_batches:
                    g8 = policy.generate(vis, s_enc, emb_id=eid_t, n_steps=8)
                    g1 = policy.generate(vis, s_enc, emb_id=eid_t, n_steps=1)
                    for _l in range(len(gt)):
                        gc += (g8[_l] == gt[_l]).float().sum().item()
                        sc += (g1[_l] == gt[_l]).float().sum().item(); gt_tot += gt[_l].numel()
            acc, loss = tc / max(1, tm), lsum / max(1, ln)
            return (acc, loss, gc / max(1, gt_tot), sc / max(1, gt_tot)) if gen else (acc, loss)
        finally:
            for m, w in zip(mods, was_t):
                if w: m.train()

    # ── eval-why-depth: per-iteration readout + direction convergence (why don't more iters help?) ──
    if args.eval_why_depth:
        if _val_loader is None:
            print("  [why-depth] no val loader — abort", flush=True); return
        for m in mods: m.eval()
        import torch as _t
        frames, states, actions, prevs, tasks, emb = next(iter(_val_loader))
        nfp = min(64, frames.shape[0])
        frames=frames[:nfp].to(dev); states=states[:nfp].to(dev); actions=actions[:nfp].to(dev)
        prevs=prevs[:nfp].to(dev); tasks=tasks[:nfp]; emb=emb[:nfp]
        with _t.no_grad():
            vis = encode_modalities(frames, tasks, emb, mod_drop=0.0)
            s_enc = encode_state(states, emb, noisy=False)
            gt = encode_targets(actions, prevs, emb)
        eid = _t.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb], dtype=_t.long, device=dev)
        T_l = gt[0].shape[1]
        mk = [(_t.rand(nfp, T_l, device=dev) < 0.5)]                       # 50% mask = informative regime
        rows = policy.diagnose_depth_usage(vis, s_enc, indices_list=gt, mask_list=mk, emb_id=eid, n_iter=20)
        print(f"\n  [why-depth] inner map per-iteration (50%-masked). LayerScale γ={[round(x,3) for x in policy._layerscale_stats()]}")
        print(f"  {'iter k':>6} {'CE':>7} {'acc%':>6} {'cos→final':>10} {'‖z_k‖':>8} {'‖g-z‖/‖z‖':>10} {'xExCos':>7} {'stableRank':>10}")
        for k, ce, ac, cos, zn, gr, xc, sr in rows:
            if k <= 6 or k % 5 == 0:
                print(f"  {k:>6} {ce:>7.3f} {ac:>6.1f} {cos:>10.4f} {zn:>8.1f} {gr:>10.3f} {xc:>7.3f} {sr:>10.2f}", flush=True)
        return

    # ── eval-maskgit-variants: greedy/sampled × maxprob/entropy confidence × n_steps ──
    if args.eval_maskgit_variants:
        if _val_loader is None:
            print("  [mgv] no val loader — abort", flush=True); return
        for m in mods: m.eval()
        import torch as _t
        frames, states, actions, prevs, tasks, emb = next(iter(_val_loader))
        nfp = min(64, frames.shape[0])
        frames=frames[:nfp].to(dev); states=states[:nfp].to(dev); actions=actions[:nfp].to(dev)
        prevs=prevs[:nfp].to(dev); tasks=tasks[:nfp]; emb=emb[:nfp]
        with _t.no_grad():
            vis = encode_modalities(frames, tasks, emb, mod_drop=0.0)
            s_enc = encode_state(states, emb, noisy=False)
            gt = encode_targets(actions, prevs, emb)
        eid = _t.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb], dtype=_t.long, device=dev)
        def gacc(**kw):
            reps = 3 if kw.get('sample') else 1               # average stochastic variants
            accs=[]
            for _ in range(reps):
                with _t.no_grad(): pred = policy.generate(vis, s_enc, emb_id=eid, **kw)
                tc=tn=0.0
                for l in range(len(gt)): tc+=(pred[l]==gt[l]).float().sum().item(); tn+=gt[l].numel()
                accs.append(tc/max(tn,1)*100)
            return sum(accs)/len(accs)
        variants = [
            ('greedy  · maxprob', dict(sample=False, conf_measure='maxprob', temperature=0.0)),
            ('sampled · maxprob', dict(sample=True,  conf_measure='maxprob', temperature=1.0)),
            ('greedy  · entropy', dict(sample=False, conf_measure='entropy', temperature=0.0)),
            ('sampled · entropy', dict(sample=True,  conf_measure='entropy', temperature=1.0)),
        ]
        print(f"\n  [maskgit-variants] generation acc% (all-masked→full codes, T=4); sampled=avg of 3:")
        print(f"  {'variant':>20} | " + "  ".join(f"n={ns:<2}" for ns in [1,2,4,8]))
        for name, base in variants:
            cells=[f"{gacc(n_steps=ns, **base):5.1f}" for ns in [1,2,4,8]]
            print(f"  {name:>20} | " + "   ".join(cells), flush=True)
        return

    # ── eval-spectrum: dominant eigenvalue vs norm (explain α·σ_g) + MaskGIT n_steps sanity ──
    if args.eval_spectrum:
        if _val_loader is None:
            print("  [spectrum] no val loader — abort", flush=True); return
        for m in mods: m.eval()
        import torch as _t
        frames, states, actions, prevs, tasks, emb = next(iter(_val_loader))
        nfp = min(48, frames.shape[0])
        frames=frames[:nfp].to(dev); states=states[:nfp].to(dev); actions=actions[:nfp].to(dev)
        prevs=prevs[:nfp].to(dev); tasks=tasks[:nfp]; emb=emb[:nfp]
        with _t.no_grad():
            vis = encode_modalities(frames, tasks, emb, mod_drop=0.0)
            s_enc = encode_state(states, emb, noisy=False)
            gt = encode_targets(actions, prevs, emb)
        eid = _t.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb], dtype=_t.long, device=dev)
        T_l = gt[0].shape[1]
        # iterate damped to z* (inner map, z_H=0), then measure spectrum there
        mk = [_t.ones(nfp, T_l, dtype=_t.bool, device=dev)]
        kv = policy._build_kv(vis, s_enc); y = policy._y_embed(nfp, dev, gt, mk); ctx = y
        alpha,_ = policy._rhos(); z = _t.zeros_like(y)
        with _t.no_grad():
            for _ in range(40): z = (1-alpha)*z + alpha*policy.g(z, ctx, kv)
        sig = policy._spectral_norm_jz(z, ctx, kv, iters=12)
        absl, re = policy._dominant_eig_jz(z, ctx, kv, iters=40)
        a = float(alpha.mean()) if _t.is_tensor(alpha) else float(alpha)
        print(f"\n  [spectrum] at z* (α={a:.3f}):")
        print(f"    σ_g (spectral NORM ‖J‖)      = {sig:.3f}")
        print(f"    |λ_dom| (dominant eigenvalue) = {absl:.3f}   ratio |λ|/‖J‖ = {absl/max(sig,1e-9):.3f}  (≪1 ⇒ non-normal)")
        print(f"    Re(λ_dom) (Rayleigh v·Jv)     = {re:.3f}   (Re<0 ⇒ oscillatory/dampable; ≥1 ⇒ undampable)")
        print(f"    α·σ_g = {a*sig:.3f}   α·|λ_dom| = {a*absl:.3f}")
        # MaskGIT n_steps sanity: more steps should help (or ≥) if context-conditioning works
        print(f"  [maskgit] generation acc vs n_steps (all-masked start → full action codes):")
        for ns in [1,2,4,8,16]:
            with _t.no_grad():
                pred = policy.generate(vis, s_enc, emb_id=eid, n_steps=ns)
            tc=tn=0.0
            for l in range(len(gt)):
                tc += (pred[l]==gt[l]).float().sum().item(); tn += gt[l].numel()
            print(f"    n_steps={ns:>2}: gen_acc={tc/max(tn,1)*100:.2f}%", flush=True)
        return

    # ── eval-depth-sweep: does the NUMBER of recursion iterations (L inner × H outer) move acc/CE? ──
    if args.eval_depth_sweep:
        if _val_loader is None:
            print("  [depth] no val loader — abort", flush=True); return
        for m in mods: m.eval()
        import torch as _t
        frames, states, actions, prevs, tasks, emb = next(iter(_val_loader))
        nfp = min(64, frames.shape[0])
        frames = frames[:nfp].to(dev); states = states[:nfp].to(dev)
        actions = actions[:nfp].to(dev); prevs = prevs[:nfp].to(dev); tasks = tasks[:nfp]; emb = emb[:nfp]
        with _t.no_grad():
            vis = encode_modalities(frames, tasks, emb, mod_drop=0.0)
            s_enc = encode_state(states, emb, noisy=False)
            gt = encode_targets(actions, prevs, emb)
        eid = _t.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb], dtype=_t.long, device=dev)
        T_l = gt[0].shape[1]
        def ce_acc(logits, mk):
            tce=tc=tn=0.0
            for l in range(len(gt)):
                m=mk[l]; lg=logits[l][..., :K]
                tce+=_t.nn.functional.cross_entropy(lg[m], gt[l][m], reduction='sum').item()
                tc+=(lg[m].argmax(-1)==gt[l][m]).float().sum().item(); tn+=m.sum().item()
            return tce/max(tn,1), tc/max(tn,1)*100
        Hs=[1,2,3,4,6,8]; Ls=[1,2,5,10]
        for label, mk in [('ALL-masked', [_t.ones(nfp,T_l,dtype=_t.bool,device=dev)]),
                          ('50%-masked', [(_t.rand(nfp,T_l,device=dev)<0.5)])]:
            print(f"\n  [depth-sweep] {label} — masked-CE (acc%) over (H_outer × L_inner):")
            print("      L_inner: " + "  ".join(f"{L:>11}" for L in Ls))
            _Lsave = policy.L_inner
            for H in Hs:
                cells=[]
                for L in Ls:
                    policy.L_inner = L                          # damped _inner loops self.L_inner, NOT n_inner
                    with _t.no_grad():
                        logits = policy.forward(gt, vis, s_enc, mask_list=mk, n_outer=H, n_inner=L, emb_id=eid)[-1]
                    ce,ac = ce_acc(logits, mk); cells.append(f"{ce:5.2f}({ac:4.1f})")
                print(f"  H={H:>2}:  " + "  ".join(cells), flush=True)
            policy.L_inner = _Lsave
        return

    # ── eval-invariance: probe whether the readout actually depends on the latent z ──
    if args.eval_invariance:
        if _val_loader is None:
            print("  [invariance] no val loader — abort", flush=True); return
        for m in mods: m.eval()
        frames, states, actions, prevs, tasks, emb = next(iter(_val_loader))
        nfp = min(32, frames.shape[0])
        frames = frames[:nfp].to(dev); states = states[:nfp].to(dev)
        actions = actions[:nfp].to(dev); prevs = prevs[:nfp].to(dev); tasks = tasks[:nfp]; emb = emb[:nfp]
        with torch.no_grad():
            vis = encode_modalities(frames, tasks, emb, mod_drop=0.0)
            s_enc = encode_state(states, emb, noisy=False)
            gt = encode_targets(actions, prevs, emb)
        eid = torch.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb], dtype=torch.long, device=dev)
        T_l = gt[0].shape[1]
        import torch as _t
        for label, mk in [('ALL-masked (deploy)', [_t.ones(nfp, T_l, dtype=_t.bool, device=dev)]),
                          ('50%-masked', [(_t.rand(nfp, T_l, device=dev) < 0.5)])]:
            d = policy.diagnose_iteration_invariance(vis, s_enc, indices_list=gt, mask_list=mk, emb_id=eid)
            print(f"\n  [invariance] {label}  (mask_frac={d['mask_frac']:.2f})  — readout CE / acc on masked positions:")
            print(f"  {'latent fed to head':>20} | {'WITH out-norm (deployed)':>26} | {'raw z (no out-norm)':>22}")
            for nm in ('z*', 'zero', 'random', '2·z*', 'z_accum'):
                ce_w, ac_w = d['with_outnorm'][nm]; ce_r, ac_r = d['raw_z'][nm]
                print(f"  {nm:>20} | CE {ce_w:6.3f}  acc {ac_w*100:5.1f}%      | CE {ce_r:6.3f}  acc {ac_r*100:5.1f}%", flush=True)
            zz = d['z_accum_vs_zstar']
            print(f"  z_accum vs z*: rel-dist={zz['rel_dist']:.3f}  cosine={zz['cosine']:.3f}  "
                  f"(low-impact if zero/random ≈ z*)", flush=True)
        return

    # ── eval-iter-schemes: drive the SAME trained g through 3 fixed-point schemes, plot, exit ──
    if args.eval_iter_schemes:
        if _val_loader is None:
            print("  [iter-schemes] no val loader — abort", flush=True); return
        for m in mods: m.eval()
        frames, states, actions, prevs, tasks, emb = next(iter(_val_loader))
        nfp = min(32, frames.shape[0])
        frames = frames[:nfp].to(dev); states = states[:nfp].to(dev)
        actions = actions[:nfp].to(dev); prevs = prevs[:nfp].to(dev); tasks = tasks[:nfp]; emb = emb[:nfp]
        with torch.no_grad():
            vis = encode_modalities(frames, tasks, emb, mod_drop=0.0)
            s_enc = encode_state(states, emb, noisy=False)
            gt = encode_targets(actions, prevs, emb)
        eid = torch.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb], dtype=torch.long, device=dev)
        mk = [torch.ones(nfp, gt[0].shape[1], dtype=torch.bool, device=dev)]   # all-masked = deploy condition
        res = policy.compare_iteration_schemes(vis, s_enc, indices_list=gt, mask_list=mk, emb_id=eid,
                                               n_iter=args.iter_n, beta=args.iter_beta)
        print(f"\n  [iter-schemes] α(learned)={res['damped']['alpha']:.3f}  β={args.iter_beta}  n_iter={args.iter_n}")
        print(f"  {'scheme':>12} {'final_step_resid':>16} {'final_fp_resid‖g(z)-z‖':>22} {'final_‖z‖':>10} {'readout_CE':>11} {'acc':>6}")
        for sch in ('accumulator', 'damped', 'nesterov'):
            d = res[sch]
            print(f"  {sch:>12} {d['step_resid'][-1]:>16.3e} {d['fp_resid'][-1]:>22.3e} "
                  f"{d['z_norm'][-1]:>10.1f} {d['ce']:>11.4f} {d['acc']*100:>5.1f}%", flush=True)
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
            col = {'accumulator': 'tab:green', 'damped': 'tab:blue', 'nesterov': 'tab:red'}
            for sch in ('accumulator', 'damped', 'nesterov'):
                d = res[sch]; xs = range(1, len(d['fp_resid']) + 1)
                ax[0].plot(xs, d['fp_resid'], '-o', ms=3, color=col[sch], label=sch)
                ax[1].plot(xs, d['step_resid'], '-o', ms=3, color=col[sch], label=sch)
                ax[2].plot(xs, d['z_norm'], '-o', ms=3, color=col[sch], label=sch)
            ax[0].set_title('TRUE fixed-point residual ‖g(z)-z‖ (the honest metric)'); ax[0].set_yscale('log')
            ax[1].set_title('step residual ‖Δz‖ (accumulator →0 trivially)'); ax[1].set_yscale('log')
            ax[2].set_title('‖z‖ (does it blow up?)')
            for a in ax: a.set_xlabel('iteration'); a.grid(alpha=0.3, which='both'); a.legend(fontsize=8)
            plt.tight_layout(); plt.savefig('data/logs/iter_schemes.png', dpi=110)
            print("  [iter-schemes] saved data/logs/iter_schemes.png", flush=True)
        except Exception as _e:
            print(f"  [iter-schemes] plot failed: {_e}", flush=True)
        return

    # ── eval-only: run the probe once on the loaded weights and exit (NO training step) ──
    if args.eval_only:
        if _val_loader is None:
            print("  [eval-only] no val probe configured — abort", flush=True); return
        vacc, vloss, vgen, vsingle = _probe(_val_loader, gen=True, n_batches=max(_probe_nb, 8))
        tacc, tloss = _probe(_train_loader, n_batches=max(_probe_nb, 8)) if _train_loader is not None else (float('nan'), float('nan'))
        print(f"  [eval-only] H_outer={args.H_outer}: train_acc={tacc*100:.2f}%/loss={tloss:.4f}  "
              f"val_acc={vacc*100:.2f}%/loss={vloss:.4f}  gap={(tacc-vacc)*100:.1f}pt/{vloss-tloss:.3f}  "
              f"GEN(maskgit)={vgen*100:.2f}%  gen(single)={vsingle*100:.2f}%", flush=True)
        return

    # ── training loop ──
    _early_stop = False
    revive_dead = {'L': 0, 'H': 0}; revive_cd = {'L': 0, 'H': 0}
    revive_peak = {'L': 0.1, 'H': 0.1}
    step = _step; t0 = time.perf_counter(); wl = wc = wa = wn = 0
    # Guardrail: register an atexit handler that always saves a ckpt on process exit,
    # whether normal, exception, OOM, or kill. Closure captures the latest step counter.
    import atexit as _atexit
    _emergency_save_state = {'step': step}
    def _emergency_save():
        try:
            s = _emergency_save_state['step']
            print(f"  [emergency-save] writing ckpt at step {s} ...", flush=True)
            save_ckpt(args, mods, keys, opt, ema_state, s)
        except Exception as _e:
            print(f"  [emergency-save] FAILED: {type(_e).__name__}: {_e}", flush=True)
    _atexit.register(_emergency_save)
    # Install SIGTERM handler that translates SIGTERM into sys.exit(), so atexit fires.
    # Default Python behavior on SIGTERM = terminate immediately, atexit DOES NOT run.
    # With this handler, when our stop-when-90 watcher (or any external SIGTERM)
    # asks training to stop, _emergency_save writes a final ckpt before exit.
    import signal as _signal, sys as _sys
    def _sigterm(signum, frame):
        print(f"  [sigterm] received SIGTERM at step {_emergency_save_state['step']} — "
              f"calling sys.exit() to trigger emergency-save", flush=True)
        _sys.exit(0)
    _signal.signal(_signal.SIGTERM, _sigterm)
    # LR schedule. 'constant' = 150-step warmup then flat (legacy).
    # 'cosine' = linear warmup (--lr-warmup-frac of --steps) then cosine decay to --lr-min (default 0.1·lr).
    _lr_min = args.lr_min if args.lr_min is not None else args.lr * 0.1
    _warmup_constant = 150
    _warmup_cosine = int(args.steps * args.lr_warmup_frac)
    def _lr_at(s):
        if args.lr_schedule == 'cosine':
            if s < _warmup_cosine:
                return args.lr * s / max(_warmup_cosine, 1)
            if s >= args.steps:
                return _lr_min
            progress = (s - _warmup_cosine) / max(args.steps - _warmup_cosine, 1)
            return _lr_min + 0.5 * (args.lr - _lr_min) * (1 + math.cos(math.pi * progress))
        # constant (legacy)
        return args.lr * min(1.0, s / _warmup_constant)
    if args.lr_schedule == 'cosine':
        print(f"  [lr-schedule] cosine: warmup {_warmup_cosine} steps ({args.lr_warmup_frac*100:.0f}% of {args.steps}); "
              f"peak {args.lr:.2e} → min {_lr_min:.2e} (×{_lr_min/args.lr:.2f})", flush=True)
    warmup = _warmup_constant                                                     # kept for legacy print compat
    print(f"\n=== training: {args.steps} steps, batch {args.batch_size}, {args.num_workers} workers ===")
    # Guardrails: periodic empty_cache fights CUDA allocator fragmentation; memory watchdog
    # checks usage at intervals and force-clears when near full. The try/except below catches
    # CUDA OOM and other exceptions, saves a final emergency ckpt so we don't lose progress,
    # then re-raises so the caller still knows training failed. See [[feedback-training-guardrails]].
    CACHE_CLEAR_EVERY = 1000
    OOM_FRACTION_LIMIT = 0.93
    while step < args.steps and not _early_stop:
        for frames, states, actions, prevs, tasks, emb_robots, _ in loader:
            if step >= args.steps: break
            step += 1
            frames = frames.to(dev, non_blocking=True)
            # vision regularization (training-only): prev-frame dropout + APR amplitude-randomization.
            if args.vision == 'vit':
                if args.prev_frame_drop > 0.0:                 # zero prev frame sometimes (rollout step-0 robustness)
                    dm = torch.rand(frames.shape[0], device=dev) < args.prev_frame_drop
                    frames[dm, 0] = 0.0
                if args.apr_prob > 0.0:                        # APR per-frame (flatten the 2-frame axis)
                    B0, Fr = frames.shape[0], frames.shape[1]
                    fr = augment.apr_augment(frames.reshape(B0 * Fr, *frames.shape[2:]),
                                             p=args.apr_prob, eta_max=args.apr_eta_max)
                    frames = fr.reshape(B0, Fr, *fr.shape[1:])
            elif args.apr_prob > 0.0:
                frames = augment.apr_augment(frames, p=args.apr_prob, eta_max=args.apr_eta_max)
            states = states.to(dev, non_blocking=True)
            actions = actions.to(dev, non_blocking=True)
            prevs = prevs.to(dev, non_blocking=True)
            # state encoding (per-emb) + dropout
            if args.dropout_prob > 0.0:
                keep = (torch.rand(states.shape[0], 1, device=dev) >= args.dropout_prob).float()
                states = states * keep
            s_enc = encode_state(states, emb_robots, noisy=True)
            # targets
            with torch.no_grad():
                gt = encode_targets(actions, prevs, emb_robots)
            try:
                # bf16 autocast: ~halves activation memory, no GradScaler needed (bf16 has fp32 dynamic range).
                # Disabled via --no-amp. Master weights stay fp32; gradients accumulate in fp32.
                _amp_ctx = torch.autocast('cuda', dtype=torch.bfloat16) if args.amp else _NullCtx()
                with _amp_ctx:
                    vis = encode_modalities(frames, tasks, emb_robots, mod_drop=args.dropout_prob)
                    emb_id_t = torch.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb_robots],
                                            dtype=torch.long, device=dev)
                    # Both STRMPolicy and STRMPolicyVAE now accept emb_id + label_smoothing kwargs.
                    loss, per, _ = policy.forward_loss(gt, vis, s_enc,
                                                       n_inner=args.L_inner, h_max=args.h_max,
                                                       emb_id=emb_id_t,
                                                       label_smoothing=args.label_smoothing,
                                                       mask_sampler=args.mask_sampler)
                # NaN-safety: if loss/grads went bad, skip the step entirely (don't poison params).
                if not torch.isfinite(loss):
                    if step % args.log_every == 0:
                        print(f"  [skip] non-finite loss at step {step}; zeroing grads", flush=True)
                    opt.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    # SMART gradient clipping — runs BOTH:
                    # 1. HARD ceiling (max_norm=args.grad_clip_max) — never lets ‖g‖ exceed this absolute value.
                    #    Cheap insurance against exploding gradients that smart-median would track upward.
                    # 2. SPIKE-vs-median clip — only triggers when ‖g‖ jumps >5× median of recent 20 steps.
                    #    Preserves LARS's behavior in the normal regime; catches transient spikes.
                    trainable_params = [p for m in mods for p in m.parameters() if p.requires_grad]
                    total_grad_norm = torch.norm(
                        torch.stack([p.grad.detach().norm() for p in trainable_params if p.grad is not None])
                    ).item()
                    # (1) hard ceiling — applied first, unconditional
                    if total_grad_norm > args.grad_clip_max and args.grad_clip_max > 0:
                        scale = args.grad_clip_max / total_grad_norm
                        for p in trainable_params:
                            if p.grad is not None: p.grad.mul_(scale)
                        if step % args.log_every == 0:
                            print(f"  [grad-hard-clip] step {step}: ||g||={total_grad_norm:.2f} > "
                                  f"max {args.grad_clip_max:.1f}, scaled by {scale:.4f}", flush=True)
                        total_grad_norm = args.grad_clip_max
                    # (2) spike vs running median (still useful for smaller spikes inside hard limit)
                    if 'grad_norm_history' not in locals():
                        grad_norm_history = []
                    grad_norm_history.append(total_grad_norm)
                    if len(grad_norm_history) > 50: grad_norm_history.pop(0)
                    if len(grad_norm_history) >= 20:
                        median_recent = sorted(grad_norm_history[-20:])[10]
                        spike_threshold = 5.0 * median_recent
                        if total_grad_norm > spike_threshold and median_recent > 0:
                            scale = median_recent / total_grad_norm
                            for p in trainable_params:
                                if p.grad is not None: p.grad.mul_(scale)
                            if step % args.log_every == 0:
                                print(f"  [grad-spike] step {step}: ||g||={total_grad_norm:.2f} > {spike_threshold:.2f} "
                                      f"(5× median {median_recent:.2f}), clipped to median", flush=True)
                    for g in opt.param_groups: g['lr'] = _lr_at(step)
                    opt.step(); opt.zero_grad(set_to_none=True)
                    # POST-step NaN guard: if any param went NaN/Inf despite all safety, log loud + abort the step
                    if any(not torch.isfinite(p).all() for m in mods for p in m.parameters() if p.requires_grad):
                        print(f"  [PARAM-NaN] step {step}: a param became non-finite POST-step; this should not happen "
                              f"with hard clipping. Likely needs lower lr or smaller batch.", flush=True)
            except torch.cuda.OutOfMemoryError as _oom_e:
                # OOM-safe: log, zero grads, empty cache, skip rest of this step (EMA/log/ckpt) via continue.
                # The step counter is preserved so ckpt-every / log-every / val cadence isn't disturbed.
                print(f"  [OOM] step {step}: {_oom_e} — zero_grad + empty_cache + skipping step", flush=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                _emergency_save_state['step'] = step
                continue
            # EMA
            if ema_state is not None:
                d = args.ema_decay
                with torch.no_grad():
                    for nm, m in zip(keys, mods):
                        for pname, p in m.named_parameters():
                            if p.requires_grad:
                                ema_state[f"{nm}.{pname}"].mul_(d).add_(p.detach(), alpha=1 - d)
            # ρ revival
            if args.revive:
                with torch.no_grad():
                    # Use the ACTUAL ρ/α value (after parametrization), not the raw param.
                    # For per-dim α (vector), check the mean — if mean is below threshold,
                    # the gate is effectively dead globally.
                    rL_val, rH_val = policy._rhos()
                    for nm, raw, cur in (('L', policy.rho_L_raw, rL_val.mean().item()),
                                         ('H', policy.rho_H_raw, rH_val.mean().item())):
                        if revive_cd[nm] > 0:
                            revive_cd[nm] -= 1; continue
                        revive_dead[nm] = revive_dead[nm] + 1 if cur < args.revive_thresh else 0
                        if revive_dead[nm] >= args.revive_patience:
                            revive_dead[nm] = 0; revive_cd[nm] = args.revive_cooldown
                            rt = revive_peak[nm]
                            if rt <= args.revive_thresh: continue
                            # Convert the target ρ/α value back to RAW space for the parametrization in use.
                            if policy.update_mode == 'damped' and policy.alpha_parametrization == 'sigmoid':
                                target_raw = math.log(rt / (1 - rt))      # logit
                            elif policy.weighting == 'linear':
                                target_raw = rt                            # raw IS slope
                            else:
                                target_raw = rt                            # clamp_direct: raw IS ρ
                            raw.data.fill_(target_raw); opt.state.pop(raw, None)
                            revive_peak[nm] = rt * args.revive_decay
                            print(f"  [revive] ρ_{nm} → {rt:.3f} (next {revive_peak[nm]:.3f}) @ step {step}", flush=True)
            wl += loss.item(); wn += 1
            for l in range(len(seq_lens)):
                wc += per[l]['mask_correct']; wa += per[l]['mask_total']
            if step % args.log_every == 0 or step == 1:
                print(f"  step {step:>5}/{args.steps}  loss={wl/wn:.3f}  acc={wc/max(wa,1)*100:.1f}%  mask={getattr(policy,'_last_mask_frac',0.0):.2f}  "
                      f"ρ_L={policy._rhos()[0].mean().item():.3f} ρ_H={policy._rhos()[1].mean().item():.3f}  "
                      f"[{time.perf_counter()-t0:.0f}s]", flush=True)
                wl = wc = wa = wn = 0
            if step % args.ckpt_every == 0:
                save_ckpt(args, mods, keys, opt, ema_state, step)
            # OOD-recovery probe + objective-aligned early-stop
            if _ood is not None and step % args.ood_probe_every == 0:
                ratio = _ood_ratio()
                improved = ratio < _ood['best'] - args.ood_min_delta
                tag = ''
                if ratio < _ood['best']:
                    _ood['best'] = ratio
                    _orig = args.ckpt_path; args.ckpt_path = _ood['path']
                    save_ckpt(args, mods, keys, opt, ema_state, step)
                    args.ckpt_path = _orig; tag = ' (best, saved)'
                _ood['noimp'] = 0 if improved else _ood['noimp'] + 1
                print(f"  [ood-probe] step {step}: sim→real ratio={ratio:.3f}  "
                      f"best={_ood['best']:.3f}  noimp={_ood['noimp']}{tag}", flush=True)
                if args.ood_early_stop_patience > 0 and _ood['noimp'] >= args.ood_early_stop_patience:
                    print(f"  [early-stop] OOD ratio plateaued ({_ood['noimp']} probes w/o "
                          f"≥{args.ood_min_delta} gain) — stopping at step {step}", flush=True)
                    _early_stop = True; break
            # ── val-acc probe on disjoint val_eps (early-stop on true generalization) ──
            if _val_loader is not None and step % args.val_probe_every == 0:
                vacc, vloss, vgen, vsingle = _probe(_val_loader, gen=True)
                tacc, tloss = _probe(_train_loader) if _train_loader is not None else (float('nan'), float('nan'))
                improved = vloss < _val_best['best'] - args.val_min_delta   # select on val LOSS (lower=better)
                tag = ''
                if vloss < _val_best['best']:
                    _val_best['best'] = vloss
                    _orig = args.ckpt_path; args.ckpt_path = _val_best['path']
                    save_ckpt(args, mods, keys, opt, ema_state, step)
                    args.ckpt_path = _orig; tag = ' (best, saved)'
                _val_best['noimp'] = 0 if improved else _val_best['noimp'] + 1
                print(f"  [val-probe] step {step}: train_loss={tloss:.4f}/acc={tacc*100:.2f}%  "
                      f"val_loss={vloss:.4f}/acc={vacc*100:.2f}%  gap={vloss-tloss:.3f}/{(tacc-vacc)*100:.1f}pt  "
                      f"GEN(maskgit)={vgen*100:.2f}%  gen(single)={vsingle*100:.2f}%  "
                      f"best_loss={_val_best['best']:.4f}  noimp={_val_best['noimp']}{tag}", flush=True)
                if args.val_early_stop_patience > 0 and _val_best['noimp'] >= args.val_early_stop_patience:
                    print(f"  [early-stop] val_loss plateaued ({_val_best['noimp']} probes w/o "
                          f"≥{args.val_min_delta:.3f} loss drop) — stopping at step {step}", flush=True)
                    _early_stop = True; break
            # ── fixed-point / Lipschitz probe (damped mode only) ──
            if (args.fp_probe_every > 0 and _val_loader is not None
                    and getattr(policy, 'update_mode', None) == 'damped'
                    and step % args.fp_probe_every == 0):
                was_t = [m.training for m in mods]
                for m in mods: m.eval()
                try:
                    frames, states, actions, prevs, tasks, emb = next(iter(_val_loader))  # fresh batch
                    nfp = min(16, frames.shape[0])
                    frames = frames[:nfp].to(dev); states = states[:nfp].to(dev)
                    actions = actions[:nfp].to(dev); prevs = prevs[:nfp].to(dev)
                    tasks = tasks[:nfp]; emb = emb[:nfp]
                    with torch.no_grad():
                        vfp = encode_modalities(frames, tasks, emb, mod_drop=0.0)
                        sfp = encode_state(states, emb, noisy=False)
                        gt = encode_targets(actions, prevs, emb)
                    eid_t = torch.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb],
                                         dtype=torch.long, device=dev)
                    T_l = gt[0].shape[1]
                    mk = torch.ones(nfp, T_l, dtype=torch.bool, device=dev)   # all-masked (deploy condition)
                    fpd = policy.fixed_point_diagnostics(
                        vfp, sfp, indices_list=gt, mask_list=[mk], emb_id=eid_t)
                    r = fpd['resid']; pe = fpd['resid_pe']; zn = fpd['z_norm']
                    print(f"  [fp-probe] step {step}: σ_g={fpd['sigma_g']:.3f} "
                          f"contraction={fpd['contraction']:.3f} Lip(T)≤{fpd['lip_T_bound']:.3f} "
                          f"α={fpd['alpha']:.3f}  resid {r[0]:.2e}→{r[-1]:.2e} "
                          f"{'CONVERGING' if r[-1] < r[0] else 'DIVERGING'}  "
                          f"conv={fpd['frac_converged']*100:.0f}% "
                          f"per-ex resid(med/p90/max)={pe['median']:.1e}/{pe['p90']:.1e}/{pe['max']:.1e} "
                          f"‖z*‖(min/med/max)={zn['min']:.1f}/{zn['median']:.1f}/{zn['max']:.1f}  "
                          f"LS={[round(x,3) for x in fpd['layerscale']]}", flush=True)
                except Exception as _e:
                    print(f"  [fp-probe] step {step}: FAILED {type(_e).__name__}: {_e}", flush=True)
                finally:
                    for m, w in zip(mods, was_t):
                        if w: m.train()
            # Periodic CUDA allocator clear + memory watchdog.
            if torch.cuda.is_available() and step % CACHE_CLEAR_EVERY == 0:
                free, total = torch.cuda.mem_get_info()
                used_frac = 1.0 - free / total
                if used_frac >= OOM_FRACTION_LIMIT:
                    print(f"  [watchdog] step {step}: GPU at {used_frac*100:.0f}% — forcing empty_cache", flush=True)
                torch.cuda.empty_cache()
            _emergency_save_state['step'] = step   # keep atexit handler current
    save_ckpt(args, mods, keys, opt, ema_state, step)
    print(f"saved {args.ckpt_path} @ step {step}", flush=True)


def save_ckpt(args, mods, keys, opt, ema_state, step):
    os.makedirs(os.path.dirname(args.ckpt_path) or '.', exist_ok=True)
    tmp = args.ckpt_path + '.tmp'
    out = {k: m.state_dict() for k, m in zip(keys, mods)}
    out.update({'step': step, 'opt': opt.state_dict(), 'args': vars(args)})
    if ema_state is not None:
        out['ema_params'] = {k: v.detach().cpu() for k, v in ema_state.items()}
        out['ema_decay'] = args.ema_decay
    torch.save(out, tmp); os.replace(tmp, args.ckpt_path)


if __name__ == '__main__':
    main()
