#!/usr/bin/env python3
"""LARS update-size diagnostic — find which parameters are getting too-large
relative updates (||Δw|| / ||w||) per training step.

Standard sanity rule of thumb (LARS/Goyal/You):
  - ||Δw|| / ||w|| ≈ 1e-3 to 5e-3 per step ≈ "trust ratio works"
  - > 1e-2 per step = aggressive (model can survive but drifts fast)
  - > 5e-2 per step = blow-up regime (instability likely within hundreds of steps)
  - < 1e-5 per step = effectively frozen (under-learning)

We split by:
  - 2D matrix weights (LARS w/ NS): trust = (||W||+ε)/(||NS(upd)||+ε); Δw = lr·trust·NS(upd)
  - 1D scalar/vector weights (no LARS, no WD): Δw = lr · (grad + m·buf)
  - Embeddings: nn.Embedding.weight is 2D but semantically different; flag.

Loads the latest ckpt, runs one full forward+backward on real data, simulates
ONE optimizer step internally (without actually updating params), prints a
table sorted by ||Δw||/||W|| descending.
"""
import os, sys, glob, random
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm,
                             ActionVQVAE1d, VQ1d_EMA, MuSGD_LARS)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm import augment
from babygroot_strm.optimizer import _newton_schulz

CKPT = 'data/ckpts/oxe_policy_v14_widowx_v2.pt'   # current (post-divergence) ckpt
LR = 8.97e-3
MOMENTUM = 0.95
WEIGHT_DECAY = 2e-3
TRUST_EPS = 0.1
ROBOT = 'widowx'
BS = 64        # smaller batch for diag — same gradient stats, faster
NW = 4
torch.set_num_threads(4); random.seed(0); torch.manual_seed(0)


def main():
    print(f"[diag] loading ckpt {CKPT} ...", flush=True)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    args = ck['args']
    step = ck.get('step', '?')
    print(f"[diag]   step at save: {step}")

    DIM = args['dim']; HEADS = args['heads']; KV = args.get('kv_heads', HEADS)
    FF = args.get('ff_hidden')
    DEPTH = args['depth']; L_I = args['L_inner']; H_O = args['H_outer']
    N_EMB = len(EMBODIMENTS) + 1
    IMG = args['img_size']; MAX_TEXT = args['max_text']; PFX = args['n_emb_prefix']

    # ── build modules + load ckpt ──
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=args['cnn_expand'],
                       out_dim=args['cnn_out_dim'], norm=args['cnn_norm'], pos_emb=args['cnn_pe'],
                       img_size=IMG, dropout=0.0,
                       n_embodiments=N_EMB if args.get('cnn_film_by_emb') else 0)
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(args['cnn_out_dim'], DIM)
    text_proj = nn.Linear(512, DIM)
    kv_norm = ScaleNorm(DIM)
    se_keys = sorted({k.split('.')[0] for k in ck['state_encoders'].keys()})
    state_encoders = nn.ModuleDict({e: nn.Sequential(nn.Linear(8, DIM), nn.GELU(), nn.Linear(DIM, DIM))
                                     for e in se_keys})
    emb_id_emb = nn.Embedding(N_EMB, DIM * PFX)
    n_vis = (IMG // 32) ** 2
    max_prefix = n_vis + MAX_TEXT + 16 + PFX
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=DIM, heads=HEADS,
                        kv_heads=KV, ff_hidden=FF, depth=DEPTH, L_inner=L_I, H_outer=H_O,
                        state_dim=DIM, max_prefix=max_prefix,
                        weighting=args['weighting'], update_mode=args['update_mode'],
                        alpha_parametrization=args['alpha_parametrization'],
                        alpha_per_dim=args['alpha_per_dim'],
                        n_embodiments=N_EMB, per_emb_head=args['per_emb_head'], dropout=0.0,
                        g_input_noise=args.get('g_input_noise', 0.0))
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    # strict=False so we tolerate (a) missing keys (new params added since the ckpt)
    # and (b) unexpected keys (params removed since the ckpt — e.g. qk-norm biases
    # we just removed in policy.py). Mismatches are printed for verification.
    for k, m in zip(keys, mods):
        res = m.load_state_dict(ck[k], strict=False)
        if res.missing_keys: print(f"  [load {k}] missing: {res.missing_keys[:3]}{'...' if len(res.missing_keys)>3 else ''}")
        if res.unexpected_keys: print(f"  [load {k}] unexpected: {res.unexpected_keys[:3]}{'...' if len(res.unexpected_keys)>3 else ''}")
    for m in mods: m.cuda().train()

    # ── one real batch ──
    print(f"[diag] loading data...")
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=4)
            if sp.chunk_index and sp.robot == ROBOT: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    rng = random.Random(123); pool = list(range(len(ds))); rng.shuffle(pool)
    samples = []
    vae_c = torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt', map_location='cpu', weights_only=False)
    adim = vae_c['action_dim']
    for idx in pool:
        if len(samples) >= BS: break
        try:
            fr, st, ac, pv, tk, eid_, di = ds[idx]
            if ac.shape[-1] != adim or st.shape[-1] != 8: continue
            from PIL import Image
            pil = fr.convert('RGB').resize((IMG, IMG))
            ft = torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).float() / 255.
            samples.append((ft, st, ac, pv, tk))
        except Exception: pass
    Bf = len(samples)
    frames = torch.stack([s[0] for s in samples]).cuda()
    states = torch.stack([s[1] for s in samples]).cuda()
    actions = torch.stack([s[2] for s in samples]).cuda()
    prevs = torch.stack([s[3] for s in samples]).cuda()
    tasks = [s[4] for s in samples]
    print(f"[diag]   B={Bf} samples")

    vae = ActionVQVAE1d(action_dim=adim, vq_cls=VQ1d_EMA, k=vae_c.get('k', 128)).cuda().eval()
    vae.load_state_dict(vae_c['vae'])
    var_global = vae_c['action_var_global'].view(1, 1, -1).cuda()

    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    t5s = torch.zeros(9, Bf, MAX_TEXT, 512, device='cuda')
    for b, tk in enumerate(tasks):
        e = t5['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float().cuda()
        t = min(h.shape[1], MAX_TEXT); L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)
    img_n = normalize_image(frames, img_var['var_global'].cuda())

    eid_t = torch.tensor([EMBODIMENT_ID.get(ROBOT, len(EMBODIMENTS))] * Bf, dtype=torch.long).cuda()

    # ── forward + backward ──
    print(f"[diag] forward+backward...")
    nT = actions.shape[1]
    m_pv = prevs.mean(dim=1, keepdim=True)
    S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True)
    lam = nT / (S + nT * var_global)
    xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad():
        cd, _ = vae.encode_with_soft(xn, tau=0.1)
    gt = [cd[0]]

    with torch.autocast('cuda', dtype=torch.bfloat16):
        vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
        tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
        etok = emb_id_emb(eid_t).view(Bf, PFX, DIM)
        vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
        s_enc = state_encoders[ROBOT](states)
        loss, per, _ = policy.forward_loss(gt, vis, s_enc,
            n_inner=L_I, n_outer=H_O, h_max=H_O,
            mask_ratio_max=1.0, emb_id=eid_t,
            label_smoothing=0.10, mask_sampler='cosine')
    print(f"[diag]   loss = {loss.item():.4f}")
    loss.backward()

    # Simulate one MuSGD_LARS step (mirror optimizer.py logic; don't actually apply)
    # We don't have prior momentum buffer state (this is a fresh sim), so just use
    # grad alone for the "buf" — represents the FIRST step's update size which is
    # the most conservative (no momentum amplification yet). Add a second pass that
    # assumes steady-state momentum buffer = grad / (1-m) — pessimistic upper bound.
    rows = []
    for mname, m in zip(keys, mods):
        for pname, p in m.named_parameters():
            if p.grad is None: continue
            grad = p.grad.detach().float()
            w = p.detach().float()
            wn = w.norm().item()
            gn = grad.norm().item()
            full = f"{mname}.{pname}"
            dim = p.dim()
            shape = tuple(p.shape)
            numel = p.numel()
            # First-step update (no buf): upd = grad + m*0 = grad → conservative
            # Steady-state pessimistic: upd_ss = grad + m*(grad/(1-m)) = grad/(1-m)
            ss_factor = 1.0 / (1.0 - MOMENTUM)
            # NEW OPTIMIZER LOGIC: LARS trust applies to ALL params (1D + 2D + 4D).
            # NS orthogonalization is 2D-only.
            upd_first = grad
            upd_ss = grad * ss_factor
            if dim == 2:
                upd_first = _newton_schulz(upd_first.clone(), steps=5)
                upd_ss = _newton_schulz(upd_ss.clone(), steps=5)
                kind = '2D-LARS+NS'
            elif dim == 1:
                kind = '1D-LARS'
            elif dim == 4:
                kind = '4D-LARS'
            else:
                kind = f'{dim}D-LARS'
            un_first = upd_first.norm().item()
            un_ss = upd_ss.norm().item()
            trust_first = (wn + TRUST_EPS) / (un_first + TRUST_EPS)
            trust_ss = (wn + TRUST_EPS) / (un_ss + TRUST_EPS)
            dw_first = LR * trust_first * un_first
            dw_ss = LR * trust_ss * un_ss
            rel_first = dw_first / (wn + 1e-12)
            rel_ss = dw_ss / (wn + 1e-12)
            rows.append((full, kind, numel, shape, wn, gn, gn/(wn+1e-12),
                         trust_first, dw_first, rel_first, dw_ss, rel_ss))

    # ── print top problem layers ──
    rows.sort(key=lambda r: -r[-1])   # by rel_ss (steady-state relative update)
    print(f"\n{'='*120}")
    print(f"LARS UPDATE DIAGNOSTIC (LR={LR}, m={MOMENTUM}, ε={TRUST_EPS})")
    print(f"{'='*120}")
    print(f"\nTop 20 by STEADY-STATE ||Δw||/||W|| (momentum-saturated update fraction per step):\n")
    print(f"  {'name':<50s} {'kind':>9s} {'||W||':>9s} {'||g||':>9s} {'g/w':>8s} "
          f"{'trust':>7s} {'Δw/w (first)':>12s} {'Δw/w (ss)':>10s}")
    for full, kind, n, sh, wn, gn, gw, tr, dw1, r1, dws, rs in rows[:20]:
        flag = ''
        if rs > 5e-2: flag = ' ⚠ BLOWUP'
        elif rs > 1e-2: flag = ' aggressive'
        print(f"  {full:<50s} {kind:>9s} {wn:>9.2e} {gn:>9.2e} {gw:>8.1e} "
              f"{tr:>7.2f} {r1:>12.2e} {rs:>10.2e}{flag}")

    # Bottom (smallest updates — possibly frozen)
    print(f"\nBottom 10 (smallest ||Δw||/||W|| — possibly under-learning):\n")
    for full, kind, n, sh, wn, gn, gw, tr, dw1, r1, dws, rs in rows[-10:]:
        print(f"  {full:<50s} {kind:>9s} {wn:>9.2e} {gn:>9.2e} {gw:>8.1e} "
              f"{tr:>7.2f} {r1:>12.2e} {rs:>10.2e}")

    # Split 1D vs 2D summary
    rows_1d = [r for r in rows if r[1] == '1D-LARS']
    rows_2d = [r for r in rows if r[1] == '2D-LARS+NS']
    rows_4d = [r for r in rows if r[1] == '4D-LARS']
    print(f"\n{'─'*120}")
    print(f"SUMMARY: split by LARS treatment")
    print(f"{'─'*120}")
    def stats(rs, label):
        if not rs: return
        rels_first = sorted([r[9] for r in rs])
        rels_ss = sorted([r[11] for r in rs])
        print(f"  {label} (n={len(rs)}): "
              f"rel-Δw/w first-step median={rels_first[len(rels_first)//2]:.2e} "
              f"max={max(rels_first):.2e}")
        print(f"  {' '*len(label)}             ss-bound  median={rels_ss[len(rels_ss)//2]:.2e} "
              f"max={max(rels_ss):.2e}")
    stats(rows_1d, "1D-LARS params (biases, norms, ρ-gates)")
    stats(rows_2d, "2D-LARS+NS params (weight matrices)")
    stats(rows_4d, "4D-LARS params (conv weights)")

    print(f"\n{'─'*120}")
    print("INTERPRETATION:")
    print("  - LARS trust ratio targets ||Δw||/||W|| ≈ LR for 2D matrices (here ~{:.1e}).".format(LR))
    print("  - 1D params (no LARS) have raw update size = LR·||grad+m·buf|| — depends on grad scale.")
    print("  - If 1D max ||Δw||/||W|| ≫ 2D max → 1D params are destabilizing the training.")
    print("    Fix: reduce LR for 1D-group, or add LARS-equivalent scaling, or scale them per the")
    print("    largest 2D group's trust.")
    print("  - If a few 2D params have very LOW trust → their effective LR is much smaller than the")
    print("    base lr; they're under-learning relative to others.")


if __name__ == '__main__':
    main()
