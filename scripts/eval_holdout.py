#!/usr/bin/env python3
"""TRUE disjoint real held-out eval for the widowx policy.

The only provably-unseen real bridge data: episodes of length EXACTLY 32.
LeRobot's chunker skips episodes with max_start = length-(16+16) <= 0, so the
2051 length==32 episodes were never turned into a training chunk — never seen by
training, EMA, or the sentinel (whose "val" was drawn from the SAME chunked pool
that overlapped training). Each length==32 episode has one valid window at
start=16 (lookback frames 0-15, predicted chunk frames 16-31) — identical
structure to training, so the masked-CE accuracy is directly comparable.

Reuses the sentinel's policy build + masked-CE eval, fed these disjoint windows.
Run on multiple ckpts to compare (v3-best vs CNN-finetuned, raw vs EMA).

Run: .venv/bin/python scripts/eval_holdout.py --ckpt-path data/ckpts/XXX.pt [--n 256] [--ema] [--robot widowx]
"""
import os, sys, json, glob, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import importlib.util
# Import the sentinel module (sets CPU mode + provides build_policy_from_ckpt / run_eval).
_spec = importlib.util.spec_from_file_location('sentinel', os.path.join(THIS, 'early_stop_sentinel.py'))
sent = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sent)   # forces CUDA_VISIBLE_DEVICES=''
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from babygroot_strm import ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset, EMBODIMENTS, EMBODIMENT_ID
from babygroot_strm.perimg_norm import normalize_image
from PIL import Image
import math as _math


def episode_ids_in_range(bridge_dir, lo, hi):
    ids = []
    with open(os.path.join(bridge_dir, 'meta', 'episodes.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            L = r.get('length') or r.get('num_frames') or 0
            if lo <= L <= hi: ids.append(r['episode_index'])
    return ids


def build_holdout_context(ckpt_path, robot, n, seed, ctrl_range=None, start=16, eps_list=None):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ck['args']
    c = torch.load(f'data/ckpts/oxe_vqvae_{robot}.pt', map_location='cpu', weights_only=False)
    vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c.get('k', 128),
                        binary_last=c.get('binary_gripper', False))
    vae.load_state_dict(c['vae']); vae.eval()
    var_global = c['action_var_global'].view(1, 1, -1); adim = c['action_dim']
    _t5_path = (args.get('t5_cache') if isinstance(args, dict) else getattr(args, 't5_cache', None)) \
               or 'data/cache/t5_text_cache_paraphrased.pt'
    t5 = torch.load(_t5_path, map_location='cpu', weights_only=False)
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)

    import json as _json
    def _robot_of(d):
        try: return _json.load(open(os.path.join(d, 'meta', 'info.json'))).get('robot_type')
        except Exception: return None
    bridge_dir = next(d for d in sorted(glob.glob('data/oxe/*'))
                      if os.path.isfile(os.path.join(d, 'meta', 'info.json')) and _robot_of(d) == robot)
    sp = load_dataset_spec(bridge_dir, chunk_len=16, lookback=16, chunk_stride=16)
    if eps_list is not None:
        eps = list(eps_list)                                # explicit episode ids (e.g. UNSEEN no-video eps)
        label = f'{len(eps)} explicit episodes (UNSEEN)'
    elif ctrl_range is None:
        eps = episode_ids_in_range(bridge_dir, 32, 32)      # DISJOINT: length==32 (never chunked)
        label = 'length==32 disjoint'
    else:
        eps = episode_ids_in_range(bridge_dir, *ctrl_range)  # CONTROL: trained episodes, start=16 was used
        label = f'length∈[{ctrl_range[0]},{ctrl_range[1]}] CONTROL (trained)'
    sp.chunk_index = [(ep, start) for ep in eps]
    ds = MultiOXEDataset([sp], chunk_len=16, lookback=16)
    print(f"[holdout] {len(eps)} episodes ({label}) @start={start}", flush=True)

    rng = random.Random(seed); pool = list(range(len(ds))); rng.shuffle(pool)
    samples = []
    for idx in pool:
        if len(samples) >= n: break
        try:
            fr, st, ac, pv, tk, eid, di = ds[idx]
            if ac.shape[-1] != adim: continue
            pil = fr.convert('RGB').resize((args['img_size'], args['img_size']))
            ft = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).float() / 255.
            samples.append((ft, st, ac, pv, tk))
        except Exception: pass
    print(f"[holdout] collected {len(samples)} disjoint eval samples", flush=True)

    frames = torch.stack([s[0] for s in samples]); states = torch.stack([s[1] for s in samples])
    actions = torch.stack([s[2] for s in samples]); prevs = torch.stack([s[3] for s in samples])
    tasks = [s[4] for s in samples]; Bf = len(samples)

    T_text = args['max_text']; t5s = torch.zeros(9, Bf, T_text, 512)
    for b, tk in enumerate(tasks):
        e = t5['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float(); t = min(h.shape[1], T_text); L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]
    img_n = normalize_image(frames, img_var['var_global'])

    nT = actions.shape[1]; m_pv = prevs.mean(dim=1, keepdim=True)
    S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True); lam = nT / (S + nT * var_global)
    xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad():
        cd, _ = vae.encode_with_soft(xn, tau=0.1)
    gt_codes = [cd[0]]; T_l = gt_codes[0].shape[1]

    fixed_masks = []
    for s in range(sent.MASK_SEEDS_PER_BATCH):
        g = torch.Generator(); g.manual_seed(s * 31337 + 7)
        u = torch.rand(Bf, generator=g); r = torch.cos(_math.pi * u / 2).clamp(min=1.0 / T_l)
        noise = torch.rand(Bf, T_l, generator=g); m = noise < r.unsqueeze(1)
        m[torch.arange(Bf), noise.argmin(1)] = True
        fixed_masks.append(m)
    eid_t = torch.tensor([EMBODIMENT_ID.get(robot, len(EMBODIMENTS))] * Bf, dtype=torch.long)
    return dict(args=args, robot=robot, img_n=img_n, t5s=t5s, states=states, gt_codes=gt_codes,
                fixed_masks=fixed_masks, eid_t=eid_t, Bf=Bf), ck


def apply_ema(mods, ck):
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    ep = ck.get('ema_params', {}); nset = 0
    for nm, m in zip(keys, mods):
        sd = m.state_dict()
        for k in list(sd.keys()):
            full = f'{nm}.{k}'
            if full in ep: sd[k] = ep[full].to(sd[k].dtype); nset += 1
        m.load_state_dict(sd); m.eval()
    return nset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-path', required=True)
    ap.add_argument('--robot', default='widowx')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seed', type=int, default=98765)
    ap.add_argument('--ema', action='store_true', help="apply EMA weights before eval")
    ap.add_argument('--control', action='store_true',
                    help="HARNESS CHECK: eval on TRAINED episodes (length 40-60, start=16 window was in "
                         "training) instead of the disjoint length==32 set. Should reproduce ~train acc.")
    ap.add_argument('--len-min', type=int, default=None)
    ap.add_argument('--len-max', type=int, default=None)
    ap.add_argument('--start', type=int, default=16, help="chunk start frame; off-stride (e.g. 18) = unseen in-dist target")
    ap.add_argument('--eps-file', default=None, help="JSON list of episode ids to eval on (e.g. unseen no-video eps)")
    args = ap.parse_args()
    eps_list = json.load(open(args.eps_file)) if args.eps_file else None

    if args.len_min is not None:
        ctrl_range = (args.len_min, args.len_max if args.len_max is not None else args.len_min)
    elif args.control:
        ctrl_range = (40, 60)
    else:
        ctrl_range = None
    ctx, ck = build_holdout_context(args.ckpt_path, args.robot, args.n, args.seed, ctrl_range, args.start, eps_list)
    mods = sent.build_policy_from_ckpt(ck, ctx)
    tag = 'RAW'
    if args.ema:
        nset = apply_ema(mods, ck); tag = f'EMA({nset} params)'
    acc, loss = sent.run_eval(ctx, mods)
    step = ck.get('step', '?')
    print('=' * 60)
    print(f"HELD-OUT (length==32 disjoint)  ckpt={os.path.basename(args.ckpt_path)} step={step} [{tag}]")
    print(f"  masked-CE acc = {acc*100:.2f}%   val_loss = {loss:.4f}   (N={ctx['Bf']})")
    print('=' * 60)


if __name__ == '__main__':
    main()
