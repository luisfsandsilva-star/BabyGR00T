#!/usr/bin/env python3
"""Compute per-channel pixel variance over a sample of training frames.

Persisted result feeds the per-image RevIN normalization (var_global floor).
Usage: python -m scripts.compute_image_var_global [--n 2000] [--out data/cache/image_var_global.pt]
"""
import os, sys, glob, json, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch
from PIL import Image
from babygroot_strm.multi_oxe import load_dataset_spec, _decode_frame, _episode_paths
from babygroot_strm.perimg_norm import compute_image_var_global

ap = argparse.ArgumentParser()
ap.add_argument('--oxe-root', default='data/oxe')
ap.add_argument('--n', type=int, default=2000, help="# frames to average over")
ap.add_argument('--img-size', type=int, default=224)
ap.add_argument('--out', default='data/cache/image_var_global.pt')
args = ap.parse_args()

# discover datasets on disk
specs = []
for ds_dir in sorted(glob.glob(os.path.join(args.oxe_root, '*'))):
    info_p = os.path.join(ds_dir, 'meta', 'info.json')
    if not os.path.isfile(info_p): continue
    try: specs.append(load_dataset_spec(ds_dir, chunk_len=16, lookback=16))
    except Exception as e: print(f"  skip {ds_dir}: {e}")
print(f"  {len(specs)} datasets, {sum(len(sp.chunk_index) for sp in specs)} total chunks")

# sample frames across datasets uniformly
random.seed(0)
frames = []
for _ in range(args.n):
    sp = random.choice(specs)
    if not sp.chunk_index: continue
    ep, start = random.choice(sp.chunk_index)
    _, vid = _episode_paths(sp, ep)
    if not os.path.isfile(vid): continue
    try:
        pil = _decode_frame(vid, start).convert('RGB').resize((args.img_size, args.img_size))
        x = torch.from_numpy(np.asarray(pil)).permute(2,0,1).float() / 255.   # (3,H,W)
        frames.append(x)
    except Exception as e:
        pass

X = torch.stack(frames)
print(f"  collected {X.shape[0]} frames at {args.img_size}x{args.img_size}")
# per-image per-channel variance, averaged
v = X.var(dim=(-2,-1), unbiased=False).mean(dim=0)
print(f"  per-channel pixel var: R={v[0]:.5f} G={v[1]:.5f} B={v[2]:.5f}")
print(f"  per-channel std:       R={v[0].sqrt():.4f} G={v[1].sqrt():.4f} B={v[2].sqrt():.4f}")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({'var_global': v, 'n_frames': X.shape[0], 'img_size': args.img_size}, args.out)
print(f"  saved {args.out}")
