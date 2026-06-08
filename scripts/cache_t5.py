#!/usr/bin/env python3
"""Cache frozen-T5 per-layer hidden states for every unique task string.

Stores ALL encoder layers (embeddings + each block) so the (trained, reused)
LayerAggregator can combine them at train time — exactly like the InternVL
layer cache did, but for T5 text. Frozen T5 → encode once here, never in the
training loop.

Out: data/cache/t5_text_cache.pt = {
  'model_id', 'dim', 'n_layers',
  'embeddings': { task_str: {'hidden': (n_layers, T, dim) fp16, 'mask': (T,)} }
}
"""
import os, sys, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import torch
from babygroot_strm import load_lerobot_episodes
from babygroot_strm.text_encoder import T5TextEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', default='google/flan-t5-base')
    ap.add_argument('--oxe-dataset-id', default='IPEC-COMMUNITY/bridge_orig_lerobot')
    ap.add_argument('--oxe-camera', default='observation.images.image_0')
    ap.add_argument('--n-eps-cap', type=int, default=1800)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default='data/cache/t5_text_cache.pt')
    args = ap.parse_args()

    print(f"Loading episodes (cap={args.n_eps_cap}) to collect task strings ...")
    eps = load_lerobot_episodes(args.oxe_dataset_id, camera_key=args.oxe_camera,
                                load_video=False, n_episodes=args.n_eps_cap)
    strings = sorted({ep[3] for ep in eps})
    print(f"  {len(eps)} episodes → {len(strings)} unique task strings")

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    te = T5TextEncoder(args.model_id, device=dev)
    print(f"  T5 {args.model_id}: dim={te.dim}")

    cache = {}
    for i in range(0, len(strings), args.batch):
        chunk = strings[i:i + args.batch]
        stacked, mask = te(chunk, all_layers=True)        # (L, B, T, dim), (B, T)
        for j, s in enumerate(chunk):
            ln = int(mask[j].sum().item())
            cache[s] = {'hidden': stacked[:, j, :ln, :].bfloat16().cpu().contiguous(),
                        'mask':   mask[j, :ln].cpu().contiguous()}
        if (i // args.batch) % 5 == 0:
            print(f"  encoded {min(i+args.batch, len(strings))}/{len(strings)}", flush=True)

    n_layers = next(iter(cache.values()))['hidden'].shape[0]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'model_id': args.model_id, 'dim': te.dim, 'n_layers': n_layers,
                'embeddings': cache}, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"\nSaved {len(cache)} strings × {n_layers} layers → {args.out}  ({sz:.0f} MB)")


if __name__ == '__main__':
    main()
