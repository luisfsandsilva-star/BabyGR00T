#!/usr/bin/env python3
"""Combined T5 cache for ALL OXE datasets on disk.

Walks every dataset's meta/tasks*.jsonl to collect the union of unique
instruction strings, then encodes them all with flan-t5-small (matches the
existing single-dataset cache).

Usage: python -m scripts.cache_t5_oxe [--oxe-root data/oxe] [--out data/cache/t5_text_cache.pt]
"""
import os, sys, glob, json, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import torch
from babygroot_strm.text_encoder import T5TextEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oxe-root', default='data/oxe')
    ap.add_argument('--model-id', default='google/flan-t5-small')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default='data/cache/t5_text_cache.pt')
    args = ap.parse_args()

    # collect unique instructions
    strings = set()
    per_dataset = []
    for ds_dir in sorted(glob.glob(os.path.join(args.oxe_root, '*'))):
        info_p = os.path.join(ds_dir, 'meta', 'info.json')
        if not os.path.isfile(info_p): continue
        name = os.path.basename(ds_dir)
        ds_strs = set()
        for p in glob.glob(os.path.join(ds_dir, 'meta', 'tasks*.jsonl')):
            for line in open(p):
                r = json.loads(line)
                s = r.get('task') or r.get('task_str') or ''
                if s: ds_strs.add(s)
        per_dataset.append((name, len(ds_strs)))
        strings |= ds_strs
    strings = sorted(strings)
    print(f"  {len(strings)} unique instructions across {len(per_dataset)} datasets")
    for n, c in per_dataset: print(f"    {n}: {c} unique")
    if not strings:
        print("  no instructions found — nothing to cache."); return

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    te = T5TextEncoder(args.model_id, device=dev)
    print(f"  T5 {args.model_id}: dim={te.dim}")

    cache = {}
    for i in range(0, len(strings), args.batch):
        chunk = strings[i:i + args.batch]
        stacked, mask = te(chunk, all_layers=True)
        for j, s in enumerate(chunk):
            ln = int(mask[j].sum().item())
            cache[s] = {'hidden': stacked[:, j, :ln, :].bfloat16().cpu().contiguous(),
                        'mask':   mask[j, :ln].cpu().contiguous()}
        if (i // args.batch) % 5 == 0:
            print(f"    encoded {min(i+args.batch, len(strings))}/{len(strings)}", flush=True)

    n_layers = next(iter(cache.values()))['hidden'].shape[0]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'model_id': args.model_id, 'dim': te.dim, 'n_layers': n_layers,
                'embeddings': cache}, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"\nSaved {len(cache)} strings × {n_layers} layers → {args.out}  ({sz:.0f} MB)")


if __name__ == '__main__':
    main()
