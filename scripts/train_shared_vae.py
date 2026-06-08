#!/usr/bin/env python3
"""Train ONE shared conditional VQ-VAE on actions from ALL OXE embodiments.

Approach: shared encoder + shared codebook + decoder conditioned on embodiment-id.
Each chunk is precision-normalized using its own embodiment's `var_global`
(same form as training; per-emb prior) so the encoder sees inputs in a
broadly-comparable normalized space.

Usage: python -m scripts.train_shared_vae [--steps 8000] [--out data/ckpts/oxe_shared_vae.pt]
"""
import os, sys, time, glob, json, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pyarrow.parquet as pq
from babygroot_strm import RevIN, VQ1d_EMA
from babygroot_strm.fsq import FSQ1d
from babygroot_strm.cond_vae import CondActionVQVAE1d
from babygroot_strm.multi_oxe import EMBODIMENT_ID, EMBODIMENTS


def gather_all_action_pairs(oxe_root: str, chunk_len: int = 16, max_per_dataset: int = 50000):
    """Collect (cur, prev, emb_id) tuples across all datasets, cap each dataset to be fair."""
    cur_all, prev_all, emb_all = [], [], []
    per_emb_count = {}
    matched = []
    for ds_dir in sorted(glob.glob(os.path.join(oxe_root, '*'))):
        info_p = os.path.join(ds_dir, 'meta', 'info.json')
        if not os.path.isfile(info_p): continue
        info = json.load(open(info_p))
        emb = info.get('robot_type', 'unknown')
        eid = EMBODIMENT_ID.get(emb, len(EMBODIMENTS))
        cur_list, prev_list = [], []
        for pq_path in sorted(glob.glob(os.path.join(ds_dir, 'data', 'chunk-*', '*.parquet'))):
            try:
                t = pq.read_table(pq_path, columns=['action'])
                ac_flat = np.stack(t.column('action').to_pylist())
                T, A = ac_flat.shape
                if T < 2 * chunk_len: continue
                n_full = T // chunk_len
                ac_chunks = torch.from_numpy(ac_flat[:n_full*chunk_len].reshape(n_full, chunk_len, A)).float()
                cur_list.append(ac_chunks[1:]); prev_list.append(ac_chunks[:-1])
                if sum(c.shape[0] for c in cur_list) > max_per_dataset: break
            except: pass
        if cur_list:
            c = torch.cat(cur_list, dim=0)[:max_per_dataset]
            p = torch.cat(prev_list, dim=0)[:max_per_dataset]
            cur_all.append(c); prev_all.append(p)
            emb_all.append(torch.full((c.shape[0],), eid, dtype=torch.long))
            per_emb_count[emb] = per_emb_count.get(emb, 0) + c.shape[0]
            matched.append((os.path.basename(ds_dir), emb, c.shape[0]))
    return torch.cat(cur_all, 0), torch.cat(prev_all, 0), torch.cat(emb_all, 0), matched, per_emb_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oxe-root', default='data/oxe')
    ap.add_argument('--out', default='data/ckpts/oxe_shared_vae.pt')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--max-per-dataset', type=int, default=50000,
                    help="cap each dataset to this many pairs (prevents single big dataset domination)")
    ap.add_argument('--k', type=int, default=512, help="codebook size (shared)")
    ap.add_argument('--quantizer', choices=['vq', 'fsq'], default='vq')
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--balanced-sampling', action='store_true', default=True,
                    help="Use weighted random sampler so each embodiment is equally represented per batch "
                         "(prevents data-rich embodiments from dominating gradient and starving small ones).")
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"gathering actions across all OXE datasets (cap {args.max_per_dataset}/ds)...")
    cur, prev, emb_id, matched, per_emb_count = gather_all_action_pairs(
        args.oxe_root, max_per_dataset=args.max_per_dataset)
    print(f"  total pairs: {cur.shape[0]}")
    print(f"  per-embodiment counts:")
    for emb, c in sorted(per_emb_count.items(), key=lambda x: -x[1]):
        print(f"    {emb:<14s}: {c:>8,}")
    action_dim = cur.shape[-1]
    print(f"  action_dim={action_dim}, chunk_len={cur.shape[1]}")

    # per-embodiment var_global for the precision norm — keep them separate
    var_globals = {}
    for emb in per_emb_count:
        eid = EMBODIMENT_ID.get(emb, len(EMBODIMENTS))
        mask = emb_id == eid
        var_globals[eid] = cur[mask].var(dim=(0, 1)).clamp(min=1e-8).to(dev).view(1, 1, -1)
        print(f"  [{emb}] var_global per-dim: {[round(x,4) for x in var_globals[eid].sqrt().flatten().tolist()]}")

    vq_cls = FSQ1d if args.quantizer == 'fsq' else VQ1d_EMA
    n_emb = max(EMBODIMENT_ID.values()) + 2     # +1 unknown bucket
    vae = CondActionVQVAE1d(action_dim=action_dim, n_embodiments=n_emb,
                            k=args.k, vq_cls=vq_cls).to(dev)
    n_p = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"  CondActionVQVAE1d params: {n_p:.2f}M  bottleneck T={vae.bottleneck_T} × D={vae.d*4}  K={args.k}  quant={args.quantizer}")

    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-4)
    # weighted sampler: give each embodiment equal expected count per batch (otherwise
    # data-rich embodiments dominate and starve the small ones — observed widowx/ur5 collapse).
    if args.balanced_sampling:
        unique_eids, counts = torch.unique(emb_id, return_counts=True)
        weight_per_eid = {int(e): 1.0 / int(c) for e, c in zip(unique_eids, counts)}
        sample_weights = torch.tensor([weight_per_eid[int(e)] for e in emb_id], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(emb_id), replacement=True)
        loader = DataLoader(TensorDataset(cur, prev, emb_id), batch_size=args.batch_size,
                            sampler=sampler, drop_last=True, num_workers=0)
        print(f"  balanced sampler: each embodiment ~1/{len(unique_eids)} of each batch")
    else:
        loader = DataLoader(TensorDataset(cur, prev, emb_id), batch_size=args.batch_size,
                            shuffle=True, drop_last=True, num_workers=0)
    print(f"  {len(loader)} batches/epoch; total {len(loader) * args.batch_size:,} per epoch")

    vae.train()
    step = 0; t0 = time.perf_counter(); wr = wq = wn = 0
    code_hits = torch.zeros(args.k, dtype=torch.long, device=dev)
    while step < args.steps:
        for ac, pv, eid in loader:
            if step >= args.steps: break
            ac = ac.to(dev, non_blocking=True); pv = pv.to(dev, non_blocking=True)
            eid = eid.to(dev, non_blocking=True)
            n = ac.shape[1]
            m = pv.mean(dim=1, keepdim=True)
            S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
            # per-emb precision norm
            vg = torch.stack([var_globals[int(e)].squeeze(0).squeeze(0) for e in eid]).unsqueeze(1)  # (B,1,A)
            lam = n / (S + n * vg)
            x = ((ac - m) * lam.sqrt()).transpose(1, 2)
            embs, vql, idxs = vae.encode(x, eid)
            recon = vae.decode(embs, eid)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + vql
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            wr += recon_loss.item(); wq += float(vql); wn += 1
            # track codebook usage live
            for i in idxs: code_hits.scatter_add_(0, i.flatten(), torch.ones_like(i.flatten()))
            step += 1
            if step % args.log_every == 0 or step == 1:
                used = (code_hits > 0).sum().item()
                elapsed = time.perf_counter() - t0
                print(f"  step {step:>6}/{args.steps}  recon={wr/wn:.4f}  vq={wq/wn:.4f}  "
                      f"codebook used={used}/{args.k} ({100*used/args.k:.0f}%)  [{elapsed:.0f}s]", flush=True)
                wr = wq = wn = 0
                code_hits.zero_()                                  # reset per window

    # save with same schema the rest expects, plus embodiment info
    torch.save({'kind': 'cond_vqvae', 'quantizer': args.quantizer,
                'vae': vae.state_dict(),
                'action_dim': action_dim,
                'action_var_globals': {int(k): v.squeeze().cpu() for k, v in var_globals.items()},
                'n_embodiments': n_emb,
                'seq_lens': vae.seq_lens, 'k': args.k,
                'per_emb_count': per_emb_count,
                'source_datasets': matched}, args.out)
    print(f"\nSaved {args.out}")


if __name__ == '__main__':
    main()
