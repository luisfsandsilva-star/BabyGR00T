#!/usr/bin/env python3
"""MaskGIT iterative decode vs single-shot, on a trained Sudoku checkpoint.

Single-shot: one forward, argmax all blanks at once (what evaluate() does).
MaskGIT:     givens frozen from step 0; each round commit the most-confident blanks per the
             cosine schedule γ(r)=cos(π/2·r), re-feed, repeat. The decode the model is trained for.

  python -m scripts.eval_sudoku_maskgit --ckpt data/ckpt/sudoku_v2_best.pt
"""
import os, sys, math, argparse
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_sudoku_v2 import SudokuSTRM, make_tokens, evaluate, MASK_ID

@torch.no_grad()
def maskgit_decode(model, P, S, dev, n_steps=8, n_H=None, n_L=None, bs=128):
    model.eval(); solve = cellc = cellt = 0
    for i in range(0, len(P), bs):
        p, s = P[i:i+bs].to(dev), S[i:i+bs].to(dev); B = p.shape[0]
        given = (p != 0); blanks = ~given
        cur = torch.where(given, p - 1, torch.full_like(p, MASK_ID))   # 0-8 at givens, MASK at blanks
        masked = blanks.clone()
        n_blank = blanks.sum(1).float()
        for t in range(n_steps):
            vis = torch.where(masked, torch.full_like(cur, MASK_ID), cur)
            logits = model(vis, n_H=n_H, n_L=n_L)[-1]
            pconf, pred = logits.softmax(-1).max(-1)                    # confidence + argmax (0-8)
            r = (t + 1) / n_steps; gamma = math.cos(math.pi / 2 * r)
            if gamma < 1e-6: gamma = 0.0
            n_keep = torch.ceil(gamma * n_blank).long()                # blanks to KEEP masked this round
            conf_m = torch.where(masked, pconf, torch.full_like(pconf, float('inf')))  # committed sort last
            thr = conf_m.sort(1).values.gather(1, (n_keep - 1).clamp(0, 80).unsqueeze(1)).squeeze(1)
            keep_masked = masked & (conf_m <= thr.unsqueeze(1)) & (n_keep.unsqueeze(1) > 0)
            newly = masked & ~keep_masked
            cur = torch.where(newly, pred, cur); masked = keep_masked
        grid = cur + 1
        cellc += ((grid == s) & blanks).sum().item(); cellt += blanks.sum().item()
        solve += (grid == s).all(1).sum().item()
    return cellc / max(cellt, 1) * 100, solve / len(P) * 100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='data/ckpt/sudoku_v2_best.pt')
    ap.add_argument('--data', default='data/cache/sudoku.pt')
    args = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location=dev); a = ck['args']
    model = SudokuSTRM(a['dim'], a['depth'], a['heads'], a['kv_heads'], a['ff'], a['L'], a['H']).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    d = torch.load(args.data); Pte, Ste = d['Pte'], d['Ste']
    print(f"ckpt step {ck['step']}  ({len(Pte)} test puzzles, clues~{(Pte[0]!=0).sum().item()})", flush=True)

    ss_cell, ss_solve = evaluate(model, Pte, Ste, dev, n_H=a['H'], n_L=a['L'])
    print(f"single-shot (argmax all blanks): cell={ss_cell:.1f}%  solve={ss_solve:.1f}%", flush=True)
    print("MaskGIT iterative decode:", flush=True)
    for T in [4, 8, 16, 32]:
        c, sv = maskgit_decode(model, Pte, Ste, dev, n_steps=T, n_H=a['H'], n_L=a['L'])
        print(f"  n_steps={T:>2}: cell={c:.1f}%  solve={sv:.1f}%", flush=True)

if __name__ == '__main__':
    main()
