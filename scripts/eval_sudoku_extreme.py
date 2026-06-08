#!/usr/bin/env python3
"""Eval a trained Sudoku checkpoint on Sudoku-Extreme (sapientinc/sudoku-extreme) test set —
the hard benchmark TRM reports ~87% on. Zero-shot if the ckpt was trained on easier puzzles.

  python -m scripts.eval_sudoku_extreme --ckpt data/ckpt/sudoku_v2_best.pt --n 5000
"""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_sudoku_v2 import SudokuSTRM, evaluate, MASK_ID
from scripts.eval_sudoku_maskgit import maskgit_decode

def load_extreme(path, n=None, seed=0):
    P, S = [], []
    with open(path) as fh:
        next(fh)                                                   # header
        rows = fh.readlines()
    if n is not None and n < len(rows):
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in rng.choice(len(rows), n, replace=False)]
    for line in rows:
        parts = line.rstrip("\n").split(",")
        q, a = parts[1], parts[2]
        if len(q) != 81 or len(a) != 81:
            continue
        P.append([0 if c == "." else int(c) for c in q])
        S.append([int(c) for c in a])
    return torch.tensor(P, dtype=torch.long), torch.tensor(S, dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='data/ckpt/sudoku_v2_best.pt')
    ap.add_argument('--csv', default='data/sudoku_extreme/test.csv')
    ap.add_argument('--n', type=int, default=5000)
    args = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Pte, Ste = load_extreme(args.csv, args.n)
    clues = (Pte != 0).sum(1).float()
    print(f"Sudoku-Extreme test: {len(Pte)} puzzles, clues mean={clues.mean():.1f} min={int(clues.min())} max={int(clues.max())} "
          f"(blanks mean={81-clues.mean():.1f})", flush=True)

    ck = torch.load(args.ckpt, map_location=dev); a = ck['args']
    model = SudokuSTRM(a['dim'], a['depth'], a['heads'], a['kv_heads'], a['ff'], a['L'], a['H']).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    print(f"ckpt step {ck['step']} (trained on clues~30 — this is ZERO-SHOT transfer to Extreme)", flush=True)

    ss_cell, ss_solve = evaluate(model, Pte, Ste, dev, n_H=a['H'], n_L=a['L'])
    print(f"single-shot: cell={ss_cell:.1f}%  solve={ss_solve:.2f}%", flush=True)
    for T in [8, 16, 32]:
        c, sv = maskgit_decode(model, Pte, Ste, dev, n_steps=T, n_H=a['H'], n_L=a['L'])
        print(f"MaskGIT n_steps={T:>2}: cell={c:.1f}%  solve={sv:.2f}%", flush=True)

if __name__ == '__main__':
    main()
