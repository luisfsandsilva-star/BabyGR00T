#!/usr/bin/env python3
"""Decisive test of our recursion on Sudoku — the lineage's home turf (HRM/TRM solve it via
iterative constraint propagation, where a single forward pass fails). Reuses the EXACT TRMNet g
(concat-context, L2-attn, LayerScale, damped Banach iteration, one-step/JFB grad) so this is a
faithful test of OUR recursion implementation — not a reimplementation.

The crux: at eval we sweep n_outer (recursion depth). If solve-rate JUMPS with more iterations,
the recursion works (and the robot task was simply non-iterative). If it stays flat, the recursion
implementation itself doesn't exploit depth.

  python -m scripts.train_sudoku --steps 20000 --clues 34 --dim 256 --depth 2 --H 8 --L 4
"""
import os, sys, time, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from babygroot_strm.policy import STRMPolicy

# ── Sudoku generation ──────────────────────────────────────────────────────────────────────
def _solve(grid, count_stop=2):
    """Backtracking solver; returns number of solutions (capped at count_stop)."""
    for i in range(81):
        if grid[i] == 0:
            r, c = divmod(i, 9); b = (r//3)*3 + c//3
            used = set()
            for k in range(9):
                used.add(grid[r*9+k]); used.add(grid[k*9+c])
                used.add(grid[(b//3*3+k//3)*9 + (b%3*3+k%3)])
            n = 0
            for v in range(1, 10):
                if v not in used:
                    grid[i] = v
                    n += _solve(grid, count_stop)
                    grid[i] = 0
                    if n >= count_stop: return n
            return n
    return 1

def _full_solution(rng):
    g = [0]*81
    def fill(i):
        if i == 81: return True
        r, c = divmod(i, 9); b = (r//3)*3 + c//3
        used = set()
        for k in range(9):
            used.add(g[r*9+k]); used.add(g[k*9+c]); used.add(g[(b//3*3+k//3)*9 + (b%3*3+k%3)])
        vs = [v for v in range(1,10) if v not in used]; rng.shuffle(vs)
        for v in vs:
            g[i] = v
            if fill(i+1): return True
            g[i] = 0
        return False
    fill(0); return g

def make_puzzle(rng, clues):
    """Full solution + dig holes to `clues` givens, keeping a UNIQUE solution."""
    sol = _full_solution(rng)
    puz = sol[:]
    order = list(range(81)); rng.shuffle(order)
    removed = 0
    for i in order:
        if 81 - removed <= clues: break
        save = puz[i]; puz[i] = 0
        g = puz[:]
        if _solve(g, 2) != 1: puz[i] = save           # keep unique
        else: removed += 1
    return puz, sol

def gen_dataset(n, clues, seed):
    rng = random.Random(seed); P = np.zeros((n,81), np.int64); S = np.zeros((n,81), np.int64)
    for j in range(n):
        p, s = make_puzzle(rng, clues); P[j] = p; S[j] = s
    return torch.from_numpy(P), torch.from_numpy(S)

def augment_batch(P, S, dev):
    """Validity-preserving Sudoku symmetry aug (makes data effectively infinite ⇒ forces the model
    to learn the RELATIONAL algorithm, not memorize). Per-example digit relabel + per-batch
    band/row/col/stack permutation + transpose. P:(B,81) vals 0-9 (0=blank), S:(B,81) vals 1-9."""
    B = P.shape[0]; P = P.to(dev).clone(); S = S.to(dev).clone()
    perms = torch.stack([torch.randperm(9, device=dev) + 1 for _ in range(B)])      # (B,9) relabel 1-9
    mp = torch.cat([torch.zeros(B, 1, dtype=perms.dtype, device=dev), perms], 1)    # (B,10), col0→0
    P = torch.gather(mp, 1, P.long()); S = torch.gather(mp, 1, S.long())
    def rowperm():
        out = []
        for b in torch.randperm(3).tolist():
            for r in torch.randperm(3).tolist(): out.append(b*3 + r)
        return out
    rp, cp = rowperm(), rowperm()
    Pg = P.view(B,9,9)[:, rp][:, :, cp]; Sg = S.view(B,9,9)[:, rp][:, :, cp]
    if torch.rand(()) < 0.5: Pg = Pg.transpose(1,2); Sg = Sg.transpose(1,2)
    return Pg.reshape(B,81), Sg.reshape(B,81)

# ── Model: SudokuTRM reusing the exact TRMNet g ────────────────────────────────────────────
class SudokuSTRM(nn.Module):
    """Thin wrapper around the ACTUAL STRMPolicy — NO architecture changes, only Sudoku I/O.
    Sudoku IS the MaskGIT setup: given cells = visible (unmasked) tokens, blanks = masked positions
    to predict; target = the solution. seq_lens=[81], k=9 (digits→classes 0-8). The vision/state KV
    is a single learned context token (no vision); reasoning flows through STRMPolicy's self-attention
    over the 81 cells + the puzzle embedding (_y_embed). forward returns per-H-cycle logits ⇒ the model's
    own DEEP SUPERVISION, with its 1-step gradient, L×H loops, y/z latents — all intact."""
    def __init__(self, dim=512, depth=4, heads=8, kv_heads=2, ff=2048, L=4, H=8,
                 one_step=True, output_scalenorm=False):
        super().__init__()
        self.dim = dim
        self.policy = STRMPolicy(seq_lens=[81], k_codebook=9, dim=dim, heads=heads, kv_heads=kv_heads,
                                 ff_hidden=ff, depth=depth, L_inner=L, H_outer=H, state_dim=dim,
                                 max_prefix=8, update_mode='damped', alpha_parametrization='sigmoid',
                                 layerscale_init=0.1, one_step_grad=one_step, output_scalenorm=output_scalenorm)
        self.ctx = nn.Parameter(torch.zeros(1, 1, dim))                 # minimal learned "vision" context
    def forward(self, sol0, blank_mask, n_outer=None, n_inner=None):
        B = sol0.shape[0]; dev = sol0.device
        vis = self.ctx.expand(B, -1, -1)                               # (B,1,dim)
        state = torch.zeros(B, self.dim, device=dev)
        # indices_list reveals GIVENS at unmasked positions (blanks get mask_idx inside _y_embed)
        return self.policy.forward([sol0], vis, state, mask_list=[blank_mask],
                                   n_outer=n_outer, n_inner=n_inner)     # list over H cycles, each [(B,81,9)]

# ── Train / eval ───────────────────────────────────────────────────────────────────────────
def evaluate(model, P, S, dev, n_outer=None):
    model.eval()
    with torch.no_grad():
        cell=solve=tot=0
        for i in range(0, len(P), 128):
            p=P[i:i+128].to(dev); s=S[i:i+128].to(dev)
            blank=(p==0); sol0=(s-1).long()                            # 0-8; givens revealed, blanks masked
            logits = model(sol0, blank, n_outer=n_outer)[-1][0]        # last cycle, level 0: (B,81,9)
            pred=logits.argmax(-1)+1
            full=torch.where(blank, pred, p)
            cell += ((pred==s) & blank).sum().item(); tot += blank.sum().item()
            solve += (full==s).all(dim=1).sum().item()
    model.train()
    return cell/max(tot,1)*100, solve/len(P)*100

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--clues', type=int, default=34); ap.add_argument('--n-train', type=int, default=20000)
    ap.add_argument('--n-test', type=int, default=1000); ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--kv-heads', type=int, default=2); ap.add_argument('--ff', type=int, default=1024)
    ap.add_argument('--H', type=int, default=8); ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--no-one-step', action='store_true')
    ap.add_argument('--cur-extra', type=int, default=25, help='curriculum: extra givens revealed early')
    ap.add_argument('--data', default='data/cache/sudoku.pt'); ap.add_argument('--seed', type=int, default=0)
    args=ap.parse_args()
    dev='cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(args.data):
        d=torch.load(args.data); Ptr,Str,Pte,Ste=d['Ptr'],d['Str'],d['Pte'],d['Ste']
        print(f"loaded {args.data}: {len(Ptr)} train / {len(Pte)} test, clues~{(Ptr[0]!=0).sum().item()}", flush=True)
    else:
        print(f"generating {args.n_train}+{args.n_test} puzzles (clues={args.clues})...", flush=True)
        t=time.time(); Ptr,Str=gen_dataset(args.n_train,args.clues,args.seed)
        Pte,Ste=gen_dataset(args.n_test,args.clues,args.seed+99999)
        os.makedirs(os.path.dirname(args.data),exist_ok=True)
        torch.save(dict(Ptr=Ptr,Str=Str,Pte=Pte,Ste=Ste),args.data)
        print(f"  generated in {time.time()-t:.0f}s → {args.data}", flush=True)

    model=SudokuSTRM(args.dim,args.depth,args.heads,args.kv_heads,args.ff,args.L,args.H,
                     one_step=not args.no_one_step).to(dev)
    print(f"SudokuSTRM (wraps STRMPolicy) params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"(dim{args.dim}/d{args.depth}/H{args.H}/L{args.L}, one_step={not args.no_one_step})", flush=True)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    import math as _math
    _warm = max(200, int(0.03 * args.steps))                  # cosine warmup + decay (match the policy)
    def _lr_at(s):
        if s < _warm: return args.lr * s / _warm
        prog = (s - _warm) / max(args.steps - _warm, 1)
        return args.lr * 0.1 + 0.5 * (args.lr - args.lr * 0.1) * (1 + _math.cos(_math.pi * prog))
    rng=np.random.default_rng(args.seed)
    t0=time.time()
    for step in range(1, args.steps+1):
        for g in opt.param_groups: g['lr'] = _lr_at(step)
        idx=rng.integers(0,len(Ptr),args.batch)
        p, s = augment_batch(Ptr[idx], Str[idx], dev)            # p 0-9 (0=blank), s 1-9; aug ⇒ no memorization
        # DIFFICULTY CURRICULUM: reveal `extra` extra givens early (easy→hard over first half) to bootstrap
        extra = int(round(args.cur_extra * max(0.0, 1 - step/(args.steps*0.5))))
        if extra > 0:
            rsc = torch.rand(p.shape, device=dev); rsc[p!=0] = -1.0      # only blanks eligible
            kth = rsc.topk(min(extra,81), dim=1).values[:, -1:]
            reveal = (rsc >= kth) & (p==0)
            p = torch.where(reveal, s, p)                                # reveal solution at those cells
        blank = (p==0); sol0 = (s-1).long()                      # classes 0-8
        all_logits = model(sol0, blank)                          # list over H cycles ⇒ DEEP SUPERVISION
        tgt = sol0[blank]                                        # loss on BLANKS only (force solving)
        loss = sum(F.cross_entropy(lg[0][blank], tgt) for lg in all_logits) / len(all_logits)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%500==0 or step==1:
            ca,sv=evaluate(model,Pte,Ste,dev)
            print(f"step {step:>6}/{args.steps} loss={loss.item():.3f}  test cell_acc={ca:.1f}%  solve={sv:.1f}%  [{time.time()-t0:.0f}s]", flush=True)
    # ── the decisive sweep: does solve-rate rise with recursion depth H? ──
    print("\n=== iteration sweep (test) — does MORE recursion help? ===", flush=True)
    for H in [1,2,4,8,16,32]:
        ca,sv=evaluate(model,Pte,Ste,dev,n_outer=H)
        print(f"  H_outer={H:>2}: cell_acc={ca:.1f}%  solve={sv:.1f}%", flush=True)

if __name__=='__main__':
    main()
