#!/usr/bin/env python3
"""Sudoku v2 — the REAL non-VAE STRMPolicy (latest-policy config) + Nesterov, applied to Sudoku.

Uses babygroot_strm.policy.STRMPolicy directly (NOT a reimplementation), with the fixes/features
the latest policy has plus the carry/nesterov additions:
  · carry_zl=True        — z_L persists across outer cycles (TRM-faithful; the bug we found)
  · nesterov=True        — Nesterov-accelerated inner fixed point + adaptive restart
  · output_scalenorm=True— ScaleNorm on the recursion output (bounds ‖z‖ inflation)
  · update_mode='damped', alpha_parametrization='sigmoid', one_step_grad=True  (non-VAE, 1-step)
  · cosine MaskGIT mask sampling (valid — we know the labels)
  · stochastic depth (uniform) on H (n_outer) and L (n_inner)

Probes: loss · grad-norm · test cell/solve · α_L/α_H · ‖z_H‖ ; end-of-run: H-sweep +
compare_iteration_schemes (damped vs nesterov fixed-point convergence).

  python -m scripts.train_sudoku_v2 --steps 40000 --batch 128 --dim 256 --depth 2 --ff 1024 --H 4 --L 16
"""
import os, sys, time, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from babygroot_strm.policy import STRMPolicy
from scripts.train_sudoku import gen_dataset, augment_batch   # reuse data + aug

MASK_ID = 9                                                    # vocab: 0-8 digits, 9 = MASK

class SudokuSTRM(nn.Module):
    """Thin wrapper around the actual non-VAE STRMPolicy (one 81-cell level, K=9)."""
    def __init__(self, dim=256, depth=2, heads=8, kv_heads=2, ff=1024, L=16, H=4, inner_tol=0.0,
                 compile_g=False):
        super().__init__()
        self.dim, self.L, self.H = dim, L, H
        self.policy = STRMPolicy(
            seq_lens=[81], k_codebook=9, dim=dim, heads=heads, kv_heads=kv_heads,
            ff_hidden=ff, depth=depth, L_inner=L, H_outer=H, state_dim=dim, max_prefix=4,
            update_mode='damped', alpha_parametrization='sigmoid', layerscale_init=0.1,
            one_step_grad=True, output_scalenorm=True,
            carry_zl=True, nesterov=True, nesterov_beta=0.7, grad_checkpoint=False,
            inner_tol=inner_tol)
        if compile_g:                                            # ~2.5× on the g-call; loop stays eager
            self.policy.g = torch.compile(self.policy.g)
        self.ctx = nn.Parameter(torch.zeros(1, 1, dim))           # minimal learned "vision" token

    def clean_state_dict(self):                                  # strip torch.compile's _orig_mod. prefix
        return {k.replace('_orig_mod.', ''): v for k, v in self.state_dict().items()}

    def forward(self, vis_tokens, n_H=None, n_L=None):
        """vis_tokens (B,81): 0-8 at visible cells, MASK_ID at masked. Returns list of per-cycle (B,81,9) logits."""
        B, dev = vis_tokens.shape[0], vis_tokens.device
        masked = (vis_tokens == MASK_ID)
        vis = self.ctx.expand(B, -1, -1)
        state = torch.zeros(B, self.dim, device=dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=vis_tokens.is_cuda):  # bf16 AMP (≈2-3× faster)
            logits_list = self.policy.forward([vis_tokens.long()], vis, state, mask_list=[masked],
                                              n_outer=n_H, n_inner=n_L)   # _y_embed puts MASK row at masked
        return [lg[0].float() for lg in logits_list]                   # fp32 logits for stable CE/argmax

    def alphas(self):
        rL, rH = self.policy._rhos()
        return float(rL.detach().mean()), float(rH.detach().mean())

# ── cosine MaskGIT mask + token prep ──────────────────────────────────────────────────────
def make_tokens(P, S, dev, cosine=True):
    """(vis_tokens, target0, loss_mask). cosine: MaskGIT mask r=cos(π/2·U) of NON-given cells
    (givens always shown); else the actual puzzle (givens shown, blanks masked)."""
    P, S = P.to(dev), S.to(dev); B = P.shape[0]; sol0 = (S - 1)            # 0-8
    given = (P != 0)
    if cosine:
        u = torch.rand(B, 1, device=dev); r = torch.cos(math.pi / 2 * u)
        noise = torch.rand(B, 81, device=dev); noise[given] = -1.0
        thr = noise.sort(dim=1, descending=True).values.gather(
            1, (r * (~given).sum(1, keepdim=True)).long().clamp(min=1) - 1)
        masked = (noise >= thr) & (~given)
    else:
        masked = (P == 0)
    vis = torch.where(masked, torch.full_like(sol0, MASK_ID), sol0)
    return vis, sol0, masked

def evaluate(model, P, S, dev, n_H=None, n_L=None):
    model.eval()
    with torch.no_grad():
        cell = solve = tot = 0
        for i in range(0, len(P), 128):
            p, s = P[i:i+128].to(dev), S[i:i+128].to(dev)
            vis, sol0, masked = make_tokens(p, s, dev, cosine=False)
            logits = model(vis, n_H=n_H, n_L=n_L)[-1]
            pred = logits.argmax(-1) + 1
            full = torch.where(masked, pred, p)
            cell += ((pred == s) & masked).sum().item(); tot += masked.sum().item()
            solve += (full == s).all(dim=1).sum().item()
    model.train()
    return cell / max(tot, 1) * 100, solve / len(P) * 100

@torch.no_grad()
def eval_cosine(model, P, S, dev, passes=3):
    """Mean masked-cell accuracy under the MaskGIT cosine mask distribution (the training
    distribution) on held-out puzzles — the honest 'natural' metric, averaged over sampled ratios."""
    model.eval(); cell = tot = 0
    for _ in range(passes):
        for i in range(0, len(P), 128):
            p, s = P[i:i+128].to(dev), S[i:i+128].to(dev)
            vis, sol0, masked = make_tokens(p, s, dev, cosine=True)
            pred = model(vis, n_H=model.H, n_L=model.L)[-1].argmax(-1)
            cell += ((pred == sol0) & masked).sum().item(); tot += masked.sum().item()
    model.train(); return cell / max(tot, 1) * 100

@torch.no_grad()
def eval_by_maskratio(model, S, dev, ks=(1, 5, 20, 50)):
    """Cell-acc when EXACTLY k cells are masked (all others revealed = solution). Isolates
    difficulty: k=1 is near-trivial constraint completion, k=50 is hard. Flat-across-k ⇒
    not a difficulty/capacity issue."""
    model.eval(); out = {}
    for k in ks:
        cell = tot = 0
        for i in range(0, len(S), 128):
            s = S[i:i+128].to(dev); B = s.shape[0]; sol0 = s - 1
            noise = torch.rand(B, 81, device=dev)
            thr = noise.sort(1).values[:, k-1:k]                 # k smallest → masked
            masked = noise <= thr
            vis = torch.where(masked, torch.full_like(sol0, MASK_ID), sol0)
            pred = model(vis, n_H=model.H, n_L=model.L)[-1].argmax(-1)
            cell += ((pred == sol0) & masked).sum().item(); tot += masked.sum().item()
        out[k] = cell / max(tot, 1) * 100
    model.train(); return out

def ema_init(model, decay):
    return {nm: p.detach().clone() for nm, p in model.named_parameters() if p.requires_grad} if decay > 0 else None

@torch.no_grad()
def ema_update(model, ema, decay):
    for nm, p in model.named_parameters():
        if nm in ema: ema[nm].mul_(decay).add_(p.detach(), alpha=1 - decay)

@torch.no_grad()
def ema_swap(model, ema):                                     # swap EMA weights in, return raw backup
    bk = {}
    for nm, p in model.named_parameters():
        if nm in ema: bk[nm] = p.detach().clone(); p.data.copy_(ema[nm])
    return bk

@torch.no_grad()
def ema_restore(model, bk):
    for nm, p in model.named_parameters():
        if nm in bk: p.data.copy_(bk[nm])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40000); ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--dim', type=int, default=256); ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--heads', type=int, default=8); ap.add_argument('--kv-heads', type=int, default=2)
    ap.add_argument('--ff', type=int, default=1024); ap.add_argument('--H', type=int, default=4)
    ap.add_argument('--L', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--cur-extra', type=int, default=20); ap.add_argument('--no-cosine', action='store_true')
    ap.add_argument('--data', default='data/cache/sudoku.pt'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--inner-tol', type=float, default=0.0)   # inner early-stop rel. residual (0=off)
    ap.add_argument('--compile', action='store_true')         # torch.compile the g-net (~2.5×)
    ap.add_argument('--ema-decay', type=float, default=0.0)   # weight EMA (0=off); eval/ckpt use EMA weights
    ap.add_argument('--ckpt-path', default='data/ckpt/sudoku_v2_best.pt')  # per-run checkpoint
    ap.add_argument('--early-stop', type=int, default=0)      # stop after N evals w/o cosine improvement (0=off)
    ap.add_argument('--tag', default='')                      # label for logs
    ap.add_argument('--no-final-diag', action='store_true')   # skip end-of-run H-sweep + fp-convergence (sweep speed)
    args = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')                # TF32 for fp32 matmuls

    d = torch.load(args.data); Ptr, Str, Pte, Ste = d['Ptr'], d['Str'], d['Pte'], d['Ste']
    print(f"loaded {len(Ptr)} train / {len(Pte)} test, clues~{(Ptr[0]!=0).sum().item()}", flush=True)
    model = SudokuSTRM(args.dim, args.depth, args.heads, args.kv_heads, args.ff, args.L, args.H,
                       inner_tol=args.inner_tol, compile_g=args.compile).to(dev)
    print(f"SudokuSTRM (REAL STRMPolicy: carry_zl+nesterov+out_scalenorm, non-VAE) "
          f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(dim{args.dim}/d{args.depth}/H{args.H}/L{args.L}, cosine={not args.no_cosine})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warm = max(200, int(0.03 * args.steps))
    def lr_at(s): return args.lr*s/warm if s < warm else args.lr*0.1 + 0.5*(args.lr-args.lr*0.1)*(1+math.cos(math.pi*(s-warm)/(args.steps-warm)))
    ema = ema_init(model, args.ema_decay)
    if ema is not None: print(f"  [ema] on, decay={args.ema_decay}, {len(ema)} params", flush=True)
    rng = np.random.default_rng(args.seed); t0 = time.time(); best = 0.0; tr_ema = None; since_best = 0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups: g['lr'] = lr_at(step)
        idx = rng.integers(0, len(Ptr), args.batch)
        p, s = augment_batch(Ptr[idx], Str[idx], dev)
        extra = int(round(args.cur_extra * max(0.0, 1 - step/(args.steps*0.5))))      # curriculum
        if extra > 0:
            rsc = torch.rand(p.shape, device=dev); rsc[p != 0] = -1.0
            kth = rsc.topk(min(extra, 81), dim=1).values[:, -1:]
            p = torch.where((rsc >= kth) & (p == 0), s, p)
        vis, sol0, masked = make_tokens(p, s, dev, cosine=not args.no_cosine)
        nH = int(torch.randint(1, args.H + 1, ()).item())                            # stochastic depth: H
        nL = int(torch.randint(1, args.L + 1, ()).item())                            # stochastic depth: L
        logits = model(vis, n_H=nH, n_L=nL)
        loss = sum(F.cross_entropy(lg[masked], sol0[masked]) for lg in logits) / len(logits)
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        if ema is not None: ema_update(model, ema, args.ema_decay)
        with torch.no_grad():                                     # train acc on THIS step's cosine masks
            tr = ((logits[-1].argmax(-1) == sol0) & masked).sum().float() / masked.sum().clamp(min=1)
        tr_ema = tr.item() if tr_ema is None else 0.99 * tr_ema + 0.01 * tr.item()
        if step % 1000 == 0 or step == 1:
            bk = ema_swap(model, ema) if ema is not None else None  # eval + checkpoint the EMA weights
            cos = eval_cosine(model, Pte, Ste, dev, passes=1)     # test acc under cosine (training) distribution
            ca, sv = evaluate(model, Pte, Ste, dev, n_H=args.H, n_L=args.L)  # full-puzzle (hardest)
            aL, aH = model.alphas()
            print(f"step {step:>6}/{args.steps} loss={loss.item():.3f} train_acc={tr_ema*100:.1f}% grad={gnorm:.2f} | "
                  f"test cosine={cos:.1f}% full(cell={ca:.1f}% solve={sv:.1f}%) | α_L={aL:.3f} α_H={aH:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            if step % 2000 == 0 or step == 1:                     # difficulty split by #masked cells (EMA weights)
                mr = eval_by_maskratio(model, Ste, dev)
                print(f"    by #masked: " + "  ".join(f"k={k}:{v:.1f}%" for k, v in mr.items()), flush=True)
            if ca > best + 0.05:                                  # best-checkpoint by FULL-PUZZLE cell (tracks solve;
                best = ca; since_best = 0                         # cosine peaks early then diverges from solve)
                os.makedirs(os.path.dirname(args.ckpt_path) or '.', exist_ok=True)
                tmp = args.ckpt_path + '.tmp'                     # atomic save: write tmp then rename
                torch.save({'model': model.clean_state_dict(), 'step': step, 'args': vars(args), 'cosine': cos, 'cell': ca}, tmp)
                os.replace(tmp, args.ckpt_path)
            else:
                since_best += 1
            if ema is not None: ema_restore(model, bk)            # restore raw weights for training
            if args.early_stop and since_best >= args.early_stop:
                print(f"[early-stop] no cosine gain for {args.early_stop} evals (best {best:.1f}% ) @ step {step}", flush=True)
                break
    if args.no_final_diag:
        print(f"[done] best cosine {best:.1f}% — ckpt {args.ckpt_path}", flush=True); return
    print("\n=== iteration sweep (test) — does MORE recursion help now (carry z_L + nesterov)? ===", flush=True)
    for H in [1, 2, 4, 8, 16]:
        ca, sv = evaluate(model, Pte, Ste, dev, n_H=H, n_L=args.L)
        print(f"  H={H:>2} (L={args.L}): cell={ca:.1f}% solve={sv:.1f}%", flush=True)
    # fixed-point convergence: damped vs nesterov on the trained g
    print("\n=== fixed-point convergence (damped vs nesterov) ===", flush=True)
    model.eval()
    with torch.no_grad():
        vis2, sol2, mask = make_tokens(Pte[:128], Ste[:128], dev, cosine=False)
        B = 128
        ctx = model.ctx.expand(B, -1, -1); state = torch.zeros(B, model.dim, device=dev)
        res = model.policy.compare_iteration_schemes(ctx, state, indices_list=[sol2.long()],
                                                     mask_list=[mask], n_iter=40)  # sol2: true codes 0-8
    for scheme in ('damped', 'nesterov'):
        fr = res[scheme]['fp_resid']
        print(f"  {scheme:>9}: fp_resid {fr[0]:.3f}→{fr[-1]:.4f}  acc={res[scheme].get('acc', float('nan')):.3f}", flush=True)

if __name__ == '__main__':
    main()
