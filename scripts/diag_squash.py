#!/usr/bin/env python3
"""Compare parameterizations f: ℝ → [0,1] for the ρ gate.

What we need:
  * Smooth & differentiable
  * Bounded to [0, 1] (numerical stability)
  * NON-VANISHING gradient near the boundaries (so the optimizer can both
    escape ρ=0 and approach ρ=1 without dying)
  * Reasonable gradient in the middle too

We measure:
  (a) Forward shape: f(x) for x ∈ [-10, 10]
  (b) df/dx at FIXED target ρ values {0.002, 0.05, 0.1, 0.5, 0.9, 0.95, 0.998}
  (c) Synthetic optimization: how many SGD steps to go from ρ=0.002 → ρ=0.1
      with the same scalar loss landscape (∇L w.r.t. ρ = some fixed constant).
"""
import math, torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Squash functions: each returns (forward(x), inverse(rho), name).
# The forward maps any real to (0, 1) (or [0, 1] for hard ones).
# inverse finds the raw value for a target ρ — used to seed the synth test.

def make_funcs():
    funcs = []

    # 1. Sigmoid (the current parameterization)
    def sigmoid_inv(r): return math.log(r / (1 - r))
    funcs.append(('sigmoid', torch.sigmoid, sigmoid_inv))

    # 2. Algebraic (a.k.a. Elliott): 0.5 + 0.5 x / (1 + |x|)
    def algebraic(x): return 0.5 + 0.5 * x / (1 + x.abs())
    def algebraic_inv(r):
        u = 2 * r - 1                        # in (-1, 1)
        return u / (1 - abs(u))
    funcs.append(('algebraic', algebraic, algebraic_inv))

    # 3. Arctan: 0.5 + arctan(x) / π
    def arctan(x): return 0.5 + torch.atan(x) / math.pi
    def arctan_inv(r): return math.tan(math.pi * (r - 0.5))
    funcs.append(('arctan', arctan, arctan_inv))

    # 4. Gaussian CDF (Φ): 0.5 (1 + erf(x/√2))
    def gauss_cdf(x): return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    def gauss_cdf_inv(r):
        # math has no erfinv; use torch.special.erfinv
        return (math.sqrt(2) * torch.special.erfinv(torch.tensor(2.0 * r - 1.0))).item()
    funcs.append(('gauss_cdf', gauss_cdf, gauss_cdf_inv))

    # 5. Hard sigmoid: clamp(0.5 + 0.2 x, 0, 1)
    def hard_sigmoid(x): return (0.5 + 0.2 * x).clamp(0, 1)
    def hard_sigmoid_inv(r): return (r - 0.5) / 0.2
    funcs.append(('hard_sigmoid', hard_sigmoid, hard_sigmoid_inv))

    # 6. Direct clamp: ρ = clamp(raw, eps, 1-eps)  — gradient = 1 inside, 0 outside
    def clamp_raw(x): return x.clamp(1e-3, 1 - 1e-3)
    def clamp_inv(r): return r
    funcs.append(('clamp_direct', clamp_raw, clamp_inv))

    # 7. Softplus-based: 1 - exp(-softplus(x))  → smooth, no zero saturation
    def softplus_norm(x): return 1 - torch.exp(-F.softplus(x))
    def softplus_norm_inv(r):
        # softplus(x) = -log(1-r); for x>0 softplus(x)≈x. So x = log(exp(-log(1-r))-1) = log(-1/(1-r) - 1)
        sp = -math.log(max(1 - r, 1e-12))
        if sp > 50: return sp
        return math.log(math.exp(sp) - 1)
    funcs.append(('softplus_norm', softplus_norm, softplus_norm_inv))

    # 8. Tanh-shifted: 0.5 (1 + tanh(x))  — same as sigmoid up to scale of x
    def tanh_sh(x): return 0.5 * (1 + torch.tanh(x))
    def tanh_sh_inv(r): return 0.5 * math.log(r / (1 - r))     # = sigmoid_inv/2
    funcs.append(('tanh_shifted', tanh_sh, tanh_sh_inv))

    return funcs


# ──────────────────────────────────────────────────────────────────────
# (a) + (b): boundary gradient via autograd at exact target ρ values
def boundary_grads():
    print("=== gradient |df/dx| at target ρ values ===")
    funcs = make_funcs()
    targets = [0.002, 0.05, 0.1, 0.5, 0.9, 0.95, 0.998]
    print(f"  {'function':<16s} | " + " | ".join(f'ρ={r:<6.3f}' for r in targets))
    print('  ' + '-' * 95)
    for name, f, inv in funcs:
        row = f'  {name:<16s} |'
        for rho in targets:
            raw = inv(rho)
            x = torch.tensor(float(raw), requires_grad=True)
            y = f(x)
            try:
                g = torch.autograd.grad(y, x, create_graph=False)[0].item()
            except Exception:
                g = 0.0
            row += f' {abs(g):>8.4f} |'
        print(row)
    print()


# ──────────────────────────────────────────────────────────────────────
# (c) Synthetic: ρ → minimum at ρ_target. SGD on raw.
# Loss(ρ) = (ρ - ρ_target)²  → dL/dρ = 2(ρ - ρ_target)
# dL/draw = dL/dρ * dρ/draw
# Steps to converge from ρ=ρ_start to ρ_target.
def synth_optim_curve():
    print("=== synthetic SGD: steps to go from ρ_start=0.002 → ρ=0.5 (loss = (ρ-0.5)²) ===")
    print("  (lr = 3.8e-3, plain SGD, no momentum)")
    funcs = make_funcs()
    rho_start = 0.002
    rho_target = 0.5
    lr = 3.8e-3
    max_steps = 50000
    print(f"  {'function':<16s} | {'steps_to_within_0.01':>22s} | {'final_ρ':>10s}")
    print('  ' + '-' * 60)
    for name, f, inv in funcs:
        raw = torch.tensor(float(inv(rho_start)), requires_grad=True)
        steps = max_steps
        for step in range(1, max_steps + 1):
            rho = f(raw)
            loss = (rho - rho_target) ** 2
            if (rho - rho_target).abs().item() < 0.01:
                steps = step; break
            g = torch.autograd.grad(loss, raw)[0]
            with torch.no_grad():
                raw -= lr * g
            raw.requires_grad_(True)
        final = f(raw).item()
        marker = '✓' if steps < max_steps else '✗'
        print(f"  {name:<16s} | {steps:>22d}  | {final:>10.4f}  {marker}")
    print()


# ──────────────────────────────────────────────────────────────────────
# (d) Synthetic: now with LARS-style per-param adaptive LR.
# Effective lr = trust_coef * |raw| / |grad|, then step.
def synth_lars_curve():
    print("=== synthetic LARS: same task with effective_lr = trust * |raw| / |grad| ===")
    print("  (trust_coef = 0.001, no momentum)")
    funcs = make_funcs()
    rho_start = 0.002
    rho_target = 0.5
    trust = 0.001
    max_steps = 50000
    print(f"  {'function':<16s} | {'steps':>10s} | {'final_ρ':>10s}")
    print('  ' + '-' * 50)
    for name, f, inv in funcs:
        raw = torch.tensor(float(inv(rho_start)), requires_grad=True)
        steps = max_steps
        for step in range(1, max_steps + 1):
            rho = f(raw)
            if (rho - rho_target).abs().item() < 0.01:
                steps = step; break
            loss = (rho - rho_target) ** 2
            g = torch.autograd.grad(loss, raw)[0]
            with torch.no_grad():
                eff_lr = trust * max(abs(raw.item()), 1e-3) / max(abs(g.item()), 1e-9)
                raw -= eff_lr * g
            raw.requires_grad_(True)
        final = f(raw).item()
        marker = '✓' if steps < max_steps else '✗'
        print(f"  {name:<16s} | {steps:>10d}  | {final:>10.4f}  {marker}")
    print()


if __name__ == '__main__':
    boundary_grads()
    synth_optim_curve()
    synth_lars_curve()
