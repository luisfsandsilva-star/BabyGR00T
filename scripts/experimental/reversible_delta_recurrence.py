import torch
import torch.nn as nn
import torch.nn.functional as F


def delta_erase(x, k, beta):
    B, T, d = x.shape
    H = k.shape[2]
    xh = x.view(B, T, H, d // H)
    proj = (k * xh).sum(-1)
    xe = xh - beta.unsqueeze(-1) * k * proj.unsqueeze(-1)
    return xe.reshape(B, T, d), proj


def delta_erase_inverse(y, k, beta, buf):
    B, T, d = y.shape
    H = k.shape[2]
    yh = y.view(B, T, H, d // H)
    projy = (k * yh).sum(-1)
    xh = yh + (buf - projy).unsqueeze(-1) * k
    return xh.reshape(B, T, d)


class _StreamNet(nn.Module):
    def __init__(self, dim, n_heads, beta_max=1.0):
        super().__init__()
        self.h, self.dh, self.bmax = n_heads, dim // n_heads, beta_max
        self.norm = nn.LayerNorm(dim)
        self.trunk = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.to_w = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_b = nn.Linear(dim, n_heads, bias=False)

    def forward(self, cond):
        B, T, d = cond.shape
        h = self.trunk(self.norm(cond))
        w = self.to_w(h)
        k = self.to_k(h).view(B, T, self.h, self.dh)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        beta = self.bmax * torch.sigmoid(self.to_b(h))
        return w, k, beta


class RefinementStep(nn.Module):
    def __init__(self, dim, n_heads=4, beta_max=1.0):
        super().__init__()
        self.net_z = _StreamNet(dim, n_heads, beta_max)
        self.net_y = _StreamNet(dim, n_heads, beta_max)

    @staticmethod
    def _cond(a, b):
        return a + b

    def forward(self, z, y, x):
        w_z, k_z, b_z = self.net_z(self._cond(x, y))
        z_e, buf_z = delta_erase(z, k_z, b_z)
        z = z_e + w_z
        w_y, k_y, b_y = self.net_y(self._cond(x, z))
        y_e, buf_y = delta_erase(y, k_y, b_y)
        y = y_e + w_y
        return z, y, (buf_z, buf_y)

    def inverse(self, z, y, x, buffers):
        buf_z, buf_y = buffers
        w_y, k_y, b_y = self.net_y(self._cond(x, z))
        y = delta_erase_inverse(y - w_y, k_y, b_y, buf_y)
        w_z, k_z, b_z = self.net_z(self._cond(x, y))
        z = delta_erase_inverse(z - w_z, k_z, b_z, buf_z)
        return z, y


class _RevRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, y, x, step, n_steps):
        ctx.step, ctx.n = step, n_steps
        bufs = []
        with torch.no_grad():
            for _ in range(n_steps):
                z, y, b = step(z, y, x)
                bufs.append(b)
        ctx.save_for_backward(z.detach(), y.detach(), x.detach())
        ctx.bufs = bufs
        return z, y

    @staticmethod
    def backward(ctx, dz, dy):
        z, y, x = ctx.saved_tensors
        step = ctx.step
        dx = torch.zeros_like(x)
        params = [p for p in step.parameters() if p.requires_grad]
        for i in range(ctx.n - 1, -1, -1):
            with torch.no_grad():
                z, y = step.inverse(z, y, x, ctx.bufs[i])
            zr = z.detach().requires_grad_(True)
            yr = y.detach().requires_grad_(True)
            xr = x.detach().requires_grad_(True)
            with torch.enable_grad():
                z2, y2, _ = step(zr, yr, xr)
            g = torch.autograd.grad((z2, y2), (zr, yr, xr, *params), (dz, dy), allow_unused=True)
            dz, dy = g[0], g[1]
            if g[2] is not None:
                dx = dx + g[2]
            for p, gp in zip(params, g[3:]):
                if gp is None:
                    continue
                p.grad = gp.detach() if p.grad is None else p.grad + gp.detach()
        return dz, dy, dx, None, None


class ReversibleReasoner(nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, z, y, x, n_steps, store_everything=False):
        if store_everything:
            for _ in range(n_steps):
                z, y, _ = self.step(z, y, x)
            return z, y
        return _RevRecurrence.apply(z, y, x, self.step, n_steps)


def _gradient_check(dim=48, heads=4, n_steps=40, B=4, T=16, beta_max=1.0, tol=1e-3):
    torch.manual_seed(0)
    model = ReversibleReasoner(RefinementStep(dim, heads, beta_max))
    x = torch.randn(B, T, dim); z0 = torch.randn(B, T, dim); y0 = torch.randn(B, T, dim)
    tz = torch.randn(B, T, dim); ty = torch.randn(B, T, dim)

    def loss_from(zf, yf):
        return ((zf - tz) ** 2).mean() + ((yf - ty) ** 2).mean()

    def run(store):
        model.zero_grad(set_to_none=True)
        zi = z0.clone().requires_grad_(True); yi = y0.clone().requires_grad_(True); xi = x.clone().requires_grad_(True)
        zf, yf = model(zi, yi, xi, n_steps, store_everything=store)
        loss_from(zf, yf).backward()
        gp = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        gi = {'z0': zi.grad.clone(), 'y0': yi.grad.clone(), 'x': xi.grad.clone()}
        return gp, gi

    gp_b, gi_b = run(store=True); gp_r, gi_r = run(store=False)
    wc, wr = 1.0, 0.0
    for d_b, d_r in [(gp_b, gp_r), (gi_b, gi_r)]:
        for k in d_b:
            a, b = d_b[k].flatten(), d_r[k].flatten()
            wc = min(wc, (a @ b / (a.norm() * b.norm() + 1e-20)).item())
            wr = max(wr, ((b - a).norm() / (a.norm() + 1e-20)).item())
    print(f"steps={n_steps} beta_max={beta_max} | worst cos={wc:.6f} worst rel-err={wr:.2e} -> {'PASS' if wr < tol else 'FAIL'}")
    return wr


if __name__ == "__main__":
    _gradient_check(beta_max=1.0, n_steps=40)
    _gradient_check(beta_max=0.9, n_steps=40)
    _gradient_check(beta_max=1.0, n_steps=200)
