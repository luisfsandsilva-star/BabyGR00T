#!/usr/bin/env python3
"""Plot raw vs EMA-smoothed training loss for v14_widowx_v2.
Combines original run (steps 1-7600 before crash) and resumed run (5000+).
Marks the resume boundary and ρ_H collapse region.
"""
import re, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOGS = [
    ('original (broken-opt then fix, crashed @ 7600)', '/tmp/v14_widowx_v2.log', 'C0', '-'),
    ('resumed (no-revive, reset-opt) from step 5000', '/tmp/v14_widowx_v2_resumed.log', 'C3', '-'),
]
SENTINEL_LOGS = [
    ('orig sentinel val_acc', '/tmp/sentinel_widowx_v2.log', 'C0'),
    ('resume sentinel val_acc', '/tmp/sentinel_widowx_v2_resumed.log', 'C3'),
]
PAT = re.compile(r'step\s+(\d+)/\d+\s+loss=([\d.]+)\s+acc=([\d.]+)%')
SPAT = re.compile(r'step\s+(\d+):\s+val_acc=([\d.]+)%')

def parse(path):
    steps, losses, accs = [], [], []
    if not os.path.exists(path): return steps, losses, accs
    for line in open(path):
        m = PAT.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            accs.append(float(m.group(3)))
    return steps, losses, accs

def ema(xs, beta=0.9):
    out = []; s = None
    for x in xs:
        s = x if s is None else beta * s + (1 - beta) * x
        out.append(s)
    return out

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
ax_loss, ax_acc = axes

# loss panel
all_max_step = 0
for label, path, color, style in LOGS:
    s, l, a = parse(path)
    if not s: continue
    all_max_step = max(all_max_step, max(s))
    ax_loss.plot(s, l, color=color, ls=style, lw=0.5, alpha=0.35, label=f'{label} (raw)')
    ax_loss.plot(s, ema(l), color=color, ls=style, lw=2.0, label=f'{label} (EMA β=0.9)')
    ax_acc.plot(s, a, color=color, ls=style, lw=0.5, alpha=0.35)
    ax_acc.plot(s, ema(a), color=color, ls=style, lw=2.0, label=f'{label} (train_acc EMA)')

# overlay sentinel val_acc as scatter points
for label, path, color in SENTINEL_LOGS:
    if not os.path.exists(path): continue
    pts = []
    for line in open(path):
        m = SPAT.search(line)
        if m: pts.append((int(m.group(1)), float(m.group(2))))
    if pts:
        sx, sy = zip(*pts)
        ax_acc.scatter(sx, sy, color=color, marker='*', s=140, edgecolor='black', linewidth=0.7,
                       zorder=10, label=label)

# annotate key events
ax_loss.axvline(5000, color='black', ls=':', alpha=0.5)
ax_loss.text(5000, 7.5, ' resume @ 5000\n (val_acc 18.30%)', fontsize=8, va='top')
ax_loss.axvspan(5400, 7600, color='orange', alpha=0.10, label='ρ_H collapse region (orig run)')
ax_loss.axvspan(5400, 7600, color='orange', alpha=0)  # only adds to legend once

ax_loss.set_ylabel('CE loss')
ax_loss.set_title('v14_widowx_v2: raw vs EMA loss (single-emb, lr=8.97e-3, σ_g=0.03, 13.58M params)')
ax_loss.legend(loc='upper right', fontsize=8)
ax_loss.grid(True, alpha=0.3)
ax_loss.set_ylim(3.5, 8.0)

ax_acc.axvline(5000, color='black', ls=':', alpha=0.5)
ax_acc.axhline(19.33, color='C3', ls=':', alpha=0.5)
ax_acc.text(all_max_step * 0.6, 19.5, ' val best 19.33% @ step 7000 (resumed)', fontsize=8, color='C3')
ax_acc.set_xlabel('training step')
ax_acc.set_ylabel('accuracy (%)')
ax_acc.legend(loc='lower right', fontsize=8)
ax_acc.grid(True, alpha=0.3)
ax_acc.set_ylim(0, 25)

plt.tight_layout()
out = '/tmp/v14_widowx_v2_loss.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f"saved → {out}")
print(f"  orig run: {len(parse(LOGS[0][1])[0])} log points, max step {max(parse(LOGS[0][1])[0] or [0])}")
print(f"  resumed:  {len(parse(LOGS[1][1])[0])} log points, max step {max(parse(LOGS[1][1])[0] or [0])}")
