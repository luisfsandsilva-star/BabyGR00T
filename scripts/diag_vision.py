#!/usr/bin/env python3
"""Vision-path diagnostics for the S-TRM policy:
  1. gradient flow into aggregator / resampler / vis_proj / policy (+ into the
     raw vision features), and policy cross-attn(vision) vs self-attn grads.
  2. output sensitivity to vision: real vs zeroed vs shuffled vision.
  3. LayerAggregator's learned 25-layer softmax weighting.
Usage: python -m scripts.diag_vision <ckpt> [cache_dir] [vae_ckpt] [N]
"""
import os, sys, math, random, json
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import torch, torch.nn.functional as F
from babygroot_strm import (RevIN, ActionVQVAE1d, ActionRQUNet1d, VQ1d_EMA,
    STRMPolicy, STRMPolicyVAE, LayerAggregator, PerceiverResampler,
    VIS_HIDDEN_DIM, NUM_RESAMPLER_LATENTS, load_lerobot_episodes, make_loader)

ckpt = sys.argv[1]
cache_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/cache/oxe_vision_cache_v2'
vae_ckpt  = sys.argv[3] if len(sys.argv) > 3 else 'data/ckpts/oxe_vqvae_1800ep_16k.pt'
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

c = torch.load(ckpt, map_location=dev, weights_only=False)
L=c.get('L_inner',5); H=c.get('H_outer',4); depth=c.get('depth',2); dim=c.get('dim',768)
vis_dim=c.get('vis_dim',dim); rL=c.get('rho_L',0.1); rH=c.get('rho_H',0.1)
sdim=c.get('state_dim',8); is_vae=c.get('vae_latent',False); beta=c.get('beta',1e-3); fb=c.get('free_bits',0.1)
vck=torch.load(vae_ckpt,map_location=dev,weights_only=False)
adim=vck.get('action_dim',6); vk=vck.get('kind','cqvae')
vae=(ActionVQVAE1d(action_dim=adim,vq_cls=VQ1d_EMA) if vk=='vqvae'
     else ActionRQUNet1d(action_dim=adim,vq_cls=VQ1d_EMA)).to(dev)
revin=RevIN(adim).to(dev); vae.load_state_dict(vck['vae']); revin.load_state_dict(vck['revin'])
vae.eval(); revin.eval()
seq_lens=tuple(vae.seq_lens); K=vae.vqs[0].K
agg=LayerAggregator(hidden_dim=VIS_HIDDEN_DIM,n_layers=25).to(dev)
res=PerceiverResampler(input_dim=VIS_HIDDEN_DIM,dim=vis_dim,num_latents=NUM_RESAMPLER_LATENTS).to(dev)
vproj=(torch.nn.Identity() if vis_dim==dim else torch.nn.Linear(vis_dim,dim)).to(dev)
Cls=STRMPolicyVAE if is_vae else STRMPolicy
extra=dict(beta=beta,free_bits=fb) if is_vae else {}
pol=Cls(seq_lens=seq_lens,k_codebook=K,dim=dim,heads=8,depth=depth,L_inner=L,H_outer=H,
        rho_L=rL,rho_H=rH,max_prefix=NUM_RESAMPLER_LATENTS+16,state_dim=sdim,**extra).to(dev)
agg.load_state_dict(c['aggregator']); res.load_state_dict(c['resampler'])
if c.get('vis_proj') is not None and not isinstance(vproj,torch.nn.Identity):
    vproj.load_state_dict(c['vis_proj'])
pol.load_state_dict(c['policy'])
print(f"loaded {os.path.basename(ckpt)} step={c.get('step')} dim={dim} vis_dim={vis_dim} "
      f"vae={is_vae} seq_lens={seq_lens} K={K} "
      f"ρ_L={torch.sigmoid(pol.rho_L_raw).item():.3f} ρ_H={torch.sigmoid(pol.rho_H_raw).item():.3f}")

cn=None; mp=os.path.join(cache_dir,'meta.json')
if os.path.exists(mp): cn=int(json.load(open(mp)).get('n_episodes') or 0) or None
eps=load_lerobot_episodes(c.get('oxe_dataset_id','IPEC-COMMUNITY/bridge_orig_lerobot'),
                          camera_key=c.get('oxe_camera','observation.images.image_0'),
                          load_video=False, n_episodes=min(cn or 32,32))
torch.manual_seed(0); random.seed(0)
loader=make_loader(cache_dir, eps[:32], batch_size=N, num_workers=0, shuffle=True, lru_size=16)
batch=next(iter(loader))   # chunk_collate pads variable N_tok across the batch
hidden=batch['hidden'].to(dev).float(); state=batch['state'].to(dev).float(); action=batch['action'].to(dev).float()
print(f"shapes: hidden={tuple(hidden.shape)} state={tuple(state.shape)} action={tuple(action.shape)}")

def vis_pipe(h):
    return vproj(res(agg([h[:,l] for l in range(h.shape[1])])))

with torch.no_grad():
    x=revin(action,'norm').transpose(1,2); gt,_=vae.encode_with_soft(x,tau=0.1)

# ---- 1. GRADIENT FLOW ----
agg.train(); res.train(); pol.train()
for m in (agg,res,pol): m.zero_grad(set_to_none=True)
if not isinstance(vproj,torch.nn.Identity): vproj.train(); vproj.zero_grad(set_to_none=True)
hin=hidden.clone().requires_grad_(True)
vis=vis_pipe(hin)
loss,_,_=pol.forward_loss(gt, vis, state, n_outer=H, n_inner=L, mask_ratio_max=1.0)
loss.backward()
def stats(mod):
    gs=[p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
    wn=sum(p.norm().item()**2 for p in mod.parameters())**0.5
    return ((sum(g*g for g in gs))**0.5, len(gs), len(list(mod.parameters())), wn)
print(f"\n=== 1. GRADIENT FLOW  (loss={loss.item():.3f}) ===")
for name,mod in [('aggregator',agg),('resampler',res),('vis_proj',vproj),('policy',pol)]:
    if isinstance(mod,torch.nn.Identity): print(f"  {name:11s}: Identity (no params)"); continue
    g,nw,npar,wn=stats(mod)
    print(f"  {name:11s}: grad_norm={g:.2e}  weight_norm={wn:.2e}  grad/weight={g/max(wn,1e-12):.2e}  ({nw}/{npar} w/grad)")
print(f"  grad wrt RAW vision features (hidden input): {hin.grad.norm().item():.2e}")
ca=sum(p.grad.norm().item()**2 for n,p in pol.named_parameters() if p.grad is not None and ('ca' in n.split('.')[-2:][0] or 'ca_norm' in n))
sa=sum(p.grad.norm().item()**2 for n,p in pol.named_parameters() if p.grad is not None and ('sa' in n.split('.')[-2:][0] or 'sa_norm' in n))
print(f"  policy cross-attn(vision-consuming) grad={ca**0.5:.2e}   self-attn grad={sa**0.5:.2e}")

# ---- 2. VISION SENSITIVITY ----
pol.eval(); agg.eval(); res.eval()
if not isinstance(vproj,torch.nn.Identity): vproj.eval()
with torch.no_grad():
    vr=vis_pipe(hidden); vz=torch.zeros_like(vr)
    perm=torch.randperm(hidden.shape[0]); vs=vis_pipe(hidden[perm])
    def lo(v): return pol(None, v, state, mask_list=None, n_outer=H, n_inner=L)[-1]
    lr=lo(vr); lz=lo(vz); lsf=lo(vs)
    base=sum(a.abs().mean().item() for a in lr)/len(lr)
    print(f"\n=== 2. VISION SENSITIVITY (does the output depend on vision?) ===")
    print(f"  mean |logit| = {base:.4f}")
    def cmp(lb,tag):
        d=sum((a-b).abs().mean().item() for a,b in zip(lr,lb))/len(lr)
        ch=sum((a[...,:K].argmax(-1)!=b[...,:K].argmax(-1)).float().mean().item() for a,b in zip(lr,lb))/len(lr)
        print(f"  real vs {tag:14s}: mean|Δlogit|={d:.4f} ({d/max(base,1e-9)*100:5.1f}% of |logit|)  argmax-change={ch*100:.0f}%")
    cmp(lz,'ZEROED vision'); cmp(lsf,'SHUFFLED vision')
    vstd=vr.std(0).mean().item(); vmean=vr.abs().mean().item()
    print(f"  resampler output across the {N} inputs: std={vstd:.4f} |mean|={vmean:.4f} "
          f"(std/|mean|={vstd/max(vmean,1e-9):.2f}; →0 means vision collapses to a near-constant)")

# ---- 3. AGGREGATOR LAYER WEIGHTING ----
with torch.no_grad():
    stacked=torch.stack([hidden[:,l] for l in range(hidden.shape[1])],0)
    alpha=F.softmax(agg.gate_proj(stacked)+agg.bias_pre.view(hidden.shape[1],1,1,-1),dim=0)
    per_layer=alpha.mean(dim=(1,2,3))
    print(f"\n=== 3. LayerAggregator: mean softmax weight per LLM layer (uniform={1/hidden.shape[1]:.3f}) ===")
    print("  " + " ".join(f"{w:.3f}" for w in per_layer.tolist()))
    tk=torch.topk(per_layer,3)
    print(f"  top-3 layers: {tk.indices.tolist()} = {[round(x,3) for x in tk.values.tolist()]}  "
          f"(entropy {(-(per_layer*per_layer.clamp_min(1e-9).log()).sum()).item():.2f}/{math.log(hidden.shape[1]):.2f})")
