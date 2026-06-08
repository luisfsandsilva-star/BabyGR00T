#!/usr/bin/env python3
"""Build the in-distribution REFERENCE for the sim-real OOD test.

On real held-out bridge_orig (widowx) frames, collects:
  - pooled CNN features (mean over vis tokens)  → the perception manifold
  - full vis (CNN+text+emb) → policy code logits → prediction confidence
    (max softmax prob per code position, and entropy)
  - predicted code distribution (which of K codes get picked)

Saves to /tmp/ood_reference.npz for the sim side (run_ood_sim, sim venv) to load.
"""
import os, sys, glob, random
os.environ['CUDA_VISIBLE_DEVICES'] = ''
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm, ActionVQVAE1d, VQ1d_EMA)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset, EMBODIMENTS, EMBODIMENT_ID)

torch.set_num_threads(4); random.seed(0); torch.manual_seed(0)
CKPT = 'data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt'
ROBOT = 'widowx'; N = 128


def build(ck):
    a = ck['args']; DIM=a['dim']; PFX=a['n_emb_prefix']; IMG=a['img_size']; MAX_TEXT=a['max_text']
    H,L=a['H_outer'],a['L_inner']; N_EMB=len(EMBODIMENTS)+1
    cnn=EfficientCNN(dims=[24,48,96,192],depths=[1,1,1,1],expand=a['cnn_expand'],out_dim=a['cnn_out_dim'],
                     norm=a['cnn_norm'],pos_emb=a['cnn_pe'],img_size=IMG,dropout=0.0,n_embodiments=0)
    text_agg=LayerAggregator(hidden_dim=512,n_layers=9); cnn_proj=nn.Linear(a['cnn_out_dim'],DIM)
    text_proj=nn.Linear(512,DIM); kv_norm=ScaleNorm(DIM)
    se_keys=sorted({k.split('.')[0] for k in ck['state_encoders'].keys()})
    state_encoders=nn.ModuleDict({e:nn.Sequential(nn.Linear(8,DIM),nn.GELU(),nn.Linear(DIM,DIM)) for e in se_keys})
    emb_id_emb=nn.Embedding(N_EMB,DIM*PFX); n_vis=(IMG//32)**2
    policy=STRMPolicy(seq_lens=(4,),k_codebook=256,dim=DIM,heads=a['heads'],
                      kv_heads=a.get('kv_heads'),ff_hidden=a.get('ff_hidden'),depth=a['depth'],L_inner=L,H_outer=H,
                      state_dim=DIM,max_prefix=n_vis+MAX_TEXT+16+PFX,weighting=a['weighting'],update_mode=a['update_mode'],
                      alpha_parametrization=a['alpha_parametrization'],alpha_per_dim=a['alpha_per_dim'],
                      n_embodiments=N_EMB,per_emb_head=a['per_emb_head'],dropout=0.0,g_input_noise=0.0)
    mods=[('cnn',cnn),('text_agg',text_agg),('cnn_proj',cnn_proj),('text_proj',text_proj),('kv_norm',kv_norm),
          ('state_encoders',state_encoders),('emb_id_emb',emb_id_emb),('policy',policy)]
    for nm,m in mods: m.load_state_dict(ck[nm])
    if 'ema_params' in ck:
        for nm,m in mods:
            sd=m.state_dict()
            for k in list(sd.keys()):
                full=f'{nm}.{k}'
                if full in ck['ema_params']: sd[k]=ck['ema_params'][full].to(sd[k].dtype)
            m.load_state_dict(sd)
    for nm,m in mods: m.eval()
    return dict(mods), a, (DIM,PFX,IMG,MAX_TEXT,H,L)


def main():
    ck=torch.load(CKPT,map_location='cpu',weights_only=False)
    M,a,(DIM,PFX,IMG,MAX_TEXT,H,L)=build(ck)
    K=256
    img_var=torch.load('data/cache/image_var_global.pt',map_location='cpu',weights_only=False)['var_global']
    t5=torch.load('data/cache/t5_text_cache_paraphrased.pt',map_location='cpu',weights_only=False)
    eid=torch.tensor([EMBODIMENT_ID[ROBOT]],dtype=torch.long)

    specs=[]
    for d in sorted(glob.glob('data/oxe/*')):
        if 'bridge_orig' in d and os.path.isfile(os.path.join(d,'meta','info.json')):
            specs.append(load_dataset_spec(d,chunk_len=16,lookback=16,chunk_stride=16)); break
    ds=MultiOXEDataset(specs,chunk_len=16,lookback=16)
    rng=random.Random(98765); pool=list(range(len(ds))); rng.shuffle(pool)

    feats=[]; maxprobs=[]; entropies=[]; codes=[]; tok_accum=[]
    n=0
    for idx in pool:
        if n>=N: break
        try: fr,st,ac,pv,tk,_,_=ds[idx]
        except: continue
        pil=fr.convert('RGB').resize((IMG,IMG))
        x=torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).float()[None]/255.
        x=normalize_image(x,img_var)
        with torch.no_grad():
            vtok,_=M['cnn'](x); vtok=M['cnn_proj'](vtok)
            feats.append(vtok.mean(1)[0].numpy())                    # pooled CNN feature
            tok_accum.append(vtok[0].numpy())                        # (n_vis, dim) per-token
            t5s=torch.zeros(9,1,MAX_TEXT,512)
            e=t5['embeddings'].get(tk)
            if e is not None:
                h=e['hidden'].float(); tt=min(h.shape[1],MAX_TEXT); LL=min(h.shape[0],9)
                t5s[:LL,0,:tt,:]=h[:LL,:tt,:]
            ttok=M['text_proj'](M['text_agg']([t5s[l] for l in range(9)]))
            etok=M['emb_id_emb'](eid).view(1,PFX,DIM)
            vis=M['kv_norm'](torch.cat([etok,vtok,ttok],dim=1))
            s_enc=M['state_encoders'][ROBOT](st[None])
            al=M['policy'](None,vis,s_enc,mask_list=None,n_outer=H,n_inner=L,emb_id=eid)
            logits=al[-1][0][...,:K]                                  # (1,4,K)
            p=F.softmax(logits,dim=-1)
            maxprobs.append(p.max(-1).values.mean().item())
            entropies.append((-(p*torch.log(p+1e-9)).sum(-1)).mean().item())
            codes.append(logits.argmax(-1)[0].numpy())
        n+=1
    feats=np.stack(feats); codes=np.stack(codes)
    print(f'REFERENCE built on {n} real bridge frames')
    print(f'  pooled CNN feat: mean-norm={np.linalg.norm(feats,axis=1).mean():.3f}  dim={feats.shape[1]}')
    print(f'  prediction max-prob: mean={np.mean(maxprobs):.3f}  (high = confident)')
    print(f'  prediction entropy : mean={np.mean(entropies):.3f}  (low = confident; max possible={np.log(K):.2f})')
    print(f'  unique codes used: {len(np.unique(codes))}/{K}')
    toks=np.concatenate(tok_accum,0)   # (n_imgs*n_vis, dim) all real tokens
    # subsample a real-token BANK for kNN transport (cap ~6000 tokens)
    ridx=np.random.RandomState(0).permutation(len(toks))[:6000]
    tok_bank=toks[ridx].astype(np.float32)
    np.savez('/tmp/ood_reference.npz', feats=feats, maxprobs=np.array(maxprobs),
             entropies=np.array(entropies), codes=codes,
             feat_mean=feats.mean(0), feat_std=feats.std(0),
             tok_mean=toks.mean(0), tok_std=toks.std(0),
             tok_bank=tok_bank)
    print(f'  real token bank for kNN: {tok_bank.shape}')
    print(f'  per-token feature stats saved: tok_mean-norm={np.linalg.norm(toks.mean(0)):.3f}')
    print('saved /tmp/ood_reference.npz')


if __name__ == '__main__':
    main()
