#!/usr/bin/env python3
"""Offline eval on the SPECIFIC bridge task types that match the 4 SimplerEnv tasks.

Filters held-out bridge_orig frames whose task string matches each SimplerEnv task
(spoon+towel, carrot+plate, stack+block/cube, eggplant+basket), then runs the
full-mask + cosine masked-CE accuracy on each subset. Tells us whether the model
is strong on these EXACT task semantics offline (real frames) — isolating the
sim-visual gap from any task-competence gap.
"""
import os, sys, glob, random, math
os.environ['CUDA_VISIBLE_DEVICES'] = ''
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm, ActionVQVAE1d, VQ1d_EMA)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset, EMBODIMENTS, EMBODIMENT_ID)

torch.set_num_threads(4); random.seed(0); torch.manual_seed(0)
CKPT='data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt'; ROBOT='widowx'

# SimplerEnv task → (required keywords, all must appear in lowercased task string)
SIM_TASKS = {
    'spoon_on_towel':     [['spoon'], ['towel','cloth']],
    'carrot_on_plate':    [['carrot'], ['plate']],
    'stack_cube':         [['stack'], ['block','cube']],
    'eggplant_in_basket': [['eggplant'], ['basket']],
}
def matches(task, groups):
    t=task.lower()
    return all(any(kw in t for kw in g) for g in groups)


def main():
    ck=torch.load(CKPT,map_location='cpu',weights_only=False); a=ck['args']
    DIM=a['dim'];PFX=a['n_emb_prefix'];IMG=a['img_size'];MAX_TEXT=a['max_text'];H=a['H_outer'];L=a['L_inner']
    N_EMB=len(EMBODIMENTS)+1; K=256
    vck=torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt',map_location='cpu',weights_only=False)
    adim=vck['action_dim']
    vae=ActionVQVAE1d(action_dim=adim,vq_cls=VQ1d_EMA,k=K); vae.load_state_dict(vck['vae']); vae.eval()
    var_global=vck['action_var_global'].view(1,1,-1)
    img_var=torch.load('data/cache/image_var_global.pt',map_location='cpu',weights_only=False)['var_global']
    t5=torch.load('data/cache/t5_text_cache_paraphrased.pt',map_location='cpu',weights_only=False)
    eid=torch.tensor([EMBODIMENT_ID[ROBOT]],dtype=torch.long)
    cnn=EfficientCNN(dims=[24,48,96,192],depths=[1,1,1,1],expand=a['cnn_expand'],out_dim=a['cnn_out_dim'],
                     norm=a['cnn_norm'],pos_emb=a['cnn_pe'],img_size=IMG,dropout=0.0,n_embodiments=0)
    text_agg=LayerAggregator(hidden_dim=512,n_layers=9); cnn_proj=nn.Linear(a['cnn_out_dim'],DIM)
    text_proj=nn.Linear(512,DIM); kv_norm=ScaleNorm(DIM)
    se_keys=sorted({k.split('.')[0] for k in ck['state_encoders'].keys()})
    state_encoders=nn.ModuleDict({e:nn.Sequential(nn.Linear(8,DIM),nn.GELU(),nn.Linear(DIM,DIM)) for e in se_keys})
    emb_id_emb=nn.Embedding(N_EMB,DIM*PFX); n_vis=(IMG//32)**2
    policy=STRMPolicy(seq_lens=(4,),k_codebook=K,dim=DIM,heads=a['heads'],kv_heads=a.get('kv_heads'),
                      ff_hidden=a.get('ff_hidden'),depth=a['depth'],L_inner=L,H_outer=H,state_dim=DIM,
                      max_prefix=n_vis+MAX_TEXT+16+PFX,weighting=a['weighting'],update_mode=a['update_mode'],
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

    # gather frames per sim-task
    sp=None
    for d in sorted(glob.glob('data/oxe/*')):
        if 'bridge_orig' in d and os.path.isfile(os.path.join(d,'meta','info.json')):
            sp=load_dataset_spec(d,chunk_len=16,lookback=16,chunk_stride=16); break
    ds=MultiOXEDataset([sp],chunk_len=16,lookback=16)
    rng=random.Random(123); pool=list(range(len(ds))); rng.shuffle(pool)

    @torch.no_grad()
    def eval_frame(fr,st,ac,pv,tk, full_mask):
        pil=fr.convert('RGB').resize((IMG,IMG))
        x=torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).float()[None]/255.
        x=normalize_image(x,img_var)
        vtok,_=cnn(x); vtok=cnn_proj(vtok)
        t5s=torch.zeros(9,1,MAX_TEXT,512)
        e=t5['embeddings'].get(tk)
        if e is not None:
            h=e['hidden'].float(); tt=min(h.shape[1],MAX_TEXT); LL=min(h.shape[0],9); t5s[:LL,0,:tt,:]=h[:LL,:tt,:]
        ttok=text_proj(text_agg([t5s[l] for l in range(9)]))
        etok=emb_id_emb(eid).view(1,PFX,DIM)
        vis=kv_norm(torch.cat([etok,vtok,ttok],dim=1))
        s_enc=state_encoders[ROBOT](st[None])
        # GT codes
        nT=ac.shape[0]; m=pv.mean(0,keepdim=True); S=((pv-m)**2).sum(0,keepdim=True)
        lam=nT/(S+nT*var_global.squeeze(0)); xn=((ac-m)*lam.sqrt()).transpose(0,1)[None]
        cd,_=vae.encode_with_soft(xn,tau=0.1); gt=[cd[0]]; target=cd[0]
        T_l=target.shape[1]
        mask=torch.ones(1,T_l,dtype=torch.bool) if full_mask else None
        al=policy(gt if not full_mask else [torch.full((1,T_l),K,dtype=torch.long)], vis, s_enc,
                  mask_list=[mask] if mask is not None else None, n_outer=H, n_inner=L, emb_id=eid)
        logits=al[-1][0]
        preds=logits[...,:K].argmax(-1)
        if mask is None: mask=torch.ones_like(preds,dtype=torch.bool)
        correct=((preds==target)&mask).float().sum().item(); masked=mask.float().sum().item()
        return correct, masked

    print(f'{"sim task":<22s} {"#frames":>8s} {"full-mask acc":>14s}')
    overall_c=0; overall_m=0
    for name, groups in SIM_TASKS.items():
        c=0; tot=0; nf=0
        for idx in pool:
            if nf>=40: break
            try: fr,st,ac,pv,tk,_,_=ds[idx]
            except: continue
            if ac.shape[-1]!=adim or st.shape[-1]!=8: continue
            if not matches(tk, groups): continue
            cc,mm=eval_frame(fr,st,ac,pv,tk, full_mask=True)
            c+=cc; tot+=mm; nf+=1
        acc=c/max(1,tot)*100
        overall_c+=c; overall_m+=tot
        print(f'{name:<22s} {nf:>8d} {acc:>13.1f}%')
    print(f'{"OVERALL (these tasks)":<22s} {"":>8s} {overall_c/max(1,overall_m)*100:>13.1f}%')
    print()
    print('(compare: overall full-mask on RANDOM held-out frames was ~84% at this ckpt)')
    print('=> if these task-specific numbers are similarly high, the model KNOWS these tasks')
    print('   offline; the 0% sim is purely the visual domain gap, not task competence.')


if __name__ == '__main__':
    main()
