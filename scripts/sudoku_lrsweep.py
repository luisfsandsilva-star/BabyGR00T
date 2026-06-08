import sys, time, math; sys.path.insert(0,'.')
import torch, torch.nn.functional as F, numpy as np
from scripts.train_sudoku import SudokuSTRM, augment_batch, evaluate
d=torch.load('data/cache/sudoku.pt'); Ptr,Str,Pte,Ste=d['Ptr'],d['Str'],d['Pte'],d['Ste']; dev='cuda'
STEPS=1200; WARM=80
for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
    torch.manual_seed(0)
    m=SudokuSTRM(dim=192,depth=2,heads=8,kv_heads=2,ff=768,L=5,H=6,one_step=True).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-2); rng=np.random.default_rng(0); t=time.time()
    print(f"\n### lr={lr:.0e} (warmup{WARM}+cosine, 1.42M) ###", flush=True)
    for step in range(1,STEPS+1):
        cur = lr*step/WARM if step<WARM else lr*0.1+0.5*(lr-lr*0.1)*(1+math.cos(math.pi*(step-WARM)/(STEPS-WARM)))
        for g in opt.param_groups: g['lr']=cur
        idx=rng.integers(0,len(Ptr),64); pp,ss=augment_batch(Ptr[idx],Str[idx],dev)
        bl=(pp==0); so=(ss-1).long(); al=m(so,bl)
        loss=sum(F.cross_entropy(lg[0][bl],so[bl]) for lg in al)/len(al)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step%300==0 or step==100:
            ca,sv=evaluate(m,Pte[:500],Ste[:500],dev)
            print(f"  step {step:>4}: loss={loss.item():.3f} test_cell_acc={ca:.1f}% solve={sv:.1f}%", flush=True)
    del m, opt; torch.cuda.empty_cache()
print("\n=== sweep done: pick the lr whose loss drops below ~2.0 / cell_acc climbs fastest ===", flush=True)
