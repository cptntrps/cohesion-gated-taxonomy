#!/usr/bin/env python3
"""Define 'not trained correctly' by MEASUREMENT, not assertion.
Train the hyperbolic arm as we did, then inspect the geometry itself."""
import numpy as np, torch, geoopt, collections
from nltk.corpus import wordnet as wn
DIM,NEG,BATCH=8,8,100_000
dev="cuda" if torch.cuda.is_available() else "cpu"
syns=list(wn.all_synsets('n')); idx={s.name():i for i,s in enumerate(syns)}; N=len(syns)
parent=np.full(N,-1,dtype=np.int64)
for s in syns:
    h=s.hypernyms()
    if h: parent[idx[s.name()]]=idx[h[0].name()]
def chain(n):
    o=[];x=int(n);seen=set()
    while x!=-1 and x not in seen: o.append(x);seen.add(x);x=int(parent[x])
    return o
el=[[i,a] for i in range(N) for a in chain(i)[1:]]
E=torch.tensor(el,device=dev)
ball=geoopt.PoincareBall()
X=geoopt.ManifoldParameter(ball.random(N,DIM,std=1e-3).to(dev),manifold=ball)
opt=geoopt.optim.RiemannianAdam([X],lr=0.05)
print(f"{N} nodes, {len(el)} edges. lr=0.05, no burn-in (our config)\n")
print(f"{'epoch':>6}{'loss':>9}{'med radius':>12}{'max radius':>12}{'% r>0.99':>10}{'% r>0.999':>11}{'grad norm':>11}")
for ep in range(401):
    s=E[torch.randint(0,len(E),(BATCH,),device=dev)]
    u,v=s[:,0],s[:,1]; neg=torch.randint(0,N,(len(u),NEG),device=dev)
    dp=ball.dist(X[u],X[v]); dn=ball.dist(X[u].unsqueeze(1),X[neg])
    loss=torch.nn.functional.cross_entropy(torch.cat([-dp.unsqueeze(1),-dn],1),
         torch.zeros(len(u),dtype=torch.long,device=dev))
    opt.zero_grad(); loss.backward()
    gn=float(X.grad.norm())
    opt.step()
    if ep%100==0:
        r=X.detach().norm(dim=1)
        print(f"{ep:>6}{loss.item():>9.3f}{float(r.median()):>12.4f}{float(r.max()):>12.6f}"
              f"{100*float((r>0.99).float().mean()):>9.1f}%{100*float((r>0.999).float().mean()):>10.1f}%{gn:>11.4f}")
r=X.detach().norm(dim=1).cpu().numpy()
print(f"\nFINAL radius distribution: p10 {np.percentile(r,10):.4f}  p50 {np.percentile(r,50):.4f}  "
      f"p90 {np.percentile(r,90):.4f}  max {r.max():.6f}")
print(f"points at r>0.99: {100*(r>0.99).mean():.1f}%   r>0.999: {100*(r>0.999).mean():.1f}%")
# does radius encode DEPTH? (the whole point of hyperbolic hierarchy embedding)
d=np.array([len(chain(i)) for i in range(0,N,7)]); rr=r[0:N:7]
from scipy.stats import spearmanr
print(f"\nradius vs true depth Spearman: {spearmanr(d,rr).correlation:.3f}   (should be strongly POSITIVE)")
# numerical resolution check
conf=1-r**2
print(f"conformal term 1-||x||^2 : median {np.median(conf):.2e}  min {conf.min():.2e}  (float32 eps ~1.2e-7)")
print(f"fraction below float32 eps: {100*(conf<1.2e-7).mean():.1f}%")
