#!/usr/bin/env python3
"""Does burn-in + lower lr fix the boundary collapse? Falsifiable: radius-vs-depth
Spearman must become strongly positive (>0.3) and boundary pinning must drop."""
import numpy as np, torch, geoopt
from scipy.stats import spearmanr
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
depth=np.array([len(chain(i)) for i in range(N)])
ball=geoopt.PoincareBall()
def run(tag, lr, burn_ep, burn_div, epochs):
    torch.manual_seed(13)
    X=geoopt.ManifoldParameter(ball.random(N,DIM,std=1e-3).to(dev),manifold=ball)
    opt=geoopt.optim.RiemannianSGD([X],lr=lr/burn_div)
    for ep in range(epochs):
        if ep==burn_ep:
            opt=geoopt.optim.RiemannianSGD([X],lr=lr)
        s=E[torch.randint(0,len(E),(BATCH,),device=dev)]
        u,v=s[:,0],s[:,1]; neg=torch.randint(0,N,(len(u),NEG),device=dev)
        dp=ball.dist(X[u],X[v]); dn=ball.dist(X[u].unsqueeze(1),X[neg])
        loss=torch.nn.functional.cross_entropy(torch.cat([-dp.unsqueeze(1),-dn],1),
             torch.zeros(len(u),dtype=torch.long,device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    r=X.detach().norm(dim=1).cpu().numpy()
    rho=spearmanr(depth,r).correlation
    print(f"{tag:38} loss {loss.item():.3f}  med_r {np.median(r):.4f}  %r>0.99 {100*(r>0.99).mean():5.1f}%  radius~depth rho {rho:+.3f}")
    return rho
print(f"{N} nodes, {len(el)} edges\n")
print("GOAL: radius~depth rho > 0.3 and boundary pinning down => collapse fixed\n")
run("OUR CONFIG (RAdam lr .05, no burn-in)",0.05,0,1,400)
run("RSGD lr 0.3, burn-in 20ep @ lr/10",0.3,20,10,400)
run("RSGD lr 0.1, burn-in 50ep @ lr/10",0.1,50,10,400)
run("RSGD lr 0.01, burn-in 50ep @ lr/10",0.01,50,10,400)
