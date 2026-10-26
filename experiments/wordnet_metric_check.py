#!/usr/bin/env python3
"""Decisive diagnostic: is our CODE broken, or is the METRIC the whole story?
Train once per geometry, then score BOTH:
  (a) MAP-style reconstruction rank of the true parent (the PUBLISHED metric)
  (b) global tree-distance Spearman (OUR metric)
hyperbolic wins (a) -> code sound, metric explains the disagreement.
hyperbolic loses both -> our hyperbolic training is broken."""
import collections, numpy as np, torch, geoopt
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn
DIM, NEG, EPOCHS, BATCH = 8, 8, 400, 100_000
dev = "cuda" if torch.cuda.is_available() else "cpu"
syns = list(wn.all_synsets('n')); idx={s.name():i for i,s in enumerate(syns)}; N=len(syns)
parent=np.full(N,-1,dtype=np.int64)
for s in syns:
    h=s.hypernyms()
    if h: parent[idx[s.name()]]=idx[h[0].name()]
def chain(n):
    o=[];x=int(n);seen=set()
    while x!=-1 and x not in seen: o.append(x);seen.add(x);x=int(parent[x])
    return o
el=[[i,a] for i in range(N) for a in chain(i)[1:]]
edges=torch.tensor(el,device=dev); print(f"{N} synsets, {len(el)} edges",flush=True)
kids=[i for i in range(N) if parent[i]!=-1]
rng=np.random.default_rng(0); probe=rng.choice(kids,1500,replace=False)
pairs=[]
ch_cache={}
for _ in range(1200):
    a=int(rng.integers(0,N)); c=chain(a)
    for lvl in range(1,len(c)):
        pairs.append((a,int(c[lvl])))
pa=np.array([p[0] for p in pairs]); pb=np.array([p[1] for p in pairs])
td=np.array([chain(a).index(b) if b in chain(a) else 99 for a,b in zip(pa,pb)])
k=td<99; pa,pb,td=pa[k],pb[k],td[k]
for geom in ["euclidean","hyperbolic"]:
    if geom=="hyperbolic":
        ball=geoopt.PoincareBall()
        X=geoopt.ManifoldParameter(ball.random(N,DIM,std=1e-3).to(dev),manifold=ball)
        opt=geoopt.optim.RiemannianAdam([X],lr=0.05); dist=ball.dist
    else:
        X=torch.nn.Parameter(torch.randn(N,DIM,device=dev)*1e-3)
        opt=torch.optim.Adam([X],lr=0.05); dist=lambda a,b:(a-b).norm(dim=-1)
    for ep in range(EPOCHS):
        s=edges[torch.randint(0,len(edges),(BATCH,),device=dev)]
        u,v=s[:,0],s[:,1]; neg=torch.randint(0,N,(len(u),NEG),device=dev)
        dp=dist(X[u],X[v]); dn=dist(X[u].unsqueeze(1),X[neg])
        loss=torch.nn.functional.cross_entropy(torch.cat([-dp.unsqueeze(1),-dn],1),
             torch.zeros(len(u),dtype=torch.long,device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    X=X.detach()
    # (a) reconstruction rank of true parent among 500 random candidates
    ranks=[]
    for i in probe[:600]:
        cand=torch.tensor(np.append(rng.choice(N,499,replace=False),parent[i]),device=dev)
        d=dist(X[int(i)].unsqueeze(0),X[cand]).cpu().numpy()
        ranks.append(1.0/(1+int((d<d[-1]).sum())))
    mrr=float(np.mean(ranks))
    # (b) global tree distance
    ed=dist(X[torch.tensor(pa,device=dev)],X[torch.tensor(pb,device=dev)]).cpu().numpy()
    rho=spearmanr(td,ed).correlation
    print(f"{geom:>12}  MRR(parent-rank, PUBLISHED-style) {mrr:.3f}   tree-dist rho (OURS) {rho:.3f}",flush=True)
