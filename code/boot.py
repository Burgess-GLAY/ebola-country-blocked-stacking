import pickle, numpy as np, json
from sklearn.metrics import roc_auc_score
R=pickle.load(open('main_42.pkl','rb')); rng=np.random.RandomState(0)
B=2000; folds=[r['fold'] for r in R]
auc={f:{'stack':[],'rf':[]} for f in folds}
for r in R:
    y=r['y']; ps=r['meta']['group']['p']; pr=r['base'][:,1]
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    for b in range(B):
        i=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
        auc[r['fold']]['stack'].append(roc_auc_score(y[i],ps[i])); auc[r['fold']]['rf'].append(roc_auc_score(y[i],pr[i]))
out={'folds':{}}
for f in folds:
    s=np.array(auc[f]['stack']); d=s-np.array(auc[f]['rf'])
    out['folds'][f]={'stack':[float(np.mean(s)),*np.percentile(s,[2.5,97.5]).tolist()],'diff':[float(np.mean(d)),*np.percentile(d,[2.5,97.5]).tolist()]}
S=np.mean([auc[f]['stack'] for f in folds],axis=0); Rr=np.mean([auc[f]['rf'] for f in folds],axis=0)
d=S-Rr; out['mean_diff']=[float(d.mean()),*np.percentile(d,[2.5,97.5]).tolist()]
out['mean_stack']=[float(S.mean()),*np.percentile(S,[2.5,97.5]).tolist()]
# point estimate in this run
pt={r['fold']:(roc_auc_score(r['y'],r['meta']['group']['p']),roc_auc_score(r['y'],r['base'][:,1])) for r in R}
out['point']=pt; out['point_mean_diff']=float(np.mean([a-b for a,b in pt.values()]))
json.dump(out,open('boot.json','w'),indent=1); print(json.dumps(out,indent=1))
