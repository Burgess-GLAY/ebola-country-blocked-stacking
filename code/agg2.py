import pickle, numpy as np, json, glob
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, cohen_kappa_score
out={}
def L(p): return {s:pickle.load(open(f'{p}_{s}.pkl','rb')) for s in (42,43,44) if glob.glob(f'{p}_{s}.pkl')}
def foldaucs(res,v): return [roc_auc_score(r['y'],r['meta'][v]['p']) for r in res]
# main for context
M=L('main'); out['country_group']=[float(np.mean(foldaucs(M[s],'group'))) for s in M]; out['country_old']=[float(np.mean(foldaucs(M[s],'old'))) for s in M]
out['country_sd']=[float(np.std(foldaucs(M[s],'group'),ddof=1)) for s in M]
A=L('abl')
if A:
    folds=[r['fold'] for r in M[42]]
    for v in ('group','old'):
        out['abl_'+v]={f:float(np.mean([foldaucs(A[s],v)[i] for s in A])) for i,f in enumerate(folds)}
        out['abl_'+v]['mean']=float(np.mean([np.mean(foldaucs(A[s],v)) for s in A]))
    out['full_group']={f:float(np.mean([foldaucs(M[s],'group')[i] for s in M])) for i,f in enumerate(folds)}; out['full_group']['mean']=float(np.mean(out['country_group']))
    out['abl_seed_means']=[float(np.mean(foldaucs(A[s],'group'))) for s in A]
for nm,v in (('district','gkf'),('random','strat')):
    S=L(nm)
    if S:
        out[nm+'_new']=[float(np.mean(foldaucs(S[s],v))) for s in S]; out[nm+'_old']=[float(np.mean(foldaucs(S[s],'old'))) for s in S]
        out[nm+'_sd']=[float(np.std(foldaucs(S[s],v),ddof=1)) for s in S]
if glob.glob('roll3.pkl'):
    R=pickle.load(open('roll3.pkl','rb')); out['roll']={}
    for h,W in R.items():
        y=np.concatenate([w['y'] for w in W]); p=np.concatenate([w['p'] for w in W]); yh=(p>=.5).astype(int); rp=np.concatenate([w['rf'] for w in W])
        ev=[w for w in W if len(set(w['y']))==2]
        wa=[roc_auc_score(w['y'],w['p']) for w in ev]
        out['roll'][h]={'n_windows':len(W),'n_evaluable':len(ev),'n_rows':int(len(y)),'n_pos':int(y.sum()),'auc_pooled':float(roc_auc_score(y,p)),'auc_sd_windows':float(np.std(wa,ddof=1)) if len(wa)>1 else None,'f1':float(f1_score(y,yh)),'brier':float(brier_score_loss(y,p)),'kappa':float(cohen_kappa_score(y,yh)),'rf_auc_pooled':float(roc_auc_score(y,rp)),'fallback_windows':int(sum(w['fb'] for w in W))}
json.dump(out,open('agg2.json','w'),indent=1); print(json.dumps(out,indent=1))
