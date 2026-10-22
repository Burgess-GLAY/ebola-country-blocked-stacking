import pickle, numpy as np, glob, json
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, cohen_kappa_score
def M(y,p):
    yh=(p>=.5).astype(int)
    return dict(auc=roc_auc_score(y,p),f1=f1_score(y,yh,zero_division=0),pr=average_precision_score(y,p),br=brier_score_loss(y,p),k=cohen_kappa_score(y,yh))
def models(row,variant):
    o={'lr':row['lr'],'lr_sl':row['lr_sl'],'xgb':row['base'][:,0],'rf':row['base'][:,1],'gb':row['base'][:,2]}
    for v in row['meta']: o['stack_'+v]=row['meta'][v]['p']
    return o
def load(prefix,seeds=(42,43,44)):
    return {s:pickle.load(open(f'results/{prefix}_{s}.pkl','rb')) for s in seeds if glob.glob(f'results/{prefix}_{s}.pkl')}
if __name__=='__main__':
    R=load('main'); print('seeds',list(R))
    names=None; tab={}; pc={}
    for s,res in R.items():
        for row in res:
            for m,p in models(row,None).items():
                x=M(row['y'],p); 
                for k,v in x.items(): tab.setdefault((m,k),{}).setdefault(row['fold'],[]).append(v)
    ms=sorted({m for m,_ in tab})
    folds=['Guinea','Liberia','Sierra Leone','Uganda','DRC']
    for m in ms:
        line=[]
        for k in ('auc','f1','pr','br','k'):
            per=[np.mean(tab[(m,k)][f]) for f in folds]
            line.append(f'{np.mean(per):.3f}±{np.std(per,ddof=1):.3f}')
        print(f'{m:14s}',*line, '| per-country AUC', [round(float(np.mean(tab[(m,'auc')][f])),3) for f in folds])
    # seed means for stack variants
    for v in ('old','strat','group'):
        sm=[np.mean([np.mean(tab[('stack_'+v,'auc')][f][i]) for f in folds]) for i in range(len(R))]
        print(v,'seed means',np.round(sm,3))
    coef=[(row['fold'],{v:np.round(row['meta'][v]['coef'],2).tolist() for v in row['meta']}) for row in R[42]]
    print(coef)
