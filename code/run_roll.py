import sys; sys.path.insert(0,'/tmp/rp1/v2')
from core import *
from sklearn.metrics import roc_auc_score
s=42
d=D[D.country!='DRC'].copy()
d['t']=d.groupby('country').apply(lambda g:(g.year*12+g.month)-(g.year*12+g.month).min()).reset_index(level=0,drop=True)
out={}
for h in (1,2,3):
    W=[]
    for T in range(5,int(d.t.max())):
        tr=d[d.t<=T]; te=d[d.t==T+h]
        if len(te)<5 or te.outbreak_binary.nunique()<2 or tr.outbreak_binary.nunique()<2: continue
        Xtr=tr[FALL].values; ytr=tr.outbreak_binary.values
        try:
            Pte=base_fp(Xtr,ytr,[te[FALL].values],s)[0]
            Po,yo=meta_oof(Xtr,ytr,'group',tr.country.values,s)
            if len(np.unique(tr.country))<3: raise ValueError
        except Exception as e:
            continue
        lr=LR(C=1,max_iter=2000).fit(Po,yo)
        W.append({'T':T,'y':te.outbreak_binary.values,'p':lr.predict_proba(Pte)[:,1],'rf':Pte[:,1]})
        print('roll',h,T,flush=True)
    out[h]=W
pickle.dump(out,open('results/roll.pkl','wb'))
