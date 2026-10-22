import pandas as pd, numpy as np, warnings, pickle, sys, os
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingClassifier as GB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut, cross_val_predict
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier as XG
from joblib import Parallel, delayed
D=pd.read_csv('data/evd_final_feature_matrix.csv')
D['district']=D['country']+'|'+D['shapeName'].astype(str)
FALL=['era5_tp','era5_t2m','era5_swvl1','worldpop_pop_density','gfw_loss_rate','gfw_treecover2000','prior_outbreak_12m','spatial_lag_cases','month_sin','month_cos']
def models(s):
    return [XG(n_estimators=350,max_depth=2,learning_rate=0.187,subsample=0.707,colsample_bytree=0.85,random_state=s,eval_metric='logloss',n_jobs=1),
            RF(n_estimators=600,random_state=s,n_jobs=1),
            GB(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=s)]
def base_fp(Xtr,ytr,Xtes,s):
    X,y=SMOTEENN(random_state=s).fit_resample(Xtr,ytr)
    outs=[[] for _ in Xtes]
    for m in models(s):
        m.fit(X,y)
        for i,Xt in enumerate(Xtes): outs[i].append(m.predict_proba(Xt)[:,1])
    return [np.c_[tuple(o)] for o in outs]
def _inner(Xtr,ytr,tri,vai,s):
    return vai, base_fp(Xtr[tri],ytr[tri],[Xtr[vai]],s)[0]
def meta_oof(Xtr,ytr,mode,groups,s):
    n=len(ytr)
    if mode=='old':
        X,y=SMOTEENN(random_state=s).fit_resample(Xtr,ytr)
        P=np.c_[tuple(cross_val_predict(m,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=s),method='predict_proba')[:,1] for m in models(s))]
        return P,y
    if mode=='strat': sp=list(StratifiedKFold(5,shuffle=True,random_state=s).split(Xtr,ytr))
    elif mode=='group': sp=list(LeaveOneGroupOut().split(Xtr,ytr,groups))
    elif mode=='gkf': sp=list(GroupKFold(5).split(Xtr,ytr,groups))
    r=Parallel(n_jobs=2)(delayed(_inner)(Xtr,ytr,a,b,s) for a,b in sp)
    P=np.zeros((n,3))
    for vai,p in r: P[vai]=p
    return P,ytr
def run(outer, feats, variants, s, tag):
    """outer: list of (name, train_idx, test_idx, inner_group_array or None per variant mode)"""
    res=[]
    for name,tr,te,grp in outer:
        dtr=D.iloc[tr]; dte=D.iloc[te]
        Xtr=dtr[feats].values; ytr=dtr.outbreak_binary.values; Xte=dte[feats].values
        Pte=base_fp(Xtr,ytr,[Xte],s)[0]
        row={'fold':name,'y':dte.outbreak_binary.values,'base':Pte,'idx':te,'meta':{}}
        for v,gk in variants:
            g=None if gk is None else dtr[gk].values
            Po,yo=meta_oof(Xtr,ytr,v,g,s)
            lr=LR(C=1,max_iter=2000).fit(Po,yo)
            row['meta'][v]={'p':lr.predict_proba(Pte)[:,1],'coef':lr.coef_[0].tolist(),'b':float(lr.intercept_[0])}
        # linear baselines
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        cols0=[c for c in feats if c!='spatial_lag_cases']
        for nm,cs in (('lr',cols0),('lr_sl',feats)):
            ii=[feats.index(c) for c in cs if c in feats]
            m=make_pipeline(StandardScaler(),LR(max_iter=2000)); m.fit(Xtr[:,ii],ytr); row[nm]=m.predict_proba(Xte[:,ii])[:,1]
        res.append(row); print(tag,s,name,flush=True)
    return res
def country_outer():
    return [(c,np.where(D.country!=c)[0],np.where(D.country==c)[0],None) for c in ['Guinea','Liberia','Sierra Leone','Uganda','DRC']]
