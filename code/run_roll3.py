import sys; sys.path.insert(0,'code')
from core import *
s=42
d=D[D.country!='DRC'].copy()
d['t']=d.groupby('country').apply(lambda g:(g.year*12+g.month)-(g.year*12+g.month).min()).reset_index(level=0,drop=True)
out={1:[],2:[],3:[]}
for T in range(5,int(d.t.max())-0):
    tr=d[d.t<=T]
    if tr.outbreak_binary.nunique()<2: continue
    tes={h:d[d.t==T+h] for h in (1,2,3)}
    if all(len(v)==0 for v in tes.values()): continue
    Xtr=tr[FALL].values; ytr=tr.outbreak_binary.values
    try: Pte=base_fp(Xtr,ytr,[tes[h][FALL].values if len(tes[h]) else np.zeros((0,len(FALL))) for h in (1,2,3)],s)
    except Exception as e: continue
    fb=0
    try: Po,yo=meta_oof(Xtr,ytr,'group',tr.country.values,s)
    except Exception:
        fb=1
        try: Po,yo=meta_oof(Xtr,ytr,'strat',None,s)
        except Exception: continue
    lr=LR(C=1,max_iter=2000).fit(Po,yo)
    for h,P in zip((1,2,3),Pte):
        if len(tes[h])==0: continue
        out[h].append({'T':T,'fb':fb,'y':tes[h].outbreak_binary.values,'p':lr.predict_proba(P)[:,1],'rf':P[:,1]})
    print('roll',T,flush=True)
pickle.dump(out,open('results/roll3.pkl','wb'))
