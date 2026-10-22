import sys; sys.path.insert(0,'/tmp/rp1/v2')
from core import *
import shap
out={}
X=D[FALL].values; y=D.outbreak_binary.values
for s in (42,43,44):
    Po,yo=meta_oof(X,y,'group',D.country.values,s)
    lr=LR(C=1,max_iter=2000).fit(Po,yo)
    bg=shap.kmeans(Po,30)
    f=lambda z: lr.predict_proba(z)[:,1]
    ex=shap.KernelExplainer(f,bg)
    sv=ex.shap_values(Po[np.random.RandomState(s).choice(len(Po),600,replace=False)],nsamples=200)
    m=np.abs(sv).mean(0)
    out[s]={'mean_abs':m.tolist(),'share':(m/m.sum()).tolist(),'coef':lr.coef_[0].tolist(),'b':float(lr.intercept_[0]),'corr':np.corrcoef(Po.T).round(3).tolist()}
    # old (leaky) for comparison
    Pl,yl=meta_oof(X,y,'old',None,s); lr2=LR(C=1,max_iter=2000).fit(Pl,yl)
    ex2=shap.KernelExplainer(lambda z: lr2.predict_proba(z)[:,1],shap.kmeans(Pl,30))
    sv2=ex2.shap_values(Pl[np.random.RandomState(s).choice(len(Pl),600,replace=False)],nsamples=200); m2=np.abs(sv2).mean(0)
    out[s]['old']={'mean_abs':m2.tolist(),'share':(m2/m2.sum()).tolist(),'coef':lr2.coef_[0].tolist()}
    print(s,out[s]['share'],out[s]['old']['share'],flush=True)
    if s==42: np.save('results/Po42.npy',Po)
json.dump(out,open('results/l2.json','w')) if (json:=__import__('json')) else 0
