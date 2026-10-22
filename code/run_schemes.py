import sys; sys.path.insert(0,'/tmp/rp1/v2')
from core import *
s=int(sys.argv[1])
y=D.outbreak_binary.values
# district-blocked
outer=[]
for k,(a,b) in enumerate(GroupKFold(5).split(D,y,D.district)): outer.append((f'd{k}',a,b,None))
r=run(outer,FALL,[('old',None),('gkf','district')],s,'district')
pickle.dump(r,open(f'results/district_{s}.pkl','wb'))
outer=[]
for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=s).split(D,y)): outer.append((f'r{k}',a,b,None))
r=run(outer,FALL,[('old',None),('strat',None)],s,'random')
pickle.dump(r,open(f'results/random_{s}.pkl','wb'))
