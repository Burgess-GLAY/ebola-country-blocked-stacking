import sys; sys.path.insert(0,'/tmp/rp1/v2')
from core import *
s=int(sys.argv[1])
res=run(country_outer(),FALL,[('old',None),('strat',None),('group','country')],s,'main')
pickle.dump(res,open(f'results/main_{s}.pkl','wb'))
