import sys; sys.path.insert(0,'code')
from core import *
s=int(sys.argv[1]); F=[f for f in FALL if f!='spatial_lag_cases']
r=run(country_outer(),F,[('old',None),('group','country')],s,'abl')
pickle.dump(r,open(f'results/abl_{s}.pkl','wb'))
