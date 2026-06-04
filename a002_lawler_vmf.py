#%%
def vmf_fun(xvec,yvec,zvec,probmasses_array):
    import numpy as np
    from scipy.optimize import fsolve
    nobj = len(xvec)
    Sx = np.sum(xvec)
    Sy = np.sum(yvec)
    Sz = np.sum(zvec)
    R = np.sqrt(Sx**2+Sy**2+Sz**2)
    xcc = Sx/R
    ycc = Sy/R
    zcc = Sz/R
    Rbar = R/nobj
    kappahat = (nobj-1)/(nobj-R)
    fun = lambda K: Rbar + 1/K - 1/np.tanh(K)
    Kout = fsolve(fun,kappahat)
    dsum = 0
    for iobj in range(nobj):
        dsum = dsum + (xvec[iobj]*xcc+yvec[iobj]*ycc+zvec[iobj]*zcc)**2
    d = 1 - 1/nobj * dsum
    sigmahat = np.sqrt(d/(nobj*Rbar**2))
    A = 1 - probmasses_array
    angledegs = np.degrees(np.arcsin(sigmahat*np.sqrt(-np.log(A))))
    iradcc = np.arccos(zcc)
    sini = np.sin(iradcc)
    Wradcc = np.arctan2(ycc/sini,xcc/sini)
    idegcc = np.degrees(iradcc)
    Wdegcc = np.degrees(Wradcc)
    return idegcc,Wdegcc,xcc,ycc,zcc,angledegs,sigmahat,Rbar,Kout[0],kappahat
#%%
import numpy as np
import pandas as pd
libs = ['gplus','gminus','gnone','gplusminusnone','gall']
probmasses_array = np.array([0.68,0.95,0.997])
infile_none = 'a000_lawler_StablePlutinos.txt'
df_none = pd.read_csv('a000_lawler_StablePlutinos.txt',delim_whitespace=True)
df_flippers = pd.read_csv('a000_lawler_KozaiFlippers.txt',delim_whitespace=True)
df_plusminus = pd.read_csv('a000_lawler_StableKozaiPlutinos.txt',delim_whitespace=True)
df_plusindex = pd.read_csv('b001_lawler_gplus_indices.txt')
plus_indices = df_plusindex['index'].to_list()
df_minusindex = pd.read_csv('b001_lawler_gminus_indices.txt')
minus_indices = df_minusindex['index'].to_list()
ideg_none = np.array(df_none['inc'].to_list())
Wdeg_none = np.array(df_none['Omega'].to_list())
q_none = np.sin(np.radians(ideg_none))*np.cos(np.radians(Wdeg_none))
p_none = np.sin(np.radians(ideg_none))*np.sin(np.radians(Wdeg_none))
s_none = np.cos(np.radians(ideg_none))
ideg_flippers = np.array(df_flippers['inc'].to_list())
Wdeg_flippers = np.array(df_flippers['Omega'].to_list())
q_flippers = np.sin(np.radians(ideg_flippers))*np.cos(np.radians(Wdeg_flippers))
p_flippers = np.sin(np.radians(ideg_flippers))*np.sin(np.radians(Wdeg_flippers))
s_flippers = np.cos(np.radians(ideg_flippers))
ideg_plusminus = np.array(df_plusminus['inc'].to_list())
Wdeg_plusminus = np.array(df_plusminus['Omega'].to_list())
q_plusminus = np.sin(np.radians(ideg_plusminus))*np.cos(np.radians(Wdeg_plusminus))
p_plusminus = np.sin(np.radians(ideg_plusminus))*np.sin(np.radians(Wdeg_plusminus))
s_plusminus = np.cos(np.radians(ideg_plusminus))
q_plusminusnone = np.hstack([q_plusminus,q_none])
p_plusminusnone = np.hstack([p_plusminus,p_none])
s_plusminusnone = np.hstack([s_plusminus,s_none])
q_all = np.hstack([q_plusminusnone,q_flippers])
p_all = np.hstack([p_plusminusnone,p_flippers])
s_all = np.hstack([s_plusminusnone,s_flippers])
q_plus = q_plusminus[plus_indices]
p_plus = p_plusminus[plus_indices]
s_plus = s_plusminus[plus_indices]
q_minus = q_plusminus[minus_indices]
p_minus = p_plusminus[minus_indices]
s_minus = s_plusminus[minus_indices]
xvecs = [q_plus,q_minus,q_none,q_plusminusnone,q_all]
yvecs = [p_plus,p_minus,p_none,p_plusminusnone,p_all]
zvecs = [s_plus,s_minus,s_none,s_plusminusnone,s_all]
idegcc_list = []
Wdegcc_list = []
qcc_list = []
pcc_list = []
scc_list = []
angledeg68_list = []
angledeg95_list = []
angledeg997_list = []
sigmahat_list = []
Rbar_list = []
Kout_list = []
kappahat_list = []
for ivec in range(len(xvecs)):
    idegcc,Wdegcc,xcc,ycc,zcc,angledegs,sigmahat,Rbar,Kout,kappahat = \
        vmf_fun(xvecs[ivec],yvecs[ivec],zvecs[ivec],probmasses_array)
    idegcc_list.append(idegcc)
    Wdegcc_list.append(Wdegcc)
    qcc_list.append(xcc)
    pcc_list.append(ycc)
    scc_list.append(zcc)
    angledeg68_list.append(angledegs[0])
    angledeg95_list.append(angledegs[1])
    angledeg997_list.append(angledegs[2])
    sigmahat_list.append(sigmahat)
    Rbar_list.append(Rbar)
    Kout_list.append(Kout)
    kappahat_list.append(kappahat)
dictionary = {'libs':libs,'idegcc':idegcc_list,'Wdegcc':Wdegcc_list,\
              'qcc':qcc_list,'pcc':pcc_list,'scc':scc_list,'angledeg68':angledeg68_list,\
              'angledeg95':angledeg95_list,'angledeg997':angledeg997_list,\
              'sigmahat':sigmahat_list,'Rbar':Rbar_list,'Kout':Kout_list,'kappahat':kappahat_list}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b002_lawler_vmf_results.csv',index=False)
