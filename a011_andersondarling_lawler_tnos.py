#%%
def shift_to_pole(q_array,p_array,s_array,idegmean,Wdegmean):
    import numpy as np
    iradmean = np.radians(idegmean)
    Wradmean = np.radians(Wdegmean)
    A = np.array([[np.cos(iradmean)*np.cos(Wradmean),np.cos(iradmean)*np.sin(Wradmean),-np.sin(iradmean)],\
                              [-np.sin(Wradmean),np.cos(Wradmean),0],\
                              [np.sin(iradmean)*np.cos(Wradmean),np.sin(iradmean)*np.sin(Wradmean),np.cos(iradmean)]])
    qvec_pre = np.array([q_array,p_array,s_array])
    qvec_post = np.matmul(A,qvec_pre)
    qrel_array = qvec_post[0,:]
    prel_array = qvec_post[1,:]
    srel_array = qvec_post[2,:]
    irel_rad = np.arccos(srel_array)
    Wrel_rad = np.arctan2(prel_array/np.sin(irel_rad),qrel_array/np.sin(irel_rad))
    ireldeg = np.degrees(irel_rad)
    Wreldeg = np.degrees(Wrel_rad)
    return A,qrel_array,prel_array,srel_array,ireldeg,Wreldeg
#%%
import numpy as np
import pandas as pd
#%%
infile_lawler = 'b002_lawler_vmf_results.csv'
df_lawler = pd.read_csv(infile_lawler)
qmean_gplus_lawler = df_lawler['qcc'][0]
qmean_gminus_lawler = df_lawler['qcc'][1]
qmean_gplusminusnone_lawler = df_lawler['qcc'][3]
pmean_gplus_lawler = df_lawler['pcc'][0]
pmean_gminus_lawler = df_lawler['pcc'][1]
pmean_gplusminusnone_lawler = df_lawler['pcc'][3]
idegmean_gplus_lawler = df_lawler['idegcc'][0]
idegmean_gminus_lawler = df_lawler['idegcc'][1]
idegmean_gplusminusnone_lawler = df_lawler['idegcc'][3]
Wdegmean_gplus_lawler = df_lawler['Wdegcc'][0]
Wdegmean_gminus_lawler = df_lawler['Wdegcc'][1]
Wdegmean_gplusminusnone_lawler = df_lawler['Wdegcc'][3]
angledeg_gplus_lawler = df_lawler['angledeg95'][0]
angledeg_gminus_lawler = df_lawler['angledeg95'][1]
angledeg_gplusminusnone_lawler = df_lawler['angledeg95'][3]
sigma_gplus_lawler = df_lawler['sigmahat'][0]
sigma_gminus_lawler = df_lawler['sigmahat'][1]
sigma_gplusminusnone_lawler = df_lawler['sigmahat'][3]
kappa_gplus_lawler = df_lawler['Kout'][0]
kappa_gminus_lawler = df_lawler['Kout'][1]
kappa_gplusminusnone_lawler = df_lawler['Kout'][3]
smean_gplus_lawler = np.sqrt(1-qmean_gplus_lawler**2-pmean_gplus_lawler**2)
smean_gminus_lawler = np.sqrt(1-qmean_gminus_lawler**2-pmean_gminus_lawler**2)
smean_gplusminusnone_lawler = np.sqrt(1-qmean_gplusminusnone_lawler**2-pmean_gplusminusnone_lawler**2)
infile_none_lawler = 'a000_lawler_StablePlutinos.txt'
df_none_lawler = pd.read_csv('a000_lawler_StablePlutinos.txt',delim_whitespace=True)
df_plusminus_lawler = pd.read_csv('a000_lawler_StableKozaiPlutinos.txt',delim_whitespace=True)
df_plusindex_lawler = pd.read_csv('b001_lawler_gplus_indices.txt')
plus_indices_lawler = df_plusindex_lawler['index'].to_list()
df_minusindex_lawler = pd.read_csv('b001_lawler_gminus_indices.txt')
minus_indices_lawler = df_minusindex_lawler['index'].to_list()
ideg_none_lawler = np.array(df_none_lawler['inc'].to_list())
Wdeg_none_lawler = np.array(df_none_lawler['Omega'].to_list())
q_none_lawler = np.sin(np.radians(ideg_none_lawler))*np.cos(np.radians(Wdeg_none_lawler))
p_none_lawler = np.sin(np.radians(ideg_none_lawler))*np.sin(np.radians(Wdeg_none_lawler))
s_none_lawler = np.cos(np.radians(ideg_none_lawler))
ideg_plusminus_lawler = np.array(df_plusminus_lawler['inc'].to_list())
Wdeg_plusminus_lawler = np.array(df_plusminus_lawler['Omega'].to_list())
q_plusminus_lawler = np.sin(np.radians(ideg_plusminus_lawler))*np.cos(np.radians(Wdeg_plusminus_lawler))
p_plusminus_lawler = np.sin(np.radians(ideg_plusminus_lawler))*np.sin(np.radians(Wdeg_plusminus_lawler))
s_plusminus_lawler = np.cos(np.radians(ideg_plusminus_lawler))
q_array_gplusminusnone_lawler = np.hstack([q_plusminus_lawler,q_none_lawler])
p_array_gplusminusnone_lawler = np.hstack([p_plusminus_lawler,p_none_lawler])
s_array_gplusminusnone_lawler = np.hstack([s_plusminus_lawler,s_none_lawler])
q_array_gplus_lawler = q_plusminus_lawler[plus_indices_lawler]
p_array_gplus_lawler = p_plusminus_lawler[plus_indices_lawler]
s_array_gplus_lawler = s_plusminus_lawler[plus_indices_lawler]
q_array_gminus_lawler = q_plusminus_lawler[minus_indices_lawler]
p_array_gminus_lawler = p_plusminus_lawler[minus_indices_lawler]
s_array_gminus_lawler = s_plusminus_lawler[minus_indices_lawler]
A_gplus_lawler,qrel_gplus_lawler,prel_gplus_lawler,srel_gplus_lawler,ireldeg_gplus_lawler,Wreldeg_gplus_lawler = \
    shift_to_pole(q_array_gplus_lawler,p_array_gplus_lawler,s_array_gplus_lawler,idegmean_gplus_lawler,Wdegmean_gplus_lawler)
A_gminus_lawler,qrel_gminus_lawler,prel_gminus_lawler,srel_gminus_lawler,ireldeg_gminus_lawler,Wreldeg_gminus_lawler = \
    shift_to_pole(q_array_gminus_lawler,p_array_gminus_lawler,s_array_gminus_lawler,idegmean_gminus_lawler,Wdegmean_gminus_lawler)
A_gplusminusnone_lawler,qrel_gplusminusnone_lawler,prel_gplusminusnone_lawler,srel_gplusminusnone_lawler,ireldeg_gplusminusnone_lawler,Wreldeg_gplusminusnone_lawler = \
    shift_to_pole(q_array_gplusminusnone_lawler,p_array_gplusminusnone_lawler,s_array_gplusminusnone_lawler,idegmean_gplusminusnone_lawler,Wdegmean_gplusminusnone_lawler)
#%%
infile_plutinos = 'b004_tnos_orbels_jd246e4.csv'
df_plutinos = pd.read_csv(infile_plutinos)
n_plutinos = df_plutinos.shape[0]
des_list_plutinos = df_plutinos['mpc_des'].tolist()
aau_array_plutinos = np.array(df_plutinos['aau_bary'].tolist())
e_array_plutinos = np.array(df_plutinos['e_bary'].tolist())
irad_array_plutinos = np.radians(np.array(df_plutinos['ideg_bary'].tolist()))
wrad_array_plutinos = np.radians(np.array(df_plutinos['wdeg_bary'].tolist()))
Wrad_array_plutinos = np.radians(np.array(df_plutinos['Wdeg_bary'].tolist()))
Mrad_array_plutinos = np.radians(np.array(df_plutinos['Mdeg_bary'].tolist()))
q_array_plutinos = np.sin(irad_array_plutinos)*np.cos(Wrad_array_plutinos)
p_array_plutinos = np.sin(irad_array_plutinos)*np.sin(Wrad_array_plutinos)
s_array_plutinos = np.cos(irad_array_plutinos)
dfind_plutinos = pd.read_csv('b004_p3q2_gplus_2026feb12_index.csv')
indices_plutinos = dfind_plutinos['index'].to_list()
des_list_gplus_plutinos = []
for index in indices_plutinos:
    des_list_gplus_plutinos.append(des_list_plutinos[index])
aau_array_gplus_plutinos = aau_array_plutinos[indices_plutinos]
e_array_gplus_plutinos = e_array_plutinos[indices_plutinos]
irad_array_gplus_plutinos = irad_array_plutinos[indices_plutinos]
wrad_array_gplus_plutinos = wrad_array_plutinos[indices_plutinos]
Wrad_array_gplus_plutinos = Wrad_array_plutinos[indices_plutinos]
Mrad_array_gplus_plutinos = Mrad_array_plutinos[indices_plutinos]
q_array_gplus_plutinos = q_array_plutinos[indices_plutinos]
p_array_gplus_plutinos = p_array_plutinos[indices_plutinos]
s_array_gplus_plutinos = s_array_plutinos[indices_plutinos]
dfind_plutinos = pd.read_csv('b004_p3q2_gminus_2026feb12_index.csv')
indices_plutinos = dfind_plutinos['index'].to_list()
des_list_gminus_plutinos = []
for index in indices_plutinos:
    des_list_gminus_plutinos.append(des_list_plutinos[index])
aau_array_gminus_plutinos = aau_array_plutinos[indices_plutinos]
e_array_gminus_plutinos = e_array_plutinos[indices_plutinos]
irad_array_gminus_plutinos = irad_array_plutinos[indices_plutinos]
wrad_array_gminus_plutinos = wrad_array_plutinos[indices_plutinos]
Wrad_array_gminus_plutinos = Wrad_array_plutinos[indices_plutinos]
Mrad_array_gminus_plutinos = Mrad_array_plutinos[indices_plutinos]
q_array_gminus_plutinos = q_array_plutinos[indices_plutinos]
p_array_gminus_plutinos = p_array_plutinos[indices_plutinos]
s_array_gminus_plutinos = s_array_plutinos[indices_plutinos]
libration = 'gplusminusnone'
dfind_plutinos = pd.read_csv('b004_p3q2_gplusminusnone_2026feb12_index.csv')
indices_plutinos = dfind_plutinos['index'].to_list()
des_list_gplusminusnone_plutinos = []
for index in indices_plutinos:
    des_list_gplusminusnone_plutinos.append(des_list_plutinos[index])
aau_array_gplusminusnone_plutinos = aau_array_plutinos[indices_plutinos]
e_array_gplusminusnone_plutinos = e_array_plutinos[indices_plutinos]
irad_array_gplusminusnone_plutinos = irad_array_plutinos[indices_plutinos]
wrad_array_gplusminusnone_plutinos = wrad_array_plutinos[indices_plutinos]
Wrad_array_gplusminusnone_plutinos = Wrad_array_plutinos[indices_plutinos]
Mrad_array_gplusminusnone_plutinos = Mrad_array_plutinos[indices_plutinos]
q_array_gplusminusnone_plutinos = q_array_plutinos[indices_plutinos]
p_array_gplusminusnone_plutinos = p_array_plutinos[indices_plutinos]
s_array_gplusminusnone_plutinos = s_array_plutinos[indices_plutinos]
df_gplus_plutinos = pd.read_csv('b006_2026feb12_plutinos_gplus_siraj.csv')
idegmean_gplus_plutinos = df_gplus_plutinos['ideg_siraj'][0]
Wdegmean_gplus_plutinos = df_gplus_plutinos['Wdeg_siraj'][0]
iradmean_gplus_plutinos = np.radians(idegmean_gplus_plutinos)
Wradmean_gplus_plutinos = np.radians(Wdegmean_gplus_plutinos)
df_gminus_plutinos = pd.read_csv('b006_2026feb12_plutinos_gminus_siraj.csv')
idegmean_gminus_plutinos = df_gminus_plutinos['ideg_siraj'][0]
Wdegmean_gminus_plutinos = df_gminus_plutinos['Wdeg_siraj'][0]
iradmean_gminus_plutinos = np.radians(idegmean_gminus_plutinos)
Wradmean_gminus_plutinos = np.radians(Wdegmean_gminus_plutinos)
df_gplusminusnone_plutinos = pd.read_csv('b006_2026feb12_plutinos_gplusminusnone_siraj.csv')
idegmean_gplusminusnone_plutinos = df_gplusminusnone_plutinos['ideg_siraj'][0]
Wdegmean_gplusminusnone_plutinos = df_gplusminusnone_plutinos['Wdeg_siraj'][0]
iradmean_gplusminusnone_plutinos = np.radians(idegmean_gplusminusnone_plutinos)
Wradmean_gplusminusnone_plutinos = np.radians(Wdegmean_gplusminusnone_plutinos)
A_gplus_plutinos,qrel_gplus_plutinos,prel_gplus_plutinos,srel_gplus_plutinos,ireldeg_gplus_plutinos,Wreldeg_gplus_plutinos = \
    shift_to_pole(q_array_gplus_plutinos,p_array_gplus_plutinos,s_array_gplus_plutinos,idegmean_gplus_plutinos,Wdegmean_gplus_plutinos)
A_gminus_plutinos,qrel_gminus_plutinos,prel_gminus_plutinos,srel_gminus_plutinos,ireldeg_gminus_plutinos,Wreldeg_gminus_plutinos = \
    shift_to_pole(q_array_gminus_plutinos,p_array_gminus_plutinos,s_array_gminus_plutinos,idegmean_gminus_plutinos,Wdegmean_gminus_plutinos)
A_gplusminusnone_plutinos,qrel_gplusminusnone_plutinos,prel_gplusminusnone_plutinos,srel_gplusminusnone_plutinos,ireldeg_gplusminusnone_plutinos,Wreldeg_gplusminusnone_plutinos = \
    shift_to_pole(q_array_gplusminusnone_plutinos,p_array_gplusminusnone_plutinos,s_array_gplusminusnone_plutinos,idegmean_gplusminusnone_plutinos,Wdegmean_gplusminusnone_plutinos)
#%%
from scipy import stats
samples_gplus = [ireldeg_gplus_plutinos,ireldeg_gplus_lawler]
res_gplus = stats.anderson_ksamp(samples_gplus,method=stats.PermutationMethod())
res_gplus_statistic = res_gplus.statistic
res_gplus_pvalue = res_gplus.pvalue
print('gplus',res_gplus_pvalue)
samples_gminus = [ireldeg_gminus_plutinos,ireldeg_gminus_lawler]
res_gminus = stats.anderson_ksamp(samples_gminus,method=stats.PermutationMethod())
res_gminus_statistic = res_gminus.statistic
res_gminus_pvalue = res_gminus.pvalue
print('gminus',res_gminus_pvalue)
samples_gplusminusnone = [ireldeg_gplusminusnone_plutinos,ireldeg_gplusminusnone_lawler]
res_gplusminusnone = stats.anderson_ksamp(samples_gplusminusnone,method=stats.PermutationMethod())
res_gplusminusnone_statistic = res_gplusminusnone.statistic
res_gplusminusnone_pvalue = res_gplusminusnone.pvalue
print('gplusminusnone',res_gplusminusnone_pvalue)
