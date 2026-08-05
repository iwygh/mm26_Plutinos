#%%
import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt
#%%
libration = 'gplus'
df = pd.read_csv('b006_2026feb12_plutinos_'+libration+'_siraj.csv')
qmean_gplus = df['q_siraj'][0]
pmean_gplus = df['p_siraj'][0]
prob0_gplus = df['probmasses'][0]
ap_gplus = df['ap_siraj'][0]
bp_gplus = df['bp_siraj'][0]
ec_gplus = df['ec_siraj'][0]
phirad_gplus = df['phi_siraj'][0]
phideg_gplus = np.degrees(phirad_gplus)
chi2scale_gplus = np.sqrt(scipy.stats.chi2.isf(1-prob0_gplus,2))
sigma_a_gplus = ap_gplus/chi2scale_gplus
sigma_b_gplus = bp_gplus/chi2scale_gplus
lambda_a_gplus = sigma_a_gplus**2
lambda_b_gplus = sigma_b_gplus**2
D_gplus = np.diag([lambda_a_gplus,lambda_b_gplus])
rotation_matrix_gplus = np.array([[np.cos(phirad_gplus),-np.sin(phirad_gplus)],\
                            [np.sin(phirad_gplus), np.cos(phirad_gplus)]])
v1_gplus = np.matmul(rotation_matrix_gplus,np.array([sigma_a_gplus,0]))
v2_gplus = np.matmul(rotation_matrix_gplus,np.array([0,sigma_b_gplus]))
v1hat_gplus = v1_gplus/np.linalg.norm(v1_gplus)
v2hat_gplus = v2_gplus/np.linalg.norm(v2_gplus)
T_gplus = np.transpose(np.array([v1hat_gplus,v2hat_gplus]))
Sigma_gplus = np.matmul(T_gplus,np.matmul(D_gplus,np.transpose(T_gplus))) # covariance matrix

libration = 'gminus'
df = pd.read_csv('b006_2026feb12_plutinos_'+libration+'_siraj.csv')
qmean_gminus = df['q_siraj'][0]
pmean_gminus = df['p_siraj'][0]
prob0_gminus = df['probmasses'][0]
ap_gminus = df['ap_siraj'][0]
bp_gminus = df['bp_siraj'][0]
ec_gminus = df['ec_siraj'][0]
phirad_gminus = df['phi_siraj'][0]
phideg_gminus = np.degrees(phirad_gminus)
chi2scale_gminus = np.sqrt(scipy.stats.chi2.isf(1-prob0_gminus,2))
sigma_a_gminus = ap_gminus/chi2scale_gminus
sigma_b_gminus = bp_gminus/chi2scale_gminus
lambda_a_gminus = sigma_a_gminus**2
lambda_b_gminus = sigma_b_gminus**2
D_gminus = np.diag([lambda_a_gminus,lambda_b_gminus])
rotation_matrix_gminus = np.array([[np.cos(phirad_gminus),-np.sin(phirad_gminus)],\
                            [np.sin(phirad_gminus), np.cos(phirad_gminus)]])
v1_gminus = np.matmul(rotation_matrix_gminus,np.array([sigma_a_gminus,0]))
v2_gminus = np.matmul(rotation_matrix_gminus,np.array([0,sigma_b_gminus]))
v1hat_gminus = v1_gminus/np.linalg.norm(v1_gminus)
v2hat_gminus = v2_gminus/np.linalg.norm(v2_gminus)
T_gminus = np.transpose(np.array([v1hat_gminus,v2hat_gminus]))
Sigma_gminus = np.matmul(T_gminus,np.matmul(D_gminus,np.transpose(T_gminus))) # covariance matrix

libration = 'gplusminusnone'
df = pd.read_csv('b006_2026feb12_plutinos_'+libration+'_siraj.csv')
qmean_gplusminusnone = df['q_siraj'][0]
pmean_gplusminusnone = df['p_siraj'][0]
prob0_gplusminusnone = df['probmasses'][0]
ap_gplusminusnone = df['ap_siraj'][0]
bp_gplusminusnone = df['bp_siraj'][0]
ec_gplusminusnone = df['ec_siraj'][0]
phirad_gplusminusnone = df['phi_siraj'][0]
phideg_gplusminusnone = np.degrees(phirad_gplusminusnone)
chi2scale_gplusminusnone = np.sqrt(scipy.stats.chi2.isf(1-prob0_gplusminusnone,2))
sigma_a_gplusminusnone = ap_gplusminusnone/chi2scale_gplusminusnone
sigma_b_gplusminusnone = bp_gplusminusnone/chi2scale_gplusminusnone
lambda_a_gplusminusnone = sigma_a_gplusminusnone**2
lambda_b_gplusminusnone = sigma_b_gplusminusnone**2
D_gplusminusnone = np.diag([lambda_a_gplusminusnone,lambda_b_gplusminusnone])
rotation_matrix_gplusminusnone = np.array([[np.cos(phirad_gplusminusnone),-np.sin(phirad_gplusminusnone)],\
                            [np.sin(phirad_gplusminusnone), np.cos(phirad_gplusminusnone)]])
v1_gplusminusnone = np.matmul(rotation_matrix_gplusminusnone,np.array([sigma_a_gplusminusnone,0]))
v2_gplusminusnone = np.matmul(rotation_matrix_gplusminusnone,np.array([0,sigma_b_gplusminusnone]))
v1hat_gplusminusnone = v1_gplusminusnone/np.linalg.norm(v1_gplusminusnone)
v2hat_gplusminusnone = v2_gplusminusnone/np.linalg.norm(v2_gplusminusnone)
T_gplusminusnone = np.transpose(np.array([v1hat_gplusminusnone,v2hat_gplusminusnone]))
Sigma_gplusminusnone = np.matmul(T_gplusminusnone,np.matmul(D_gplusminusnone,np.transpose(T_gplusminusnone))) # covariance matrix

qmean_diff_gplus_gminus = qmean_gplus - qmean_gminus
pmean_diff_gplus_gminus = pmean_gplus - pmean_gminus
Sigma_combined_gplus_gminus = Sigma_gplus + Sigma_gminus # covariance matrix of combined random variable
mu_diff_gplus_gminus = np.array([qmean_diff_gplus_gminus,pmean_diff_gplus_gminus])
prod_gplus_gminus = np.matmul(np.linalg.inv(Sigma_combined_gplus_gminus),mu_diff_gplus_gminus)
prod2_gplus_gminus = np.matmul(np.transpose(mu_diff_gplus_gminus),prod_gplus_gminus)
mahalanobis_gplus_gminus = np.sqrt(prod2_gplus_gminus)
probmass_inside_gplus_gminus = 1 - np.exp(mahalanobis_gplus_gminus**2 / -2)
probmass_outside_gplus_gminus = 1 - probmass_inside_gplus_gminus

qmean_diff_gplus_gplusminusnone = qmean_gplus - qmean_gplusminusnone
pmean_diff_gplus_gplusminusnone = pmean_gplus - pmean_gplusminusnone
Sigma_combined_gplus_gplusminusnone = Sigma_gplus + Sigma_gplusminusnone # covariance matrix of combined random variable
mu_diff_gplus_gplusminusnone = np.array([qmean_diff_gplus_gplusminusnone,pmean_diff_gplus_gplusminusnone])
prod_gplus_gplusminusnone = np.matmul(np.linalg.inv(Sigma_combined_gplus_gplusminusnone),mu_diff_gplus_gplusminusnone)
prod2_gplus_gplusminusnone = np.matmul(np.transpose(mu_diff_gplus_gplusminusnone),prod_gplus_gplusminusnone)
mahalanobis_gplus_gplusminusnone = np.sqrt(prod2_gplus_gplusminusnone)
probmass_inside_gplus_gplusminusnone = 1 - np.exp(mahalanobis_gplus_gplusminusnone**2 / -2)
probmass_outside_gplus_gplusminusnone = 1 - probmass_inside_gplus_gplusminusnone

qmean_diff_gplusminusnone_gminus = qmean_gplusminusnone - qmean_gminus
pmean_diff_gplusminusnone_gminus = pmean_gplusminusnone - pmean_gminus
Sigma_combined_gplusminusnone_gminus = Sigma_gplusminusnone + Sigma_gminus # covariance matrix of combined random variable
mu_diff_gplusminusnone_gminus = np.array([qmean_diff_gplusminusnone_gminus,pmean_diff_gplusminusnone_gminus])
prod_gplusminusnone_gminus = np.matmul(np.linalg.inv(Sigma_combined_gplusminusnone_gminus),mu_diff_gplusminusnone_gminus)
prod2_gplusminusnone_gminus = np.matmul(np.transpose(mu_diff_gplusminusnone_gminus),prod_gplusminusnone_gminus)
mahalanobis_gplusminusnone_gminus = np.sqrt(prod2_gplusminusnone_gminus)
probmass_inside_gplusminusnone_gminus = 1 - np.exp(mahalanobis_gplusminusnone_gminus**2 / -2)
probmass_outside_gplusminusnone_gminus = 1 - probmass_inside_gplusminusnone_gminus
#%%
infile = 'b004_tnos_orbels_jd246e4.csv'
df = pd.read_csv(infile)
n = df.shape[0]
des_list = df['mpc_des'].tolist()
aau_array = np.array(df['aau_bary'].tolist())
e_array = np.array(df['e_bary'].tolist())
irad_array = np.radians(np.array(df['ideg_bary'].tolist()))
wrad_array = np.radians(np.array(df['wdeg_bary'].tolist()))
Wrad_array = np.radians(np.array(df['Wdeg_bary'].tolist()))
Mrad_array = np.radians(np.array(df['Mdeg_bary'].tolist()))
q_array = np.sin(irad_array)*np.cos(Wrad_array)
p_array = np.sin(irad_array)*np.sin(Wrad_array)
s_array = np.cos(irad_array)
dfind = pd.read_csv('b004_p3q2_gplus_2026feb12_index.csv')
indices = dfind['index'].to_list()
des_list_gplus = []
for index in indices:
    des_list_gplus.append(des_list[index])
aau_array_gplus = aau_array[indices]
e_array_gplus = e_array[indices]
irad_array_gplus = irad_array[indices]
wrad_array_gplus = wrad_array[indices]
Wrad_array_gplus = Wrad_array[indices]
Mrad_array_gplus = Mrad_array[indices]
q_array_gplus = q_array[indices]
p_array_gplus = p_array[indices]
s_array_gplus = s_array[indices]
dfind = pd.read_csv('b004_p3q2_gminus_2026feb12_index.csv')
indices = dfind['index'].to_list()
des_list_gminus = []
for index in indices:
    des_list_gminus.append(des_list[index])
aau_array_gminus = aau_array[indices]
e_array_gminus = e_array[indices]
irad_array_gminus = irad_array[indices]
wrad_array_gminus = wrad_array[indices]
Wrad_array_gminus = Wrad_array[indices]
Mrad_array_gminus = Mrad_array[indices]
q_array_gminus = q_array[indices]
p_array_gminus = p_array[indices]
s_array_gminus = s_array[indices]
libration = 'gplusminusnone'
dfind = pd.read_csv('b004_p3q2_gplusminusnone_2026feb12_index.csv')
indices = dfind['index'].to_list()
des_list_gplusminusnone = []
for index in indices:
    des_list_gplusminusnone.append(des_list[index])
aau_array_gplusminusnone = aau_array[indices]
e_array_gplusminusnone = e_array[indices]
irad_array_gplusminusnone = irad_array[indices]
wrad_array_gplusminusnone = wrad_array[indices]
Wrad_array_gplusminusnone = Wrad_array[indices]
Mrad_array_gplusminusnone = Mrad_array[indices]
q_array_gplusminusnone = q_array[indices]
p_array_gplusminusnone = p_array[indices]
s_array_gplusminusnone = s_array[indices]
Sx_gplus = np.sum(q_array_gplus)
Sy_gplus = np.sum(p_array_gplus)
Sz_gplus = np.sum(s_array_gplus)
R_gplus = np.linalg.norm(np.array([Sx_gplus,Sy_gplus,Sz_gplus]))
qmean_gplus = Sx_gplus/R_gplus
pmean_gplus = Sy_gplus/R_gplus
smean_gplus = Sz_gplus/R_gplus
Sx_gminus = np.sum(q_array_gminus)
Sy_gminus = np.sum(p_array_gminus)
Sz_gminus = np.sum(s_array_gminus)
R_gminus = np.linalg.norm(np.array([Sx_gminus,Sy_gminus,Sz_gminus]))
qmean_gminus = Sx_gminus/R_gminus
pmean_gminus = Sy_gminus/R_gminus
smean_gminus = Sz_gminus/R_gminus
Sx_gplusminusnone = np.sum(q_array_gplusminusnone)
Sy_gplusminusnone = np.sum(p_array_gplusminusnone)
Sz_gplusminusnone = np.sum(s_array_gplusminusnone)
R_gplusminusnone = np.linalg.norm(np.array([Sx_gplusminusnone,Sy_gplusminusnone,Sz_gplusminusnone]))
qmean_gplusminusnone = Sx_gplusminusnone/R_gplusminusnone
pmean_gplusminusnone = Sy_gplusminusnone/R_gplusminusnone
smean_gplusminusnone = Sz_gplusminusnone/R_gplusminusnone
qmean_diff_gplus_gminus = qmean_gplus - qmean_gminus
pmean_diff_gplus_gminus = pmean_gplus - pmean_gminus
qmean_diff_gplus_gplusminusnone = qmean_gplus - qmean_gplusminusnone
pmean_diff_gplus_gplusminusnone = pmean_gplus - pmean_gplusminusnone
qmean_diff_gplusminusnone_gminus = qmean_gplusminusnone - qmean_gminus
pmean_diff_gplusminusnone_gminus = pmean_gplusminusnone - pmean_gminus
n_gplus = len(q_array_gplus)
n_gminus = len(q_array_gminus)
n_gplusminusnone = len(q_array_gplusminusnone)
S_gplus = np.cov(q_array_gplus,p_array_gplus)
S_gminus = np.cov(q_array_gminus,p_array_gminus)
S_gplusminusnone = np.cov(q_array_gplusminusnone,p_array_gplusminusnone)
Stilde_gplus = S_gplus/n_gplus
Stilde_gminus = S_gminus/n_gminus
Stilde_gplusminusnone = S_gplusminusnone/n_gplusminusnone
Spooled_gplus_gminus = Stilde_gplus + Stilde_gminus
Spooled_gplus_gplusminusnone = Stilde_gplus + Stilde_gplusminusnone
Spooled_gplusminusnone_gminus = Stilde_gplusminusnone + Stilde_gminus
p1_gplus_gminus = np.matmul(np.linalg.inv(Spooled_gplus_gminus),mu_diff_gplus_gminus)
p1_gplus_gplusminusnone = np.matmul(np.linalg.inv(Spooled_gplus_gplusminusnone),mu_diff_gplus_gplusminusnone)
p1_gplusminusnone_gminus = np.matmul(np.linalg.inv(Spooled_gplusminusnone_gminus),mu_diff_gplusminusnone_gminus)
T2_gplus_gminus = np.matmul(np.transpose(mu_diff_gplus_gminus),p1_gplus_gminus)
T2_gplus_gplusminusnone = np.matmul(np.transpose(mu_diff_gplus_gplusminusnone),p1_gplus_gplusminusnone)
T2_gplusminusnone_gminus = np.matmul(np.transpose(mu_diff_gplusminusnone_gminus),p1_gplusminusnone_gminus)
pdof = 2
trace1_gplus_gminus = np.linalg.trace(np.matmul(Spooled_gplus_gminus,Spooled_gplus_gminus))
trace2_gplus_gminus = np.linalg.trace(Spooled_gplus_gminus)
top_gplus_gminus = trace1_gplus_gminus + trace2_gplus_gminus**2
term1_gplus_gminus = 1/(n_gplus-1) * (np.linalg.trace(np.matmul(Stilde_gplus,Stilde_gplus))+np.linalg.trace(Stilde_gplus)**2)
term2_gplus_gminus = 1/(n_gminus-1) * (np.linalg.trace(np.matmul(Stilde_gminus,Stilde_gminus))+np.linalg.trace(Stilde_gminus)**2)
bottom_gplus_gminus = term1_gplus_gminus + term2_gplus_gminus
vdof_gplus_gminus = top_gplus_gminus/bottom_gplus_gminus
trace1_gplus_gplusminusnone = np.linalg.trace(np.matmul(Spooled_gplus_gplusminusnone,Spooled_gplus_gplusminusnone))
trace2_gplus_gplusminusnone = np.linalg.trace(Spooled_gplus_gplusminusnone)
top_gplus_gplusminusnone = trace1_gplus_gplusminusnone + trace2_gplus_gplusminusnone**2
term1_gplus_gplusminusnone = 1/(n_gplus-1) * (np.linalg.trace(np.matmul(Stilde_gplus,Stilde_gplus))+np.linalg.trace(Stilde_gplus)**2)
term2_gplus_gplusminusnone = 1/(n_gplusminusnone-1) * (np.linalg.trace(np.matmul(Stilde_gplusminusnone,Stilde_gplusminusnone))+np.linalg.trace(Stilde_gplusminusnone)**2)
bottom_gplus_gplusminusnone = term1_gplus_gplusminusnone + term2_gplus_gplusminusnone
vdof_gplus_gplusminusnone = top_gplus_gplusminusnone/bottom_gplus_gplusminusnone
trace1_gplusminusnone_gminus = np.linalg.trace(np.matmul(Spooled_gplusminusnone_gminus,Spooled_gplusminusnone_gminus))
trace2_gplusminusnone_gminus = np.linalg.trace(Spooled_gplusminusnone_gminus)
top_gplusminusnone_gminus = trace1_gplusminusnone_gminus + trace2_gplusminusnone_gminus**2
term1_gplusminusnone_gminus = 1/(n_gplusminusnone-1) * (np.linalg.trace(np.matmul(Stilde_gplusminusnone,Stilde_gplusminusnone))+np.linalg.trace(Stilde_gplusminusnone)**2)
term2_gplusminusnone_gminus = 1/(n_gminus-1) * (np.linalg.trace(np.matmul(Stilde_gminus,Stilde_gminus))+np.linalg.trace(Stilde_gminus)**2)
bottom_gplusminusnone_gminus = term1_gplusminusnone_gminus + term2_gplusminusnone_gminus
vdof_gplusminusnone_gminus = top_gplusminusnone_gminus/bottom_gplusminusnone_gminus
Fval_gplus_gminus = T2_gplus_gminus * (vdof_gplus_gminus-pdof+1) / (vdof_gplus_gminus*pdof)
Fval_gplus_gplusminusnone = T2_gplus_gplusminusnone * (vdof_gplus_gplusminusnone-pdof+1) / (vdof_gplus_gplusminusnone*pdof)
Fval_gplusminusnone_gminus = T2_gplusminusnone_gminus * (vdof_gplusminusnone_gminus-pdof+1) / (vdof_gplusminusnone_gminus*pdof)
Fdof1 = pdof
Fdof2_gplus_gminus = vdof_gplus_gminus-pdof+1
Fdof2_gplus_gplusminusnone = vdof_gplus_gplusminusnone-pdof+1
Fdof2_gplusminusnone_gminus = vdof_gplusminusnone_gminus-pdof+1
from scipy.stats import f
F_gplus_gminus = f(Fdof1,Fdof2_gplus_gminus)
F_gplus_gplusminusnone = f(Fdof1,Fdof2_gplus_gplusminusnone)
F_gplusminusnone_gminus = f(Fdof1,Fdof2_gplusminusnone_gminus)
pval_gplus_gminus = 1 - F_gplus_gminus.cdf(Fval_gplus_gminus)
pval_gplus_gplusminusnone = 1 - F_gplus_gplusminusnone.cdf(Fval_gplus_gplusminusnone)
pval_gplusminusnone_gminus = 1 - F_gplusminusnone_gminus.cdf(Fval_gplusminusnone_gminus)
print(pval_gplus_gminus)
print(pval_gplus_gplusminusnone)
print(pval_gplusminusnone_gminus)
#%%
import numpy as np
import pandas as pd
from pingouin import multivariate_normality
result_gplus = multivariate_normality(np.array([q_array_gplus,p_array_gplus,s_array_gplus]),alpha=0.05)
print(f"HZ Statistic gplus: {result_gplus.hz}")
print(f"P-value gplus:      {result_gplus.pval}")
print(f"Normal gplus?       {result_gplus.normal}")
result_gminus = multivariate_normality(np.array([q_array_gminus,p_array_gminus,s_array_gminus]),alpha=0.05)
print(f"HZ Statistic gminus: {result_gminus.hz}")
print(f"P-value gminus:      {result_gminus.pval}")
print(f"Normal gminus?       {result_gminus.normal}")
result_gplusminusnone = multivariate_normality(np.array([q_array_gplusminusnone,p_array_gplusminusnone,s_array_gplusminusnone]),alpha=0.05)
print(f"HZ Statistic gplusminusnone: {result_gplusminusnone.hz}")
print(f"P-value gplusminusnone:      {result_gplusminusnone.pval}")
print(f"Normal gplusminusnone?       {result_gplusminusnone.normal}")

# # 1. Generate fake multivariate normal data
# np.random.seed(42)
# mean = [0, 0, 0]
# covariance = [[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]]
# data = np.random.multivariate_normal(mean, covariance, size=100)
# # Convert to DataFrame (or leave as NumPy array)
# df = pd.DataFrame(data, columns=['Var1', 'Var2', 'Var3'])
# # 2. Perform the Multivariate Normality Test
# result = multivariate_normality(df, alpha=0.05)
# # 3. Print results
# print(f"HZ Statistic: {result.hz}")
# print(f"P-value:      {result.pval}")
# print(f"Normal?       {result.normal}")

