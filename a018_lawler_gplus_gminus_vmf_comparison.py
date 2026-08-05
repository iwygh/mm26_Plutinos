#%%
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import chi2
#%%
df_plusminus = pd.read_csv('a000_lawler_StableKozaiPlutinos.txt',delim_whitespace=True)
df_plusindex = pd.read_csv('b001_lawler_gplus_indices.txt')
plus_indices = df_plusindex['index'].to_list()
df_minusindex = pd.read_csv('b001_lawler_gminus_indices.txt')
minus_indices = df_minusindex['index'].to_list()
ideg_plusminus = np.array(df_plusminus['inc'].to_list())
Wdeg_plusminus = np.array(df_plusminus['Omega'].to_list())
q_plusminus = np.sin(np.radians(ideg_plusminus))*np.cos(np.radians(Wdeg_plusminus))
p_plusminus = np.sin(np.radians(ideg_plusminus))*np.sin(np.radians(Wdeg_plusminus))
s_plusminus = np.cos(np.radians(ideg_plusminus))
q_plus = q_plusminus[plus_indices]
p_plus = p_plusminus[plus_indices]
s_plus = s_plusminus[plus_indices]
q_minus = q_plusminus[minus_indices]
p_minus = p_plusminus[minus_indices]
s_minus = s_plusminus[minus_indices]
df_none = pd.read_csv('a000_lawler_StablePlutinos.txt',delim_whitespace=True)
ideg_none = np.array(df_none['inc'].to_list())
Wdeg_none = np.array(df_none['Omega'].to_list())
q_none = np.sin(np.radians(ideg_none))*np.cos(np.radians(Wdeg_none))
p_none = np.sin(np.radians(ideg_none))*np.sin(np.radians(Wdeg_none))
s_none = np.cos(np.radians(ideg_none))
q_plusminusnone = np.hstack([q_plusminus,q_none])
p_plusminusnone = np.hstack([p_plusminus,p_none])
s_plusminusnone = np.hstack([s_plusminus,s_none])
# sample data from rumcheva and presnell 2017 doi:10.1111/anzs.12183
# q_plus = np.array([+0.551,+0.456,+0.439,+0.178,+0.513,+0.526,+0.177,+0.406,+0.436,\
#                    +0.457,+0.519,+0.274,+0.456,+0.309,+0.611])
# p_plus = np.array([-0.794,-0.872,-0.818,-0.974,-0.761,-0.598,-0.967,-0.869,-0.755,\
#                    -0.804,-0.824,-0.899,-0.879,-0.925,-0.745])
# s_plus = np.array([-0.257,-0.177,-0.371,-0.143,-0.397,-0.605,-0.185,-0.281,-0.490,\
#                    -0.381,-0.227,-0.340,-0.138,-0.222,-0.268])
# q_minus = np.array([+0.204,+0.311,+0.302,+0.288,+0.259,+0.251,+0.390,+0.355,+0.338,\
#                     +0.180,+0.372,+0.362,+0.268,+0.282,+0.384,+0.537,+0.220,+0.281])
# p_minus = np.array([-0.962,-0.887,-0.936,-0.910,-0.945,-0.929,-0.890,-0.873,-0.845,\
#                     -0.948,-0.912,-0.877,-0.930,-0.860,-0.923,-0.812,-0.975,-0.900])
# s_minus = np.array([-0.181,-0.341,-0.181,-0.297,-0.199,-0.271,-0.238,-0.335,-0.415,\
#                     -0.261,-0.174,-0.315,-0.250,-0.424,-0.009,-0.228,-0.024,-0.334])
#%%
d = 3 # number of dimensions
n_plus = len(q_plus)
Sx_plus = np.sum(q_plus)
Sy_plus = np.sum(p_plus)
Sz_plus = np.sum(s_plus)
# R_plus = np.sqrt(Sx_plus**2+Sy_plus**2+Sz_plus**2)
R_plus = np.linalg.norm(np.array([Sx_plus,Sy_plus,Sz_plus]))
qmean_plus = Sx_plus/R_plus
pmean_plus = Sy_plus/R_plus
smean_plus = Sz_plus/R_plus
Rbar_plus = R_plus/n_plus
kappahat_plus = (n_plus-1)/(n_plus-R_plus)
fun_plus = lambda K_plus: Rbar_plus + 1/K_plus - 1/np.tanh(K_plus)
Kout_plus = fsolve(fun_plus,kappahat_plus)
Kout_plus = Kout_plus[0]
dsum_plus = 0
for iobj_plus in range(n_plus):
    dsum_plus = dsum_plus + (q_plus[iobj_plus]*qmean_plus+p_plus[iobj_plus]*pmean_plus+s_plus[iobj_plus]*smean_plus)**2
d_plus = 1 - 1/n_plus * dsum_plus
sigmahat_plus = np.sqrt(d_plus/(n_plus*Rbar_plus**2))
A68_plus = 1 - 0.68
sin_anglerad68_plus = sigmahat_plus*np.sqrt(-np.log(A68_plus))
chi2scale_plus = np.sqrt(chi2.isf(A68_plus,2))
sigma_plus = sin_anglerad68_plus/chi2scale_plus
lambda_plus = sigma_plus**2
D_plus = np.diag([lambda_plus,lambda_plus])
phirad_plus = 0
rotation_matrix_plus = np.array([[np.cos(phirad_plus),-np.sin(phirad_plus)],\
                            [np.sin(phirad_plus), np.cos(phirad_plus)]])
v1_plus = np.matmul(rotation_matrix_plus,np.array([sigma_plus,0]))
v2_plus = np.matmul(rotation_matrix_plus,np.array([0,sigma_plus]))
v1hat_plus = v1_plus/np.linalg.norm(v1_plus)
v2hat_plus = v2_plus/np.linalg.norm(v2_plus)
T_plus = np.transpose(np.array([v1hat_plus,v2hat_plus]))
Sigma_plus = np.matmul(T_plus,np.matmul(D_plus,np.transpose(T_plus))) # covariance matrix

n_minus = len(q_minus)
Sx_minus = np.sum(q_minus)
Sy_minus = np.sum(p_minus)
Sz_minus = np.sum(s_minus)
# R_minus = np.sqrt(Sx_minus**2+Sy_minus**2+Sz_minus**2)
R_minus = np.linalg.norm(np.array([Sx_minus,Sy_minus,Sz_minus]))
qmean_minus = Sx_minus/R_minus
pmean_minus = Sy_minus/R_minus
smean_minus = Sz_minus/R_minus
Rbar_minus = R_minus/n_minus
kappahat_minus = (n_minus-1)/(n_minus-R_minus)
fun_minus = lambda K_minus: Rbar_minus + 1/K_minus - 1/np.tanh(K_minus)
Kout_minus = fsolve(fun_minus,kappahat_minus)
Kout_minus = Kout_minus[0]
dsum_minus = 0
for iobj_minus in range(n_minus):
    dsum_minus = dsum_minus + (q_minus[iobj_minus]*qmean_minus+p_minus[iobj_minus]*pmean_minus+s_minus[iobj_minus]*smean_minus)**2
d_minus = 1 - 1/n_minus * dsum_minus
sigmahat_minus = np.sqrt(d_minus/(n_minus*Rbar_minus**2))
A68_minus = 1 - 0.68
sin_anglerad68_minus = sigmahat_minus*np.sqrt(-np.log(A68_minus))
chi2scale_minus = np.sqrt(chi2.isf(A68_minus,2))
sigma_minus = sin_anglerad68_minus/chi2scale_minus
lambda_minus = sigma_minus**2
D_minus = np.diag([lambda_minus,lambda_minus])
phirad_minus = 0
rotation_matrix_minus = np.array([[np.cos(phirad_minus),-np.sin(phirad_minus)],\
                            [np.sin(phirad_minus), np.cos(phirad_minus)]])
v1_minus = np.matmul(rotation_matrix_minus,np.array([sigma_minus,0]))
v2_minus = np.matmul(rotation_matrix_minus,np.array([0,sigma_minus]))
v1hat_minus = v1_minus/np.linalg.norm(v1_minus)
v2hat_minus = v2_minus/np.linalg.norm(v2_minus)
T_minus = np.transpose(np.array([v1hat_minus,v2hat_minus]))
Sigma_minus = np.matmul(T_minus,np.matmul(D_minus,np.transpose(T_minus))) # covariance matrix

n_plusminusnone = len(q_plusminusnone)
Sx_plusminusnone = np.sum(q_plusminusnone)
Sy_plusminusnone = np.sum(p_plusminusnone)
Sz_plusminusnone = np.sum(s_plusminusnone)
# R_plusminusnone = np.sqrt(Sx_plusminusnone**2+Sy_plusminusnone**2+Sz_plusminusnone**2)
R_plusminusnone = np.linalg.norm(np.array([Sx_plusminusnone,Sy_plusminusnone,Sz_plusminusnone]))
qmean_plusminusnone = Sx_plusminusnone/R_plusminusnone
pmean_plusminusnone = Sy_plusminusnone/R_plusminusnone
smean_plusminusnone = Sz_plusminusnone/R_plusminusnone
Rbar_plusminusnone = R_plusminusnone/n_plusminusnone
kappahat_plusminusnone = (n_plusminusnone-1)/(n_plusminusnone-R_plusminusnone)
fun_plusminusnone = lambda K_plusminusnone: Rbar_plusminusnone + 1/K_plusminusnone - 1/np.tanh(K_plusminusnone)
Kout_plusminusnone = fsolve(fun_plusminusnone,kappahat_plusminusnone)
Kout_plusminusnone = Kout_plusminusnone[0]
dsum_plusminusnone = 0
for iobj_plusminusnone in range(n_plusminusnone):
    dsum_plusminusnone = dsum_plusminusnone + (q_plusminusnone[iobj_plusminusnone]*qmean_plusminusnone+p_plusminusnone[iobj_plusminusnone]*pmean_plusminusnone+s_plusminusnone[iobj_plusminusnone]*smean_plusminusnone)**2
d_plusminusnone = 1 - 1/n_plusminusnone * dsum_plusminusnone
sigmahat_plusminusnone = np.sqrt(d_plusminusnone/(n_plusminusnone*Rbar_plusminusnone**2))
A68_plusminusnone = 1 - 0.68
sin_anglerad68_plusminusnone = sigmahat_plusminusnone*np.sqrt(-np.log(A68_plusminusnone))
chi2scale_plusminusnone = np.sqrt(chi2.isf(A68_plusminusnone,2))
sigma_plusminusnone = sin_anglerad68_plusminusnone/chi2scale_plusminusnone
lambda_plusminusnone = sigma_plusminusnone**2
D_plusminusnone = np.diag([lambda_plusminusnone,lambda_plusminusnone])
phirad_plusminusnone = 0
rotation_matrix_plusminusnone = np.array([[np.cos(phirad_plusminusnone),-np.sin(phirad_plusminusnone)],\
                            [np.sin(phirad_plusminusnone), np.cos(phirad_plusminusnone)]])
v1_plusminusnone = np.matmul(rotation_matrix_plusminusnone,np.array([sigma_plusminusnone,0]))
v2_plusminusnone = np.matmul(rotation_matrix_plusminusnone,np.array([0,sigma_plusminusnone]))
v1hat_plusminusnone = v1_plusminusnone/np.linalg.norm(v1_plusminusnone)
v2hat_plusminusnone = v2_plusminusnone/np.linalg.norm(v2_plusminusnone)
T_plusminusnone = np.transpose(np.array([v1hat_plusminusnone,v2hat_plusminusnone]))
Sigma_plusminusnone = np.matmul(T_plusminusnone,np.matmul(D_plusminusnone,np.transpose(T_plusminusnone))) # covariance matrix

# Bartlett test of homogeneity (test of same concentration), mardia jupp 2000 pg 226 (pdf pg 241)
p = 3 # number of dimensions of the sample space
q = 2 # number of sample sets to compare
n_plus_minus = n_plus + n_minus
nu_plus_minus = (p-1) * (n_plus_minus-q)
nu_plus = (p-1) * (n_plus-1)
nu_minus = (p-1) * (n_minus-1)
log_nu_plus_minus = np.log( (n_plus_minus-R_plus-R_minus)/nu_plus_minus )
log_nu_plus = np.log( (n_plus-R_plus)/nu_plus )
log_nu_minus = np.log( (n_minus-R_minus)/nu_minus )
# d = 1/(3*(q-1)) * (1/nu_plus-1/nu + 1/nu_minus-1/nu)
d_plus_minus = 1/(3*(q-1)) * (1/nu_plus-1/nu_plus_minus + 1/nu_minus)
B_plus_minus = 1/(1+d_plus_minus) * (nu_plus_minus*log_nu_plus_minus - nu_plus*log_nu_plus - nu_minus*log_nu_minus)
chi2_pvalue_B_plus_minus = chi2.sf(B_plus_minus,1)

# Bartlett test of homogeneity (test of same concentration), mardia jupp 2000 pg 226 (pdf pg 241)
p = 3 # number of dimensions of the sample space
q = 2 # number of sample sets to compare
n_plus_plusminusnone = n_plus + n_plusminusnone
nu_plus_plusminusnone = (p-1) * (n_plus_plusminusnone-q)
nu_plus = (p-1) * (n_plus-1)
nu_plusminusnone = (p-1) * (n_plusminusnone-1)
log_nu_plus_plusminusnone = np.log( (n_plus_plusminusnone-R_plus-R_plusminusnone)/nu_plus_plusminusnone )
log_nu_plus = np.log( (n_plus-R_plus)/nu_plus )
log_nu_plusminusnone = np.log( (n_plusminusnone-R_plusminusnone)/nu_plusminusnone )
# d = 1/(3*(q-1)) * (1/nu_plus-1/nu + 1/nu_plusminusnone-1/nu)
d_plus_plusminusnone = 1/(3*(q-1)) * (1/nu_plus-1/nu_plus_plusminusnone + 1/nu_plusminusnone)
B_plus_plusminusnone = 1/(1+d_plus_plusminusnone) * (nu_plus_plusminusnone*log_nu_plus_plusminusnone - nu_plus*log_nu_plus - nu_plusminusnone*log_nu_plusminusnone)
chi2_pvalue_B_plus_plusminusnone = chi2.sf(B_plus_plusminusnone,1)

# Bartlett test of homogeneity (test of same concentration), mardia jupp 2000 pg 226 (pdf pg 241)
p = 3 # number of dimensions of the sample space
q = 2 # number of sample sets to compare
n_plusminusnone_minus = n_plusminusnone + n_minus
nu_plusminusnone_minus = (p-1) * (n_plusminusnone_minus-q)
nu_plusminusnone = (p-1) * (n_plusminusnone-1)
nu_minus = (p-1) * (n_minus-1)
log_nu_plusminusnone_minus = np.log( (n_plusminusnone_minus-R_plusminusnone-R_minus)/nu_plusminusnone_minus )
log_nu_plusminusnone = np.log( (n_plusminusnone-R_plusminusnone)/nu_plusminusnone )
log_nu_minus = np.log( (n_minus-R_minus)/nu_minus )
# d = 1/(3*(q-1)) * (1/nu_plusminusnone-1/nu + 1/nu_minus-1/nu)
d_plusminusnone_minus = 1/(3*(q-1)) * (1/nu_plusminusnone-1/nu_plusminusnone_minus + 1/nu_minus)
B_plusminusnone_minus = 1/(1+d_plusminusnone_minus) * (nu_plusminusnone_minus*log_nu_plusminusnone_minus - nu_plusminusnone*log_nu_plusminusnone - nu_minus*log_nu_minus)
chi2_pvalue_B_plusminusnone_minus = chi2.sf(B_plusminusnone_minus,1)

# equality of mean directions test from rumcheva and presnell 2017 doi:10.1111/anzs.12183
d = 3 # number of dimensions of the sample space
k = 2 # number of sample sets to compare
n_plus_minus = n_plus + n_minus
Sx_plus_minus = Sx_plus + Sx_minus
Sy_plus_minus = Sy_plus + Sy_minus
Sz_plus_minus = Sz_plus + Sz_minus
# R_plus_minus = np.sqrt(Sx_plus_minus**2+Sy_plus_minus**2+Sz_plus_minus**2)
R_plus_minus = np.linalg.norm(np.array([Sx_plus_minus,Sy_plus_minus,Sz_plus_minus]))
Rbar_plus_minus = R_plus_minus/n_plus_minus
Rtilde_plus_minus = (R_plus+R_minus)/n_plus_minus
# top = (n-k) * (R_plus-R + R_minus-R)
top_plus_minus = (n_plus_minus-k) * (R_plus + R_minus - R_plus_minus)
bottom_plus_minus = (k-1) * (n_plus_minus - R_plus - R_minus)
W_plus_minus = top_plus_minus/bottom_plus_minus
factor_1_plus_minus = (n_plus_minus-k)/(k-1)
factor_2_plus_minus = (1-(d-2)**2)/(2*(d-1)**2)
factor_3_plus_minus = (Rtilde_plus_minus-Rbar_plus_minus)*(1-Rbar_plus_minus)/(1-Rtilde_plus_minus)
P_plus_minus = W_plus_minus + factor_1_plus_minus * factor_2_plus_minus * factor_3_plus_minus
# P has an F distribution with (k-1)(d-1) and (n-k)(d-1) degrees of freedom
dfn_plus_minus = (k-1)*(d-1) # might have dfn and dfd mixed up
dfd_plus_minus = (n_plus_minus-k)*(d-1)
from scipy.stats import f
pval_f_plus_minus = f.sf(P_plus_minus,dfn_plus_minus,dfd_plus_minus)
#%%
qmean_diff_plus_minus = qmean_plus - qmean_minus
pmean_diff_plus_minus = pmean_plus - pmean_minus
Sigma_combined_plus_minus = Sigma_plus + Sigma_minus # covariance matrix of combined random variable
mu_diff_plus_minus = np.array([qmean_diff_plus_minus,pmean_diff_plus_minus])
prod_plus_minus = np.matmul(np.linalg.inv(Sigma_combined_plus_minus),mu_diff_plus_minus)
prod2_plus_minus = np.matmul(np.transpose(mu_diff_plus_minus),prod_plus_minus)
mahalanobis_plus_minus = np.sqrt(prod2_plus_minus)
probmass_inside_plus_minus = 1 - np.exp(mahalanobis_plus_minus**2 / -2)
probmass_outside_plus_minus = 1 - probmass_inside_plus_minus

qmean_diff_plus_plusminusnone = qmean_plus - qmean_plusminusnone
pmean_diff_plus_plusminusnone = pmean_plus - pmean_plusminusnone
Sigma_combined_plus_plusminusnone = Sigma_plus + Sigma_plusminusnone # covariance matrix of combined random variable
mu_diff_plus_plusminusnone = np.array([qmean_diff_plus_plusminusnone,pmean_diff_plus_plusminusnone])
prod_plus_plusminusnone = np.matmul(np.linalg.inv(Sigma_combined_plus_plusminusnone),mu_diff_plus_plusminusnone)
prod2_plus_plusminusnone = np.matmul(np.transpose(mu_diff_plus_plusminusnone),prod_plus_plusminusnone)
mahalanobis_plus_plusminusnone = np.sqrt(prod2_plus_plusminusnone)
probmass_inside_plus_plusminusnone = 1 - np.exp(mahalanobis_plus_plusminusnone**2 / -2)
probmass_outside_plus_plusminusnone = 1 - probmass_inside_plus_plusminusnone

qmean_diff_plusminusnone_minus = qmean_plusminusnone - qmean_minus
pmean_diff_plusminusnone_minus = pmean_plusminusnone - pmean_minus
Sigma_combined_plusminusnone_minus = Sigma_plusminusnone + Sigma_minus # covariance matrix of combined random variable
mu_diff_plusminusnone_minus = np.array([qmean_diff_plusminusnone_minus,pmean_diff_plusminusnone_minus])
prod_plusminusnone_minus = np.matmul(np.linalg.inv(Sigma_combined_plusminusnone_minus),mu_diff_plusminusnone_minus)
prod2_plusminusnone_minus = np.matmul(np.transpose(mu_diff_plusminusnone_minus),prod_plusminusnone_minus)
mahalanobis_plusminusnone_minus = np.sqrt(prod2_plusminusnone_minus)
probmass_inside_plusminusnone_minus = 1 - np.exp(mahalanobis_plusminusnone_minus**2 / -2)
probmass_outside_plusminusnone_minus = 1 - probmass_inside_plusminusnone_minus
#%%
S_plus = np.cov(q_plus,p_plus)
S_minus = np.cov(q_minus,p_minus)
S_plusminusnone = np.cov(q_plusminusnone,p_plusminusnone)
# S_plus = Sigma_plus
# S_minus = Sigma_minus
# S_plusminusnone = Sigma_plusminusnone
Stilde_plus = S_plus/n_plus
Stilde_minus = S_minus/n_minus
Stilde_plusminusnone = S_plusminusnone/n_plusminusnone
Spooled_plus_minus = Stilde_plus + Stilde_minus
Spooled_plus_plusminusnone = Stilde_plus + Stilde_plusminusnone
Spooled_plusminusnone_minus = Stilde_plusminusnone + Stilde_minus
p1_plus_minus = np.matmul(np.linalg.inv(Spooled_plus_minus),mu_diff_plus_minus)
p1_plus_plusminusnone = np.matmul(np.linalg.inv(Spooled_plus_plusminusnone),mu_diff_plus_plusminusnone)
p1_plusminusnone_minus = np.matmul(np.linalg.inv(Spooled_plusminusnone_minus),mu_diff_plusminusnone_minus)
T2_plus_minus = np.matmul(np.transpose(mu_diff_plus_minus),p1_plus_minus)
T2_plus_plusminusnone = np.matmul(np.transpose(mu_diff_plus_plusminusnone),p1_plus_plusminusnone)
T2_plusminusnone_minus = np.matmul(np.transpose(mu_diff_plusminusnone_minus),p1_plusminusnone_minus)
pdof = 2
trace1_plus_minus = np.linalg.trace(np.matmul(Spooled_plus_minus,Spooled_plus_minus))
trace2_plus_minus = np.linalg.trace(Spooled_plus_minus)
top_plus_minus = trace1_plus_minus + trace2_plus_minus**2
term1_plus_minus = 1/(n_plus-1) * (np.linalg.trace(np.matmul(Stilde_plus,Stilde_plus))+np.linalg.trace(Stilde_plus)**2)
term2_plus_minus = 1/(n_minus-1) * (np.linalg.trace(np.matmul(Stilde_minus,Stilde_minus))+np.linalg.trace(Stilde_minus)**2)
bottom_plus_minus = term1_plus_minus + term2_plus_minus
vdof_plus_minus = top_plus_minus/bottom_plus_minus
trace1_plus_plusminusnone = np.linalg.trace(np.matmul(Spooled_plus_plusminusnone,Spooled_plus_plusminusnone))
trace2_plus_plusminusnone = np.linalg.trace(Spooled_plus_plusminusnone)
top_plus_plusminusnone = trace1_plus_plusminusnone + trace2_plus_plusminusnone**2
term1_plus_plusminusnone = 1/(n_plus-1) * (np.linalg.trace(np.matmul(Stilde_plus,Stilde_plus))+np.linalg.trace(Stilde_plus)**2)
term2_plus_plusminusnone = 1/(n_plusminusnone-1) * (np.linalg.trace(np.matmul(Stilde_plusminusnone,Stilde_plusminusnone))+np.linalg.trace(Stilde_plusminusnone)**2)
bottom_plus_plusminusnone = term1_plus_plusminusnone + term2_plus_plusminusnone
vdof_plus_plusminusnone = top_plus_plusminusnone/bottom_plus_plusminusnone
trace1_plusminusnone_minus = np.linalg.trace(np.matmul(Spooled_plusminusnone_minus,Spooled_plusminusnone_minus))
trace2_plusminusnone_minus = np.linalg.trace(Spooled_plusminusnone_minus)
top_plusminusnone_minus = trace1_plusminusnone_minus + trace2_plusminusnone_minus**2
term1_plusminusnone_minus = 1/(n_plusminusnone-1) * (np.linalg.trace(np.matmul(Stilde_plusminusnone,Stilde_plusminusnone))+np.linalg.trace(Stilde_plusminusnone)**2)
term2_plusminusnone_minus = 1/(n_minus-1) * (np.linalg.trace(np.matmul(Stilde_minus,Stilde_minus))+np.linalg.trace(Stilde_minus)**2)
bottom_plusminusnone_minus = term1_plusminusnone_minus + term2_plusminusnone_minus
vdof_plusminusnone_minus = top_plusminusnone_minus/bottom_plusminusnone_minus
Fval_plus_minus = T2_plus_minus * (vdof_plus_minus-pdof+1) / (vdof_plus_minus*pdof)
Fval_plus_plusminusnone = T2_plus_plusminusnone * (vdof_plus_plusminusnone-pdof+1) / (vdof_plus_plusminusnone*pdof)
Fval_plusminusnone_minus = T2_plusminusnone_minus * (vdof_plusminusnone_minus-pdof+1) / (vdof_plusminusnone_minus*pdof)
Fdof1 = pdof
Fdof2_plus_minus = vdof_plus_minus-pdof+1
Fdof2_plus_plusminusnone = vdof_plus_plusminusnone-pdof+1
Fdof2_plusminusnone_minus = vdof_plusminusnone_minus-pdof+1
from scipy.stats import f
F_plus_minus = f(Fdof1,Fdof2_plus_minus)
F_plus_plusminusnone = f(Fdof1,Fdof2_plus_plusminusnone)
F_plusminusnone_minus = f(Fdof1,Fdof2_plusminusnone_minus)
pval_plus_minus = 1 - F_plus_minus.cdf(Fval_plus_minus)
pval_plus_plusminusnone = 1 - F_plus_plusminusnone.cdf(Fval_plus_plusminusnone)
pval_plusminusnone_minus = 1 - F_plusminusnone_minus.cdf(Fval_plusminusnone_minus)
print(pval_plus_minus)
print(pval_plus_plusminusnone)
print(pval_plusminusnone_minus)
#%%
import numpy as np
import pandas as pd
from pingouin import multivariate_normality
result_plus = multivariate_normality(np.array([q_plus,p_plus,s_plus]),alpha=0.05)
print(f"HZ Statistic gplus: {result_plus.hz}")
print(f"P-value gplus:      {result_plus.pval}")
print(f"Normal gplus?       {result_plus.normal}")
result_minus = multivariate_normality(np.array([q_minus,p_minus,s_minus]),alpha=0.05)
print(f"HZ Statistic gminus: {result_minus.hz}")
print(f"P-value gminus:      {result_minus.pval}")
print(f"Normal gminus?       {result_minus.normal}")
import random
# Generates a list of len(q_plus) random numbers between 0 and len(q_plus)-1
random_list = [random.randint(0,len(q_plus)-1) for _ in range(len(q_plus))]
q2 = q_plusminusnone[random_list]
p2 = p_plusminusnone[random_list]
s2 = s_plusminusnone[random_list]
result_plusminusnone = multivariate_normality(np.array([q2,p2,s2]),alpha=0.05)
# result_plusminusnone = multivariate_normality(np.array([q_plusminusnone,p_plusminusnone,s_plusminusnone]),alpha=0.05)
print(f"HZ Statistic gplusminusnone: {result_plusminusnone.hz}")
print(f"P-value gplusminusnone:      {result_plusminusnone.pval}")
print(f"Normal gplusminusnone?       {result_plusminusnone.normal}")






