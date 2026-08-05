#%%
def ellipse_points(a0,b0,siga,sigb,rhoab,probmass,lenth):
    import numpy as np
    from scipy.stats import chi2 as sschi2
    cov = np.array([[siga**2,rhoab*siga*sigb],[rhoab*siga*sigb,sigb**2]])
    th = np.linspace(start=0,stop=2*np.pi,num=lenth,endpoint=True)
    costh = np.cos(th)
    sinth = np.sin(th)
    chi2val = sschi2.ppf(probmass,2)
    # print(chi2val)
    eigenvalues,eigenvectors = np.linalg.eig(cov)
    eigmin = np.min(eigenvalues)
    eigmax = np.max(eigenvalues)
    eigmaxindex = np.argmax(eigenvalues)
    eigmaxvec = eigenvectors[:,eigmaxindex]
    rotation_angle = np.arctan2(eigmaxvec[1],eigmaxvec[0])
    rotation_matrix = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle)],\
                                [np.sin(rotation_angle),np.cos(rotation_angle)]])
    semimajoraxis = np.sqrt(chi2val*eigmax)
    semiminoraxis = np.sqrt(chi2val*eigmin)
    ellipse_x_vec = costh*semimajoraxis
    ellipse_y_vec = sinth*semiminoraxis
    for i in range(lenth):
        xhere = np.array([ellipse_x_vec[i],ellipse_y_vec[i]])
        xhere2 = np.matmul(rotation_matrix.reshape(2,2),xhere.reshape(2,1))
        xherex = xhere2[0][0]
        xherey = xhere2[1][0]
        ellipse_x_vec[i] = xherex
        ellipse_y_vec[i] = xherey
    ellipse_x_vec = ellipse_x_vec + a0
    ellipse_y_vec = ellipse_y_vec + b0
    ecc = np.sqrt(1-semiminoraxis**2/semimajoraxis**2)
    phideg = np.degrees(rotation_angle)
    return ellipse_x_vec,ellipse_y_vec,chi2val,ecc,phideg,semimajoraxis,semiminoraxis
#%%
def ellipse_points_2(xvec,yvec,probmass,lenth):
    import numpy as np
    a0 = np.mean(xvec)
    b0 = np.mean(yvec)
    cov = np.cov([xvec,yvec])
    siga = np.sqrt(cov[0][0])
    sigb = np.sqrt(cov[1][1])
    rhoab = cov[0][1]/(siga*sigb)
    ellipse_x_vec,ellipse_y_vec,chi2val,ecc,phideg,semimajoraxis,semiminoraxis = \
        ellipse_points(a0,b0,siga,sigb,rhoab,probmass,lenth)
    return ellipse_x_vec,ellipse_y_vec,a0,b0,siga,sigb,rhoab,chi2val,ecc,phideg,semimajoraxis,semiminoraxis
#%%
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.optimize import fsolve
#%%
libration_siraj = 'gplusminusnone'
df_siraj = pd.read_csv('b006_2026feb12_plutinos_'+libration_siraj+'_siraj.csv')
qmean_siraj = df_siraj['q_siraj'][0]
pmean_siraj = df_siraj['p_siraj'][0]
prob0_siraj = df_siraj['probmasses'][0]
ap_siraj = df_siraj['ap_siraj'][0]
bp_siraj = df_siraj['bp_siraj'][0]
ec_siraj = df_siraj['ec_siraj'][0]
phirad_siraj = df_siraj['phi_siraj'][0]
phideg_siraj = np.degrees(phirad_siraj)
chi2scale_siraj = np.sqrt(chi2.isf(1-prob0_siraj,2))
sigma_a_siraj = ap_siraj/chi2scale_siraj
sigma_b_siraj = bp_siraj/chi2scale_siraj
lambda_a_siraj = sigma_a_siraj**2
lambda_b_siraj = sigma_b_siraj**2
D_siraj = np.diag([lambda_a_siraj,lambda_b_siraj])
rotation_matrix_siraj = np.array([[np.cos(phirad_siraj),-np.sin(phirad_siraj)],\
                            [np.sin(phirad_siraj), np.cos(phirad_siraj)]])
v1_siraj = np.matmul(rotation_matrix_siraj,np.array([sigma_a_siraj,0]))
v2_siraj = np.matmul(rotation_matrix_siraj,np.array([0,sigma_b_siraj]))
v1hat_siraj = v1_siraj/np.linalg.norm(v1_siraj)
v2hat_siraj = v2_siraj/np.linalg.norm(v2_siraj)
T_siraj = np.transpose(np.array([v1hat_siraj,v2hat_siraj]))
Sigma_siraj = np.matmul(T_siraj,np.matmul(D_siraj,np.transpose(T_siraj))) # covariance matrix

libration_vmf = 'gplusminusnone_2026feb12'
df_vmf = pd.read_csv('b004_p3q2_'+libration_vmf+'_index.csv')
indices_vmf = df_vmf['index'].to_list()
infile_vmf = 'b004_tnos_orbels_jd246e4.csv'
df_vmf = pd.read_csv(infile_vmf)
n_vmf = df_vmf.shape[0]
des_list_vmf = df_vmf['mpc_des'].tolist()
irad_vmf = np.radians(np.array(df_vmf['ideg_bary'].tolist()))
Wrad_vmf = np.radians(np.array(df_vmf['Wdeg_bary'].tolist()))
q_vmf = np.sin(irad_vmf)*np.cos(Wrad_vmf)
p_vmf = np.sin(irad_vmf)*np.sin(Wrad_vmf)
s_vmf = np.cos(irad_vmf)
des_list_2_vmf = []
for index_vmf in indices_vmf:
    des_list_2_vmf.append(des_list_vmf[index_vmf])
q_vmf = q_vmf[indices_vmf]
p_vmf = p_vmf[indices_vmf]
s_vmf = s_vmf[indices_vmf]
n_vmf = len(indices_vmf)
Sx_vmf = np.sum(q_vmf)
Sy_vmf = np.sum(p_vmf)
Sz_vmf = np.sum(s_vmf)
R_vmf = np.linalg.norm(np.array([Sx_vmf,Sy_vmf,Sz_vmf]))
qmean_vmf = Sx_vmf/R_vmf
pmean_vmf = Sy_vmf/R_vmf
smean_vmf = Sz_vmf/R_vmf
Rbar_vmf = R_vmf/n_vmf
kappahat_vmf = (n_vmf-1)/(n_vmf-R_vmf)
fun_vmf = lambda K_vmf: Rbar_vmf + 1/K_vmf - 1/np.tanh(K_vmf)
Kout_vmf = fsolve(fun_vmf,kappahat_vmf)
Kout_vmf = Kout_vmf[0]
dsum_vmf = 0
for iobj_vmf in range(n_vmf):
    dsum_vmf = dsum_vmf + (q_vmf[iobj_vmf]*qmean_vmf+p_vmf[iobj_vmf]*pmean_vmf+s_vmf[iobj_vmf]*smean_vmf)**2
d_vmf = 1 - 1/n_vmf * dsum_vmf
sigmahat_vmf = np.sqrt(d_vmf/(n_vmf*Rbar_vmf**2))
A68_vmf = 1 - 0.68
sin_anglerad68_vmf = sigmahat_vmf*np.sqrt(-np.log(A68_vmf))
chi2scale_vmf = np.sqrt(chi2.isf(A68_vmf,2))
sigma_vmf = sin_anglerad68_vmf/chi2scale_vmf
lambda_vmf = sigma_vmf**2
D_vmf = np.diag([lambda_vmf,lambda_vmf])
phirad_vmf = 0
rotation_matrix_vmf = np.array([[np.cos(phirad_vmf),-np.sin(phirad_vmf)],\
                            [np.sin(phirad_vmf), np.cos(phirad_vmf)]])
v1_vmf = np.matmul(rotation_matrix_vmf,np.array([sigma_vmf,0]))
v2_vmf = np.matmul(rotation_matrix_vmf,np.array([0,sigma_vmf]))
v1hat_vmf = v1_vmf/np.linalg.norm(v1_vmf)
v2hat_vmf = v2_vmf/np.linalg.norm(v2_vmf)
T_vmf = np.transpose(np.array([v1hat_vmf,v2hat_vmf]))
Sigma_vmf = np.matmul(T_vmf,np.matmul(D_vmf,np.transpose(T_vmf))) # covariance matrix

probmass_vm17 = 0.68
lenth_vm17 = 200
njobs_vm17 = 1000
nreps_vm17 = 40
infile_vm17 = 'b007_d_consolidated_mean_planes_vm17_2026feb12_plutinos_gplusminusnone_njobs'+\
    str(njobs_vm17)+'_nreps'+str(nreps_vm17)+'.csv'
df_vm17 = pd.read_csv(infile_vm17)
n_vm17 = df_vm17.shape[0]
i_mid_deg_vm17 = np.array(df_vm17['i_mid_deg'].to_list())
W_mid_deg_vm17 = np.array(df_vm17['node_mid_deg'].to_list())
irad_vm17 = np.radians(i_mid_deg_vm17)
Wrad_vm17 = np.radians(W_mid_deg_vm17)
q_vm17 = np.sin(irad_vm17)*np.cos(Wrad_vm17)
p_vm17 = np.sin(irad_vm17)*np.sin(Wrad_vm17)
q_ellipse_vm17,p_ellipse_vm17,a0_vm17,b0_vm17,siga_vm17,sigb_vm17,rhoab_vm17,chi2val_vm17,\
    ecc_vm17,phideg_vm17,semimajoraxis_vm17,semiminoraxis_vm17 = ellipse_points_2(q_vm17,p_vm17,probmass_vm17,lenth_vm17)
infile_vm17_1 = 'b007_fortran_mean_planes_vm17_2026feb12_plutinos_ijob1_njobs'+str(njobs_vm17)+'_nreps'+str(nreps_vm17)+'.txt'
f_vm17 = open(infile_vm17_1,'r')
flines_vm17 = f_vm17.readlines()
fsplit_vm17 = flines_vm17[0].split()
idegmean_vm17 = float(fsplit_vm17[2])
Wdegmean_vm17 = float(fsplit_vm17[3])
iradmean_vm17 = np.radians(idegmean_vm17)
Wradmean_vm17 = np.radians(Wdegmean_vm17)
qmean_vm17 = np.sin(iradmean_vm17)*np.cos(Wradmean_vm17)
pmean_vm17 = np.sin(iradmean_vm17)*np.sin(Wradmean_vm17)
smean_vm17 = np.cos(iradmean_vm17)
angledeg_vm17 = np.degrees(np.arcsin(np.sqrt(semimajoraxis_vm17*semiminoraxis_vm17)))
chi2scale_vm17 = np.sqrt(chi2val_vm17)
qmean_vm17 = np.mean(q_ellipse_vm17)
pmean_vm17 = np.mean(p_ellipse_vm17)
smean_vm17 = np.sqrt(1-qmean_vm17**2-pmean_vm17**2)
Sigma_vm17 = np.cov(q_vm17,p_vm17)
#%%
qmean_diff_vm17_siraj = qmean_vm17 - qmean_siraj
pmean_diff_vm17_siraj = pmean_vm17 - pmean_siraj
Sigma_combined_vm17_siraj = Sigma_vm17 + Sigma_siraj # covariance matrix of combined random variable
mu_diff_vm17_siraj = np.array([qmean_diff_vm17_siraj,pmean_diff_vm17_siraj])
prod_vm17_siraj = np.matmul(np.linalg.inv(Sigma_combined_vm17_siraj),mu_diff_vm17_siraj)
prod2_vm17_siraj = np.matmul(np.transpose(mu_diff_vm17_siraj),prod_vm17_siraj)
mahalanobis_vm17_siraj = np.sqrt(prod2_vm17_siraj)
probmass_inside_vm17_siraj = 1 - np.exp(mahalanobis_vm17_siraj**2 / -2)
probmass_outside_vm17_siraj = 1 - probmass_inside_vm17_siraj

qmean_diff_vm17_vmf = qmean_vm17 - qmean_vmf
pmean_diff_vm17_vmf = pmean_vm17 - pmean_vmf
Sigma_combined_vm17_vmf = Sigma_vm17 + Sigma_vmf # covariance matrix of combined random variable
mu_diff_vm17_vmf = np.array([qmean_diff_vm17_vmf,pmean_diff_vm17_vmf])
prod_vm17_vmf = np.matmul(np.linalg.inv(Sigma_combined_vm17_vmf),mu_diff_vm17_vmf)
prod2_vm17_vmf = np.matmul(np.transpose(mu_diff_vm17_vmf),prod_vm17_vmf)
mahalanobis_vm17_vmf = np.sqrt(prod2_vm17_vmf)
probmass_inside_vm17_vmf = 1 - np.exp(mahalanobis_vm17_vmf**2 / -2)
probmass_outside_vm17_vmf = 1 - probmass_inside_vm17_vmf

qmean_diff_vmf_siraj = qmean_vmf - qmean_siraj
pmean_diff_vmf_siraj = pmean_vmf - pmean_siraj
Sigma_combined_vmf_siraj = Sigma_vmf + Sigma_siraj # covariance matrix of combined random variable
mu_diff_vmf_siraj = np.array([qmean_diff_vmf_siraj,pmean_diff_vmf_siraj])
prod_vmf_siraj = np.matmul(np.linalg.inv(Sigma_combined_vmf_siraj),mu_diff_vmf_siraj)
prod2_vmf_siraj = np.matmul(np.transpose(mu_diff_vmf_siraj),prod_vmf_siraj)
mahalanobis_vmf_siraj = np.sqrt(prod2_vmf_siraj)
probmass_inside_vmf_siraj = 1 - np.exp(mahalanobis_vmf_siraj**2 / -2)
probmass_outside_vmf_siraj = 1 - probmass_inside_vmf_siraj