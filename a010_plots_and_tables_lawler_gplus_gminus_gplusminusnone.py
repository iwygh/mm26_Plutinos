#%%
def probmass_inside_circle_thru_point_vmf_2(xmean,ymean,sigma,xpt,ypt):
    import numpy as np
    zmean = np.sqrt(1-xmean**2-ymean**2)
    zpt = np.sqrt(1-xpt**2-ypt**2)
    diff_rad = np.arccos(xmean*xpt + ymean*ypt + zmean*zpt)
    A = np.exp(-(np.sin(diff_rad)/sigma)**2)
    probmass = 1 - A
    return probmass
#%%
def probmass_inside_circle_thru_point_vmf(xvec,yvec,xpt,ypt):
    import numpy as np
    n = len(xvec)
    Sx = np.sum(xvec)
    Sy = np.sum(yvec)
    zvec = np.sqrt(1-xvec**2-yvec**2)
    Sz = np.sum(zvec)
    R = np.sqrt(Sx**2+Sy**2+Sz**2)
    xhat = Sx/R
    yhat = Sy/R
    zhat = Sz/R
    Rbar = R/n
    d = 1 - 1/n * np.sum( (xvec*xhat + yvec*yhat + zvec*zhat)**2 )
    sigma = np.sqrt(d/(n*Rbar**2))
    zpt = np.sqrt(1-xpt**2-ypt**2)
    diff_rad = np.arccos(xhat*xpt + yhat*ypt + zhat*zpt)
    A = np.exp(-(np.sin(diff_rad)/sigma)**2)
    probmass = 1 - A
    return probmass
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
def curve_vec_exact(radians_vec,kappa):
    curve_vec_exact = kappa/(np.exp(kappa)-np.exp(-kappa))*\
        np.exp(kappa*np.cos(radians_vec))*np.sin(radians_vec)
    return curve_vec_exact
#%%
def shift_to_pole(q_array,p_array,s_array,idegmean,Wdegmean):
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
def angle_between_points_qp1_qp2(q1,p1,q2,p2):
    import numpy as np
    s1 = np.sqrt(1-q1**2-p1**2)
    s2 = np.sqrt(1-q2**2-p2**2)
    dot = q1*q2 + p1*p2 + s1*s2
    deg_distance = np.degrees(np.arccos(dot))
    return deg_distance
#%%
def plane_angle_qp1_qp2(q1,p1,q2,p2,qcenter,pcenter):
    dx1 = q1-qcenter
    dx2 = q2-qcenter
    dy1 = p1-pcenter
    dy2 = p2-pcenter
    dmag1 = np.sqrt(dx1**2+dy1**2)
    dmag2 = np.sqrt(dx2**2+dy2**2)
    dot = dx1*dx2 + dy1*dy2
    costheta = dot/(dmag1*dmag2)
    plane_angle_deg = np.degrees(np.arccos(costheta))
    return plane_angle_deg
#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
probmass = 0.95
#%%
infile = 'b002_lawler_vmf_results.csv'
df = pd.read_csv(infile)
qmean_gplus = df['qcc'][0]
qmean_gminus = df['qcc'][1]
qmean_gplusminusnone = df['qcc'][3]
qmean_gall = df['qcc'][4]
pmean_gplus = df['pcc'][0]
pmean_gminus = df['pcc'][1]
pmean_gplusminusnone = df['pcc'][3]
pmean_gall = df['pcc'][4]
idegmean_gplus = df['idegcc'][0]
idegmean_gminus = df['idegcc'][1]
idegmean_gplusminusnone = df['idegcc'][3]
idegmean_gall = df['idegcc'][4]
Wdegmean_gplus = df['Wdegcc'][0]
Wdegmean_gminus = df['Wdegcc'][1]
Wdegmean_gplusminusnone = df['Wdegcc'][3]
Wdegmean_gall = df['Wdegcc'][4]
angledeg_gplus = df['angledeg95'][0]
angledeg_gminus = df['angledeg95'][1]
angledeg_gplusminusnone = df['angledeg95'][3]
angledeg_gall = df['angledeg95'][4]
sigma_gplus = df['sigmahat'][0]
sigma_gminus = df['sigmahat'][1]
sigma_gplusminusnone = df['sigmahat'][3]
sigma_gall = df['sigmahat'][4]
kappa_gplus = df['Kout'][0]
kappa_gminus = df['Kout'][1]
kappa_gplusminusnone = df['Kout'][3]
kappa_gall = df['Kout'][4]
smean_gplus = np.sqrt(1-qmean_gplus**2-pmean_gplus**2)
smean_gminus = np.sqrt(1-qmean_gminus**2-pmean_gminus**2)
smean_gplusminusnone = np.sqrt(1-qmean_gplusminusnone**2-pmean_gplusminusnone**2)
smean_gall = np.sqrt(1-qmean_gall**2-pmean_gall**2)
#%%
infile_none = 'a000_lawler_StablePlutinos.txt'
df_none = pd.read_csv('a000_lawler_StablePlutinos.txt',delim_whitespace=True)
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
ideg_plusminus = np.array(df_plusminus['inc'].to_list())
Wdeg_plusminus = np.array(df_plusminus['Omega'].to_list())
q_plusminus = np.sin(np.radians(ideg_plusminus))*np.cos(np.radians(Wdeg_plusminus))
p_plusminus = np.sin(np.radians(ideg_plusminus))*np.sin(np.radians(Wdeg_plusminus))
s_plusminus = np.cos(np.radians(ideg_plusminus))
q_array_gplusminusnone = np.hstack([q_plusminus,q_none])
p_array_gplusminusnone = np.hstack([p_plusminus,p_none])
s_array_gplusminusnone = np.hstack([s_plusminus,s_none])
q_array_gplus = q_plusminus[plus_indices]
p_array_gplus = p_plusminus[plus_indices]
s_array_gplus = s_plusminus[plus_indices]
q_array_gminus = q_plusminus[minus_indices]
p_array_gminus = p_plusminus[minus_indices]
s_array_gminus = s_plusminus[minus_indices]
infile_flippers = 'a000_lawler_KozaiFlippers.txt'
df_flippers = pd.read_csv('a000_lawler_KozaiFlippers.txt',delim_whitespace=True)
ideg_flippers = np.array(df_flippers['inc'].to_list())
Wdeg_flippers = np.array(df_flippers['Omega'].to_list())
q_flippers = np.sin(np.radians(ideg_flippers))*np.cos(np.radians(Wdeg_flippers))
p_flippers = np.sin(np.radians(ideg_flippers))*np.sin(np.radians(Wdeg_flippers))
s_flippers = np.cos(np.radians(ideg_flippers))
q_array_gall = np.hstack([q_array_gplusminusnone,q_flippers])
p_array_gall = np.hstack([p_array_gplusminusnone,p_flippers])
s_array_gall = np.hstack([s_array_gplusminusnone,s_flippers])
#%%
df_invar = pd.read_csv('b001_lawler_plEndState_idegWdegqpinvar_BE.csv')
ideg_invar = df_invar['ideginvar'][0]
Wdeg_invar = df_invar['Wdeginvar'][0]
q_invar = df_invar['qinvar'][0]
p_invar = df_invar['pinvar'][0]
irad_invar = np.radians(ideg_invar)
Wrad_invar = np.radians(Wdeg_invar)
df_planets = pd.read_csv('b001_lawler_plEndState_edited_BE.csv')
ideg_bary_neptune = df_planets['ideg'][4]
Wdeg_bary_neptune = df_planets['Wdeg'][4]
irad_bary_neptune = np.radians(ideg_bary_neptune)
Wrad_bary_neptune = np.radians(Wdeg_bary_neptune)
q_neptune = np.sin(irad_bary_neptune)*np.cos(Wrad_bary_neptune)
p_neptune = np.sin(irad_bary_neptune)*np.sin(Wrad_bary_neptune)
df_laplace_rescenters = pd.read_csv('b005_laplace_lawler_rescenters.csv')
q_laplace = df_laplace_rescenters['laplace_q'][0]
p_laplace = df_laplace_rescenters['laplace_p'][0]
# ideg_laplace = df_laplace_rescenters['laplace_ideg'][0]
# Wdeg_laplace = df_laplace_rescenters['laplace_Wdeg'][0]
# irad_laplace = np.radians(ideg_laplace)
# Wrad_laplace = np.radians(Wdeg_laplace)
# q_laplace = np.sin(irad_laplace)*np.cos(Wrad_laplace)
# p_laplace = np.sin(irad_laplace)*np.sin(Wrad_laplace)
lawler_amin =  99999
lawler_amax = -99999
lawler_strs = ['StableKozaiPlutinos','StablePlutinos','KozaiFlippers']
for ilaw in range(len(lawler_strs)):
    dflaw = pd.read_csv('a000_lawler_'+lawler_strs[ilaw]+'.txt',delim_whitespace=True)
    alist = dflaw['a'].to_list()
    if np.min(alist) <= lawler_amin:
        lawler_amin = np.min(alist)
    if np.max(alist) >= lawler_amax:
        lawler_amax = np.max(alist)
dflap = pd.read_csv('b005_laplace_lawler.csv')
laplace_q_list = []
laplace_p_list = []
lq_list = np.array(dflap['lq'].to_list())
lp_list = np.array(dflap['lp'].to_list())
nlap = dflap.shape[0]
for ilap in range(nlap):
    if lawler_amin <= dflap['aau'][ilap] <= lawler_amax:
        laplace_q_list.append(lq_list[ilap])
        laplace_p_list.append(lp_list[ilap])
#%%
A_gplus,qrel_gplus,prel_gplus,srel_gplus,ireldeg_gplus,Wreldeg_gplus = shift_to_pole(q_array_gplus,p_array_gplus,s_array_gplus,idegmean_gplus,Wdegmean_gplus)
A_gminus,qrel_gminus,prel_gminus,srel_gminus,ireldeg_gminus,Wreldeg_gminus = shift_to_pole(q_array_gminus,p_array_gminus,s_array_gminus,idegmean_gminus,Wdegmean_gminus)
A_gplusminusnone,qrel_gplusminusnone,prel_gplusminusnone,srel_gplusminusnone,ireldeg_gplusminusnone,Wreldeg_gplusminusnone = shift_to_pole(q_array_gplusminusnone,p_array_gplusminusnone,s_array_gplusminusnone,idegmean_gplusminusnone,Wdegmean_gplusminusnone)
A_gall,qrel_gall,prel_gall,srel_gall,ireldeg_gall,Wreldeg_gall = shift_to_pole(q_array_gall,p_array_gall,s_array_gall,idegmean_gall,Wdegmean_gall)
#%%
probmass_gminus_wrt_gplus = probmass_inside_circle_thru_point_vmf_2(qmean_gplus,pmean_gplus,sigma_gplus,qmean_gminus,pmean_gminus)
probmass_gplusminusnone_wrt_gplus = probmass_inside_circle_thru_point_vmf_2(qmean_gplus,pmean_gplus,sigma_gplus,qmean_gplusminusnone,pmean_gplusminusnone)
probmass_invar_wrt_gplus = probmass_inside_circle_thru_point_vmf_2(qmean_gplus,pmean_gplus,sigma_gplus,q_invar,p_invar)
probmass_neptune_wrt_gplus = probmass_inside_circle_thru_point_vmf_2(qmean_gplus,pmean_gplus,sigma_gplus,q_neptune,p_neptune)
probmass_laplace_wrt_gplus = probmass_inside_circle_thru_point_vmf_2(qmean_gplus,pmean_gplus,sigma_gplus,q_laplace,p_laplace)

probmass_gplus_wrt_gminus = probmass_inside_circle_thru_point_vmf_2(qmean_gminus,pmean_gminus,sigma_gminus,qmean_gplus,pmean_gplus)
probmass_gplusminusnone_wrt_gminus = probmass_inside_circle_thru_point_vmf_2(qmean_gminus,pmean_gminus,sigma_gminus,qmean_gplusminusnone,pmean_gplusminusnone)
probmass_invar_wrt_gminus = probmass_inside_circle_thru_point_vmf_2(qmean_gminus,pmean_gminus,sigma_gminus,q_invar,p_invar)
probmass_neptune_wrt_gminus = probmass_inside_circle_thru_point_vmf_2(qmean_gminus,pmean_gminus,sigma_gminus,q_neptune,p_neptune)
probmass_laplace_wrt_gminus = probmass_inside_circle_thru_point_vmf_2(qmean_gminus,pmean_gminus,sigma_gminus,q_laplace,p_laplace)

probmass_gplus_wrt_gplusminusnone = probmass_inside_circle_thru_point_vmf_2(qmean_gplusminusnone,pmean_gplusminusnone,sigma_gplusminusnone,qmean_gplus,pmean_gplus)
probmass_gminus_wrt_gplusminusnone = probmass_inside_circle_thru_point_vmf_2(qmean_gplusminusnone,pmean_gplusminusnone,sigma_gplusminusnone,qmean_gminus,pmean_gminus)
probmass_invar_wrt_gplusminusnone = probmass_inside_circle_thru_point_vmf_2(qmean_gplusminusnone,pmean_gplusminusnone,sigma_gplusminusnone,q_invar,p_invar)
probmass_neptune_wrt_gplusminusnone = probmass_inside_circle_thru_point_vmf_2(qmean_gplusminusnone,pmean_gplusminusnone,sigma_gplusminusnone,q_neptune,p_neptune)
probmass_laplace_wrt_gplusminusnone = probmass_inside_circle_thru_point_vmf_2(qmean_gplusminusnone,pmean_gplusminusnone,sigma_gplusminusnone,q_laplace,p_laplace)
#%%
deg_distance_gplus_wrt_gplusminusnone   = angle_between_points_qp1_qp2(qmean_gplusminusnone,pmean_gplusminusnone,qmean_gplus,pmean_gplus)
deg_distance_gminus_wrt_gplusminusnone  = angle_between_points_qp1_qp2(qmean_gplusminusnone,pmean_gplusminusnone,qmean_gminus,pmean_gminus)
deg_distance_invar_wrt_gplusminusnone   = angle_between_points_qp1_qp2(qmean_gplusminusnone,pmean_gplusminusnone,q_invar,p_invar)
deg_distance_neptune_wrt_gplusminusnone = angle_between_points_qp1_qp2(qmean_gplusminusnone,pmean_gplusminusnone,q_neptune,p_neptune)
deg_distance_laplace_wrt_gplusminusnone = angle_between_points_qp1_qp2(qmean_gplusminusnone,pmean_gplusminusnone,q_laplace,p_laplace)
deg_distance_gplusminusnone_wrt_gplus   = angle_between_points_qp1_qp2(qmean_gplus,pmean_gplus,qmean_gplusminusnone,pmean_gplusminusnone)
deg_distance_gminus_wrt_gplus           = angle_between_points_qp1_qp2(qmean_gplus,pmean_gplus,qmean_gminus,pmean_gminus)
deg_distance_invar_wrt_gplus            = angle_between_points_qp1_qp2(qmean_gplus,pmean_gplus,q_invar,p_invar)
deg_distance_neptune_wrt_gplus          = angle_between_points_qp1_qp2(qmean_gplus,pmean_gplus,q_neptune,p_neptune)
deg_distance_laplace_wrt_gplus          = angle_between_points_qp1_qp2(qmean_gplus,pmean_gplus,q_laplace,p_laplace)
deg_distance_gplusminusnone_wrt_gminus  = angle_between_points_qp1_qp2(qmean_gminus,pmean_gminus,qmean_gplusminusnone,pmean_gplusminusnone)
deg_distance_gplus_wrt_gminus           = angle_between_points_qp1_qp2(qmean_gminus,pmean_gminus,qmean_gplus,pmean_gplus)
deg_distance_invar_wrt_gminus           = angle_between_points_qp1_qp2(qmean_gminus,pmean_gminus,q_invar,p_invar)
deg_distance_neptune_wrt_gminus         = angle_between_points_qp1_qp2(qmean_gminus,pmean_gminus,q_neptune,p_neptune)
deg_distance_laplace_wrt_gminus         = angle_between_points_qp1_qp2(qmean_gminus,pmean_gminus,q_laplace,p_laplace)
#%%
plane_angle_deg = plane_angle_qp1_qp2(qmean_gplus,pmean_gplus,qmean_gminus,pmean_gminus,qmean_gplusminusnone,pmean_gplusminusnone)
#%%
th = np.linspace(start=0,stop=2*np.pi,num=100,endpoint=True)
costh = np.cos(th)
sinth = np.sin(th)
#%%
plt.rcParams['font.size'] = 8
s = 10
fig = plt.figure(figsize=(2.0,2.0))

ax = fig.add_subplot(111)
ax.tick_params(axis='x',direction='in')
ax.tick_params(axis='y',direction='in')
ax.axhline(color='black',linestyle='-',linewidth=0.2,alpha=0.5)
ax.axvline(color='black',linestyle='-',linewidth=0.2,alpha=0.5)
ax.plot(costh*np.sin(np.radians(angledeg_gplus))+qmean_gplus,sinth*np.sin(np.radians(angledeg_gplus))+pmean_gplus,\
        color='forestgreen',linestyle='dashed',linewidth=0.5) # confidence circle
ax.plot(costh*np.sin(np.radians(angledeg_gminus))+qmean_gminus,sinth*np.sin(np.radians(angledeg_gminus))+pmean_gminus,\
        color='goldenrod',linestyle='dotted',linewidth=0.5) # confidence circle
ax.plot(costh*np.sin(np.radians(angledeg_gplusminusnone))+qmean_gplusminusnone,sinth*np.sin(np.radians(angledeg_gplusminusnone))+pmean_gplusminusnone,\
        color='black',linestyle='solid',linewidth=0.5) # confidence circle
# ax.plot(costh*np.sin(np.radians(angledeg_gall))+qmean_gall,sinth*np.sin(np.radians(angledeg_gall))+pmean_gall,\
#         color='magenta',linestyle='solid',linewidth=0.5) # confidence circle
ax.plot(laplace_q_list,laplace_p_list,color='magenta',linestyle='-',linewidth=0.5)
ax.scatter(qmean_gplus,pmean_gplus,color='forestgreen',s=s,marker='>')
ax.scatter(qmean_gminus,pmean_gminus,color='goldenrod',s=s,marker='v')
ax.scatter(qmean_gplusminusnone,pmean_gplusminusnone,color='black',s=s,marker='*')
# ax.scatter(qmean_gall,pmean_gall,color='magenta',s=s,marker='*')
ax.scatter(q_invar,p_invar,color='red',s=s,marker='x') # invariable pole of the solar system
ax.scatter(q_neptune,p_neptune,color='blue',s=s,marker='o') # orbit pole of neptune
ax.scatter(q_laplace,p_laplace,color='magenta',s=s,marker='+') # laplace pole at nominal resonance
ax.set_xlabel('$q=sin(i)cos(\Omega)$')
ax.set_ylabel('$p=sin(i)sin(\Omega)$',labelpad=5)
ax.text(0.95,0.95,str(probmass*100)+'%',ha='right',va='top',bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5),\
        transform=ax.transAxes)
# xmin_ellipse_plot = -0.055
# xmax_ellipse_plot = +0.025
# ymin_ellipse_plot = -0.005
# ymax_ellipse_plot = +0.06
xmin_ellipse_plot = -0.04
xmax_ellipse_plot = +0.025
ymin_ellipse_plot = -0.03
ymax_ellipse_plot = +0.065
# xmin_ellipse_plot = -0.18
# xmax_ellipse_plot = +0.18
# ymin_ellipse_plot = -0.10
# ymax_ellipse_plot = +0.26
ax.set_xlim([xmin_ellipse_plot,xmax_ellipse_plot])
ax.set_ylim([ymin_ellipse_plot,ymax_ellipse_plot])
ax.set_box_aspect((ymax_ellipse_plot-ymin_ellipse_plot)/(xmax_ellipse_plot-xmin_ellipse_plot))

plt.savefig('b010_a_plots_ellipses_'+str(probmass)+'_dots_onepanel_lawler.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()
#%%
longbins = np.arange(start=0,stop=95,step=5)
bbox_props = dict(fc='white',alpha=1,ec='black',lw=0.5)
fig = plt.figure(figsize=(1.75,1.75))
plt.rcParams['font.size'] = 6
ax = fig.add_subplot(111)
data = ireldeg_gplus
kappa = kappa_gplus
ax.hist(data,bins=longbins,color='lightblue',linewidth=0.5,edgecolor='black')
myHist = np.histogram(data,bins=longbins)
myHist_maxbinheight = np.max(myHist[0])
myHist_dense = np.histogram(data,bins=longbins,density=True)
myHist_maxdenseheight = np.max(myHist_dense[0])
radians_vec = np.radians(np.linspace(0,90,num=10000,endpoint=True))
curve_vec = curve_vec_exact(radians_vec,kappa)
curve_max = np.max(curve_vec)
ax.plot(np.degrees(radians_vec),curve_vec*myHist_maxbinheight/curve_max,color='red',linestyle='solid',lw=0.5)
ax.text(0.92,0.92,'Synthetic, g+\n$\gamma_{est}=$'+str(np.round(kappa,2)),\
        horizontalalignment='right',verticalalignment='top',bbox=bbox_props,transform=ax.transAxes)
ax.set_xlabel('Relative inclination (degrees)')
ax.set_ylabel('Count')
ax.set_xlim([0,90])
plt.tight_layout(pad=0.2,w_pad=1.5,h_pad=1.5)
plt.savefig('b010_2026feb17figure7_lawler_subplots1x1_gplus_rayleighcurve.pdf',dpi=400)
plt.show()

fig = plt.figure(figsize=(1.75,1.75))
plt.rcParams['font.size'] = 6
ax = fig.add_subplot(111)
data = ireldeg_gminus
kappa = kappa_gminus
ax.hist(data,bins=longbins,color='lightblue',linewidth=0.5,edgecolor='black')
myHist = np.histogram(data,bins=longbins)
myHist_maxbinheight = np.max(myHist[0])
myHist_dense = np.histogram(data,bins=longbins,density=True)
myHist_maxdenseheight = np.max(myHist_dense[0])
radians_vec = np.radians(np.linspace(0,90,num=10000,endpoint=True))
curve_vec = curve_vec_exact(radians_vec,kappa)
curve_max = np.max(curve_vec)
ax.plot(np.degrees(radians_vec),curve_vec*myHist_maxbinheight/curve_max,color='red',linestyle='solid',lw=0.5)
ax.text(0.92,0.92,'Synthetic, g-\n$\gamma_{est}=$'+str(np.round(kappa,2)),\
        horizontalalignment='right',verticalalignment='top',bbox=bbox_props,transform=ax.transAxes)
ax.set_xlabel('Relative inclination (degrees)')
ax.set_ylabel('Count')
ax.set_xlim([0,90])
plt.tight_layout(pad=0.2,w_pad=1.5,h_pad=1.5)
plt.savefig('b010_2026feb17figure7_lawler_subplots1x1_gminus_rayleighcurve.pdf',dpi=400)
plt.show()

fig = plt.figure(figsize=(1.75,1.75))
plt.rcParams['font.size'] = 6
ax = fig.add_subplot(111)
data = ireldeg_gplusminusnone
kappa = kappa_gplusminusnone
ax.hist(data,bins=longbins,color='lightblue',linewidth=0.5,edgecolor='black')
myHist = np.histogram(data,bins=longbins)
myHist_maxbinheight = np.max(myHist[0])
myHist_dense = np.histogram(data,bins=longbins,density=True)
myHist_maxdenseheight = np.max(myHist_dense[0])
radians_vec = np.radians(np.linspace(0,90,num=10000,endpoint=True))
curve_vec = curve_vec_exact(radians_vec,kappa)
curve_max = np.max(curve_vec)
ax.plot(np.degrees(radians_vec),curve_vec*myHist_maxbinheight/curve_max,color='red',linestyle='solid',lw=0.5)
ax.text(0.92,0.92,'Synthetic, all\n$\gamma_{est}=$'+str(np.round(kappa,2)),\
        horizontalalignment='right',verticalalignment='top',bbox=bbox_props,transform=ax.transAxes)
ax.set_xlabel('Relative inclination (degrees)')
ax.set_ylabel('Count')
ax.set_xlim([0,90])
plt.tight_layout(pad=0.2,w_pad=1.5,h_pad=1.5)
plt.savefig('b010_2026feb17figure7_lawler_subplots1x1_gplusminusnone_rayleighcurve.pdf',dpi=400)
plt.show()
#%%
print('')
print('qmean_gplusminusnone = ',np.round(qmean_gplusminusnone,3))
print('pmean_gplusminusnone = ',np.round(pmean_gplusminusnone,3))
print('idegmean_gplusminusnone = ',np.round(idegmean_gplusminusnone,2))
print('Wdegmean_gplusminusnone = ',np.round(Wdegmean_gplusminusnone,2))
print('angledeg_gplusminusnone = ',np.round(angledeg_gplusminusnone,2))
print('kappa_gplusminusnone = ',np.round(kappa_gplusminusnone,2))
print('sigmadegraleigh_gplusminusnone = ',np.round(np.degrees(kappa_gplusminusnone**-0.5),2))
print('')
print('qmean_gplus = ',np.round(qmean_gplus,3))
print('pmean_gplus = ',np.round(pmean_gplus,3))
print('idegmean_gplus = ',np.round(idegmean_gplus,2))
print('Wdegmean_gplus = ',np.round(Wdegmean_gplus,2))
print('angledeg_gplus = ',np.round(angledeg_gplus,2))
print('kappa_gplus = ',np.round(kappa_gplus,2))
print('sigmadegraleigh_gplus = ',np.round(np.degrees(kappa_gplus**-0.5),2))
print('')
print('qmean_gminus = ',np.round(qmean_gminus,3))
print('pmean_gminus = ',np.round(pmean_gminus,3))
print('idegmean_gminus = ',np.round(idegmean_gminus,2))
print('Wdegmean_gminus = ',np.round(Wdegmean_gminus,2))
print('angledeg_gminus = ',np.round(angledeg_gminus,2))
print('kappa_gminus = ',np.round(kappa_gminus,2))
print('sigmadegraleigh_gminus = ',np.round(np.degrees(kappa_gminus**-0.5),2))
print('')
print('deg_distance_gplusminusnone_wrt_gplusminusnone = 0')
print('deg_distance_gplus_wrt_gplusminusnone = ',np.round(deg_distance_gplus_wrt_gplusminusnone,2))
print('deg_distance_gminus_wrt_gplusminusnone = ',np.round(deg_distance_gminus_wrt_gplusminusnone,2))
print('deg_distance_invar_wrt_gplusminusnone = ',np.round(deg_distance_invar_wrt_gplusminusnone,2))
print('deg_distance_neptune_wrt_gplusminusnone = ',np.round(deg_distance_neptune_wrt_gplusminusnone,2))
print('deg_distance_laplace_wrt_gplusminusnone = ',np.round(deg_distance_laplace_wrt_gplusminusnone,2))
print('')
print('deg_distance_gplusminusnone_wrt_gplus = ',np.round(deg_distance_gplusminusnone_wrt_gplus,2))
print('deg_distance_gplus_wrt_gplus = 0')
print('deg_distance_gminus_wrt_gplus = ',np.round(deg_distance_gminus_wrt_gplus,2))
print('deg_distance_invar_wrt_gplus = ',np.round(deg_distance_invar_wrt_gplus,2))
print('deg_distance_neptune_wrt_gplus = ',np.round(deg_distance_neptune_wrt_gplus,2))
print('deg_distance_laplace_wrt_gplus = ',np.round(deg_distance_laplace_wrt_gplus,2))
print('')
print('deg_distance_gplusminusnone_wrt_gminus = ',np.round(deg_distance_gplusminusnone_wrt_gminus,2))
print('deg_distance_gplus_wrt_gminus = ',np.round(deg_distance_gplus_wrt_gminus,2))
print('deg_distance_gminus_wrt_gminus = 0')
print('deg_distance_invar_wrt_gminus = ',np.round(deg_distance_invar_wrt_gminus,2))
print('deg_distance_neptune_wrt_gminus = ',np.round(deg_distance_neptune_wrt_gminus,2))
print('deg_distance_laplace_wrt_gminus = ',np.round(deg_distance_laplace_wrt_gminus,2))
print('')
print('pval_gplusminusnone_wrt_gplusminusnone = N/A')
print('pval_gplus_wrt_gplusminusnone = ',1-probmass_gplus_wrt_gplusminusnone)
print('pval_gminus_wrt_gplusminusnone = ',1-probmass_gminus_wrt_gplusminusnone)
print('pval_invar_wrt_gplusminusnone = ',1-probmass_invar_wrt_gplusminusnone)
print('pval_neptune_wrt_gplusminusnone = ',1-probmass_neptune_wrt_gplusminusnone)
print('pval_laplace_wrt_gplusminusnone = ',1-probmass_laplace_wrt_gplusminusnone)
print('')
print('pval_gplusminusnone_wrt_gplus = ',1-probmass_gplusminusnone_wrt_gplus)
print('pval_gplus_wrt_gplus = N/A')
print('pval_gminus_wrt_gplus = ',1-probmass_gminus_wrt_gplus)
print('pval_invar_wrt_gplus = ',1-probmass_invar_wrt_gplus)
print('pval_neptune_wrt_gplus = ',1-probmass_neptune_wrt_gplus)
print('pval_laplace_wrt_gplus = ',1-probmass_laplace_wrt_gplus)
print('')
print('pval_gplusminusnone_wrt_gminus = ',1-probmass_gplusminusnone_wrt_gminus)
print('pval_gplus_wrt_gminus = ',1-probmass_gplus_wrt_gminus)
print('pval_gminus_wrt_gminus = N/A')
print('pval_invar_wrt_gminus = ',1-probmass_invar_wrt_gminus)
print('pval_neptune_wrt_gminus = ',1-probmass_neptune_wrt_gminus)
print('pval_laplace_wrt_gminus = ',1-probmass_laplace_wrt_gminus)
print('')
#%%
qrel_gminus = qmean_gminus - qmean_gplusminusnone
prel_gminus = pmean_gminus - pmean_gplusminusnone
qrel_gplus = qmean_gplus - qmean_gplusminusnone
prel_gplus = pmean_gplus - pmean_gplusminusnone
dot = qrel_gminus*qrel_gplus + prel_gminus*prel_gplus
magnitude_gminus = np.sqrt(qrel_gminus**2+prel_gminus**2)
magnitude_gplus = np.sqrt(qrel_gplus**2+prel_gplus**2)
angledeg_gminus_gplus = np.degrees(np.arccos(dot/(magnitude_gminus*magnitude_gplus)))
print('angledeg_gminus_gplus = ',angledeg_gminus_gplus)
angledeg_gminus = np.degrees(np.arctan2(prel_gminus,qrel_gminus))
angledeg_gplus = np.degrees(np.arctan2(prel_gplus,qrel_gplus))
signed_angledeg_gminus_gplus = angledeg_gminus - angledeg_gplus
print('signed_angledeg_gminus_gplus = ',signed_angledeg_gminus_gplus)
#%%
dictionary = {'probmass':[probmass],\
    'qmean_gplus':[qmean_gplus],'pmean_gplus':[pmean_gplus],'idegmean_gplus':[idegmean_gplus],\
    'Wdegmean_gplus':[Wdegmean_gplus],'angledeg_gplus':[angledeg_gplus],'kappa_gplus':[kappa_gplus],\
    'qmean_gminus':[qmean_gminus],'pmean_gminus':[pmean_gminus],'idegmean_gminus':[idegmean_gminus],\
    'Wdegmean_gminus':[Wdegmean_gminus],'angledeg_gminus':[angledeg_gminus],'kappa_gminus':[kappa_gminus],\
    'qmean_gplusminusnone':[qmean_gplusminusnone],'pmean_gplusminusnone':[pmean_gplusminusnone],'idegmean_gplusminusnone':[idegmean_gplusminusnone],\
    'Wdegmean_gplusminusnone':[Wdegmean_gplusminusnone],'angledeg_gplusminusnone':[angledeg_gplusminusnone],\
    'pval_gminus_wrt_gplus':[1-probmass_gminus_wrt_gplus],\
    'pval_gplusminusnone_wrt_gplus':[1-probmass_gplusminusnone_wrt_gplus],\
    'pval_invar_wrt_gplus':[1-probmass_invar_wrt_gplus],\
    'pval_neptune_wrt_gplus':[1-probmass_neptune_wrt_gplus],\
    'pval_laplace_wrt_gplus':[1-probmass_laplace_wrt_gplus],\
    'pval_gplus_wrt_gminus':[1-probmass_gplus_wrt_gminus],\
    'pval_gplusminusnone_wrt_gminus':[1-probmass_gplusminusnone_wrt_gminus],\
    'pval_invar_wrt_gminus':[1-probmass_invar_wrt_gminus],\
    'pval_neptune_wrt_gminus':[1-probmass_neptune_wrt_gminus],\
    'pval_laplace_wrt_gminus':[1-probmass_laplace_wrt_gminus],\
    'pval_gplus_wrt_gplusminusnone':[1-probmass_gplus_wrt_gplusminusnone],\
    'pval_gminus_wrt_gplusminusnone':[1-probmass_gminus_wrt_gplusminusnone],\
    'pval_invar_wrt_gplusminusnone':[1-probmass_invar_wrt_gplusminusnone],\
    'pval_neptune_wrt_gplusminusnone':[1-probmass_neptune_wrt_gplusminusnone],\
    'pval_laplace_wrt_gplusminusnone':[1-probmass_laplace_wrt_gplusminusnone],\
    'deg_distance_gplus_wrt_gplusminusnone':[deg_distance_gplus_wrt_gplusminusnone],\
    'deg_distance_gminus_wrt_gplusminusnone':[deg_distance_gminus_wrt_gplusminusnone],\
    'deg_distance_invar_wrt_gplusminusnone':[deg_distance_invar_wrt_gplusminusnone],\
    'deg_distance_neptune_wrt_gplusminusnone':[deg_distance_neptune_wrt_gplusminusnone],\
    'deg_distance_laplace_wrt_gplusminusnone':[deg_distance_laplace_wrt_gplusminusnone],\
    'deg_distance_gplusminusnone_wrt_gplus':[deg_distance_gplusminusnone_wrt_gplus],\
    'deg_distance_gminus_wrt_gplus':[deg_distance_gminus_wrt_gplus],\
    'deg_distance_invar_wrt_gplus':[deg_distance_invar_wrt_gplus],\
    'deg_distance_neptune_wrt_gplus':[deg_distance_neptune_wrt_gplus],\
    'deg_distance_laplace_wrt_gplus':[deg_distance_laplace_wrt_gplus],\
    'deg_distance_gplusminusnone_wrt_gminus':[deg_distance_gplusminusnone_wrt_gminus],\
    'deg_distance_gminus_wrt_gminus':[deg_distance_gplus_wrt_gminus],\
    'deg_distance_invar_wrt_gminus':[deg_distance_invar_wrt_gminus],\
    'deg_distance_neptune_wrt_gminus':[deg_distance_neptune_wrt_gminus],\
    'deg_distance_laplace_wrt_gminus':[deg_distance_laplace_wrt_gminus],\
    }
dfdict = pd.DataFrame.from_dict(dictionary)
dfdict.to_csv('b010_b_table_lawler_gplus_gminus_gplusminusnone.csv')