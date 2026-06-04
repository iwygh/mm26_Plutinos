#%%
def read_ellipse(q_ellipse,p_ellipse):
    import numpy as np
    from shapely.geometry import Polygon, Point
    q_ellipse = np.array(q_ellipse)
    p_ellipse = np.array(p_ellipse)
    q_mean = np.mean(q_ellipse)
    p_mean = np.mean(p_ellipse)
    q_rel = q_ellipse - q_mean
    p_rel = p_ellipse - p_mean
    range_rel = np.sqrt(q_rel**2+p_rel**2)
    a = np.min(range_rel)
    b = np.max(range_rel)
    delta_i = np.degrees(np.arcsin(0.5*(a+b)))
    sin_i_ellipse = np.sqrt(q_ellipse**2+p_ellipse**2)
    i_ellipse = np.arcsin(sin_i_ellipse)
    W_ellipse = np.arctan2(p_ellipse,q_ellipse)
    i_ellipse_degrees = np.degrees(i_ellipse)
    W_ellipse_degrees = np.degrees(W_ellipse) # -180 to + 180 degrees
    i_min_degrees = np.min(i_ellipse_degrees)
    i_max_degrees = np.max(i_ellipse_degrees)
    W_min_degrees = np.min(W_ellipse_degrees)
    W_max_degrees = np.max(W_ellipse_degrees)
    # if ellipse straddles the second and third quadrants
    if (-180<=W_min_degrees<-90) and (90<W_max_degrees<=180):
        W_ellipse_degrees = np.mod(W_ellipse_degrees,360)
        W_min_degrees = np.min(W_ellipse_degrees)
        W_max_degrees = np.max(W_ellipse_degrees)
    else:
        W_min_degrees = np.mod(W_min_degrees,360)
        W_max_degrees = np.mod(W_max_degrees,360)
    # if ellipse contains origin, Omega runs 0 to 360 degrees and imin == 0
    Ne = len(q_ellipse)
    linestring = []
    for i2 in range(Ne):
        pt = (q_ellipse[i2],p_ellipse[i2])
        linestring.append(pt)
    pt = (q_ellipse[0],p_ellipse[0])
    linestring.append(pt)
    poly = Polygon(linestring)
    pt = Point(0,0)
    checkstatus = pt.within(poly)
    if checkstatus == True:
        W_min_degrees = 0
        W_max_degrees = 360
        i_min_degrees = 0
    return i_min_degrees,i_max_degrees,W_min_degrees,W_max_degrees,delta_i
#%% import block
import numpy as np
import pandas as pd
# import aa_utilities as ut
import matplotlib.pyplot as plt
import matplotlib
#%% make iW mean plane plots
# invariable plane location used in mm23
i_invariable = 1.578694
Omega_invariable = 107.582222
savename_iW = 'b014_iW_vm17published'
matplotlib.rcParams['pdf.fonttype'] = 42 # makes text editable in svg and pdf
matplotlib.rcParams['ps.fonttype'] = 42 # makes text editable in svg and pdf
# # set up axes for iW plot
fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(3.5,2.5),\
                        gridspec_kw={'width_ratios':[2,1]})
axs[0,0].set_xlim((35,50))
axs[0,1].set_xlim((50,150))
axs[1,0].set_xlim((35,50))
axs[1,1].set_xlim((50,150))
axs[0,0].set_ylim((0,18))
axs[0,1].set_ylim((0,18))
axs[1,0].set_ylim((0,360))
axs[1,1].set_ylim((0,360))
axs[0,0].set_xticks((35,38,40,42,44,46,48,50))
axs[0,1].set_xticks((50,100,150))
axs[1,0].set_xticks((35,38,40,42,44,46,48,50))
axs[1,1].set_xticks((50,100,150))
axs[0,0].set_yticks((0,2,4,6,8,10,12,14,16,18))
axs[0,1].set_yticks((0,2,4,6,8,10,12,14,16,18))
axs[1,0].set_yticks((0,60,120,180,240,300,360))
axs[1,1].set_yticks((0,60,120,180,240,300,360))
axs[0,0].set_xticklabels('')
axs[0,1].set_xticklabels('')
axs[0,1].set_yticklabels('')
axs[1,1].set_yticklabels('')
axs[0,0].set_ylabel('inclination (deg)',fontsize=8)
axs[1,0].set_ylabel('Ω (deg)',fontsize=8)
axs[0,0].plot([35,50],[i_invariable,i_invariable],color='black',linestyle='-',linewidth=0.3)
axs[0,1].plot([50,150],[i_invariable,i_invariable],color='black',linestyle='-',linewidth=0.3)
axs[1,0].plot([35,50],[Omega_invariable,Omega_invariable],color='black',linestyle='-',linewidth=0.3)
axs[1,1].plot([50,150],[Omega_invariable,Omega_invariable],color='black',linestyle='-',linewidth=0.3)
labels = axs[1,0].get_xticks().tolist()
labels[-1] = ''
axs[1,0].set_xticklabels(labels)
axs[0,0].tick_params(direction="in")
axs[0,1].tick_params(direction="in")
axs[1,0].tick_params(direction="in")
axs[1,1].tick_params(direction="in")
axs[0,0].tick_params(top=True,right=True)
axs[0,1].tick_params(top=True,right=True)
axs[1,0].tick_params(top=True,right=True)
axs[1,1].tick_params(top=True,right=True)
axs[0,0].tick_params(labelsize='x-small')
axs[0,1].tick_params(labelsize='x-small')
axs[1,0].tick_params(labelsize='x-small')
axs[1,1].tick_params(labelsize='x-small')
fig.tight_layout(h_pad=0.5,w_pad=0)
fig.text(0.5,0.02,'a (au)',ha='center',fontsize=8)
# # plot Laplace plane on iW plot (use Laplace plane from mm23)
openfile = 'a000_laplace_plane_jsun_barycentric_20230220_2.txt'
df = pd.read_csv(openfile,delim_whitespace=True)
a_list = df['a'].tolist()
i_list = df['i_deg'].tolist()
W_list = df['W_deg'].tolist()
axs[0,0].plot(a_list,i_list,linewidth=0.5,color='blue',label='Barycentric')
axs[0,1].plot(a_list,i_list,linewidth=0.5,color='blue')
axs[1,0].plot(a_list,W_list,linewidth=0.5,color='blue',label='Barycentric')
axs[1,1].plot(a_list,W_list,linewidth=0.5,color='blue')
i_invariable = np.radians(i_invariable)
Omega_invariable = np.radians(Omega_invariable)
# # add published results from mm23 to iW plot
amin_list =         [ 35,    40.3,  42,   43,   44,    45,    45,    50,    50]
amax_list =         [ 40.3,  42,    43,   44,   45,    48,    50,    80,   150]
imin_degrees_list = [ 3.0-2.3,4.1-0.9,1.6-0.7,2.1-0.6,1.8-0.7,1.3-0.4,1.4-0.4,2.0-2.0,4.1-4.1]
imid_degrees_list = [ 3.0,   4.1,   1.6,  2.1,  1.8,   1.3,   1.4,   2.0,  4.1]
imax_degrees_list = [ 3.0+1.8,4.1+2.3,1.6+0.8,2.1+0.6,1.8+0.6,1.3+0.7,1.4+0.8,2.0+3.2,4.1+3.1]
Wmin_degrees_list = [ 121-37,272-12,64-46,96-15,85-22,3-31,355-26,114-114,138-138]
Wmid_degrees_list = [ 121,272,64,96,85,3,355,114,138]
Wmax_degrees_list = [ 121+25,272+17,64+24,96+11,85+15,3+25,355+22,114+246,138+222]
amin_str_list =         ['35',  '40.3','42','43','44','45','45','50','50']
amax_str_list =         ['40.3','42',  '43','44','45','48','50','80','150']
Nbins = len(amin_list)
amid_list = []
for i in range(Nbins):
    amin = amin_list[i]
    amax = amax_list[i]
    amid = (amin+amax)/2
    amid_list.append(amid)
i_up_errs = []
i_down_errs = []
i_left_errs = []
i_right_errs = []
W_up_errs = []
W_down_errs = []
for i in range(Nbins):
    i_up_errs.append(np.abs(imax_degrees_list[i]-imid_degrees_list[i]))
    i_down_errs.append(np.abs(imin_degrees_list[i]-imid_degrees_list[i]))
    Wmax = Wmax_degrees_list[i]
    Wmin = Wmin_degrees_list[i]
    Wmid = Wmid_degrees_list[i]
    if (Wmin<=Wmid<=Wmax):
        W_up_errs.append(Wmax-Wmid)
        W_down_errs.append(Wmid-Wmin)
    if (Wmax<=Wmin<=Wmid):
        W_up_errs.append(Wmax+360-Wmid)
        W_down_errs.append(Wmid-Wmin)
    if (Wmid<=Wmax<=Wmin):
        W_up_errs.append(Wmax-Wmid)
        W_down_errs.append(Wmid-(Wmin-360))
for i in range(Nbins):
    i_left_errs.append(np.abs(amid_list[i]-amin_list[i]))
    i_right_errs.append(np.abs(amid_list[i]-amax_list[i]))
W_left_errs = i_left_errs
W_right_errs = i_right_errs
i_x_errs = [i_left_errs,i_right_errs]
i_y_errs = [i_down_errs,i_up_errs]
W_x_errs = [W_left_errs,W_right_errs]
W_y_errs = [W_down_errs,W_up_errs]
color_mm23 = 'forestgreen'
axs[0,0].errorbar(amid_list, imid_degrees_list, xerr=i_x_errs, yerr=i_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color_mm23)
axs[0,1].errorbar(amid_list, imid_degrees_list, xerr=i_x_errs, yerr=i_y_errs,\
    marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
    color=color_mm23)
axs[1,0].errorbar(amid_list, Wmid_degrees_list, xerr=W_x_errs, yerr=W_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color_mm23)
axs[1,1].errorbar(amid_list, Wmid_degrees_list, xerr=W_x_errs, yerr=W_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color_mm23)
W2_qp = np.array(Wmid_degrees_list) + 360
axs[1,0].errorbar(amid_list, W2_qp, xerr=W_x_errs, yerr=W_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color_mm23)
axs[1,1].errorbar(amid_list, W2_qp, xerr=W_x_errs, yerr=W_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color_mm23)
mmrps = [2,5,5,7]
mmrqs = [1,2,3,4]
delta = 0.2
n_q = 400
n_p = 400
probmass = 0.682
color = 'red'
amin_list = []
amid_list = []
amax_list = []
imin_degrees_list = []
imid_degrees_list = []
imax_degrees_list = []
Wmin_degrees_list = []
Wmid_degrees_list = []
Wmax_degrees_list = []
i_up_errs = []
i_down_errs = []
i_left_errs = []
i_right_errs = []
W_up_errs = []
W_down_errs = []
nmmrs = len(mmrps)
for immr in range(nmmrs):
    mmrstr = 'p'+str(mmrps[immr])+'q'+str(mmrqs[immr])
    infile = 'b012_tnos_'+mmrstr+'_siraj.csv'
    df = pd.read_csv(infile)
    imid = df['ideg_siraj'][0]
    Wmid = df['Wdeg_siraj'][0]
    infile = 'b004_horizons_orbels_'+mmrstr+'_jd246e4_HEHIBEBI.csv'
    df = pd.read_csv(infile)
    aau_BE_list = df['aau_BE'].to_list()
    amin = np.min(aau_BE_list)
    amax = np.max(aau_BE_list)
    amid = (amin+amax)/2
    boxstr = str(mmrps[immr])+':'+str(mmrqs[immr])
    if amid < 50:
        axs[0,0].text(amid,16,boxstr,ha='center',va='top',fontsize=5,bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5))
        axs[1,0].text(amid,330,boxstr,ha='center',va='top',fontsize=5,bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5))
    else:
        axs[0,1].text(amid,16,boxstr,ha='left',va='top',fontsize=5,bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5))
        axs[1,1].text(amid,330,boxstr,ha='left',va='top',fontsize=5,bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5))
    infile_pval = 'b012_tnos_'+mmrstr+'_siraj_pvalmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    infile_q = 'b012_tnos_'+mmrstr+'_siraj_qmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    infile_p = 'b012_tnos_'+mmrstr+'_siraj_pmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    pvalmat_siraj = np.loadtxt(infile_pval,delimiter=',')
    qmat_siraj = np.loadtxt(infile_q,delimiter=',')
    pmat_siraj = np.loadtxt(infile_p,delimiter=',')
    fig_dummy = plt.figure(figsize=(10,10))
    ax_dummy = fig_dummy.add_subplot(111)
    quadset = ax_dummy.contour(qmat_siraj,pmat_siraj,pvalmat_siraj,levels=[1-probmass])
    plt.close()
    segs = quadset.allsegs[0]
    segs_array = np.array(segs)
    segs_flat = np.squeeze(segs_array)
    q_ellipse_siraj = segs_flat[:,0]
    p_ellipse_siraj = segs_flat[:,1]
    i_min_degrees,i_max_degrees,W_min_degrees,W_max_degrees,delta_i = read_ellipse(q_ellipse_siraj,p_ellipse_siraj)
    amin_list.append(amin)
    amid_list.append(amid)
    amax_list.append(amax)
    imin_degrees_list.append(i_min_degrees)
    imid_degrees_list.append(imid)
    imax_degrees_list.append(i_max_degrees)
    Wmin_degrees_list.append(W_min_degrees)
    Wmid_degrees_list.append(Wmid)
    Wmax_degrees_list.append(W_max_degrees)
    i_up_errs.append(np.abs(i_max_degrees-imid))
    i_down_errs.append(np.abs(i_min_degrees-imid))
    i_left_errs.append(np.abs(amid-amin))
    i_right_errs.append(np.abs(amid-amax))
    Wmax = W_max_degrees
    Wmin = W_min_degrees
    Wmid = Wmid
    if (Wmin<=Wmid<=Wmax):
        W_up_errs.append(Wmax-Wmid)
        W_down_errs.append(Wmid-Wmin)
    if (Wmax<=Wmin<=Wmid):
        W_up_errs.append(Wmax+360-Wmid)
        W_down_errs.append(Wmid-Wmin)
    if (Wmid<=Wmax<=Wmin):
        W_up_errs.append(Wmax-Wmid)
        W_down_errs.append(Wmid-(Wmin-360))
infile = 'b006_2026feb12_plutinos_gplusminusnone_siraj.csv'
df = pd.read_csv(infile)
imid = df['ideg_siraj'][0]
Wmid = df['Wdeg_siraj'][0]
infile = 'b004_horizons_orbels_p3q2_jd246e4_HEHIBEBI.csv'
df = pd.read_csv(infile)
aau_BE_list = df['aau_BE'].to_list()
amin = np.min(aau_BE_list)
amax = np.max(aau_BE_list)
amid = (amin+amax)/2
boxstr = '3:2'
axs[0,0].text(amid,16,boxstr,ha='center',va='top',fontsize=5,bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5))
axs[1,0].text(amid,330,boxstr,ha='center',va='top',fontsize=5,bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5))
infile_pval = 'b006_2026feb12_plutinos_gplusminusnone_siraj_pvalmat_delta0.4_nq400_np400.csv'
infile_q = 'b006_2026feb12_plutinos_gplusminusnone_siraj_qmat_delta0.4_nq400_np400.csv'
infile_p = 'b006_2026feb12_plutinos_gplusminusnone_siraj_pmat_delta0.4_nq400_np400.csv'
pvalmat_siraj = np.loadtxt(infile_pval,delimiter=',')
qmat_siraj = np.loadtxt(infile_q,delimiter=',')
pmat_siraj = np.loadtxt(infile_p,delimiter=',')
fig_dummy = plt.figure(figsize=(10,10))
ax_dummy = fig_dummy.add_subplot(111)
quadset_plutinos = ax_dummy.contour(qmat_siraj,pmat_siraj,pvalmat_siraj,levels=[1-probmass])
plt.close()
segs_plutinos = quadset_plutinos.allsegs[0]
segs_array_plutinos = np.array(segs_plutinos)
segs_flat_plutinos = np.squeeze(segs_array_plutinos)
q_ellipse_siraj = segs_flat_plutinos[:,0]
p_ellipse_siraj = segs_flat_plutinos[:,1]
i_min_degrees,i_max_degrees,W_min_degrees,W_max_degrees,delta_i = read_ellipse(q_ellipse_siraj,p_ellipse_siraj)
amin_list.append(amin)
amid_list.append(amid)
amax_list.append(amax)
imin_degrees_list.append(i_min_degrees)
imid_degrees_list.append(imid)
imax_degrees_list.append(i_max_degrees)
Wmin_degrees_list.append(W_min_degrees)
Wmid_degrees_list.append(Wmid)
Wmax_degrees_list.append(W_max_degrees)
i_up_errs.append(np.abs(i_max_degrees-imid))
i_down_errs.append(np.abs(i_min_degrees-imid))
i_left_errs.append(np.abs(amid-amin))
i_right_errs.append(np.abs(amid-amax))
Wmax = W_max_degrees
Wmin = W_min_degrees
Wmid = Wmid
if (Wmin<=Wmid<=Wmax):
    W_up_errs.append(Wmax-Wmid)
    W_down_errs.append(Wmid-Wmin)
if (Wmax<=Wmin<=Wmid):
    W_up_errs.append(Wmax+360-Wmid)
    W_down_errs.append(Wmid-Wmin)
if (Wmid<=Wmax<=Wmin):
    W_up_errs.append(Wmax-Wmid)
    W_down_errs.append(Wmid-(Wmin-360))
W_left_errs = i_left_errs
W_right_errs = i_right_errs
i_x_errs = [i_left_errs,i_right_errs]
i_y_errs = [i_down_errs,i_up_errs]
W_x_errs = [W_left_errs,W_right_errs]
W_y_errs = [W_down_errs,W_up_errs]
axs[0,0].errorbar(amid_list, imid_degrees_list, xerr=i_x_errs, yerr=i_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color)
axs[0,1].errorbar(amid_list, imid_degrees_list, xerr=i_x_errs, yerr=i_y_errs,\
    marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
    color=color)
axs[1,0].errorbar(amid_list, Wmid_degrees_list, xerr=W_x_errs, yerr=W_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color)
axs[1,1].errorbar(amid_list, Wmid_degrees_list, xerr=W_x_errs, yerr=W_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color)
W2 = np.array(Wmid_degrees_list) + 360
axs[1,0].errorbar(amid_list, W2, xerr=W_x_errs, yerr=W_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color)
axs[1,1].errorbar(amid_list, W2, xerr=W_x_errs, yerr=W_y_errs,\
        marker='o',markersize=0.5,linewidth=0.5,linestyle='none',\
        color=color)
plt.savefig(savename_iW+'.pdf' ,format='pdf' ,transparent=True,dpi=800)
plt.show()
