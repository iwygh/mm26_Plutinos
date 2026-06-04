#%%
def circle_diff(a,b,flag):
    import numpy as np
    if flag == 'deg':
        maxval = 360
    if flag == 'rad':
        maxval = 2*np.pi
    amod = np.mod(a,maxval)
    bmod = np.mod(b,maxval)
    smaller = np.min([amod,bmod])
    larger = np.max([amod,bmod])
    diff = larger - smaller
    if diff > maxval/2:
        diff = smaller + (maxval-larger)
    return diff
#%%
def solve_kepler(e, M):
    import numpy as np
    "Find E such that M = E - e sin E."
    assert(0 <= e < 1)
    assert(0 <= M <= 2*np.pi) 
    f = lambda E: E - e*np.sin(E) - M 
    E = M
    tolerance = 1e-10 
    # Newton's method 
    while (abs(f(E)) > tolerance):
        E -= f(E)/(1 - e*np.cos(E))
    return E
#%%
def rv_from_orbels(aau,qperiau,irad,wrad,Wrad,Mrad):
    import numpy as np
    e = 1 - qperiau/aau
    Erad = solve_kepler(e,Mrad) # eccentric anomaly
    thetarad = 2 * np.arctan(np.sqrt((1+e)/(1-e))*np.tan(Erad/2))
    h = np.sqrt(aau*(1-e**2))
    rperixau = h**2/(1+e*np.cos(thetarad))*np.cos(thetarad)
    rperiyau = h**2/(1+e*np.cos(thetarad))*np.sin(thetarad)
    rperizau = 0
    rperiau_vec = np.array([rperixau,rperiyau,rperizau])
    vperix = 1/h * (-np.sin(thetarad))
    vperiy = 1/h * (e+np.cos(thetarad))
    vperiz = 0
    vperi_vec = np.array([vperix,vperiy,vperiz])
    Q00 = -np.sin(Wrad)*np.cos(irad)*np.sin(wrad) + np.cos(Wrad)*np.cos(wrad)
    Q01 = -np.sin(Wrad)*np.cos(irad)*np.cos(wrad) - np.cos(Wrad)*np.sin(wrad)
    Q02 =  np.sin(Wrad)*np.sin(irad)
    Q10 =  np.cos(Wrad)*np.cos(irad)*np.sin(wrad) + np.sin(Wrad)*np.cos(wrad)
    Q11 =  np.cos(Wrad)*np.cos(irad)*np.cos(wrad) - np.sin(Wrad)*np.sin(wrad)
    Q12 = -np.cos(Wrad)*np.sin(irad)
    Q20 =  np.sin(irad)*np.sin(wrad)
    Q21 =  np.sin(irad)*np.cos(wrad)
    Q22 =  np.cos(irad)
    Qpqr_to_eci = np.array([ [Q00,Q01,Q02],\
                             [Q10,Q11,Q12],\
                             [Q20,Q21,Q22] ])
    rvec = np.matmul(Qpqr_to_eci,np.reshape(rperiau_vec,(3,1)))
    vvec = np.matmul(Qpqr_to_eci,np.reshape(vperi_vec,(3,1)))
    rvec = np.ndarray.flatten(rvec)
    vvec = np.ndarray.flatten(vvec)
    return rvec,vvec
#%%
def orbels_from_rv(rvec,vvec): # follows Curtis algorithm 4.2, pg 197
    import numpy as np
    mu = 1.0
    r = np.linalg.norm(rvec)
    v = np.linalg.norm(vvec)
    vr = np.dot(rvec,vvec)/r
    hvec = np.cross(rvec,vvec)
    h = np.linalg.norm(hvec)
    irad = np.arccos(hvec[2]/h)
    Khatvec = np.array([0,0,1])
    Nvec = np.cross(Khatvec,hvec)
    N = np.linalg.norm(Nvec)
    if Nvec[1] >= 0:
        Wrad = np.arccos(Nvec[0]/N)
    else:
        Wrad = 2*np.pi - np.arccos(Nvec[0]/N)
    evec = 1/mu * ((v**2-mu/r)*rvec - r*vr*vvec)
    e = np.linalg.norm(evec)
    if evec[2] >= 0:
        wrad = np.arccos(np.dot(Nvec,evec)/(N*e))
    else:
        wrad = 2*np.pi - np.arccos(np.dot(Nvec,evec)/(N*e))
    if vr >= 0:
        frad = np.arccos(1/e*(h**2/(mu*r)-1))
    else:
        frad = 2*np.pi - np.arccos(1/e*(h**2/(mu*r)-1))
    T = 2*np.pi/mu**2 * (h/np.sqrt(1-e**2))**3
    aau = (T*np.sqrt(mu)/(2*np.pi))**(2/3)
    qperiau = aau * (1-e)
    Erad = 2*np.arctan(np.sqrt(1-e)/np.sqrt(1+e)*np.tan(frad/2))
    Mrad = Erad - e*np.sin(Erad)
    return aau,qperiau,irad,wrad,Wrad,Mrad
#%%
def shift_to_pole_angles(x_array,y_array,z_array,iidegmean,WWdegmean):
    iradmean = np.radians(iidegmean)
    Wradmean = np.radians(WWdegmean)
    A = np.array([[np.cos(iradmean)*np.cos(Wradmean),np.cos(iradmean)*np.sin(Wradmean),-np.sin(iradmean)],\
                              [-np.sin(Wradmean),np.cos(Wradmean),0],\
                              [np.sin(iradmean)*np.cos(Wradmean),np.sin(iradmean)*np.sin(Wradmean),np.cos(iradmean)]])
    rvec_pre = np.array([x_array,y_array,z_array])
    rvec_post = np.matmul(A,rvec_pre)
    xrel_array = rvec_post[0,:]
    yrel_array = rvec_post[1,:]
    zrel_array = rvec_post[2,:]
    rrel_norm = np.sqrt(xrel_array**2+yrel_array**2+zrel_array**2)
    # xhatrel_array = xrel_array/rrel_norm
    # yhatrel_array = yrel_array/rrel_norm
    zhatrel_array = zrel_array/rrel_norm
    if np.max(zhatrel_array)>1:
        print(np.linalg.norm(rvec_pre),flush=True)
        return
    iirel_rad = np.arccos(zhatrel_array)
    # print(zrel_array[0],np.degrees(iirel_rad[0]),flush=True)
    WWrel_rad = np.arctan2(yrel_array/np.sin(iirel_rad),xrel_array/np.sin(iirel_rad))
    iireldeg = np.degrees(iirel_rad)
    WWreldeg = np.degrees(WWrel_rad)
    return A,xrel_array,yrel_array,zrel_array,iireldeg,WWreldeg
#%%
import numpy as np
import pandas as pd
import time
t00 = time.time()
jd = 246e4
jdstr = '246e4'
dfin = pd.read_csv('b004_tnos_orbels_jd'+jdstr+'.csv')
class_list = dfin['classification_gmv08method_vvl24'].to_list()
designation_list = dfin['mpc_des'].to_list()
p_list = dfin['res_p_largernumber'].to_list()
q_list = dfin['res_q_smallernumber'].to_list()
g_list = dfin['g2026feb12'].to_list()
nobj = dfin.shape[0]
res_p_list = np.array([3,5,2,7,5])
res_q_list = np.array([2,2,1,4,3])
indlists = []
deslists = []
nmmrs = len(res_p_list)
for immr in range(nmmrs):
    sp = str(res_p_list[immr])
    sq = str(res_q_list[immr])
    indlist = []
    deslist = []
    for iobj in range(nobj):
        if class_list[iobj] == 'resonant' and \
            str(p_list[iobj]) == sp and str(q_list[iobj]) == sq:
            indlist.append(iobj)
            deslist.append(designation_list[iobj])
    dictionary = {'index':indlist}
    dfout = pd.DataFrame.from_dict(dictionary)
    dfout.to_csv('b004_p'+sp+'q'+sq+'_index.csv',index=False)
    indlists.append(indlist)
    deslists.append(deslist)
gplus_index_2026feb12 = []
gminus_index_2026feb12 = []
gnone_index_2026feb12 = []
gunstable_index_2026feb12 = []
count = 0
for iobj in range(nobj):
    if class_list[iobj] == 'resonant' and \
        str(p_list[iobj]) == '3' and str(q_list[iobj]) == '2':
        count = count+1
        print(count,indlists[0][count-1],deslists[0][count-1])
        g = g_list[iobj]
        if g == 2:
            gunstable_index_2026feb12.append(iobj)
        if g == 0:
            gnone_index_2026feb12.append(iobj)
        if g == 1:
            gplus_index_2026feb12.append(iobj)
        if g == -1:
            gminus_index_2026feb12.append(iobj)
nplus_2026feb12 = len(gplus_index_2026feb12)
nminus_2026feb12 = len(gminus_index_2026feb12)
nnone_2026feb12 = len(gnone_index_2026feb12)
nunstable_2026feb12 = len(gunstable_index_2026feb12)
nall_2026feb12 = nplus_2026feb12+nminus_2026feb12+nnone_2026feb12+nunstable_2026feb12
ndiff_2026feb12 = nall_2026feb12-count
print(nplus_2026feb12,nminus_2026feb12,nnone_2026feb12,nunstable_2026feb12,\
      nall_2026feb12,count,ndiff_2026feb12)
plusminusnone_index_2026feb12 = np.hstack((gplus_index_2026feb12,gminus_index_2026feb12,gnone_index_2026feb12))
plusminusnone_index_2026feb12 = np.sort(plusminusnone_index_2026feb12)
all_index_2026feb12 = np.hstack((plusminusnone_index_2026feb12,gunstable_index_2026feb12))
all_index_2026feb12 = np.sort(all_index_2026feb12)
libs = ['gplus_2026feb12','gminus_2026feb12','gnone_2026feb12','gunstable_2026feb12','gplusminusnone_2026feb12','gall_2026feb12']
indlists_out = [gplus_index_2026feb12,gminus_index_2026feb12,gnone_index_2026feb12,\
                gunstable_index_2026feb12,plusminusnone_index_2026feb12,all_index_2026feb12]
for ilib in range(len(libs)):
    lib = libs[ilib]
    indlist = indlists_out[ilib]
    dictionary = {'index':indlist}
    dfout = pd.DataFrame.from_dict(dictionary)
    dfout.to_csv('b004_p3q2_'+lib+'_index.csv',index=False)