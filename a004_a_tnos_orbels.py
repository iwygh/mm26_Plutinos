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
from astroquery.jplhorizons import Horizons
import time
t00 = time.time()
jd = 246e4
jdstr = '246e4'
df = pd.read_csv('b003_idegWdegqpinvar_HE_ss12.csv')
qinvar = df['qinvar'][0]
pinvar = df['pinvar'][0]
ideginvar = df['ideginvar'][0]
qdeginvar = df['Wdeginvar'][0]
sinvar = np.sqrt(1-qinvar**2-pinvar**2)
hxinvar =  pinvar
hyinvar = -qinvar
hzinvar =  sinvar
iidegmean = np.degrees(np.arccos(hzinvar))
sinii = np.sin(np.radians(iidegmean))
WWdegmean = np.degrees(np.arctan2(hyinvar/sinii,hxinvar/sinii))
#%%
infile = 'a000_rnaasad22d4t1_mrt.txt'
#  1-  7 A7     ---   Des   Minor Planet Center designation (1)
#  9- 18 A10    ---   Class Gladman et al. 2008 dynamical classification
# 20- 21 A2     ---   p     Resonant integer p for resonant objects (2)
# 23- 24 I2     ---   q     Resonant integer q for resonant objects (3)
# 26- 27 A2     ---   soi   Secure (S) or insecure (I) classification (4)
# 29- 36 F8.3   au    a     Barycentric semimajor axis of the best-fit orbit
# 38- 42 F5.3   ---   e     Eccentricity of the best-fit orbit
# 44- 50 F7.3   deg   i     Ecliptic inclination of the best-fit orbit
top_lines = 22
file = open(infile,'r')
lines = file.readlines()
Nlines = len(lines)
ids_mrt = []
p_mrt = []
q_mrt = []
classification_mrt = []
for iline in range(Nlines):
    # if iline > top_lines-1 and iline < top_lines+1050:
    if iline > top_lines-1:
        line = lines[iline]
        # print(line)
        designation = line[0:7]
        designation = designation.lstrip()
        designation = designation.rstrip()
        ids_mrt.append(designation)
        classification = line[8:18]
        classification = classification.lstrip()
        classification = classification.rstrip()
        classification_mrt.append(classification)
        p = line[19:21]
        p = p.lstrip()
        p = p.rstrip()
        p_mrt.append(p)
        q = line[22:24]
        q = q.lstrip()
        q = q.rstrip()
        q_mrt.append(q)
n_mrt = len(ids_mrt)
#%%
ids_mrt_32 = []
indices_mrt_32 = []
for i in range(n_mrt):
    des = ids_mrt[i]
    classification = classification_mrt[i]
    p = p_mrt[i]
    q = q_mrt[i]
    if classification == 'resonant' and p == '3' and q == '2':
        ids_mrt_32.append(des)
        indices_mrt_32.append(i)
n_mrt_32 = len(ids_mrt_32)
aau_pl_HE = []
e_pl_HE = []
ideg_pl_HE = []
wdeg_pl_HE = []
Wdeg_pl_HE = []
Mdeg_pl_HE = []
for i in range(n_mrt_32):
    des = ids_mrt_32[i]
    idtype = 'smallbody'
    center = '500@10' # heliocenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_HE.append(el['a'][0])
    e_pl_HE.append(el['e'][0])
    ideg_pl_HE.append(el['incl'][0])
    wdeg_pl_HE.append(el['w'][0])
    Wdeg_pl_HE.append(el['Omega'][0])
    Mdeg_pl_HE.append(el['M'][0])
    print('32',i,n_mrt_32,'helio',des)
aau_pl_BE = []
e_pl_BE = []
ideg_pl_BE = []
wdeg_pl_BE = []
Wdeg_pl_BE = []
Mdeg_pl_BE = []
for i in range(n_mrt_32):
    des = ids_mrt_32[i]
    idtype = 'smallbody'
    center = '500@0' # solar system barycenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_BE.append(el['a'][0])
    e_pl_BE.append(el['e'][0])
    ideg_pl_BE.append(el['incl'][0])
    wdeg_pl_BE.append(el['w'][0])
    Wdeg_pl_BE.append(el['Omega'][0])
    Mdeg_pl_BE.append(el['M'][0])
    print('32',i,n_mrt_32,'bary',des)
xau_pl_HE = []
yau_pl_HE = []
zau_pl_HE = []
vx_pl_HE = []
vy_pl_HE = []
vz_pl_HE = []
for i in range(n_mrt_32):
    rvec,vvec = rv_from_orbels(aau_pl_HE[i],aau_pl_HE[i]*(1-e_pl_HE[i]),\
        np.radians(ideg_pl_HE[i]),np.radians(wdeg_pl_HE[i]),np.radians(Wdeg_pl_HE[i]),np.radians(Mdeg_pl_HE[i]))
    xau_pl_HE.append(rvec[0])
    yau_pl_HE.append(rvec[1])
    zau_pl_HE.append(rvec[2])
    vx_pl_HE.append(vvec[0])
    vy_pl_HE.append(vvec[1])
    vz_pl_HE.append(vvec[2])
xau_pl_HE = np.array(xau_pl_HE)
yau_pl_HE = np.array(yau_pl_HE)
zau_pl_HE = np.array(zau_pl_HE)
vx_pl_HE = np.array(vx_pl_HE)
vy_pl_HE = np.array(vy_pl_HE)
vz_pl_HE = np.array(vz_pl_HE)
xau_pl_BE = [] # heliocentric, ecliptic rvec,vvec of Sun are always 0
yau_pl_BE = []
zau_pl_BE = []
vx_pl_BE = []
vy_pl_BE = []
vz_pl_BE = []
for i in range(n_mrt_32):
    rvec,vvec = rv_from_orbels(aau_pl_BE[i],aau_pl_BE[i]*(1-e_pl_BE[i]),\
        np.radians(ideg_pl_BE[i]),np.radians(wdeg_pl_BE[i]),np.radians(Wdeg_pl_BE[i]),np.radians(Mdeg_pl_BE[i]))
    xau_pl_BE.append(rvec[0])
    yau_pl_BE.append(rvec[1])
    zau_pl_BE.append(rvec[2])
    vx_pl_BE.append(vvec[0])
    vy_pl_BE.append(vvec[1])
    vz_pl_BE.append(vvec[2])
xau_pl_BE = np.array(xau_pl_BE)
yau_pl_BE = np.array(yau_pl_BE)
zau_pl_BE = np.array(zau_pl_BE)
vx_pl_BE = np.array(vx_pl_BE)
vy_pl_BE = np.array(vy_pl_BE)
vz_pl_BE = np.array(vz_pl_BE)
xau_pl_HI = [] # heliocentric, invariable plane
yau_pl_HI = []
zau_pl_HI = []
vx_pl_HI = []
vy_pl_HI = []
vz_pl_HI = []
for i in range(n_mrt_32):
    A,xau_HI,yau_HI,zau_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_HE[i]],[yau_pl_HE[i]],[zau_pl_HE[i]],iidegmean,WWdegmean)
    xau_pl_HI.append(xau_HI[0])
    yau_pl_HI.append(yau_HI[0])
    zau_pl_HI.append(zau_HI[0])
    A,vx_HI,vy_HI,vz_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_HE[i]],[vy_pl_HE[i]],[vz_pl_HE[i]],iidegmean,WWdegmean)
    vx_pl_HI.append(vx_HI[0])
    vy_pl_HI.append(vy_HI[0])
    vz_pl_HI.append(vz_HI[0])
xau_pl_BI = [] # barycentric, invariable plane
yau_pl_BI = []
zau_pl_BI = []
vx_pl_BI = []
vy_pl_BI = []
vz_pl_BI = []
for i in range(n_mrt_32):
    A,xau_BI,yau_BI,zau_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_BE[i]],[yau_pl_BE[i]],[zau_pl_BE[i]],iidegmean,WWdegmean)
    xau_pl_BI.append(xau_BI[0])
    yau_pl_BI.append(yau_BI[0])
    zau_pl_BI.append(zau_BI[0])
    A,vx_BI,vy_BI,vz_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_BE[i]],[vy_pl_BE[i]],[vz_pl_BE[i]],iidegmean,WWdegmean)
    vx_pl_BI.append(vx_BI[0])
    vy_pl_BI.append(vy_BI[0])
    vz_pl_BI.append(vz_BI[0])
aau_pl_BI = []
e_pl_BI = []
ideg_pl_BI = []
wdeg_pl_BI = []
Wdeg_pl_BI = []
Mdeg_pl_BI = []
for i in range(n_mrt_32):
    rvec = np.array([xau_pl_BI[i],yau_pl_BI[i],zau_pl_BI[i]])
    vvec = np.array([vx_pl_BI[i],vy_pl_BI[i],vz_pl_BI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_BI.append(aau)
    e_pl_BI.append(1-qperiau/aau)
    ideg_pl_BI.append(np.degrees(irad))
    wdeg_pl_BI.append(np.degrees(wrad))
    Wdeg_pl_BI.append(np.degrees(Wrad))
    Mdeg_pl_BI.append(np.degrees(Mrad))
aau_pl_HI = []
e_pl_HI = []
ideg_pl_HI = []
wdeg_pl_HI = []
Wdeg_pl_HI = []
Mdeg_pl_HI = []
for i in range(n_mrt_32):
    rvec = np.array([xau_pl_HI[i],yau_pl_HI[i],zau_pl_HI[i]])
    vvec = np.array([vx_pl_HI[i],vy_pl_HI[i],vz_pl_HI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_HI.append(aau)
    e_pl_HI.append(1-qperiau/aau)
    ideg_pl_HI.append(np.degrees(irad))
    wdeg_pl_HI.append(np.degrees(wrad))
    Wdeg_pl_HI.append(np.degrees(Wrad))
    Mdeg_pl_HI.append(np.degrees(Mrad))
pomegadeg_pl_HE = np.array(wdeg_pl_HE) + np.array(Wdeg_pl_HE)
pomegadeg_pl_HE = np.mod(pomegadeg_pl_HE,360)
lambdadeg_pl_HE = pomegadeg_pl_HE + Mdeg_pl_HE
lambdadeg_pl_HE = np.mod(lambdadeg_pl_HE,360)
aau_pl_HE = np.array(aau_pl_HE)
e_pl_HE = np.array(e_pl_HE)
qau_pl_HE = aau_pl_HE * (1-e_pl_HE)
irad_pl_HE = np.radians(np.array(ideg_pl_HE))
wrad_pl_HE = np.radians(np.array(wdeg_pl_HE))
Wrad_pl_HE = np.radians(np.array(Wdeg_pl_HE))
Mrad_pl_HE = np.radians(np.array(Mdeg_pl_HE))
pomegarad_pl_HE = np.radians(np.array(pomegadeg_pl_HE))
q_pl_HE = np.sin(irad_pl_HE) * np.cos(Wrad_pl_HE)
p_pl_HE = np.sin(irad_pl_HE) * np.sin(Wrad_pl_HE)
s_pl_HE = np.cos(irad_pl_HE)
k_pl_HE = e_pl_HE * np.cos(pomegarad_pl_HE)
h_pl_HE = e_pl_HE * np.sin(pomegarad_pl_HE)
f_pl_HE = np.sqrt(1-k_pl_HE**2-h_pl_HE**2)
si = np.sin(irad_pl_HE)
ci = np.cos(irad_pl_HE)
sw = np.sin(wrad_pl_HE)
cw = np.cos(wrad_pl_HE)
sW = np.sin(Wrad_pl_HE)
cW = np.cos(Wrad_pl_HE)
k2_pl_HE = -sW*ci*sw+cW*cw
h2_pl_HE =  cW*ci*sw+sW*cw
f2_pl_HE =      si*sw
pomegadeg_pl_HI = np.array(wdeg_pl_HI) + np.array(Wdeg_pl_HI)
pomegadeg_pl_HI = np.mod(pomegadeg_pl_HI,360)
lambdadeg_pl_HI = pomegadeg_pl_HI + Mdeg_pl_HI
lambdadeg_pl_HI = np.mod(lambdadeg_pl_HI,360)
aau_pl_HI = np.array(aau_pl_HI)
e_pl_HI = np.array(e_pl_HI)
qau_pl_HI = aau_pl_HI * (1-e_pl_HI)
irad_pl_HI = np.radians(np.array(ideg_pl_HI))
wrad_pl_HI = np.radians(np.array(wdeg_pl_HI))
Wrad_pl_HI = np.radians(np.array(Wdeg_pl_HI))
Mrad_pl_HI = np.radians(np.array(Mdeg_pl_HI))
pomegarad_pl_HI = np.radians(np.array(pomegadeg_pl_HI))
q_pl_HI = np.sin(irad_pl_HI) * np.cos(Wrad_pl_HI)
p_pl_HI = np.sin(irad_pl_HI) * np.sin(Wrad_pl_HI)
s_pl_HI = np.cos(irad_pl_HI)
k_pl_HI = e_pl_HI * np.cos(pomegarad_pl_HI)
h_pl_HI = e_pl_HI * np.sin(pomegarad_pl_HI)
f_pl_HI = np.sqrt(1-k_pl_HI**2-h_pl_HI**2)
si = np.sin(irad_pl_HI)
ci = np.cos(irad_pl_HI)
sw = np.sin(wrad_pl_HI)
cw = np.cos(wrad_pl_HI)
sW = np.sin(Wrad_pl_HI)
cW = np.cos(Wrad_pl_HI)
k2_pl_HI = -sW*ci*sw+cW*cw
h2_pl_HI =  cW*ci*sw+sW*cw
f2_pl_HI =      si*sw
pomegadeg_pl_BE = np.array(wdeg_pl_BE) + np.array(Wdeg_pl_BE)
pomegadeg_pl_BE = np.mod(pomegadeg_pl_BE,360)
lambdadeg_pl_BE = pomegadeg_pl_BE + Mdeg_pl_BE
lambdadeg_pl_BE = np.mod(lambdadeg_pl_BE,360)
aau_pl_BE = np.array(aau_pl_BE)
e_pl_BE = np.array(e_pl_BE)
qau_pl_BE = aau_pl_BE * (1-e_pl_BE)
irad_pl_BE = np.radians(np.array(ideg_pl_BE))
wrad_pl_BE = np.radians(np.array(wdeg_pl_BE))
Wrad_pl_BE = np.radians(np.array(Wdeg_pl_BE))
Mrad_pl_BE = np.radians(np.array(Mdeg_pl_BE))
pomegarad_pl_BE = np.radians(np.array(pomegadeg_pl_BE))
q_pl_BE = np.sin(irad_pl_BE) * np.cos(Wrad_pl_BE)
p_pl_BE = np.sin(irad_pl_BE) * np.sin(Wrad_pl_BE)
s_pl_BE = np.cos(irad_pl_BE)
k_pl_BE = e_pl_BE * np.cos(pomegarad_pl_BE)
h_pl_BE = e_pl_BE * np.sin(pomegarad_pl_BE)
f_pl_BE = np.sqrt(1-k_pl_BE**2-h_pl_BE**2)
si = np.sin(irad_pl_BE)
ci = np.cos(irad_pl_BE)
sw = np.sin(wrad_pl_BE)
cw = np.cos(wrad_pl_BE)
sW = np.sin(Wrad_pl_BE)
cW = np.cos(Wrad_pl_BE)
k2_pl_BE = -sW*ci*sw+cW*cw
h2_pl_BE =  cW*ci*sw+sW*cw
f2_pl_BE =      si*sw
pomegadeg_pl_BI = np.array(wdeg_pl_BI) + np.array(Wdeg_pl_BI)
pomegadeg_pl_BI = np.mod(pomegadeg_pl_BI,360)
lambdadeg_pl_BI = pomegadeg_pl_BI + Mdeg_pl_BI
lambdadeg_pl_BI = np.mod(lambdadeg_pl_BI,360)
aau_pl_BI = np.array(aau_pl_BI)
e_pl_BI = np.array(e_pl_BI)
qau_pl_BI = aau_pl_BI * (1-e_pl_BI)
irad_pl_BI = np.radians(np.array(ideg_pl_BI))
wrad_pl_BI = np.radians(np.array(wdeg_pl_BI))
Wrad_pl_BI = np.radians(np.array(Wdeg_pl_BI))
Mrad_pl_BI = np.radians(np.array(Mdeg_pl_BI))
pomegarad_pl_BI = np.radians(np.array(pomegadeg_pl_BI))
q_pl_BI = np.sin(irad_pl_BI) * np.cos(Wrad_pl_BI)
p_pl_BI = np.sin(irad_pl_BI) * np.sin(Wrad_pl_BI)
s_pl_BI = np.cos(irad_pl_BI)
k_pl_BI = e_pl_BI * np.cos(pomegarad_pl_BI)
h_pl_BI = e_pl_BI * np.sin(pomegarad_pl_BI)
f_pl_BI = np.sqrt(1-k_pl_BI**2-h_pl_BI**2)
si = np.sin(irad_pl_BI)
ci = np.cos(irad_pl_BI)
sw = np.sin(wrad_pl_BI)
cw = np.cos(wrad_pl_BI)
sW = np.sin(Wrad_pl_BI)
cW = np.cos(Wrad_pl_BI)
k2_pl_BI = -sW*ci*sw+cW*cw
h2_pl_BI =  cW*ci*sw+sW*cw
f2_pl_BI =      si*sw
dictionary = {'ids':ids_mrt_32,'indices_mrt_32':indices_mrt_32,\
              'aau_HE':aau_pl_HE,'e_HE':e_pl_HE,'ideg_HE':ideg_pl_HE,'wdeg_HE':wdeg_pl_HE,'Wdeg_HE':Wdeg_pl_HE,'Mdeg_HE':Mdeg_pl_HE,\
              'xau_HE':xau_pl_HE,'yau_HE':yau_pl_HE,'zau_HE':zau_pl_HE,'vx_HE':vx_pl_HE,'vy_HE':vy_pl_HE,'vz_HE':vz_pl_HE,\
              'pomegadeg_HE':pomegadeg_pl_HE,'lambdadeg_HE':lambdadeg_pl_HE,'qau_HE':qau_pl_HE,'q_HE':q_pl_HE,'p_HE':p_pl_HE,'s_HE':s_pl_HE,\
              'k_HE':k_pl_HE,'h_HE':h_pl_HE,'f_HE':f_pl_HE,'k2_HE':k2_pl_HE,'h2_HE':h2_pl_HE,'f2_HE':f2_pl_HE,\
                
              'aau_HI':aau_pl_HI,'e_HI':e_pl_HI,'ideg_HI':ideg_pl_HI,'wdeg_HI':wdeg_pl_HI,'Wdeg_HI':Wdeg_pl_HI,'Mdeg_HI':Mdeg_pl_HI,\
              'xau_HI':xau_pl_HI,'yau_HI':yau_pl_HI,'zau_HI':zau_pl_HI,'vx_HI':vx_pl_HI,'vy_HI':vy_pl_HI,'vz_HI':vz_pl_HI,\
              'pomegadeg_HI':pomegadeg_pl_HI,'lambdadeg_HI':lambdadeg_pl_HI,'qau_HI':qau_pl_HI,'q_HI':q_pl_HI,'p_HI':p_pl_HI,'s_HI':s_pl_HI,\
              'k_HI':k_pl_HI,'h_HI':h_pl_HI,'f_HI':f_pl_HI,'k2_HI':k2_pl_HI,'h2_HI':h2_pl_HI,'f2_HI':f2_pl_HI,\
                
              'aau_BE':aau_pl_BE,'e_BE':e_pl_BE,'ideg_BE':ideg_pl_BE,'wdeg_BE':wdeg_pl_BE,'Wdeg_BE':Wdeg_pl_BE,'Mdeg_BE':Mdeg_pl_BE,\
              'xau_BE':xau_pl_BE,'yau_BE':yau_pl_BE,'zau_BE':zau_pl_BE,'vx_BE':vx_pl_BE,'vy_BE':vy_pl_BE,'vz_BE':vz_pl_BE,\
              'pomegadeg_BE':pomegadeg_pl_BE,'lambdadeg_BE':lambdadeg_pl_BE,'qau_BE':qau_pl_BE,'q_BE':q_pl_BE,'p_BE':p_pl_BE,'s_BE':s_pl_BE,\
              'k_BE':k_pl_BE,'h_BE':h_pl_BE,'f_BE':f_pl_BE,'k2_BE':k2_pl_BE,'h2_BE':h2_pl_BE,'f2_BE':f2_pl_BE,\
                
              'aau_BI':aau_pl_BI,'e_BI':e_pl_BI,'ideg_BI':ideg_pl_BI,'wdeg_BI':wdeg_pl_BI,'Wdeg_BI':Wdeg_pl_BI,'Mdeg_BI':Mdeg_pl_BI,\
              'xau_BI':xau_pl_BI,'yau_BI':yau_pl_BI,'zau_BI':zau_pl_BI,'vx_BI':vx_pl_BI,'vy_BI':vy_pl_BI,'vz_BI':vz_pl_BI,\
              'pomegadeg_BI':pomegadeg_pl_BI,'lambdadeg_BI':lambdadeg_pl_BI,'qau_BI':qau_pl_BI,'q_BI':q_pl_BI,'p_BI':p_pl_BI,'s_BI':s_pl_BI,\
              'k_BI':k_pl_BI,'h_BI':h_pl_BI,'f_BI':f_pl_BI,'k2_BI':k2_pl_BI,'h2_BI':h2_pl_BI,'f2_BI':f2_pl_BI,\
                  }
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b004_horizons_orbels_p3q2_jd'+jdstr+'_HEHIBEBI.csv',index=False)
#%%
ids_mrt_21 = []
indices_mrt_21 = []
for i in range(n_mrt):
    des = ids_mrt[i]
    classification = classification_mrt[i]
    p = p_mrt[i]
    q = q_mrt[i]
    if classification == 'resonant' and p == '2' and q == '1':
        ids_mrt_21.append(des)
        indices_mrt_21.append(i)
n_mrt_21 = len(ids_mrt_21)
aau_pl_HE = []
e_pl_HE = []
ideg_pl_HE = []
wdeg_pl_HE = []
Wdeg_pl_HE = []
Mdeg_pl_HE = []
for i in range(n_mrt_21):
    des = ids_mrt_21[i]
    idtype = 'smallbody'
    center = '500@10' # heliocenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_HE.append(el['a'][0])
    e_pl_HE.append(el['e'][0])
    ideg_pl_HE.append(el['incl'][0])
    wdeg_pl_HE.append(el['w'][0])
    Wdeg_pl_HE.append(el['Omega'][0])
    Mdeg_pl_HE.append(el['M'][0])
    print('21',i,n_mrt_21,'helio',des)
aau_pl_BE = []
e_pl_BE = []
ideg_pl_BE = []
wdeg_pl_BE = []
Wdeg_pl_BE = []
Mdeg_pl_BE = []
for i in range(n_mrt_21):
    des = ids_mrt_21[i]
    idtype = 'smallbody'
    center = '500@0' # solar system barycenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_BE.append(el['a'][0])
    e_pl_BE.append(el['e'][0])
    ideg_pl_BE.append(el['incl'][0])
    wdeg_pl_BE.append(el['w'][0])
    Wdeg_pl_BE.append(el['Omega'][0])
    Mdeg_pl_BE.append(el['M'][0])
    print('21',i,n_mrt_21,'bary',des)
xau_pl_HE = []
yau_pl_HE = []
zau_pl_HE = []
vx_pl_HE = []
vy_pl_HE = []
vz_pl_HE = []
for i in range(n_mrt_21):
    rvec,vvec = rv_from_orbels(aau_pl_HE[i],aau_pl_HE[i]*(1-e_pl_HE[i]),\
        np.radians(ideg_pl_HE[i]),np.radians(wdeg_pl_HE[i]),np.radians(Wdeg_pl_HE[i]),np.radians(Mdeg_pl_HE[i]))
    xau_pl_HE.append(rvec[0])
    yau_pl_HE.append(rvec[1])
    zau_pl_HE.append(rvec[2])
    vx_pl_HE.append(vvec[0])
    vy_pl_HE.append(vvec[1])
    vz_pl_HE.append(vvec[2])
xau_pl_HE = np.array(xau_pl_HE)
yau_pl_HE = np.array(yau_pl_HE)
zau_pl_HE = np.array(zau_pl_HE)
vx_pl_HE = np.array(vx_pl_HE)
vy_pl_HE = np.array(vy_pl_HE)
vz_pl_HE = np.array(vz_pl_HE)
xau_pl_BE = [] # heliocentric, ecliptic rvec,vvec of Sun are always 0
yau_pl_BE = []
zau_pl_BE = []
vx_pl_BE = []
vy_pl_BE = []
vz_pl_BE = []
for i in range(n_mrt_21):
    rvec,vvec = rv_from_orbels(aau_pl_BE[i],aau_pl_BE[i]*(1-e_pl_BE[i]),\
        np.radians(ideg_pl_BE[i]),np.radians(wdeg_pl_BE[i]),np.radians(Wdeg_pl_BE[i]),np.radians(Mdeg_pl_BE[i]))
    xau_pl_BE.append(rvec[0])
    yau_pl_BE.append(rvec[1])
    zau_pl_BE.append(rvec[2])
    vx_pl_BE.append(vvec[0])
    vy_pl_BE.append(vvec[1])
    vz_pl_BE.append(vvec[2])
xau_pl_BE = np.array(xau_pl_BE)
yau_pl_BE = np.array(yau_pl_BE)
zau_pl_BE = np.array(zau_pl_BE)
vx_pl_BE = np.array(vx_pl_BE)
vy_pl_BE = np.array(vy_pl_BE)
vz_pl_BE = np.array(vz_pl_BE)
xau_pl_HI = [] # heliocentric, invariable plane
yau_pl_HI = []
zau_pl_HI = []
vx_pl_HI = []
vy_pl_HI = []
vz_pl_HI = []
for i in range(n_mrt_21):
    A,xau_HI,yau_HI,zau_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_HE[i]],[yau_pl_HE[i]],[zau_pl_HE[i]],iidegmean,WWdegmean)
    xau_pl_HI.append(xau_HI[0])
    yau_pl_HI.append(yau_HI[0])
    zau_pl_HI.append(zau_HI[0])
    A,vx_HI,vy_HI,vz_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_HE[i]],[vy_pl_HE[i]],[vz_pl_HE[i]],iidegmean,WWdegmean)
    vx_pl_HI.append(vx_HI[0])
    vy_pl_HI.append(vy_HI[0])
    vz_pl_HI.append(vz_HI[0])
xau_pl_BI = [] # barycentric, invariable plane
yau_pl_BI = []
zau_pl_BI = []
vx_pl_BI = []
vy_pl_BI = []
vz_pl_BI = []
for i in range(n_mrt_21):
    A,xau_BI,yau_BI,zau_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_BE[i]],[yau_pl_BE[i]],[zau_pl_BE[i]],iidegmean,WWdegmean)
    xau_pl_BI.append(xau_BI[0])
    yau_pl_BI.append(yau_BI[0])
    zau_pl_BI.append(zau_BI[0])
    A,vx_BI,vy_BI,vz_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_BE[i]],[vy_pl_BE[i]],[vz_pl_BE[i]],iidegmean,WWdegmean)
    vx_pl_BI.append(vx_BI[0])
    vy_pl_BI.append(vy_BI[0])
    vz_pl_BI.append(vz_BI[0])
aau_pl_BI = []
e_pl_BI = []
ideg_pl_BI = []
wdeg_pl_BI = []
Wdeg_pl_BI = []
Mdeg_pl_BI = []
for i in range(n_mrt_21):
    rvec = np.array([xau_pl_BI[i],yau_pl_BI[i],zau_pl_BI[i]])
    vvec = np.array([vx_pl_BI[i],vy_pl_BI[i],vz_pl_BI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_BI.append(aau)
    e_pl_BI.append(1-qperiau/aau)
    ideg_pl_BI.append(np.degrees(irad))
    wdeg_pl_BI.append(np.degrees(wrad))
    Wdeg_pl_BI.append(np.degrees(Wrad))
    Mdeg_pl_BI.append(np.degrees(Mrad))
aau_pl_HI = []
e_pl_HI = []
ideg_pl_HI = []
wdeg_pl_HI = []
Wdeg_pl_HI = []
Mdeg_pl_HI = []
for i in range(n_mrt_21):
    rvec = np.array([xau_pl_HI[i],yau_pl_HI[i],zau_pl_HI[i]])
    vvec = np.array([vx_pl_HI[i],vy_pl_HI[i],vz_pl_HI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_HI.append(aau)
    e_pl_HI.append(1-qperiau/aau)
    ideg_pl_HI.append(np.degrees(irad))
    wdeg_pl_HI.append(np.degrees(wrad))
    Wdeg_pl_HI.append(np.degrees(Wrad))
    Mdeg_pl_HI.append(np.degrees(Mrad))
pomegadeg_pl_HE = np.array(wdeg_pl_HE) + np.array(Wdeg_pl_HE)
pomegadeg_pl_HE = np.mod(pomegadeg_pl_HE,360)
lambdadeg_pl_HE = pomegadeg_pl_HE + Mdeg_pl_HE
lambdadeg_pl_HE = np.mod(lambdadeg_pl_HE,360)
aau_pl_HE = np.array(aau_pl_HE)
e_pl_HE = np.array(e_pl_HE)
qau_pl_HE = aau_pl_HE * (1-e_pl_HE)
irad_pl_HE = np.radians(np.array(ideg_pl_HE))
wrad_pl_HE = np.radians(np.array(wdeg_pl_HE))
Wrad_pl_HE = np.radians(np.array(Wdeg_pl_HE))
Mrad_pl_HE = np.radians(np.array(Mdeg_pl_HE))
pomegarad_pl_HE = np.radians(np.array(pomegadeg_pl_HE))
q_pl_HE = np.sin(irad_pl_HE) * np.cos(Wrad_pl_HE)
p_pl_HE = np.sin(irad_pl_HE) * np.sin(Wrad_pl_HE)
s_pl_HE = np.cos(irad_pl_HE)
k_pl_HE = e_pl_HE * np.cos(pomegarad_pl_HE)
h_pl_HE = e_pl_HE * np.sin(pomegarad_pl_HE)
f_pl_HE = np.sqrt(1-k_pl_HE**2-h_pl_HE**2)
si = np.sin(irad_pl_HE)
ci = np.cos(irad_pl_HE)
sw = np.sin(wrad_pl_HE)
cw = np.cos(wrad_pl_HE)
sW = np.sin(Wrad_pl_HE)
cW = np.cos(Wrad_pl_HE)
k2_pl_HE = -sW*ci*sw+cW*cw
h2_pl_HE =  cW*ci*sw+sW*cw
f2_pl_HE =      si*sw
pomegadeg_pl_HI = np.array(wdeg_pl_HI) + np.array(Wdeg_pl_HI)
pomegadeg_pl_HI = np.mod(pomegadeg_pl_HI,360)
lambdadeg_pl_HI = pomegadeg_pl_HI + Mdeg_pl_HI
lambdadeg_pl_HI = np.mod(lambdadeg_pl_HI,360)
aau_pl_HI = np.array(aau_pl_HI)
e_pl_HI = np.array(e_pl_HI)
qau_pl_HI = aau_pl_HI * (1-e_pl_HI)
irad_pl_HI = np.radians(np.array(ideg_pl_HI))
wrad_pl_HI = np.radians(np.array(wdeg_pl_HI))
Wrad_pl_HI = np.radians(np.array(Wdeg_pl_HI))
Mrad_pl_HI = np.radians(np.array(Mdeg_pl_HI))
pomegarad_pl_HI = np.radians(np.array(pomegadeg_pl_HI))
q_pl_HI = np.sin(irad_pl_HI) * np.cos(Wrad_pl_HI)
p_pl_HI = np.sin(irad_pl_HI) * np.sin(Wrad_pl_HI)
s_pl_HI = np.cos(irad_pl_HI)
k_pl_HI = e_pl_HI * np.cos(pomegarad_pl_HI)
h_pl_HI = e_pl_HI * np.sin(pomegarad_pl_HI)
f_pl_HI = np.sqrt(1-k_pl_HI**2-h_pl_HI**2)
si = np.sin(irad_pl_HI)
ci = np.cos(irad_pl_HI)
sw = np.sin(wrad_pl_HI)
cw = np.cos(wrad_pl_HI)
sW = np.sin(Wrad_pl_HI)
cW = np.cos(Wrad_pl_HI)
k2_pl_HI = -sW*ci*sw+cW*cw
h2_pl_HI =  cW*ci*sw+sW*cw
f2_pl_HI =      si*sw
pomegadeg_pl_BE = np.array(wdeg_pl_BE) + np.array(Wdeg_pl_BE)
pomegadeg_pl_BE = np.mod(pomegadeg_pl_BE,360)
lambdadeg_pl_BE = pomegadeg_pl_BE + Mdeg_pl_BE
lambdadeg_pl_BE = np.mod(lambdadeg_pl_BE,360)
aau_pl_BE = np.array(aau_pl_BE)
e_pl_BE = np.array(e_pl_BE)
qau_pl_BE = aau_pl_BE * (1-e_pl_BE)
irad_pl_BE = np.radians(np.array(ideg_pl_BE))
wrad_pl_BE = np.radians(np.array(wdeg_pl_BE))
Wrad_pl_BE = np.radians(np.array(Wdeg_pl_BE))
Mrad_pl_BE = np.radians(np.array(Mdeg_pl_BE))
pomegarad_pl_BE = np.radians(np.array(pomegadeg_pl_BE))
q_pl_BE = np.sin(irad_pl_BE) * np.cos(Wrad_pl_BE)
p_pl_BE = np.sin(irad_pl_BE) * np.sin(Wrad_pl_BE)
s_pl_BE = np.cos(irad_pl_BE)
k_pl_BE = e_pl_BE * np.cos(pomegarad_pl_BE)
h_pl_BE = e_pl_BE * np.sin(pomegarad_pl_BE)
f_pl_BE = np.sqrt(1-k_pl_BE**2-h_pl_BE**2)
si = np.sin(irad_pl_BE)
ci = np.cos(irad_pl_BE)
sw = np.sin(wrad_pl_BE)
cw = np.cos(wrad_pl_BE)
sW = np.sin(Wrad_pl_BE)
cW = np.cos(Wrad_pl_BE)
k2_pl_BE = -sW*ci*sw+cW*cw
h2_pl_BE =  cW*ci*sw+sW*cw
f2_pl_BE =      si*sw
pomegadeg_pl_BI = np.array(wdeg_pl_BI) + np.array(Wdeg_pl_BI)
pomegadeg_pl_BI = np.mod(pomegadeg_pl_BI,360)
lambdadeg_pl_BI = pomegadeg_pl_BI + Mdeg_pl_BI
lambdadeg_pl_BI = np.mod(lambdadeg_pl_BI,360)
aau_pl_BI = np.array(aau_pl_BI)
e_pl_BI = np.array(e_pl_BI)
qau_pl_BI = aau_pl_BI * (1-e_pl_BI)
irad_pl_BI = np.radians(np.array(ideg_pl_BI))
wrad_pl_BI = np.radians(np.array(wdeg_pl_BI))
Wrad_pl_BI = np.radians(np.array(Wdeg_pl_BI))
Mrad_pl_BI = np.radians(np.array(Mdeg_pl_BI))
pomegarad_pl_BI = np.radians(np.array(pomegadeg_pl_BI))
q_pl_BI = np.sin(irad_pl_BI) * np.cos(Wrad_pl_BI)
p_pl_BI = np.sin(irad_pl_BI) * np.sin(Wrad_pl_BI)
s_pl_BI = np.cos(irad_pl_BI)
k_pl_BI = e_pl_BI * np.cos(pomegarad_pl_BI)
h_pl_BI = e_pl_BI * np.sin(pomegarad_pl_BI)
f_pl_BI = np.sqrt(1-k_pl_BI**2-h_pl_BI**2)
si = np.sin(irad_pl_BI)
ci = np.cos(irad_pl_BI)
sw = np.sin(wrad_pl_BI)
cw = np.cos(wrad_pl_BI)
sW = np.sin(Wrad_pl_BI)
cW = np.cos(Wrad_pl_BI)
k2_pl_BI = -sW*ci*sw+cW*cw
h2_pl_BI =  cW*ci*sw+sW*cw
f2_pl_BI =      si*sw
dictionary = {'ids':ids_mrt_21,'indices_mrt_21':indices_mrt_21,\
              'aau_HE':aau_pl_HE,'e_HE':e_pl_HE,'ideg_HE':ideg_pl_HE,'wdeg_HE':wdeg_pl_HE,'Wdeg_HE':Wdeg_pl_HE,'Mdeg_HE':Mdeg_pl_HE,\
              'xau_HE':xau_pl_HE,'yau_HE':yau_pl_HE,'zau_HE':zau_pl_HE,'vx_HE':vx_pl_HE,'vy_HE':vy_pl_HE,'vz_HE':vz_pl_HE,\
              'pomegadeg_HE':pomegadeg_pl_HE,'lambdadeg_HE':lambdadeg_pl_HE,'qau_HE':qau_pl_HE,'q_HE':q_pl_HE,'p_HE':p_pl_HE,'s_HE':s_pl_HE,\
              'k_HE':k_pl_HE,'h_HE':h_pl_HE,'f_HE':f_pl_HE,'k2_HE':k2_pl_HE,'h2_HE':h2_pl_HE,'f2_HE':f2_pl_HE,\
                
              'aau_HI':aau_pl_HI,'e_HI':e_pl_HI,'ideg_HI':ideg_pl_HI,'wdeg_HI':wdeg_pl_HI,'Wdeg_HI':Wdeg_pl_HI,'Mdeg_HI':Mdeg_pl_HI,\
              'xau_HI':xau_pl_HI,'yau_HI':yau_pl_HI,'zau_HI':zau_pl_HI,'vx_HI':vx_pl_HI,'vy_HI':vy_pl_HI,'vz_HI':vz_pl_HI,\
              'pomegadeg_HI':pomegadeg_pl_HI,'lambdadeg_HI':lambdadeg_pl_HI,'qau_HI':qau_pl_HI,'q_HI':q_pl_HI,'p_HI':p_pl_HI,'s_HI':s_pl_HI,\
              'k_HI':k_pl_HI,'h_HI':h_pl_HI,'f_HI':f_pl_HI,'k2_HI':k2_pl_HI,'h2_HI':h2_pl_HI,'f2_HI':f2_pl_HI,\
                
              'aau_BE':aau_pl_BE,'e_BE':e_pl_BE,'ideg_BE':ideg_pl_BE,'wdeg_BE':wdeg_pl_BE,'Wdeg_BE':Wdeg_pl_BE,'Mdeg_BE':Mdeg_pl_BE,\
              'xau_BE':xau_pl_BE,'yau_BE':yau_pl_BE,'zau_BE':zau_pl_BE,'vx_BE':vx_pl_BE,'vy_BE':vy_pl_BE,'vz_BE':vz_pl_BE,\
              'pomegadeg_BE':pomegadeg_pl_BE,'lambdadeg_BE':lambdadeg_pl_BE,'qau_BE':qau_pl_BE,'q_BE':q_pl_BE,'p_BE':p_pl_BE,'s_BE':s_pl_BE,\
              'k_BE':k_pl_BE,'h_BE':h_pl_BE,'f_BE':f_pl_BE,'k2_BE':k2_pl_BE,'h2_BE':h2_pl_BE,'f2_BE':f2_pl_BE,\
                
              'aau_BI':aau_pl_BI,'e_BI':e_pl_BI,'ideg_BI':ideg_pl_BI,'wdeg_BI':wdeg_pl_BI,'Wdeg_BI':Wdeg_pl_BI,'Mdeg_BI':Mdeg_pl_BI,\
              'xau_BI':xau_pl_BI,'yau_BI':yau_pl_BI,'zau_BI':zau_pl_BI,'vx_BI':vx_pl_BI,'vy_BI':vy_pl_BI,'vz_BI':vz_pl_BI,\
              'pomegadeg_BI':pomegadeg_pl_BI,'lambdadeg_BI':lambdadeg_pl_BI,'qau_BI':qau_pl_BI,'q_BI':q_pl_BI,'p_BI':p_pl_BI,'s_BI':s_pl_BI,\
              'k_BI':k_pl_BI,'h_BI':h_pl_BI,'f_BI':f_pl_BI,'k2_BI':k2_pl_BI,'h2_BI':h2_pl_BI,'f2_BI':f2_pl_BI,\
                  }
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b004_horizons_orbels_p2q1_jd'+jdstr+'_HEHIBEBI.csv',index=False)
#%%
ids_mrt_52 = []
indices_mrt_52 = []
for i in range(n_mrt):
    des = ids_mrt[i]
    classification = classification_mrt[i]
    p = p_mrt[i]
    q = q_mrt[i]
    if classification == 'resonant' and p == '5' and q == '2':
        ids_mrt_52.append(des)
        indices_mrt_52.append(i)
n_mrt_52 = len(ids_mrt_52)
aau_pl_HE = []
e_pl_HE = []
ideg_pl_HE = []
wdeg_pl_HE = []
Wdeg_pl_HE = []
Mdeg_pl_HE = []
for i in range(n_mrt_52):
    des = ids_mrt_52[i]
    idtype = 'smallbody'
    center = '500@10' # heliocenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_HE.append(el['a'][0])
    e_pl_HE.append(el['e'][0])
    ideg_pl_HE.append(el['incl'][0])
    wdeg_pl_HE.append(el['w'][0])
    Wdeg_pl_HE.append(el['Omega'][0])
    Mdeg_pl_HE.append(el['M'][0])
    print('52',i,n_mrt_52,'helio',des)
aau_pl_BE = []
e_pl_BE = []
ideg_pl_BE = []
wdeg_pl_BE = []
Wdeg_pl_BE = []
Mdeg_pl_BE = []
for i in range(n_mrt_52):
    des = ids_mrt_52[i]
    idtype = 'smallbody'
    center = '500@0' # solar system barycenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_BE.append(el['a'][0])
    e_pl_BE.append(el['e'][0])
    ideg_pl_BE.append(el['incl'][0])
    wdeg_pl_BE.append(el['w'][0])
    Wdeg_pl_BE.append(el['Omega'][0])
    Mdeg_pl_BE.append(el['M'][0])
    print('52',i,n_mrt_52,'bary',des)
xau_pl_HE = []
yau_pl_HE = []
zau_pl_HE = []
vx_pl_HE = []
vy_pl_HE = []
vz_pl_HE = []
for i in range(n_mrt_52):
    rvec,vvec = rv_from_orbels(aau_pl_HE[i],aau_pl_HE[i]*(1-e_pl_HE[i]),\
        np.radians(ideg_pl_HE[i]),np.radians(wdeg_pl_HE[i]),np.radians(Wdeg_pl_HE[i]),np.radians(Mdeg_pl_HE[i]))
    xau_pl_HE.append(rvec[0])
    yau_pl_HE.append(rvec[1])
    zau_pl_HE.append(rvec[2])
    vx_pl_HE.append(vvec[0])
    vy_pl_HE.append(vvec[1])
    vz_pl_HE.append(vvec[2])
xau_pl_HE = np.array(xau_pl_HE)
yau_pl_HE = np.array(yau_pl_HE)
zau_pl_HE = np.array(zau_pl_HE)
vx_pl_HE = np.array(vx_pl_HE)
vy_pl_HE = np.array(vy_pl_HE)
vz_pl_HE = np.array(vz_pl_HE)
xau_pl_BE = [] # heliocentric, ecliptic rvec,vvec of Sun are always 0
yau_pl_BE = []
zau_pl_BE = []
vx_pl_BE = []
vy_pl_BE = []
vz_pl_BE = []
for i in range(n_mrt_52):
    rvec,vvec = rv_from_orbels(aau_pl_BE[i],aau_pl_BE[i]*(1-e_pl_BE[i]),\
        np.radians(ideg_pl_BE[i]),np.radians(wdeg_pl_BE[i]),np.radians(Wdeg_pl_BE[i]),np.radians(Mdeg_pl_BE[i]))
    xau_pl_BE.append(rvec[0])
    yau_pl_BE.append(rvec[1])
    zau_pl_BE.append(rvec[2])
    vx_pl_BE.append(vvec[0])
    vy_pl_BE.append(vvec[1])
    vz_pl_BE.append(vvec[2])
xau_pl_BE = np.array(xau_pl_BE)
yau_pl_BE = np.array(yau_pl_BE)
zau_pl_BE = np.array(zau_pl_BE)
vx_pl_BE = np.array(vx_pl_BE)
vy_pl_BE = np.array(vy_pl_BE)
vz_pl_BE = np.array(vz_pl_BE)
xau_pl_HI = [] # heliocentric, invariable plane
yau_pl_HI = []
zau_pl_HI = []
vx_pl_HI = []
vy_pl_HI = []
vz_pl_HI = []
for i in range(n_mrt_52):
    A,xau_HI,yau_HI,zau_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_HE[i]],[yau_pl_HE[i]],[zau_pl_HE[i]],iidegmean,WWdegmean)
    xau_pl_HI.append(xau_HI[0])
    yau_pl_HI.append(yau_HI[0])
    zau_pl_HI.append(zau_HI[0])
    A,vx_HI,vy_HI,vz_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_HE[i]],[vy_pl_HE[i]],[vz_pl_HE[i]],iidegmean,WWdegmean)
    vx_pl_HI.append(vx_HI[0])
    vy_pl_HI.append(vy_HI[0])
    vz_pl_HI.append(vz_HI[0])
xau_pl_BI = [] # barycentric, invariable plane
yau_pl_BI = []
zau_pl_BI = []
vx_pl_BI = []
vy_pl_BI = []
vz_pl_BI = []
for i in range(n_mrt_52):
    A,xau_BI,yau_BI,zau_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_BE[i]],[yau_pl_BE[i]],[zau_pl_BE[i]],iidegmean,WWdegmean)
    xau_pl_BI.append(xau_BI[0])
    yau_pl_BI.append(yau_BI[0])
    zau_pl_BI.append(zau_BI[0])
    A,vx_BI,vy_BI,vz_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_BE[i]],[vy_pl_BE[i]],[vz_pl_BE[i]],iidegmean,WWdegmean)
    vx_pl_BI.append(vx_BI[0])
    vy_pl_BI.append(vy_BI[0])
    vz_pl_BI.append(vz_BI[0])
aau_pl_BI = []
e_pl_BI = []
ideg_pl_BI = []
wdeg_pl_BI = []
Wdeg_pl_BI = []
Mdeg_pl_BI = []
for i in range(n_mrt_52):
    rvec = np.array([xau_pl_BI[i],yau_pl_BI[i],zau_pl_BI[i]])
    vvec = np.array([vx_pl_BI[i],vy_pl_BI[i],vz_pl_BI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_BI.append(aau)
    e_pl_BI.append(1-qperiau/aau)
    ideg_pl_BI.append(np.degrees(irad))
    wdeg_pl_BI.append(np.degrees(wrad))
    Wdeg_pl_BI.append(np.degrees(Wrad))
    Mdeg_pl_BI.append(np.degrees(Mrad))
aau_pl_HI = []
e_pl_HI = []
ideg_pl_HI = []
wdeg_pl_HI = []
Wdeg_pl_HI = []
Mdeg_pl_HI = []
for i in range(n_mrt_52):
    rvec = np.array([xau_pl_HI[i],yau_pl_HI[i],zau_pl_HI[i]])
    vvec = np.array([vx_pl_HI[i],vy_pl_HI[i],vz_pl_HI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_HI.append(aau)
    e_pl_HI.append(1-qperiau/aau)
    ideg_pl_HI.append(np.degrees(irad))
    wdeg_pl_HI.append(np.degrees(wrad))
    Wdeg_pl_HI.append(np.degrees(Wrad))
    Mdeg_pl_HI.append(np.degrees(Mrad))
pomegadeg_pl_HE = np.array(wdeg_pl_HE) + np.array(Wdeg_pl_HE)
pomegadeg_pl_HE = np.mod(pomegadeg_pl_HE,360)
lambdadeg_pl_HE = pomegadeg_pl_HE + Mdeg_pl_HE
lambdadeg_pl_HE = np.mod(lambdadeg_pl_HE,360)
aau_pl_HE = np.array(aau_pl_HE)
e_pl_HE = np.array(e_pl_HE)
qau_pl_HE = aau_pl_HE * (1-e_pl_HE)
irad_pl_HE = np.radians(np.array(ideg_pl_HE))
wrad_pl_HE = np.radians(np.array(wdeg_pl_HE))
Wrad_pl_HE = np.radians(np.array(Wdeg_pl_HE))
Mrad_pl_HE = np.radians(np.array(Mdeg_pl_HE))
pomegarad_pl_HE = np.radians(np.array(pomegadeg_pl_HE))
q_pl_HE = np.sin(irad_pl_HE) * np.cos(Wrad_pl_HE)
p_pl_HE = np.sin(irad_pl_HE) * np.sin(Wrad_pl_HE)
s_pl_HE = np.cos(irad_pl_HE)
k_pl_HE = e_pl_HE * np.cos(pomegarad_pl_HE)
h_pl_HE = e_pl_HE * np.sin(pomegarad_pl_HE)
f_pl_HE = np.sqrt(1-k_pl_HE**2-h_pl_HE**2)
si = np.sin(irad_pl_HE)
ci = np.cos(irad_pl_HE)
sw = np.sin(wrad_pl_HE)
cw = np.cos(wrad_pl_HE)
sW = np.sin(Wrad_pl_HE)
cW = np.cos(Wrad_pl_HE)
k2_pl_HE = -sW*ci*sw+cW*cw
h2_pl_HE =  cW*ci*sw+sW*cw
f2_pl_HE =      si*sw
pomegadeg_pl_HI = np.array(wdeg_pl_HI) + np.array(Wdeg_pl_HI)
pomegadeg_pl_HI = np.mod(pomegadeg_pl_HI,360)
lambdadeg_pl_HI = pomegadeg_pl_HI + Mdeg_pl_HI
lambdadeg_pl_HI = np.mod(lambdadeg_pl_HI,360)
aau_pl_HI = np.array(aau_pl_HI)
e_pl_HI = np.array(e_pl_HI)
qau_pl_HI = aau_pl_HI * (1-e_pl_HI)
irad_pl_HI = np.radians(np.array(ideg_pl_HI))
wrad_pl_HI = np.radians(np.array(wdeg_pl_HI))
Wrad_pl_HI = np.radians(np.array(Wdeg_pl_HI))
Mrad_pl_HI = np.radians(np.array(Mdeg_pl_HI))
pomegarad_pl_HI = np.radians(np.array(pomegadeg_pl_HI))
q_pl_HI = np.sin(irad_pl_HI) * np.cos(Wrad_pl_HI)
p_pl_HI = np.sin(irad_pl_HI) * np.sin(Wrad_pl_HI)
s_pl_HI = np.cos(irad_pl_HI)
k_pl_HI = e_pl_HI * np.cos(pomegarad_pl_HI)
h_pl_HI = e_pl_HI * np.sin(pomegarad_pl_HI)
f_pl_HI = np.sqrt(1-k_pl_HI**2-h_pl_HI**2)
si = np.sin(irad_pl_HI)
ci = np.cos(irad_pl_HI)
sw = np.sin(wrad_pl_HI)
cw = np.cos(wrad_pl_HI)
sW = np.sin(Wrad_pl_HI)
cW = np.cos(Wrad_pl_HI)
k2_pl_HI = -sW*ci*sw+cW*cw
h2_pl_HI =  cW*ci*sw+sW*cw
f2_pl_HI =      si*sw
pomegadeg_pl_BE = np.array(wdeg_pl_BE) + np.array(Wdeg_pl_BE)
pomegadeg_pl_BE = np.mod(pomegadeg_pl_BE,360)
lambdadeg_pl_BE = pomegadeg_pl_BE + Mdeg_pl_BE
lambdadeg_pl_BE = np.mod(lambdadeg_pl_BE,360)
aau_pl_BE = np.array(aau_pl_BE)
e_pl_BE = np.array(e_pl_BE)
qau_pl_BE = aau_pl_BE * (1-e_pl_BE)
irad_pl_BE = np.radians(np.array(ideg_pl_BE))
wrad_pl_BE = np.radians(np.array(wdeg_pl_BE))
Wrad_pl_BE = np.radians(np.array(Wdeg_pl_BE))
Mrad_pl_BE = np.radians(np.array(Mdeg_pl_BE))
pomegarad_pl_BE = np.radians(np.array(pomegadeg_pl_BE))
q_pl_BE = np.sin(irad_pl_BE) * np.cos(Wrad_pl_BE)
p_pl_BE = np.sin(irad_pl_BE) * np.sin(Wrad_pl_BE)
s_pl_BE = np.cos(irad_pl_BE)
k_pl_BE = e_pl_BE * np.cos(pomegarad_pl_BE)
h_pl_BE = e_pl_BE * np.sin(pomegarad_pl_BE)
f_pl_BE = np.sqrt(1-k_pl_BE**2-h_pl_BE**2)
si = np.sin(irad_pl_BE)
ci = np.cos(irad_pl_BE)
sw = np.sin(wrad_pl_BE)
cw = np.cos(wrad_pl_BE)
sW = np.sin(Wrad_pl_BE)
cW = np.cos(Wrad_pl_BE)
k2_pl_BE = -sW*ci*sw+cW*cw
h2_pl_BE =  cW*ci*sw+sW*cw
f2_pl_BE =      si*sw
pomegadeg_pl_BI = np.array(wdeg_pl_BI) + np.array(Wdeg_pl_BI)
pomegadeg_pl_BI = np.mod(pomegadeg_pl_BI,360)
lambdadeg_pl_BI = pomegadeg_pl_BI + Mdeg_pl_BI
lambdadeg_pl_BI = np.mod(lambdadeg_pl_BI,360)
aau_pl_BI = np.array(aau_pl_BI)
e_pl_BI = np.array(e_pl_BI)
qau_pl_BI = aau_pl_BI * (1-e_pl_BI)
irad_pl_BI = np.radians(np.array(ideg_pl_BI))
wrad_pl_BI = np.radians(np.array(wdeg_pl_BI))
Wrad_pl_BI = np.radians(np.array(Wdeg_pl_BI))
Mrad_pl_BI = np.radians(np.array(Mdeg_pl_BI))
pomegarad_pl_BI = np.radians(np.array(pomegadeg_pl_BI))
q_pl_BI = np.sin(irad_pl_BI) * np.cos(Wrad_pl_BI)
p_pl_BI = np.sin(irad_pl_BI) * np.sin(Wrad_pl_BI)
s_pl_BI = np.cos(irad_pl_BI)
k_pl_BI = e_pl_BI * np.cos(pomegarad_pl_BI)
h_pl_BI = e_pl_BI * np.sin(pomegarad_pl_BI)
f_pl_BI = np.sqrt(1-k_pl_BI**2-h_pl_BI**2)
si = np.sin(irad_pl_BI)
ci = np.cos(irad_pl_BI)
sw = np.sin(wrad_pl_BI)
cw = np.cos(wrad_pl_BI)
sW = np.sin(Wrad_pl_BI)
cW = np.cos(Wrad_pl_BI)
k2_pl_BI = -sW*ci*sw+cW*cw
h2_pl_BI =  cW*ci*sw+sW*cw
f2_pl_BI =      si*sw
dictionary = {'ids':ids_mrt_52,'indices_mrt_52':indices_mrt_52,\
              'aau_HE':aau_pl_HE,'e_HE':e_pl_HE,'ideg_HE':ideg_pl_HE,'wdeg_HE':wdeg_pl_HE,'Wdeg_HE':Wdeg_pl_HE,'Mdeg_HE':Mdeg_pl_HE,\
              'xau_HE':xau_pl_HE,'yau_HE':yau_pl_HE,'zau_HE':zau_pl_HE,'vx_HE':vx_pl_HE,'vy_HE':vy_pl_HE,'vz_HE':vz_pl_HE,\
              'pomegadeg_HE':pomegadeg_pl_HE,'lambdadeg_HE':lambdadeg_pl_HE,'qau_HE':qau_pl_HE,'q_HE':q_pl_HE,'p_HE':p_pl_HE,'s_HE':s_pl_HE,\
              'k_HE':k_pl_HE,'h_HE':h_pl_HE,'f_HE':f_pl_HE,'k2_HE':k2_pl_HE,'h2_HE':h2_pl_HE,'f2_HE':f2_pl_HE,\
                
              'aau_HI':aau_pl_HI,'e_HI':e_pl_HI,'ideg_HI':ideg_pl_HI,'wdeg_HI':wdeg_pl_HI,'Wdeg_HI':Wdeg_pl_HI,'Mdeg_HI':Mdeg_pl_HI,\
              'xau_HI':xau_pl_HI,'yau_HI':yau_pl_HI,'zau_HI':zau_pl_HI,'vx_HI':vx_pl_HI,'vy_HI':vy_pl_HI,'vz_HI':vz_pl_HI,\
              'pomegadeg_HI':pomegadeg_pl_HI,'lambdadeg_HI':lambdadeg_pl_HI,'qau_HI':qau_pl_HI,'q_HI':q_pl_HI,'p_HI':p_pl_HI,'s_HI':s_pl_HI,\
              'k_HI':k_pl_HI,'h_HI':h_pl_HI,'f_HI':f_pl_HI,'k2_HI':k2_pl_HI,'h2_HI':h2_pl_HI,'f2_HI':f2_pl_HI,\
                
              'aau_BE':aau_pl_BE,'e_BE':e_pl_BE,'ideg_BE':ideg_pl_BE,'wdeg_BE':wdeg_pl_BE,'Wdeg_BE':Wdeg_pl_BE,'Mdeg_BE':Mdeg_pl_BE,\
              'xau_BE':xau_pl_BE,'yau_BE':yau_pl_BE,'zau_BE':zau_pl_BE,'vx_BE':vx_pl_BE,'vy_BE':vy_pl_BE,'vz_BE':vz_pl_BE,\
              'pomegadeg_BE':pomegadeg_pl_BE,'lambdadeg_BE':lambdadeg_pl_BE,'qau_BE':qau_pl_BE,'q_BE':q_pl_BE,'p_BE':p_pl_BE,'s_BE':s_pl_BE,\
              'k_BE':k_pl_BE,'h_BE':h_pl_BE,'f_BE':f_pl_BE,'k2_BE':k2_pl_BE,'h2_BE':h2_pl_BE,'f2_BE':f2_pl_BE,\
                
              'aau_BI':aau_pl_BI,'e_BI':e_pl_BI,'ideg_BI':ideg_pl_BI,'wdeg_BI':wdeg_pl_BI,'Wdeg_BI':Wdeg_pl_BI,'Mdeg_BI':Mdeg_pl_BI,\
              'xau_BI':xau_pl_BI,'yau_BI':yau_pl_BI,'zau_BI':zau_pl_BI,'vx_BI':vx_pl_BI,'vy_BI':vy_pl_BI,'vz_BI':vz_pl_BI,\
              'pomegadeg_BI':pomegadeg_pl_BI,'lambdadeg_BI':lambdadeg_pl_BI,'qau_BI':qau_pl_BI,'q_BI':q_pl_BI,'p_BI':p_pl_BI,'s_BI':s_pl_BI,\
              'k_BI':k_pl_BI,'h_BI':h_pl_BI,'f_BI':f_pl_BI,'k2_BI':k2_pl_BI,'h2_BI':h2_pl_BI,'f2_BI':f2_pl_BI,\
                  }
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b004_horizons_orbels_p5q2_jd'+jdstr+'_HEHIBEBI.csv',index=False)
#%%
ids_mrt_74 = []
indices_mrt_74 = []
for i in range(n_mrt):
    des = ids_mrt[i]
    classification = classification_mrt[i]
    p = p_mrt[i]
    q = q_mrt[i]
    if classification == 'resonant' and p == '7' and q == '4':
        ids_mrt_74.append(des)
        indices_mrt_74.append(i)
n_mrt_74 = len(ids_mrt_74)
aau_pl_HE = []
e_pl_HE = []
ideg_pl_HE = []
wdeg_pl_HE = []
Wdeg_pl_HE = []
Mdeg_pl_HE = []
for i in range(n_mrt_74):
    des = ids_mrt_74[i]
    idtype = 'smallbody'
    center = '500@10' # heliocenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_HE.append(el['a'][0])
    e_pl_HE.append(el['e'][0])
    ideg_pl_HE.append(el['incl'][0])
    wdeg_pl_HE.append(el['w'][0])
    Wdeg_pl_HE.append(el['Omega'][0])
    Mdeg_pl_HE.append(el['M'][0])
    print('74',i,n_mrt_74,'helio',des)
aau_pl_BE = []
e_pl_BE = []
ideg_pl_BE = []
wdeg_pl_BE = []
Wdeg_pl_BE = []
Mdeg_pl_BE = []
for i in range(n_mrt_74):
    des = ids_mrt_74[i]
    idtype = 'smallbody'
    center = '500@0' # solar system barycenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_BE.append(el['a'][0])
    e_pl_BE.append(el['e'][0])
    ideg_pl_BE.append(el['incl'][0])
    wdeg_pl_BE.append(el['w'][0])
    Wdeg_pl_BE.append(el['Omega'][0])
    Mdeg_pl_BE.append(el['M'][0])
    print('74',i,n_mrt_74,'bary',des)
xau_pl_HE = []
yau_pl_HE = []
zau_pl_HE = []
vx_pl_HE = []
vy_pl_HE = []
vz_pl_HE = []
for i in range(n_mrt_74):
    rvec,vvec = rv_from_orbels(aau_pl_HE[i],aau_pl_HE[i]*(1-e_pl_HE[i]),\
        np.radians(ideg_pl_HE[i]),np.radians(wdeg_pl_HE[i]),np.radians(Wdeg_pl_HE[i]),np.radians(Mdeg_pl_HE[i]))
    xau_pl_HE.append(rvec[0])
    yau_pl_HE.append(rvec[1])
    zau_pl_HE.append(rvec[2])
    vx_pl_HE.append(vvec[0])
    vy_pl_HE.append(vvec[1])
    vz_pl_HE.append(vvec[2])
xau_pl_HE = np.array(xau_pl_HE)
yau_pl_HE = np.array(yau_pl_HE)
zau_pl_HE = np.array(zau_pl_HE)
vx_pl_HE = np.array(vx_pl_HE)
vy_pl_HE = np.array(vy_pl_HE)
vz_pl_HE = np.array(vz_pl_HE)
xau_pl_BE = [] # heliocentric, ecliptic rvec,vvec of Sun are always 0
yau_pl_BE = []
zau_pl_BE = []
vx_pl_BE = []
vy_pl_BE = []
vz_pl_BE = []
for i in range(n_mrt_74):
    rvec,vvec = rv_from_orbels(aau_pl_BE[i],aau_pl_BE[i]*(1-e_pl_BE[i]),\
        np.radians(ideg_pl_BE[i]),np.radians(wdeg_pl_BE[i]),np.radians(Wdeg_pl_BE[i]),np.radians(Mdeg_pl_BE[i]))
    xau_pl_BE.append(rvec[0])
    yau_pl_BE.append(rvec[1])
    zau_pl_BE.append(rvec[2])
    vx_pl_BE.append(vvec[0])
    vy_pl_BE.append(vvec[1])
    vz_pl_BE.append(vvec[2])
xau_pl_BE = np.array(xau_pl_BE)
yau_pl_BE = np.array(yau_pl_BE)
zau_pl_BE = np.array(zau_pl_BE)
vx_pl_BE = np.array(vx_pl_BE)
vy_pl_BE = np.array(vy_pl_BE)
vz_pl_BE = np.array(vz_pl_BE)
xau_pl_HI = [] # heliocentric, invariable plane
yau_pl_HI = []
zau_pl_HI = []
vx_pl_HI = []
vy_pl_HI = []
vz_pl_HI = []
for i in range(n_mrt_74):
    A,xau_HI,yau_HI,zau_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_HE[i]],[yau_pl_HE[i]],[zau_pl_HE[i]],iidegmean,WWdegmean)
    xau_pl_HI.append(xau_HI[0])
    yau_pl_HI.append(yau_HI[0])
    zau_pl_HI.append(zau_HI[0])
    A,vx_HI,vy_HI,vz_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_HE[i]],[vy_pl_HE[i]],[vz_pl_HE[i]],iidegmean,WWdegmean)
    vx_pl_HI.append(vx_HI[0])
    vy_pl_HI.append(vy_HI[0])
    vz_pl_HI.append(vz_HI[0])
xau_pl_BI = [] # barycentric, invariable plane
yau_pl_BI = []
zau_pl_BI = []
vx_pl_BI = []
vy_pl_BI = []
vz_pl_BI = []
for i in range(n_mrt_74):
    A,xau_BI,yau_BI,zau_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_BE[i]],[yau_pl_BE[i]],[zau_pl_BE[i]],iidegmean,WWdegmean)
    xau_pl_BI.append(xau_BI[0])
    yau_pl_BI.append(yau_BI[0])
    zau_pl_BI.append(zau_BI[0])
    A,vx_BI,vy_BI,vz_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_BE[i]],[vy_pl_BE[i]],[vz_pl_BE[i]],iidegmean,WWdegmean)
    vx_pl_BI.append(vx_BI[0])
    vy_pl_BI.append(vy_BI[0])
    vz_pl_BI.append(vz_BI[0])
aau_pl_BI = []
e_pl_BI = []
ideg_pl_BI = []
wdeg_pl_BI = []
Wdeg_pl_BI = []
Mdeg_pl_BI = []
for i in range(n_mrt_74):
    rvec = np.array([xau_pl_BI[i],yau_pl_BI[i],zau_pl_BI[i]])
    vvec = np.array([vx_pl_BI[i],vy_pl_BI[i],vz_pl_BI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_BI.append(aau)
    e_pl_BI.append(1-qperiau/aau)
    ideg_pl_BI.append(np.degrees(irad))
    wdeg_pl_BI.append(np.degrees(wrad))
    Wdeg_pl_BI.append(np.degrees(Wrad))
    Mdeg_pl_BI.append(np.degrees(Mrad))
aau_pl_HI = []
e_pl_HI = []
ideg_pl_HI = []
wdeg_pl_HI = []
Wdeg_pl_HI = []
Mdeg_pl_HI = []
for i in range(n_mrt_74):
    rvec = np.array([xau_pl_HI[i],yau_pl_HI[i],zau_pl_HI[i]])
    vvec = np.array([vx_pl_HI[i],vy_pl_HI[i],vz_pl_HI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_HI.append(aau)
    e_pl_HI.append(1-qperiau/aau)
    ideg_pl_HI.append(np.degrees(irad))
    wdeg_pl_HI.append(np.degrees(wrad))
    Wdeg_pl_HI.append(np.degrees(Wrad))
    Mdeg_pl_HI.append(np.degrees(Mrad))
pomegadeg_pl_HE = np.array(wdeg_pl_HE) + np.array(Wdeg_pl_HE)
pomegadeg_pl_HE = np.mod(pomegadeg_pl_HE,360)
lambdadeg_pl_HE = pomegadeg_pl_HE + Mdeg_pl_HE
lambdadeg_pl_HE = np.mod(lambdadeg_pl_HE,360)
aau_pl_HE = np.array(aau_pl_HE)
e_pl_HE = np.array(e_pl_HE)
qau_pl_HE = aau_pl_HE * (1-e_pl_HE)
irad_pl_HE = np.radians(np.array(ideg_pl_HE))
wrad_pl_HE = np.radians(np.array(wdeg_pl_HE))
Wrad_pl_HE = np.radians(np.array(Wdeg_pl_HE))
Mrad_pl_HE = np.radians(np.array(Mdeg_pl_HE))
pomegarad_pl_HE = np.radians(np.array(pomegadeg_pl_HE))
q_pl_HE = np.sin(irad_pl_HE) * np.cos(Wrad_pl_HE)
p_pl_HE = np.sin(irad_pl_HE) * np.sin(Wrad_pl_HE)
s_pl_HE = np.cos(irad_pl_HE)
k_pl_HE = e_pl_HE * np.cos(pomegarad_pl_HE)
h_pl_HE = e_pl_HE * np.sin(pomegarad_pl_HE)
f_pl_HE = np.sqrt(1-k_pl_HE**2-h_pl_HE**2)
si = np.sin(irad_pl_HE)
ci = np.cos(irad_pl_HE)
sw = np.sin(wrad_pl_HE)
cw = np.cos(wrad_pl_HE)
sW = np.sin(Wrad_pl_HE)
cW = np.cos(Wrad_pl_HE)
k2_pl_HE = -sW*ci*sw+cW*cw
h2_pl_HE =  cW*ci*sw+sW*cw
f2_pl_HE =      si*sw
pomegadeg_pl_HI = np.array(wdeg_pl_HI) + np.array(Wdeg_pl_HI)
pomegadeg_pl_HI = np.mod(pomegadeg_pl_HI,360)
lambdadeg_pl_HI = pomegadeg_pl_HI + Mdeg_pl_HI
lambdadeg_pl_HI = np.mod(lambdadeg_pl_HI,360)
aau_pl_HI = np.array(aau_pl_HI)
e_pl_HI = np.array(e_pl_HI)
qau_pl_HI = aau_pl_HI * (1-e_pl_HI)
irad_pl_HI = np.radians(np.array(ideg_pl_HI))
wrad_pl_HI = np.radians(np.array(wdeg_pl_HI))
Wrad_pl_HI = np.radians(np.array(Wdeg_pl_HI))
Mrad_pl_HI = np.radians(np.array(Mdeg_pl_HI))
pomegarad_pl_HI = np.radians(np.array(pomegadeg_pl_HI))
q_pl_HI = np.sin(irad_pl_HI) * np.cos(Wrad_pl_HI)
p_pl_HI = np.sin(irad_pl_HI) * np.sin(Wrad_pl_HI)
s_pl_HI = np.cos(irad_pl_HI)
k_pl_HI = e_pl_HI * np.cos(pomegarad_pl_HI)
h_pl_HI = e_pl_HI * np.sin(pomegarad_pl_HI)
f_pl_HI = np.sqrt(1-k_pl_HI**2-h_pl_HI**2)
si = np.sin(irad_pl_HI)
ci = np.cos(irad_pl_HI)
sw = np.sin(wrad_pl_HI)
cw = np.cos(wrad_pl_HI)
sW = np.sin(Wrad_pl_HI)
cW = np.cos(Wrad_pl_HI)
k2_pl_HI = -sW*ci*sw+cW*cw
h2_pl_HI =  cW*ci*sw+sW*cw
f2_pl_HI =      si*sw
pomegadeg_pl_BE = np.array(wdeg_pl_BE) + np.array(Wdeg_pl_BE)
pomegadeg_pl_BE = np.mod(pomegadeg_pl_BE,360)
lambdadeg_pl_BE = pomegadeg_pl_BE + Mdeg_pl_BE
lambdadeg_pl_BE = np.mod(lambdadeg_pl_BE,360)
aau_pl_BE = np.array(aau_pl_BE)
e_pl_BE = np.array(e_pl_BE)
qau_pl_BE = aau_pl_BE * (1-e_pl_BE)
irad_pl_BE = np.radians(np.array(ideg_pl_BE))
wrad_pl_BE = np.radians(np.array(wdeg_pl_BE))
Wrad_pl_BE = np.radians(np.array(Wdeg_pl_BE))
Mrad_pl_BE = np.radians(np.array(Mdeg_pl_BE))
pomegarad_pl_BE = np.radians(np.array(pomegadeg_pl_BE))
q_pl_BE = np.sin(irad_pl_BE) * np.cos(Wrad_pl_BE)
p_pl_BE = np.sin(irad_pl_BE) * np.sin(Wrad_pl_BE)
s_pl_BE = np.cos(irad_pl_BE)
k_pl_BE = e_pl_BE * np.cos(pomegarad_pl_BE)
h_pl_BE = e_pl_BE * np.sin(pomegarad_pl_BE)
f_pl_BE = np.sqrt(1-k_pl_BE**2-h_pl_BE**2)
si = np.sin(irad_pl_BE)
ci = np.cos(irad_pl_BE)
sw = np.sin(wrad_pl_BE)
cw = np.cos(wrad_pl_BE)
sW = np.sin(Wrad_pl_BE)
cW = np.cos(Wrad_pl_BE)
k2_pl_BE = -sW*ci*sw+cW*cw
h2_pl_BE =  cW*ci*sw+sW*cw
f2_pl_BE =      si*sw
pomegadeg_pl_BI = np.array(wdeg_pl_BI) + np.array(Wdeg_pl_BI)
pomegadeg_pl_BI = np.mod(pomegadeg_pl_BI,360)
lambdadeg_pl_BI = pomegadeg_pl_BI + Mdeg_pl_BI
lambdadeg_pl_BI = np.mod(lambdadeg_pl_BI,360)
aau_pl_BI = np.array(aau_pl_BI)
e_pl_BI = np.array(e_pl_BI)
qau_pl_BI = aau_pl_BI * (1-e_pl_BI)
irad_pl_BI = np.radians(np.array(ideg_pl_BI))
wrad_pl_BI = np.radians(np.array(wdeg_pl_BI))
Wrad_pl_BI = np.radians(np.array(Wdeg_pl_BI))
Mrad_pl_BI = np.radians(np.array(Mdeg_pl_BI))
pomegarad_pl_BI = np.radians(np.array(pomegadeg_pl_BI))
q_pl_BI = np.sin(irad_pl_BI) * np.cos(Wrad_pl_BI)
p_pl_BI = np.sin(irad_pl_BI) * np.sin(Wrad_pl_BI)
s_pl_BI = np.cos(irad_pl_BI)
k_pl_BI = e_pl_BI * np.cos(pomegarad_pl_BI)
h_pl_BI = e_pl_BI * np.sin(pomegarad_pl_BI)
f_pl_BI = np.sqrt(1-k_pl_BI**2-h_pl_BI**2)
si = np.sin(irad_pl_BI)
ci = np.cos(irad_pl_BI)
sw = np.sin(wrad_pl_BI)
cw = np.cos(wrad_pl_BI)
sW = np.sin(Wrad_pl_BI)
cW = np.cos(Wrad_pl_BI)
k2_pl_BI = -sW*ci*sw+cW*cw
h2_pl_BI =  cW*ci*sw+sW*cw
f2_pl_BI =      si*sw
dictionary = {'ids':ids_mrt_74,'indices_mrt_74':indices_mrt_74,\
              'aau_HE':aau_pl_HE,'e_HE':e_pl_HE,'ideg_HE':ideg_pl_HE,'wdeg_HE':wdeg_pl_HE,'Wdeg_HE':Wdeg_pl_HE,'Mdeg_HE':Mdeg_pl_HE,\
              'xau_HE':xau_pl_HE,'yau_HE':yau_pl_HE,'zau_HE':zau_pl_HE,'vx_HE':vx_pl_HE,'vy_HE':vy_pl_HE,'vz_HE':vz_pl_HE,\
              'pomegadeg_HE':pomegadeg_pl_HE,'lambdadeg_HE':lambdadeg_pl_HE,'qau_HE':qau_pl_HE,'q_HE':q_pl_HE,'p_HE':p_pl_HE,'s_HE':s_pl_HE,\
              'k_HE':k_pl_HE,'h_HE':h_pl_HE,'f_HE':f_pl_HE,'k2_HE':k2_pl_HE,'h2_HE':h2_pl_HE,'f2_HE':f2_pl_HE,\
                
              'aau_HI':aau_pl_HI,'e_HI':e_pl_HI,'ideg_HI':ideg_pl_HI,'wdeg_HI':wdeg_pl_HI,'Wdeg_HI':Wdeg_pl_HI,'Mdeg_HI':Mdeg_pl_HI,\
              'xau_HI':xau_pl_HI,'yau_HI':yau_pl_HI,'zau_HI':zau_pl_HI,'vx_HI':vx_pl_HI,'vy_HI':vy_pl_HI,'vz_HI':vz_pl_HI,\
              'pomegadeg_HI':pomegadeg_pl_HI,'lambdadeg_HI':lambdadeg_pl_HI,'qau_HI':qau_pl_HI,'q_HI':q_pl_HI,'p_HI':p_pl_HI,'s_HI':s_pl_HI,\
              'k_HI':k_pl_HI,'h_HI':h_pl_HI,'f_HI':f_pl_HI,'k2_HI':k2_pl_HI,'h2_HI':h2_pl_HI,'f2_HI':f2_pl_HI,\
                
              'aau_BE':aau_pl_BE,'e_BE':e_pl_BE,'ideg_BE':ideg_pl_BE,'wdeg_BE':wdeg_pl_BE,'Wdeg_BE':Wdeg_pl_BE,'Mdeg_BE':Mdeg_pl_BE,\
              'xau_BE':xau_pl_BE,'yau_BE':yau_pl_BE,'zau_BE':zau_pl_BE,'vx_BE':vx_pl_BE,'vy_BE':vy_pl_BE,'vz_BE':vz_pl_BE,\
              'pomegadeg_BE':pomegadeg_pl_BE,'lambdadeg_BE':lambdadeg_pl_BE,'qau_BE':qau_pl_BE,'q_BE':q_pl_BE,'p_BE':p_pl_BE,'s_BE':s_pl_BE,\
              'k_BE':k_pl_BE,'h_BE':h_pl_BE,'f_BE':f_pl_BE,'k2_BE':k2_pl_BE,'h2_BE':h2_pl_BE,'f2_BE':f2_pl_BE,\
                
              'aau_BI':aau_pl_BI,'e_BI':e_pl_BI,'ideg_BI':ideg_pl_BI,'wdeg_BI':wdeg_pl_BI,'Wdeg_BI':Wdeg_pl_BI,'Mdeg_BI':Mdeg_pl_BI,\
              'xau_BI':xau_pl_BI,'yau_BI':yau_pl_BI,'zau_BI':zau_pl_BI,'vx_BI':vx_pl_BI,'vy_BI':vy_pl_BI,'vz_BI':vz_pl_BI,\
              'pomegadeg_BI':pomegadeg_pl_BI,'lambdadeg_BI':lambdadeg_pl_BI,'qau_BI':qau_pl_BI,'q_BI':q_pl_BI,'p_BI':p_pl_BI,'s_BI':s_pl_BI,\
              'k_BI':k_pl_BI,'h_BI':h_pl_BI,'f_BI':f_pl_BI,'k2_BI':k2_pl_BI,'h2_BI':h2_pl_BI,'f2_BI':f2_pl_BI,\
                  }
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b004_horizons_orbels_p7q4_jd'+jdstr+'_HEHIBEBI.csv',index=False)
#%%
ids_mrt_53 = []
indices_mrt_53 = []
for i in range(n_mrt):
    des = ids_mrt[i]
    classification = classification_mrt[i]
    p = p_mrt[i]
    q = q_mrt[i]
    if classification == 'resonant' and p == '5' and q == '3':
        ids_mrt_53.append(des)
        indices_mrt_53.append(i)
n_mrt_53 = len(ids_mrt_53)
aau_pl_HE = []
e_pl_HE = []
ideg_pl_HE = []
wdeg_pl_HE = []
Wdeg_pl_HE = []
Mdeg_pl_HE = []
for i in range(n_mrt_53):
    des = ids_mrt_53[i]
    idtype = 'smallbody'
    center = '500@10' # heliocenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U53A': # 3316063,K20U53A=2020UA53,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_HE.append(el['a'][0])
    e_pl_HE.append(el['e'][0])
    ideg_pl_HE.append(el['incl'][0])
    wdeg_pl_HE.append(el['w'][0])
    Wdeg_pl_HE.append(el['Omega'][0])
    Mdeg_pl_HE.append(el['M'][0])
    print('53',i,n_mrt_53,'helio',des)
aau_pl_BE = []
e_pl_BE = []
ideg_pl_BE = []
wdeg_pl_BE = []
Wdeg_pl_BE = []
Mdeg_pl_BE = []
for i in range(n_mrt_53):
    des = ids_mrt_53[i]
    idtype = 'smallbody'
    center = '500@0' # solar system barycenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U53A': # 3316063,K20U53A=2020UA53,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    horizons = Horizons(id=des,location=center,epochs=jd)
    el = horizons.elements()
    aau_pl_BE.append(el['a'][0])
    e_pl_BE.append(el['e'][0])
    ideg_pl_BE.append(el['incl'][0])
    wdeg_pl_BE.append(el['w'][0])
    Wdeg_pl_BE.append(el['Omega'][0])
    Mdeg_pl_BE.append(el['M'][0])
    print('53',i,n_mrt_53,'bary',des)
xau_pl_HE = []
yau_pl_HE = []
zau_pl_HE = []
vx_pl_HE = []
vy_pl_HE = []
vz_pl_HE = []
for i in range(n_mrt_53):
    rvec,vvec = rv_from_orbels(aau_pl_HE[i],aau_pl_HE[i]*(1-e_pl_HE[i]),\
        np.radians(ideg_pl_HE[i]),np.radians(wdeg_pl_HE[i]),np.radians(Wdeg_pl_HE[i]),np.radians(Mdeg_pl_HE[i]))
    xau_pl_HE.append(rvec[0])
    yau_pl_HE.append(rvec[1])
    zau_pl_HE.append(rvec[2])
    vx_pl_HE.append(vvec[0])
    vy_pl_HE.append(vvec[1])
    vz_pl_HE.append(vvec[2])
xau_pl_HE = np.array(xau_pl_HE)
yau_pl_HE = np.array(yau_pl_HE)
zau_pl_HE = np.array(zau_pl_HE)
vx_pl_HE = np.array(vx_pl_HE)
vy_pl_HE = np.array(vy_pl_HE)
vz_pl_HE = np.array(vz_pl_HE)
xau_pl_BE = [] # heliocentric, ecliptic rvec,vvec of Sun are always 0
yau_pl_BE = []
zau_pl_BE = []
vx_pl_BE = []
vy_pl_BE = []
vz_pl_BE = []
for i in range(n_mrt_53):
    rvec,vvec = rv_from_orbels(aau_pl_BE[i],aau_pl_BE[i]*(1-e_pl_BE[i]),\
        np.radians(ideg_pl_BE[i]),np.radians(wdeg_pl_BE[i]),np.radians(Wdeg_pl_BE[i]),np.radians(Mdeg_pl_BE[i]))
    xau_pl_BE.append(rvec[0])
    yau_pl_BE.append(rvec[1])
    zau_pl_BE.append(rvec[2])
    vx_pl_BE.append(vvec[0])
    vy_pl_BE.append(vvec[1])
    vz_pl_BE.append(vvec[2])
xau_pl_BE = np.array(xau_pl_BE)
yau_pl_BE = np.array(yau_pl_BE)
zau_pl_BE = np.array(zau_pl_BE)
vx_pl_BE = np.array(vx_pl_BE)
vy_pl_BE = np.array(vy_pl_BE)
vz_pl_BE = np.array(vz_pl_BE)
xau_pl_HI = [] # heliocentric, invariable plane
yau_pl_HI = []
zau_pl_HI = []
vx_pl_HI = []
vy_pl_HI = []
vz_pl_HI = []
for i in range(n_mrt_53):
    A,xau_HI,yau_HI,zau_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_HE[i]],[yau_pl_HE[i]],[zau_pl_HE[i]],iidegmean,WWdegmean)
    xau_pl_HI.append(xau_HI[0])
    yau_pl_HI.append(yau_HI[0])
    zau_pl_HI.append(zau_HI[0])
    A,vx_HI,vy_HI,vz_HI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_HE[i]],[vy_pl_HE[i]],[vz_pl_HE[i]],iidegmean,WWdegmean)
    vx_pl_HI.append(vx_HI[0])
    vy_pl_HI.append(vy_HI[0])
    vz_pl_HI.append(vz_HI[0])
xau_pl_BI = [] # barycentric, invariable plane
yau_pl_BI = []
zau_pl_BI = []
vx_pl_BI = []
vy_pl_BI = []
vz_pl_BI = []
for i in range(n_mrt_53):
    A,xau_BI,yau_BI,zau_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([xau_pl_BE[i]],[yau_pl_BE[i]],[zau_pl_BE[i]],iidegmean,WWdegmean)
    xau_pl_BI.append(xau_BI[0])
    yau_pl_BI.append(yau_BI[0])
    zau_pl_BI.append(zau_BI[0])
    A,vx_BI,vy_BI,vz_BI,iireldeg,WWreldeg = \
        shift_to_pole_angles([vx_pl_BE[i]],[vy_pl_BE[i]],[vz_pl_BE[i]],iidegmean,WWdegmean)
    vx_pl_BI.append(vx_BI[0])
    vy_pl_BI.append(vy_BI[0])
    vz_pl_BI.append(vz_BI[0])
aau_pl_BI = []
e_pl_BI = []
ideg_pl_BI = []
wdeg_pl_BI = []
Wdeg_pl_BI = []
Mdeg_pl_BI = []
for i in range(n_mrt_53):
    rvec = np.array([xau_pl_BI[i],yau_pl_BI[i],zau_pl_BI[i]])
    vvec = np.array([vx_pl_BI[i],vy_pl_BI[i],vz_pl_BI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_BI.append(aau)
    e_pl_BI.append(1-qperiau/aau)
    ideg_pl_BI.append(np.degrees(irad))
    wdeg_pl_BI.append(np.degrees(wrad))
    Wdeg_pl_BI.append(np.degrees(Wrad))
    Mdeg_pl_BI.append(np.degrees(Mrad))
aau_pl_HI = []
e_pl_HI = []
ideg_pl_HI = []
wdeg_pl_HI = []
Wdeg_pl_HI = []
Mdeg_pl_HI = []
for i in range(n_mrt_53):
    rvec = np.array([xau_pl_HI[i],yau_pl_HI[i],zau_pl_HI[i]])
    vvec = np.array([vx_pl_HI[i],vy_pl_HI[i],vz_pl_HI[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_HI.append(aau)
    e_pl_HI.append(1-qperiau/aau)
    ideg_pl_HI.append(np.degrees(irad))
    wdeg_pl_HI.append(np.degrees(wrad))
    Wdeg_pl_HI.append(np.degrees(Wrad))
    Mdeg_pl_HI.append(np.degrees(Mrad))
pomegadeg_pl_HE = np.array(wdeg_pl_HE) + np.array(Wdeg_pl_HE)
pomegadeg_pl_HE = np.mod(pomegadeg_pl_HE,360)
lambdadeg_pl_HE = pomegadeg_pl_HE + Mdeg_pl_HE
lambdadeg_pl_HE = np.mod(lambdadeg_pl_HE,360)
aau_pl_HE = np.array(aau_pl_HE)
e_pl_HE = np.array(e_pl_HE)
qau_pl_HE = aau_pl_HE * (1-e_pl_HE)
irad_pl_HE = np.radians(np.array(ideg_pl_HE))
wrad_pl_HE = np.radians(np.array(wdeg_pl_HE))
Wrad_pl_HE = np.radians(np.array(Wdeg_pl_HE))
Mrad_pl_HE = np.radians(np.array(Mdeg_pl_HE))
pomegarad_pl_HE = np.radians(np.array(pomegadeg_pl_HE))
q_pl_HE = np.sin(irad_pl_HE) * np.cos(Wrad_pl_HE)
p_pl_HE = np.sin(irad_pl_HE) * np.sin(Wrad_pl_HE)
s_pl_HE = np.cos(irad_pl_HE)
k_pl_HE = e_pl_HE * np.cos(pomegarad_pl_HE)
h_pl_HE = e_pl_HE * np.sin(pomegarad_pl_HE)
f_pl_HE = np.sqrt(1-k_pl_HE**2-h_pl_HE**2)
si = np.sin(irad_pl_HE)
ci = np.cos(irad_pl_HE)
sw = np.sin(wrad_pl_HE)
cw = np.cos(wrad_pl_HE)
sW = np.sin(Wrad_pl_HE)
cW = np.cos(Wrad_pl_HE)
k2_pl_HE = -sW*ci*sw+cW*cw
h2_pl_HE =  cW*ci*sw+sW*cw
f2_pl_HE =      si*sw
pomegadeg_pl_HI = np.array(wdeg_pl_HI) + np.array(Wdeg_pl_HI)
pomegadeg_pl_HI = np.mod(pomegadeg_pl_HI,360)
lambdadeg_pl_HI = pomegadeg_pl_HI + Mdeg_pl_HI
lambdadeg_pl_HI = np.mod(lambdadeg_pl_HI,360)
aau_pl_HI = np.array(aau_pl_HI)
e_pl_HI = np.array(e_pl_HI)
qau_pl_HI = aau_pl_HI * (1-e_pl_HI)
irad_pl_HI = np.radians(np.array(ideg_pl_HI))
wrad_pl_HI = np.radians(np.array(wdeg_pl_HI))
Wrad_pl_HI = np.radians(np.array(Wdeg_pl_HI))
Mrad_pl_HI = np.radians(np.array(Mdeg_pl_HI))
pomegarad_pl_HI = np.radians(np.array(pomegadeg_pl_HI))
q_pl_HI = np.sin(irad_pl_HI) * np.cos(Wrad_pl_HI)
p_pl_HI = np.sin(irad_pl_HI) * np.sin(Wrad_pl_HI)
s_pl_HI = np.cos(irad_pl_HI)
k_pl_HI = e_pl_HI * np.cos(pomegarad_pl_HI)
h_pl_HI = e_pl_HI * np.sin(pomegarad_pl_HI)
f_pl_HI = np.sqrt(1-k_pl_HI**2-h_pl_HI**2)
si = np.sin(irad_pl_HI)
ci = np.cos(irad_pl_HI)
sw = np.sin(wrad_pl_HI)
cw = np.cos(wrad_pl_HI)
sW = np.sin(Wrad_pl_HI)
cW = np.cos(Wrad_pl_HI)
k2_pl_HI = -sW*ci*sw+cW*cw
h2_pl_HI =  cW*ci*sw+sW*cw
f2_pl_HI =      si*sw
pomegadeg_pl_BE = np.array(wdeg_pl_BE) + np.array(Wdeg_pl_BE)
pomegadeg_pl_BE = np.mod(pomegadeg_pl_BE,360)
lambdadeg_pl_BE = pomegadeg_pl_BE + Mdeg_pl_BE
lambdadeg_pl_BE = np.mod(lambdadeg_pl_BE,360)
aau_pl_BE = np.array(aau_pl_BE)
e_pl_BE = np.array(e_pl_BE)
qau_pl_BE = aau_pl_BE * (1-e_pl_BE)
irad_pl_BE = np.radians(np.array(ideg_pl_BE))
wrad_pl_BE = np.radians(np.array(wdeg_pl_BE))
Wrad_pl_BE = np.radians(np.array(Wdeg_pl_BE))
Mrad_pl_BE = np.radians(np.array(Mdeg_pl_BE))
pomegarad_pl_BE = np.radians(np.array(pomegadeg_pl_BE))
q_pl_BE = np.sin(irad_pl_BE) * np.cos(Wrad_pl_BE)
p_pl_BE = np.sin(irad_pl_BE) * np.sin(Wrad_pl_BE)
s_pl_BE = np.cos(irad_pl_BE)
k_pl_BE = e_pl_BE * np.cos(pomegarad_pl_BE)
h_pl_BE = e_pl_BE * np.sin(pomegarad_pl_BE)
f_pl_BE = np.sqrt(1-k_pl_BE**2-h_pl_BE**2)
si = np.sin(irad_pl_BE)
ci = np.cos(irad_pl_BE)
sw = np.sin(wrad_pl_BE)
cw = np.cos(wrad_pl_BE)
sW = np.sin(Wrad_pl_BE)
cW = np.cos(Wrad_pl_BE)
k2_pl_BE = -sW*ci*sw+cW*cw
h2_pl_BE =  cW*ci*sw+sW*cw
f2_pl_BE =      si*sw
pomegadeg_pl_BI = np.array(wdeg_pl_BI) + np.array(Wdeg_pl_BI)
pomegadeg_pl_BI = np.mod(pomegadeg_pl_BI,360)
lambdadeg_pl_BI = pomegadeg_pl_BI + Mdeg_pl_BI
lambdadeg_pl_BI = np.mod(lambdadeg_pl_BI,360)
aau_pl_BI = np.array(aau_pl_BI)
e_pl_BI = np.array(e_pl_BI)
qau_pl_BI = aau_pl_BI * (1-e_pl_BI)
irad_pl_BI = np.radians(np.array(ideg_pl_BI))
wrad_pl_BI = np.radians(np.array(wdeg_pl_BI))
Wrad_pl_BI = np.radians(np.array(Wdeg_pl_BI))
Mrad_pl_BI = np.radians(np.array(Mdeg_pl_BI))
pomegarad_pl_BI = np.radians(np.array(pomegadeg_pl_BI))
q_pl_BI = np.sin(irad_pl_BI) * np.cos(Wrad_pl_BI)
p_pl_BI = np.sin(irad_pl_BI) * np.sin(Wrad_pl_BI)
s_pl_BI = np.cos(irad_pl_BI)
k_pl_BI = e_pl_BI * np.cos(pomegarad_pl_BI)
h_pl_BI = e_pl_BI * np.sin(pomegarad_pl_BI)
f_pl_BI = np.sqrt(1-k_pl_BI**2-h_pl_BI**2)
si = np.sin(irad_pl_BI)
ci = np.cos(irad_pl_BI)
sw = np.sin(wrad_pl_BI)
cw = np.cos(wrad_pl_BI)
sW = np.sin(Wrad_pl_BI)
cW = np.cos(Wrad_pl_BI)
k2_pl_BI = -sW*ci*sw+cW*cw
h2_pl_BI =  cW*ci*sw+sW*cw
f2_pl_BI =      si*sw
dictionary = {'ids':ids_mrt_53,'indices_mrt_53':indices_mrt_53,\
              'aau_HE':aau_pl_HE,'e_HE':e_pl_HE,'ideg_HE':ideg_pl_HE,'wdeg_HE':wdeg_pl_HE,'Wdeg_HE':Wdeg_pl_HE,'Mdeg_HE':Mdeg_pl_HE,\
              'xau_HE':xau_pl_HE,'yau_HE':yau_pl_HE,'zau_HE':zau_pl_HE,'vx_HE':vx_pl_HE,'vy_HE':vy_pl_HE,'vz_HE':vz_pl_HE,\
              'pomegadeg_HE':pomegadeg_pl_HE,'lambdadeg_HE':lambdadeg_pl_HE,'qau_HE':qau_pl_HE,'q_HE':q_pl_HE,'p_HE':p_pl_HE,'s_HE':s_pl_HE,\
              'k_HE':k_pl_HE,'h_HE':h_pl_HE,'f_HE':f_pl_HE,'k2_HE':k2_pl_HE,'h2_HE':h2_pl_HE,'f2_HE':f2_pl_HE,\
                
              'aau_HI':aau_pl_HI,'e_HI':e_pl_HI,'ideg_HI':ideg_pl_HI,'wdeg_HI':wdeg_pl_HI,'Wdeg_HI':Wdeg_pl_HI,'Mdeg_HI':Mdeg_pl_HI,\
              'xau_HI':xau_pl_HI,'yau_HI':yau_pl_HI,'zau_HI':zau_pl_HI,'vx_HI':vx_pl_HI,'vy_HI':vy_pl_HI,'vz_HI':vz_pl_HI,\
              'pomegadeg_HI':pomegadeg_pl_HI,'lambdadeg_HI':lambdadeg_pl_HI,'qau_HI':qau_pl_HI,'q_HI':q_pl_HI,'p_HI':p_pl_HI,'s_HI':s_pl_HI,\
              'k_HI':k_pl_HI,'h_HI':h_pl_HI,'f_HI':f_pl_HI,'k2_HI':k2_pl_HI,'h2_HI':h2_pl_HI,'f2_HI':f2_pl_HI,\
                
              'aau_BE':aau_pl_BE,'e_BE':e_pl_BE,'ideg_BE':ideg_pl_BE,'wdeg_BE':wdeg_pl_BE,'Wdeg_BE':Wdeg_pl_BE,'Mdeg_BE':Mdeg_pl_BE,\
              'xau_BE':xau_pl_BE,'yau_BE':yau_pl_BE,'zau_BE':zau_pl_BE,'vx_BE':vx_pl_BE,'vy_BE':vy_pl_BE,'vz_BE':vz_pl_BE,\
              'pomegadeg_BE':pomegadeg_pl_BE,'lambdadeg_BE':lambdadeg_pl_BE,'qau_BE':qau_pl_BE,'q_BE':q_pl_BE,'p_BE':p_pl_BE,'s_BE':s_pl_BE,\
              'k_BE':k_pl_BE,'h_BE':h_pl_BE,'f_BE':f_pl_BE,'k2_BE':k2_pl_BE,'h2_BE':h2_pl_BE,'f2_BE':f2_pl_BE,\
                
              'aau_BI':aau_pl_BI,'e_BI':e_pl_BI,'ideg_BI':ideg_pl_BI,'wdeg_BI':wdeg_pl_BI,'Wdeg_BI':Wdeg_pl_BI,'Mdeg_BI':Mdeg_pl_BI,\
              'xau_BI':xau_pl_BI,'yau_BI':yau_pl_BI,'zau_BI':zau_pl_BI,'vx_BI':vx_pl_BI,'vy_BI':vy_pl_BI,'vz_BI':vz_pl_BI,\
              'pomegadeg_BI':pomegadeg_pl_BI,'lambdadeg_BI':lambdadeg_pl_BI,'qau_BI':qau_pl_BI,'q_BI':q_pl_BI,'p_BI':p_pl_BI,'s_BI':s_pl_BI,\
              'k_BI':k_pl_BI,'h_BI':h_pl_BI,'f_BI':f_pl_BI,'k2_BI':k2_pl_BI,'h2_BI':h2_pl_BI,'f2_BI':f2_pl_BI,\
                  }
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b004_horizons_orbels_p5q3_jd'+jdstr+'_HEHIBEBI.csv',index=False)
#%%
infile = 'a000_rnaasad22d4t1_mrt.txt'
#  1-  7 A7     ---   Des   Minor Planet Center designation (1)
#  9- 18 A10    ---   Class Gladman et al. 2008 dynamical classification
# 20- 21 A2     ---   p     Resonant integer p for resonant objects (2)
# 23- 24 I2     ---   q     Resonant integer q for resonant objects (3)
# 26- 27 A2     ---   soi   Secure (S) or insecure (I) classification (4)
# 29- 36 F8.3   au    a     Barycentric semimajor axis of the best-fit orbit
# 38- 42 F5.3   ---   e     Eccentricity of the best-fit orbit
# 44- 50 F7.3   deg   i     Ecliptic inclination of the best-fit orbit
top_lines = 22
file = open(infile,'r')
lines = file.readlines()
Nlines = len(lines)
designation_list = []
class_list = []
p_list = []
q_list = []
soi_list = []
for iline in range(Nlines):
    # if iline > top_lines-1 and iline < top_lines+1050:
    if iline > top_lines-1:
        line = lines[iline]
        # print(line)
        designation = line[0:7]
        designation = designation.lstrip()
        designation = designation.rstrip()
        designation_list.append(designation)
        classification = line[8:18]
        classification = classification.lstrip()
        classification = classification.rstrip()
        class_list.append(classification)
        p = line[19:21]
        p = p.lstrip()
        p = p.rstrip()
        p_list.append(p)
        q = line[22:24]
        q = q.lstrip()
        q = q.rstrip()
        q_list.append(q)
        soi = line[25:27]
        soi = soi.lstrip()
        soi = soi.rstrip()
        soi_list.append(soi)
        # print(designation,classification,p,q,soi)
nobj = len(designation_list)
jd = 246e4
jdstr = '246e4'
from astroquery.jplhorizons import Horizons
import numpy as np
aau_bary_list = []
e_bary_list = []
ideg_bary_list = []
wdeg_bary_list = []
Wdeg_bary_list = []
Mdeg_bary_list = []
aau_helio_list = []
e_helio_list = []
ideg_helio_list = []
wdeg_helio_list = []
Wdeg_helio_list = []
Mdeg_helio_list = []
for iobj in range(nobj):
    des = designation_list[iobj]
    idtype = 'smallbody'
    center = '500@0' # solar system barycenter
    if des == '134340' : # pluto itself
        des = '9' # pluto-charon barycenter
        idtype = None
    elif des == 'K20U74A': # 3316063,K20U74A=2020UA74,K02PH0S=2002PS170 are the same
        des = 'K02PH0S'
    obj = Horizons(id=des,location=center,epochs=jd,id_type=idtype)
    el = obj.elements()
    aau_bary_list.append(float(el['a']))
    e_bary_list.append(float(el['e']))
    ideg_bary_list.append(float(el['incl']))
    wdeg_bary_list.append(np.mod(float(el['w']),360))
    Wdeg_bary_list.append(np.mod(float(el['Omega']),360))
    Mdeg_bary_list.append(np.mod(float(el['M']),360))
    center = '500@10' # heliocenter
    obj = Horizons(id=des,location=center,epochs=jd,id_type=idtype)
    el = obj.elements()
    aau_helio_list.append(float(el['a']))
    e_helio_list.append(float(el['e']))
    ideg_helio_list.append(float(el['incl']))
    wdeg_helio_list.append(np.mod(float(el['w']),360))
    Wdeg_helio_list.append(np.mod(float(el['Omega']),360))
    Mdeg_helio_list.append(np.mod(float(el['M']),360))
    print(iobj,nobj,des)
print('done with horizons')
#%%
import pandas as pd
infile = 'a000_gstat2026feb12.txt'
df = pd.read_csv(infile,sep=r"\s+")
des_gstat_list = df['des'].to_list()
g_gstat_list = df['g'].to_list()
nobj_gstat = df.shape[0]
no_match_list = []
no_match_ind = []
g_list = -99999*np.ones(nobj)
for iobj in range(nobj_gstat):
    des_gstat = des_gstat_list[iobj]
    if des_gstat not in designation_list:
        no_match_list.append(des_gstat)
        no_match_ind.append(iobj)
        print(iobj,nobj_gstat,des_gstat)
    else:
        des_ind = designation_list.index(des_gstat)
        g_list[des_ind] = g_gstat_list[iobj]
print('no_match_list',no_match_list)
print('no_match_ind',no_match_ind)
print('done matching gstat2026feb12')
dictionary = {'mpc_des':designation_list,\
              'classification_gmv08method_vvl24':class_list,\
              'res_p_largernumber':p_list, 'res_q_smallernumber':q_list,\
              'Secure_or_Insecure':soi_list,\
              'aau_bary':aau_bary_list,\
              'e_bary':e_bary_list,\
              'ideg_bary':ideg_bary_list,\
              'wdeg_bary':wdeg_bary_list,\
              'Wdeg_bary':Wdeg_bary_list,\
              'Mdeg_bary':Mdeg_bary_list,\
              'aau_helio':aau_helio_list,\
              'e_helio':e_helio_list,\
              'ideg_helio':ideg_helio_list,\
              'wdeg_helio':wdeg_helio_list,\
              'Wdeg_helio':Wdeg_helio_list,\
              'Mdeg_helio':Mdeg_helio_list,\
              'g2026feb12':g_list
                  }
df_out = pd.DataFrame.from_dict(dictionary)
df_out.to_csv('b004_tnos_orbels_jd'+jdstr+'.csv',index=False)