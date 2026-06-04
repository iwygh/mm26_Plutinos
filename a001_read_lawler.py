'''
StablePlutinos.txt contains osculating orbital elements, phi_32 libration centres 
and libration amplitudes for all of the non-Kozai Plutinos at the end of the 4Gyr simulation.

StableKozaiPlutinos.txt contains osculating orbital elements, phi_32 libration 
centres and libration amplitudes, and omega libration centres and libration 
amplitudes for all of the stable Kozai Plutinos at the end of the 4Gyr simulation.
'''
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
import numpy as np
import pandas as pd
df_stable = pd.read_csv('a000_lawler_StablePlutinos.txt',delim_whitespace=True)
n_stable = df_stable.shape[0]
a_stable = np.array(df_stable['a'].to_list())
e_stable = np.array(df_stable['e'].to_list())
ideg_stable = np.array(df_stable['inc'].to_list())
wdeg_stable = np.array(df_stable['omega'].to_list())
Wdeg_stable = np.array(df_stable['Omega'].to_list())
Mdeg_stable = np.array(df_stable['Manom'].to_list())
libcendeg_stable = np.array(df_stable['LibCen'].to_list())
libampdeg_stable = np.array(df_stable['LibAmp'].to_list())
df_kozai = pd.read_csv('a000_lawler_StableKozaiPlutinos.txt',delim_whitespace=True)
n_kozai = df_kozai.shape[0]
a_kozai = np.array(df_kozai['a'].to_list())
e_kozai = np.array(df_kozai['e'].to_list())
ideg_kozai = np.array(df_kozai['inc'].to_list())
wdeg_kozai = np.array(df_kozai['omega'].to_list())
Wdeg_kozai = np.array(df_kozai['Omega'].to_list())
Mdeg_kozai = np.array(df_kozai['Manom'].to_list())
libcendeg_kozai = np.array(df_kozai['LibCen'].to_list())
libampdeg_kozai = np.array(df_kozai['LibAmp'].to_list())
kozcendeg_kozai = np.array(df_kozai['KozCen'].to_list())
kozampdeg_kozai = np.array(df_kozai['KozAmp'].to_list())
gplus_indices = []
gminus_indices = []
for i_kozai in range(n_kozai):
    kozcendeg = kozcendeg_kozai[i_kozai]
    if kozcendeg < 180:
        gplus_indices.append(i_kozai)
    else:
        gminus_indices.append(i_kozai)
dictionary = {'index':gplus_indices}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b001_lawler_gplus_indices.txt',index=False)
dictionary = {'index':gminus_indices}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b001_lawler_gminus_indices.txt',index=False)
#%%
import rebound
sim = rebound.Simulation()
bodies = ['Sun','Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
for body in bodies:
    sim.add(body)
N = len(bodies)
masses = np.zeros(N)
for i in range(N):
    masses[i] = sim.particles[i].m
m_sun = np.sum(masses[0:5])
m_sun_2 = m_sun/m_sun
m_jupiter_2 = masses[5]/m_sun
m_saturn_2 = masses[6]/m_sun
m_uranus_2 = masses[7]/m_sun
m_neptune_2 = masses[8]/m_sun
masses_2 = [m_sun_2,m_jupiter_2,m_saturn_2,m_uranus_2,m_neptune_2]
np.savetxt('b001_lawler_masses.csv',masses,delimiter=',')
np.savetxt('b001_lawler_masses_reduced.csv',masses_2,delimiter=',')
#%%
df = pd.read_csv('a000_lawler_plEndState_edited_HE.csv')
df.to_csv('b003_lawler_plEndState_edited_HE.csv',index=False)
aau_pl_HE = np.array(df['aau'].to_list()) # HE = heliocentric, ecliptic
e_pl_HE = np.array(df['e'].to_list())
irad_pl_HE = np.radians(np.array(df['ideg'].to_list()))
wrad_pl_HE = np.radians(np.array(df['wdeg'].to_list()))
Wrad_pl_HE = np.radians(np.array(df['Wdeg'].to_list()))
Mrad_pl_HE = np.radians(np.array(df['Mdeg'].to_list()))
qperiau_pl_HE = aau_pl_HE * (1-e_pl_HE)
masses = np.loadtxt('b001_lawler_masses_reduced.csv',delimiter=',')
# masses = masses[1:]
xau_pl_HE = [0] # heliocentric, ecliptic rvec,vvec of Sun are always 0
yau_pl_HE = [0]
zau_pl_HE = [0]
vx_pl_HE = [0]
vy_pl_HE = [0]
vz_pl_HE = [0]
n = len(aau_pl_HE)
for i in range(n):
    rvec,vvec = rv_from_orbels(aau_pl_HE[i],qperiau_pl_HE[i],irad_pl_HE[i],wrad_pl_HE[i],Wrad_pl_HE[i],Mrad_pl_HE[i])
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
xau_bary_HE = np.dot(xau_pl_HE,masses)/np.sum(masses)
yau_bary_HE = np.dot(yau_pl_HE,masses)/np.sum(masses)
zau_bary_HE = np.dot(zau_pl_HE,masses)/np.sum(masses)
vx_bary_HE = np.dot(vx_pl_HE,masses)/np.sum(masses)
vy_bary_HE = np.dot(vy_pl_HE,masses)/np.sum(masses)
vz_bary_HE = np.dot(vz_pl_HE,masses)/np.sum(masses)
xau_pl_BE = xau_pl_HE - xau_bary_HE # barycentric, ecliptic
yau_pl_BE = yau_pl_HE - yau_bary_HE # barycentric, ecliptic
zau_pl_BE = zau_pl_HE - zau_bary_HE # barycentric, ecliptic
vx_pl_BE = vx_pl_HE - vx_bary_HE # barycentric, ecliptic
vy_pl_BE = vy_pl_HE - vy_bary_HE # barycentric, ecliptic
vz_pl_BE = vz_pl_HE - vz_bary_HE # barycentric, ecliptic
aau_pl_BE = []
e_pl_BE = []
ideg_pl_BE = []
wdeg_pl_BE = []
Wdeg_pl_BE = []
Mdeg_pl_BE = []
for i in range(n+1):
    rvec = np.array([xau_pl_BE[i],yau_pl_BE[i],zau_pl_BE[i]])
    vvec = np.array([vx_pl_BE[i],vy_pl_BE[i],vz_pl_BE[i]])
    aau,qperiau,irad,wrad,Wrad,Mrad = orbels_from_rv(rvec,vvec)
    aau_pl_BE.append(aau)
    e_pl_BE.append(1-qperiau/aau)
    ideg_pl_BE.append(np.degrees(irad))
    wdeg_pl_BE.append(np.degrees(wrad))
    Wdeg_pl_BE.append(np.degrees(Wrad))
    Mdeg_pl_BE.append(np.degrees(Mrad))
dictionary = {'masses':masses,'aau':aau_pl_BE,'e':e_pl_BE,'ideg':ideg_pl_BE,\
              'wdeg':wdeg_pl_BE,'Wdeg':Wdeg_pl_BE,'Mdeg':Mdeg_pl_BE}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b001_lawler_plEndState_edited_BE.csv',index=False)
Lvec = np.array([0,0,0])
for i in range(n+1):
    rterm = np.array([xau_pl_BE[i],yau_pl_BE[i],zau_pl_BE[i]])
    vterm = np.array([vx_pl_BE[i],vy_pl_BE[i],vz_pl_BE[i]])
    Lvec = Lvec + np.cross(rterm,vterm) * masses[i]
hvec_invar = Lvec/np.linalg.norm(Lvec)
qinvar = -hvec_invar[1]
pinvar =  hvec_invar[0]
sinvar =  hvec_invar[2]
irad_invar = np.arccos(sinvar)
sini = np.sin(irad_invar)
Wrad_invar = np.arctan2(pinvar/sini,qinvar/sini)
ideginvar = np.degrees(irad_invar)
Wdeginvar = np.degrees(Wrad_invar)
iWqp_invar = np.array([ideginvar,Wdeginvar,qinvar,pinvar])
dictionary = {'ideginvar':[ideginvar],'Wdeginvar':[Wdeginvar],'qinvar':[qinvar],'pinvar':[pinvar]}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b001_lawler_plEndState_idegWdegqpinvar_BE.csv',index=False)


