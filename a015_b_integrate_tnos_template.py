#%%
def circular_mean(angledeg_vec):
    import numpy as np
    anglerad_vec = np.radians(np.array(angledeg_vec))
    x_vec = np.cos(anglerad_vec)
    y_vec = np.sin(anglerad_vec)
    x_mean = np.mean(x_vec)
    y_mean = np.mean(y_vec)
    anglemeanrad = np.arctan2(y_mean,x_mean)
    anglemeandeg = np.degrees(anglemeanrad)
    return anglemeandeg
#%%
def check_libration(gdeg_vec,center_deg):
    import numpy as np
    gdeg_vec = np.array(gdeg_vec)
    n = len(gdeg_vec)
    gdeg_vec = np.mod(gdeg_vec,360)
    center_deg = np.mod(center_deg,360)
    gdiff_deg = gdeg_vec - center_deg
    gdiff_deg = np.mod(gdiff_deg,360)
    gdiff_deg_2 = []
    for i in range(n):
        g = gdiff_deg[i]
        if g <= 180:
            gdiff_deg_2.append(g)
        else:
            g2 = -(360-g)
            gdiff_deg_2.append(g2)
    gdiff_deg_2 = np.array(gdiff_deg_2)
    # if gdiff_deg_2 is in the range -180 to +180, something librating around 
    # center_deg should have gmin < 0 and gmax > 0,
    # and its mean should be closer to 0 than to +90 or to -90, ie it should be
    # in the range (-45,+45)
    gmin = np.min(gdiff_deg_2)
    gmax = np.max(gdiff_deg_2)
    grange = np.max(gdiff_deg_2) - np.min(gdiff_deg_2)
    gmean_deg_2 = circular_mean(gdiff_deg_2)
    # if grange<179:
    #     check = 1
    # else:
    #     check = 0
    if gmin<0 and gmax>0 and grange<179 and -45<gmean_deg_2<+45:
        check = 1
    else:
        check = 0
    # if gmin<0 and gmax>0 and grange<179:
    #     check = 1
    # else:
    #     check = 0
    return check,center_deg,grange,gmin,gmax
#%%
def classify_glibration(gdeg_vec):
    check0,centerdeg0,grange0,gmin0,gmax0 = check_libration(wdeg_vec_BI,0)
    check90,centerdeg90,grange90,gmin90,gmax90 = check_libration(wdeg_vec_BI,90)
    check180,centerdeg180,grange180,gmin180,gmax180 = check_libration(wdeg_vec_BI,180)
    check270,centerdeg270,grange270,gmin270,gmax270 = check_libration(wdeg_vec_BI,270)
    check_vec = [check0,check90,check180,check270]
    if sum(check_vec) == 0:
        checknone = 1
    else:
        checknone = 0
    if sum(check_vec) > 1:
        checkmulti = 1
    else:
        checkmulti = 0
    return check0,check90,check180,check270,checknone,checkmulti
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
    import numpy as np
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
    rmag_array = np.linalg.norm(rvec_post)
    xhatrel_array = xrel_array/rmag_array
    yhatrel_array = yrel_array/rmag_array
    zhatrel_array = zrel_array/rmag_array
    iirel_rad = np.arccos(zhatrel_array)
    WWrel_rad = np.arctan2(yhatrel_array/np.sin(iirel_rad),xhatrel_array/np.sin(iirel_rad))
    iireldeg = np.degrees(iirel_rad)
    WWreldeg = np.degrees(WWrel_rad)
    return A,xrel_array,yrel_array,zrel_array,iireldeg,WWreldeg
#%%
import pandas as pd
import numpy as np
import time
import rebound
t00 = time.time()
mmrstr = "p3q2"
iobj = 0
jdstr = 'jd246e4'
tyrsmax = "1e7"
tstepyrs = '5e2'
dtdays = '5'
tyrsstr = jdstr+'_'+tyrsmax+'yr_'+tstepyrs+'yr_'+dtdays+'d'
print(mmrstr,tyrsstr)
df_mmr = pd.read_csv('b004_horizons_orbels_'+mmrstr+"_"+jdstr+"_HEHIBEBI.csv")
n = df_mmr.shape[0]
df_planets = pd.read_csv('b003_planets_orbels_'+jdstr+'_HEHIBEBI.csv')
aau_planets_HI = np.array(df_planets['aau_HI'].to_list())
e_planets_HI = np.array(df_planets['e_HI'].to_list())
irad_planets_HI = np.radians(np.array(df_planets['ideg_HI'].to_list()))
wrad_planets_HI = np.radians(np.array(df_planets['wdeg_HI'].to_list()))
Wrad_planets_HI = np.radians(np.array(df_planets['Wdeg_HI'].to_list()))
Mrad_planets_HI = np.radians(np.array(df_planets['Mdeg_HI'].to_list()))
ids_mmr = df_mmr['ids'].to_list()
id_here = ids_mmr[iobj]
aau_mmr_HI = np.array(df_mmr['aau_HI'].to_list())
e_mmr_HI = np.array(df_mmr['e_HI'].to_list())
irad_mmr_HI = np.radians(np.array(df_mmr['ideg_HI'].to_list()))
wrad_mmr_HI = np.radians(np.array(df_mmr['wdeg_HI'].to_list()))
Wrad_mmr_HI = np.radians(np.array(df_mmr['Wdeg_HI'].to_list()))
Mrad_mmr_HI = np.radians(np.array(df_mmr['Mdeg_HI'].to_list()))
time_yrs  = int(float(tyrsmax))
tstep_yrs = int(float(tstepyrs)) # same tstep as in malhotra ito 2025
dt_days = float(dtdays)
dt_yrs = dt_days/365.25
tyrsvec = np.arange(start=0,stop=time_yrs+tstep_yrs,step=tstep_yrs) # t0=0, tf = time_yrs
nt = len(tyrsvec)
mass_types = ['m8ss12']
# mass_types = ['m4mi25']
n_mass_types = len(mass_types)
for mass_index in range(n_mass_types):
    mass_type = mass_types[mass_index]
    masses_file = 'b003_'+mass_type+'.csv'
    masses = np.loadtxt(masses_file) # sun thru neptune
    nplanets = len(masses)-1
    if nplanets == 8:
        offset = 1
    if nplanets == 4:
        offset = 5
    aau_vec_BI = np.zeros(nt)
    e_vec_BI = np.zeros(nt)
    ideg_vec_BI = np.zeros(nt)
    wdeg_vec_BI = np.zeros(nt)
    Wdeg_vec_BI = np.zeros(nt)
    Mdeg_vec_BI = np.zeros(nt)
    pomegadeg_vec_BI = np.zeros(nt)
    lambdadeg_vec_BI = np.zeros(nt)
    lambdadeg_vec_N_BI = np.zeros(nt)
    sim = rebound.Simulation()
    sim.add(m = 1,hash = '0')
    sim.integrator = 'whfast'
    for iplanet in range(nplanets):
        GM = masses[iplanet+1]
        aau  =  aau_planets_HI[iplanet+offset]
        e    =    e_planets_HI[iplanet+offset]
        irad = irad_planets_HI[iplanet+offset]
        wrad = wrad_planets_HI[iplanet+offset]
        Wrad = Wrad_planets_HI[iplanet+offset]
        Mrad = Mrad_planets_HI[iplanet+offset]
        sim.add(primary=sim.particles[0],m=GM,a=aau,e=e,inc=irad,omega=wrad,Omega=Wrad,M=Mrad)
        sim.move_to_com()
    aau = aau_mmr_HI[iobj]
    e = e_mmr_HI[iobj]
    irad = irad_mmr_HI[iobj]
    wrad = wrad_mmr_HI[iobj]
    Wrad = Wrad_mmr_HI[iobj]
    Mrad = Mrad_mmr_HI[iobj]
    sim.add(primary=sim.particles[0],m=0,a=aau,e=e,inc=irad,omega=wrad,Omega=Wrad,M=Mrad)
    sim.dt = dt_yrs * 2*np.pi
    sim.N_active = nplanets + 1 # ie, planets plus sun
    sim.move_to_com()
    for it, t in enumerate(tyrsvec):
        sim.integrate(t*2*np.pi,exact_finish_time=True)
        orbits = sim.orbits(primary=sim.com())
        o = orbits[nplanets]
        aau_vec_BI[it] = o.a
        e_vec_BI[it] = o.e
        ideg_vec_BI[it] = np.degrees(o.inc)
        wdeg_vec_BI[it] = np.mod(np.degrees(o.omega),360)
        Wdeg_vec_BI[it] = np.mod(np.degrees(o.Omega),360)
        Mdeg_vec_BI[it] = np.mod(np.degrees(o.M),360)
        pomegadeg_vec_BI[it] = np.mod(np.degrees(o.pomega),360)
        lambdadeg_vec_BI[it] = np.mod(np.degrees(o.l),360)
        o = orbits[nplanets-1]
        lambdadeg_vec_N_BI[it] = np.mod(np.degrees(o.l),360)
        sim.move_to_com()
    zero_list = []
    plus_list = []
    one80_list = []
    minus_list = []
    none_list = []
    multi_list = []
    all_list = []
    check0,check90,check180,check270,checknone,checkmulti = \
        classify_glibration(wdeg_vec_BI)
    zero_list.append(check0)
    plus_list.append(check90)
    one80_list.append(check180)
    minus_list.append(check270)
    none_list.append(checknone)
    multi_list.append(checkmulti)
    all_list.append(1)
    # dictionary = {'tyrs':tyrsvec,'wdeg_BI':wdeg_vec_BI,'sigma32deg_BI':sigma32deg_vec_BI}
    dictionary = {'g0':zero_list,\
                  'g90':plus_list,'g180':one80_list,'g270':minus_list,\
                  'gnone':none_list,'gmulti':multi_list,'gall':all_list}
    dfout = pd.DataFrame.from_dict(dictionary)
    dfout.to_csv('b015_glibration_classification_tnos'+'_'+mmrstr+'_i'+str(iobj)+'_n'+str(n)+'_'+tyrsstr+'_'+mass_type+'_BI.csv',index=False)
    # np.savetxt('b003_wdeg_tnos'+'_'+mmrstr+'_i'+str(iobj)+'_n'+str(n)+'_'+tyrsstr+'_'+mass_type+'_BI.csv',wdeg_vec_BI,delimiter=',')
    t1 = time.time()
    time_here = (t1-t00)/60
    time_here = np.round(time_here,3)
    print(mmrstr,jdstr,iobj,n,id_here,mass_index,n_mass_types,mass_type,\
          np.round(aau_vec_BI[-1],3),'dt_minutes',time_here,flush=True)
    print(check0,check90,check180,check270,checknone,checkmulti,flush=True)
