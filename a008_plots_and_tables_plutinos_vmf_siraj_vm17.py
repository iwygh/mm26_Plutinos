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
def logL(xin,aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array):
    import numpy as np
    import scipy
    nobj = len(aau_array)
    rmat = np.zeros((3,nobj))
    vmat = np.zeros((3,nobj))
    Jhatmat = np.zeros((3,nobj))
    for iobj in range(nobj):
        aau = aau_array[iobj]
        qperiau = qperiau_array[iobj]
        irad = irad_array[iobj]
        wrad = wrad_array[iobj]
        Wrad = Wrad_array[iobj]
        Mrad = Mrad_array[iobj]
        rvec,vvec = rv_from_orbels(aau,qperiau,irad,wrad,Wrad,Mrad)
        Jhat_vec = Jhat_from_rv(rvec,vvec)
        rmat[:,iobj] = rvec
        vmat[:,iobj] = vvec
        Jhatmat[:,iobj] = Jhat_vec
    gamma = xin[0]
    mhat = xin[1:]
    Jhat_sum = np.sum(Jhatmat,axis=1)
    term1 = gamma * np.dot(mhat,Jhat_sum)
    term2 = 0
    for iobj in range(nobj):
        rmag = np.linalg.norm(rmat[:,iobj])
        rhatvec = rmat[:,iobj]/rmag
        rhatvec = np.ndarray.flatten(rhatvec)
        val = gamma * np.sin(np.arccos(np.dot(mhat,rhatvec)))
        term2 = term2 + np.log(scipy.special.i0(val))
    logL = term1 - term2
    return logL
#%%
def logL_ref(xin,aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array,xref,yref,zref):
    import numpy as np
    import scipy
    nobj = len(aau_array)
    rmat = np.zeros((3,nobj))
    vmat = np.zeros((3,nobj))
    Jhatmat = np.zeros((3,nobj))
    for iobj in range(nobj):
        aau = aau_array[iobj]
        qperiau = qperiau_array[iobj]
        irad = irad_array[iobj]
        wrad = wrad_array[iobj]
        Wrad = Wrad_array[iobj]
        Mrad = Mrad_array[iobj]
        rvec,vvec = rv_from_orbels(aau,qperiau,irad,wrad,Wrad,Mrad)
        Jhat_vec = Jhat_from_rv(rvec,vvec)
        rmat[:,iobj] = rvec
        vmat[:,iobj] = vvec
        Jhatmat[:,iobj] = Jhat_vec
    gamma = xin
    mhat = np.array([xref,yref,zref])
    Jhat_sum = np.sum(Jhatmat,axis=1)
    term1 = gamma * np.dot(mhat,Jhat_sum)
    term2 = 0
    for iobj in range(nobj):
        rmag = np.linalg.norm(rmat[:,iobj])
        rhatvec = rmat[:,iobj]/rmag
        rhatvec = np.ndarray.flatten(rhatvec)
        val = gamma * np.sin(np.arccos(np.dot(mhat,rhatvec)))
        term2 = term2 + np.log(scipy.special.i0(val))
    logL = term1 - term2
    return logL
#%%
def maxlogL(aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array,delta,n_q):
    # from scipy.optimize import minimize
    # import numpy as np
    # fun = lambda x: -logL(x,aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array)
    # x0_list = [10,0,0,1]
    # x0 = np.array(x0_list)
    # idegmax = 30
    # cosi = np.cos(np.radians(idegmax))
    # sini = np.sin(np.radians(idegmax))
    # bnds = ((0,None),(-sini,sini),(-sini,sini),(cosi,1))
    # cons = ({'type':'eq','fun':lambda x: x[1]**2+x[2]**2+x[3]**2-1})
    # res = minimize(fun,x0,method='trust-constr',bounds=bnds,constraints=cons)
    from scipy.optimize import minimize
    import numpy as np
    iteration = 0
    # delta = 0.4
    # n_q = 400
    step = 2*delta/n_q
    tol = step/10
    diff = 1e9
    idegmax = 30
    cosi = np.cos(np.radians(idegmax))
    sini = np.sin(np.radians(idegmax))
    fun = lambda x: -logL(x,aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array)
    q_array = np.sin(irad_array)*np.cos(Wrad_array)
    p_array = np.sin(irad_array)*np.sin(Wrad_array)
    # s_array = np.cos(irad_array)
    hx_array = -p_array
    hy_array = +q_array
    # hz_array = +s_array
    gamma_guess = 20
    hx_guess = np.mean(hx_array)
    hy_guess = np.mean(hy_array)
    hz_guess = np.sqrt(1-hx_guess**2-hy_guess**2)
    x_in = [gamma_guess,hx_guess,hy_guess,hz_guess]
    x_in = [10,0,0,1] # gamma, hx, hy, hz
    # x_in = [20,np.mean(q_array),np.mean(p_array),np.sqrt(1-np.mean(q_array)**2-np.mean(p_array)**2)]
    while diff > tol:
        iteration = iteration + 1
        print('starting maxlogL iteration',iteration)
        bnds = ((0,None),(-sini,sini),(-sini,sini),(cosi,1))
        cons = ({'type':'eq','fun':lambda x: x[1]**2+x[2]**2+x[3]**2-1})
        res = minimize(fun,x_in,method='trust-constr',bounds=bnds,constraints=cons)
        # res = minimize(fun,x_in,method='L-BFGS-B',constraints=cons,bounds=bnds)
        x_out = res.x
        diff = (x_out[1]-x_in[1])**2 + (x_out[2]-x_in[2])**2 + (x_out[3]-x_in[3])**2
        print('end maxlogL iteration',iteration)
        print('diff = ',diff)
        x_in = x_out
    print(res.x)
    return res
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
def Jhat_from_rv(rvec,vvec):
    import numpy as np
    r2 = np.reshape(rvec,(1,3))
    v2 = np.reshape(vvec,(1,3))
    Jvec = np.cross(r2,v2)
    Jvec = np.ndarray.flatten(Jvec)
    Jmag = np.linalg.norm(Jvec)
    Jhat_vec = Jvec/Jmag
    return Jhat_vec
#%%
def el2xv(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array):
    n = len(aau_array)
    GM = 1
    a = aau_array
    e = e_array
    inc = irad_array
    capom = Wrad_array
    omega = wrad_array
    capmnq = Mrad_array
    sp = np.sin(omega)
    cp = np.cos(omega)
    so = np.sin(capom)
    co = np.cos(capom)
    si = np.sin(inc)
    ci = np.cos(inc)
    d11 = cp*co - sp*so*ci
    d12 = cp*so + sp*co*ci
    d13 = sp*si
    d21 = -sp*co - cp*so*ci
    d22 = -sp*so + cp*co*ci
    d23 = cp*si
    cape = np.zeros(n)
    for iobj in range(n):
        cape[iobj] = solve_kepler(e[iobj],capmnq[iobj]) # mean anomaly to eccentric anomaly
    scap = np.sin(cape)
    ccap = np.cos(cape)
    sqe = np.sqrt(1.0-e*e)
    sqgma = np.sqrt(GM*a)
    xfac1 = a*(ccap - e)
    xfac2 = a*sqe*scap
    ri = 1.0/(a*(1.0 - e*ccap))
    vfac1 = -ri * sqgma * scap
    vfac2 = ri * sqgma * sqe * ccap
    x =  d11*xfac1 + d21*xfac2
    y =  d12*xfac1 + d22*xfac2
    z =  d13*xfac1 + d23*xfac2
    vx = d11*vfac1 + d21*vfac2
    vy = d12*vfac1 + d22*vfac2
    vz = d13*vfac1 + d23*vfac2
    X_array = x
    Y_array = y
    Z_array = z
    VX_array = vx
    VY_array = vy
    VZ_array = vz
    return X_array,Y_array,Z_array,VX_array,VY_array,VZ_array
# ******************************************************************************
# subroutine el2xv(GM,a,e,inc,capom,omega,capmnq,x,y,z,vx,vy,vz)
# implicit none
# !..............................................................................
# ! arguments
# real*8 GM,a,e,inc,capom,omega,capmnq
# real*8 x,y,z,vx,vy,vz
# !..............................................................................
# ! internal variables
# real*8 ehybrid,cape,fhybrid,capf,zget,zpara
# real*8 sp,cp,so,co,si,ci
# real*8 d11,d12,d13,d21,d22,d23
# real*8 scap,ccap,shcap,chcap
# real*8 sqe,sqgma,xfac1,xfac2,ri,vfac1,vfac2
# !------------------------------------------------------------------------------
# ! Generate rotation matrices (on p. 42 of Fitzpatrick)
# call scget(omega,sp,cp)
# call scget(capom,so,co)
# call scget(inc,si,ci)
# d11 = cp*co - sp*so*ci
# d12 = cp*so + sp*co*ci
# d13 = sp*si
# d21 = -sp*co - cp*so*ci
# d22 = -sp*so + cp*co*ci
# d23 = cp*si
# ! Get the other quantities depending on orbit type ( i.e. IALPHA)
# ! only valid for ellipses
# cape = ehybrid(e,capmnq)
# call scget(cape,scap,ccap)
# sqe = dsqrt(1.d0 -e*e)
# sqgma = dsqrt(GM*a)
# xfac1 = a*(ccap - e)
# xfac2 = a*sqe*scap
# ri = 1.d0/(a*(1.d0 - e*ccap))
# vfac1 = -ri * sqgma * scap
# vfac2 = ri * sqgma * sqe * ccap
# x =  d11*xfac1 + d21*xfac2
# y =  d12*xfac1 + d22*xfac2
# z =  d13*xfac1 + d23*xfac2
# vx = d11*vfac1 + d21*vfac2
# vy = d12*vfac1 + d22*vfac2
# vz = d13*vfac1 + d23*vfac2
# return
# end subroutine
#%%
def orbels_to_cartesian(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array):
    import numpy as np
    n = len(aau_array)
    aau = aau_array
    e = e_array
    i = irad_array
    w = wrad_array
    W = Wrad_array
    M = Mrad_array
    p = aau*(1-e**2)
    E = np.zeros(n)
    for iobj in range(n):
        E[iobj] = solve_kepler(e[iobj],M[iobj]) # mean anomaly to eccentric anomaly
    nu = 2 * np.arctan( np.sqrt((1+e)/(1-e)) * np.tan(E/2) ) # true anomaly
    r = p/(1+e*np.cos(nu))
    # perifocal plane coordinates
    x = r*np.cos(nu)
    y = r*np.sin(nu)
    vx = -np.sin(nu)/np.sqrt(p)
    vy = (e+np.cos(nu))/np.sqrt(p)
    # ecliptic coordinates
    X_array = (np.cos(W) * np.cos(w) - np.sin(W) * np.sin(w) * np.cos(i)) * x + (-np.cos(W) * np.sin(w) - np.sin(W) * np.cos(w) * np.cos(i)) * y;
    Y_array = (np.sin(W) * np.cos(w) + np.cos(W) * np.sin(w) * np.cos(i)) * x + (-np.sin(W) * np.sin(w) + np.cos(W) * np.cos(w) * np.cos(i)) * y;
    Z_array = (np.sin(w) * np.sin(i)) * x + (np.cos(w) * np.sin(i)) * y;
    VX_array = (np.cos(W) * np.cos(w) - np.sin(W) * np.sin(w) * np.cos(i)) * vx + (-np.cos(W) * np.sin(w) - np.sin(W) * np.cos(w) * np.cos(i)) * vy;
    VY_array = (np.sin(W) * np.cos(w) + np.cos(W) * np.sin(w) * np.cos(i)) * vx + (-np.sin(W) * np.sin(w) + np.cos(W) * np.cos(w) * np.cos(i)) * vy;
    VZ_array = (np.sin(w) * np.sin(i)) * vx + (np.cos(w) * np.sin(i)) * vy;
    return X_array,Y_array,Z_array,VX_array,VY_array,VZ_array
#%%
def sky_velocity_vectors_curtis(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array):
    import numpy as np
    n = len(aau_array)
    vtx_array = np.zeros(n)
    vty_array = np.zeros(n)
    vtz_array = np.zeros(n)
    hxhat_array =  np.sin(irad_array)*np.sin(Wrad_array) # x component of orbit normal vector
    hyhat_array = -np.sin(irad_array)*np.cos(Wrad_array) # y component of orbit normal vector
    hzhat_array =  np.cos(irad_array) # z component of orbit normal vector
    x_array,y_array,z_array,vx_array,vy_array,vz_array = \
        el2xv(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array)
    for iobj in range(n):
        rvec = np.array([x_array[iobj],y_array[iobj],z_array[iobj]])
        rhatvec = rvec/np.linalg.norm(rvec)
        hhatvec = np.array([hxhat_array[iobj],hyhat_array[iobj],hzhat_array[iobj]])
        vtvec = np.cross(hhatvec,rhatvec)
        vtx_array[iobj] = vtvec[0]
        vty_array[iobj] = vtvec[1]
        vtz_array[iobj] = vtvec[2]
    return vtx_array,vty_array,vtz_array
#%%
def solve_kepler_alt(e,M,ratio): # Curtis algorithm 3.1, really just Newton-Raphson
    import numpy as np
    Etol = 1e-13
    # ratio = 1
    if M < np.pi:
        E = M + e/2
    else:
        E = M - e/2
    while np.abs(ratio) > Etol:
        top = E - e*np.sin(E) - M
        bottom = 1 - e*np.cos(E)
        ratio = top/bottom
        E = E - ratio
    return E,ratio
#%%
def sky_velocity_vectors_schaubandjunkins(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array):
    import numpy as np
    n = len(aau_array)
    vtx_array = np.zeros(n)
    vty_array = np.zeros(n)
    vtz_array = np.zeros(n)
    for iobj in range(n):
        hxhat =  np.sin(irad_array[iobj])*np.sin(Wrad_array[iobj]) # x component of orbit normal vector
        hyhat = -np.sin(irad_array[iobj])*np.cos(Wrad_array[iobj]) # y component of orbit normal vector
        hzhat =  np.cos(irad_array[iobj]) # z component of orbit normal vector
        Erad = solve_kepler(e_array[iobj],Mrad_array[iobj]) # mean anomaly to eccentric anomaly
        frad = 2 * np.arctan( np.sqrt(1+e_array[iobj])/np.sqrt(1-e_array[iobj]) * np.tan(Erad/2) ) # true anomaly
        thetarad = wrad_array[iobj] + frad
        xhat = np.cos(Wrad_array[iobj])*np.cos(thetarad) - \
            np.sin(Wrad_array[iobj])*np.sin(thetarad)*np.cos(irad_array[iobj]) # unit position vectors, Schaub & Junkins eq 9.164
        yhat = np.sin(Wrad_array[iobj])*np.cos(thetarad) + \
            np.cos(Wrad_array[iobj])*np.sin(thetarad)*np.cos(irad_array[iobj])
        zhat = np.sin(thetarad)*np.sin(irad_array[iobj])
        rhatvec = np.array([xhat,yhat,zhat])
        hhatvec = np.array([hxhat,hyhat,hzhat])
        vtvec = np.cross(hhatvec,rhatvec)
        vtx_array[iobj] = vtvec[0]
        vty_array[iobj] = vtvec[1]
        vtz_array[iobj] = vtvec[2]
    return vtx_array,vty_array,vtz_array
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
def probmass_inside_ellipse_thru_point_vm17(xvec,yvec,xpt,ypt):
    import numpy as np
    from scipy.stats import chi2 as sschi2
    cov = np.cov([xvec,yvec])
    diff = np.array([xpt-np.mean(xvec),ypt-np.mean(yvec)])
    inv = np.linalg.inv(cov)
    tdiff = np.transpose(diff)
    mabis = np.sqrt(np.matmul(tdiff,np.matmul(inv,diff))) # mahalanobis distance
    probmass = sschi2.cdf(mabis**2,2)
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
def probmass_siraj(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,xpt,ypt):
    import numpy as np
    delta = 0.4
    n_q = 400
    qperiau_array = aau_array * (1-e_array)
    res_out_array = maxlogL(aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array,delta,n_q)
    logLmax_siraj = -res_out_array.fun
    gamma_siraj = res_out_array.x[0]
    zpt = np.sqrt(1-xpt**2-ypt**2)
    hx = ypt
    hy = -xpt
    hz = zpt
    xin = gamma_siraj
    logL_pt = logL_ref(xin,aau_array,qperiau_array,irad_array,\
            wrad_array,Wrad_array,Mrad_array,hx,hy,hz)
    sigma_pt = np.sqrt(2*(logLmax_siraj-logL_pt))
    pval_pt = np.exp(-1/2*sigma_pt**2)
    probmass = 1-pval_pt
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
    import numpy as np
    curve_vec_exact = kappa/(np.exp(kappa)-np.exp(-kappa))*\
        np.exp(kappa*np.cos(radians_vec))*np.sin(radians_vec)
    return curve_vec_exact
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
def angle_between_points_qp1_qp2(q1,p1,q2,p2):
    import numpy as np
    s1 = np.sqrt(1-q1**2-p1**2)
    s2 = np.sqrt(1-q2**2-p2**2)
    dot = q1*q2 + p1*p2 + s1*s2
    deg_distance = np.degrees(np.arccos(dot))
    return deg_distance
#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
th = np.linspace(start=0,stop=2*np.pi,num=100,endpoint=True)
costh = np.cos(th)
sinth = np.sin(th)
probmass = 0.95
delta = 0.4
n_q = 400
#%%
lenth = 200
njobs = 1000
nreps = 40
infile_vm17 = 'b007_d_consolidated_mean_planes_vm17_2026feb12_plutinos_gplusminusnone_njobs'+\
    str(njobs)+'_nreps'+str(nreps)+'.csv'
df_vm17 = pd.read_csv(infile_vm17)
n_vm17 = df_vm17.shape[0]
i_mid_deg_vm17 = np.array(df_vm17['i_mid_deg'].to_list())
W_mid_deg_vm17 = np.array(df_vm17['node_mid_deg'].to_list())
irad_vm17 = np.radians(i_mid_deg_vm17)
Wrad_vm17 = np.radians(W_mid_deg_vm17)
q_vm17 = np.sin(irad_vm17)*np.cos(Wrad_vm17)
p_vm17 = np.sin(irad_vm17)*np.sin(Wrad_vm17)
q_ellipse_vm17,p_ellipse_vm17,a0,b0,siga,sigb,rhoab,chi2val,\
    ecc,phideg,semimajoraxis,semiminoraxis = ellipse_points_2(q_vm17,p_vm17,probmass,lenth)
infile_vm17_1 = 'b007_fortran_mean_planes_vm17_2026feb12_plutinos_ijob1_njobs'+str(njobs)+'_nreps'+str(nreps)+'.txt'
f = open(infile_vm17_1,'r')
flines = f.readlines()
fsplit = flines[0].split()
idegmean_vm17 = float(fsplit[2])
Wdegmean_vm17 = float(fsplit[3])
iradmean_vm17 = np.radians(idegmean_vm17)
Wradmean_vm17 = np.radians(Wdegmean_vm17)
qmean_vm17 = np.sin(iradmean_vm17)*np.cos(Wradmean_vm17)
pmean_vm17 = np.sin(iradmean_vm17)*np.sin(Wradmean_vm17)
smean_vm17 = np.cos(iradmean_vm17)
angledeg_vm17 = np.degrees(np.arcsin(np.sqrt(semimajoraxis*semiminoraxis)))
#%%
libration = 'gplusminusnone_2026feb12'
# libration = 'gall_2026feb12'
dfind = pd.read_csv('b004_p3q2_'+libration+'_index.csv')
indices = dfind['index'].to_list()
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
des_list_2 = []
for index in indices:
    des_list_2.append(des_list[index])
aau_array = aau_array[indices]
e_array = e_array[indices]
irad_array = irad_array[indices]
wrad_array = wrad_array[indices]
Wrad_array = Wrad_array[indices]
Mrad_array = Mrad_array[indices]
q_array = q_array[indices]
p_array = p_array[indices]
s_array = s_array[indices]
idegmean_vmf,Wdegmean_vmf,qmean_vmf,pmean_vmf,smean_vmf,angledegs,sigmahat,Rbar,kappa_vmf,kappahat = \
    vmf_fun(q_array,p_array,s_array,np.array([probmass]))
iradmean_vmf = np.radians(idegmean_vmf)
Wradmean_vmf = np.radians(Wdegmean_vmf)
angledeg_vmf = angledegs[0]
#%%
qmat_siraj = np.loadtxt('b006_2026feb12_plutinos_gplusminusnone_siraj_qmat_delta0.4_nq400_np400.csv',delimiter=',')
pmat_siraj = np.loadtxt('b006_2026feb12_plutinos_gplusminusnone_siraj_pmat_delta0.4_nq400_np400.csv',delimiter=',')
pvalmat_siraj = np.loadtxt('b006_2026feb12_plutinos_gplusminusnone_siraj_pvalmat_delta0.4_nq400_np400.csv',delimiter=',')
fig_dummy = plt.figure(figsize=(10,10))
ax_dummy = fig_dummy.add_subplot(111)
quadset = ax_dummy.contour(qmat_siraj,pmat_siraj,pvalmat_siraj,levels=[1-probmass])
plt.close()
segs = quadset.allsegs[0]
segs_array = np.array(segs)
segs_flat = np.squeeze(segs_array)
q_ellipse_siraj = segs_flat[:,0]
p_ellipse_siraj = segs_flat[:,1]
df_siraj = pd.read_csv('b006_2026feb12_plutinos_gplusminusnone_siraj.csv')
idegmean_siraj = df_siraj['ideg_siraj'][0]
Wdegmean_siraj = df_siraj['Wdeg_siraj'][0]
iradmean_siraj = np.radians(idegmean_siraj)
Wradmean_siraj = np.radians(Wdegmean_siraj)
qmean_siraj = df_siraj['q_siraj'][0]
pmean_siraj = df_siraj['p_siraj'][0]
smean_siraj = np.sqrt(1-qmean_siraj**2-pmean_siraj**2)
kappa_siraj = df_siraj['gamma_siraj'][0]
angledeg_siraj = df_siraj['angledegs_siraj'][1]
#%%
q_ellipse_covariance,p_ellipse_covariance,a0,b0,siga,sigb,rhoab,chi2val,ecc,phideg,semimajoraxis,semiminoraxis = \
    ellipse_points_2(q_array,p_array,probmass,lenth)
#%%
df_invar = pd.read_csv('b003_idegWdegqpinvar_HE_ss12.csv')
ideg_invar = df_invar['ideginvar'][0]
Wdeg_invar = df_invar['Wdeginvar'][0]
q_invar = df_invar['qinvar'][0]
p_invar = df_invar['pinvar'][0]
irad_invar = np.radians(ideg_invar)
Wrad_invar = np.radians(Wdeg_invar)
# ideg_invar = df_planets['ideginvar_Wdeginvar_qinvar_pinvar'][0]
# Wdeg_invar = df_planets['ideginvar_Wdeginvar_qinvar_pinvar'][1]
# irad_invar = np.radians(ideg_invar)
# Wrad_invar = np.radians(Wdeg_invar)
# q_invar = df_planets['ideginvar_Wdeginvar_qinvar_pinvar'][2]
# p_invar = df_planets['ideginvar_Wdeginvar_qinvar_pinvar'][3]
df_planets = pd.read_csv('b003_planets_orbels_jd246e4_HEHIBEBI.csv')
ideg_bary_neptune = df_planets['ideg_BE'][8]
Wdeg_bary_neptune = df_planets['Wdeg_BE'][8]
irad_bary_neptune = np.radians(ideg_bary_neptune)
Wrad_bary_neptune = np.radians(Wdeg_bary_neptune)
q_neptune = np.sin(irad_bary_neptune)*np.cos(Wrad_bary_neptune)
p_neptune = np.sin(irad_bary_neptune)*np.sin(Wrad_bary_neptune)
df_laplace_rescenters = pd.read_csv('b005_laplace_bary_rescenters_jd246e4.csv')
ideg_laplace = df_laplace_rescenters['laplace_ideg'][0]
Wdeg_laplace = df_laplace_rescenters['laplace_Wdeg'][0]
irad_laplace = np.radians(ideg_laplace)
Wrad_laplace = np.radians(Wdeg_laplace)
q_laplace = np.sin(irad_laplace)*np.cos(Wrad_laplace)
p_laplace = np.sin(irad_laplace)*np.sin(Wrad_laplace)
q_laplace = df_laplace_rescenters['laplace_q'][0]
p_laplace = df_laplace_rescenters['laplace_p'][0]
lawler_amin =  99999
lawler_amax = -99999
lawler_strs = ['StableKozaiPlutinos','StablePlutinos']
for ilaw in range(len(lawler_strs)):
    dflaw = pd.read_csv('a000_lawler_'+lawler_strs[ilaw]+'.txt',delim_whitespace=True)
    alist = dflaw['a'].to_list()
    if np.min(alist) <= lawler_amin:
        lawler_amin = np.min(alist)
    if np.max(alist) >= lawler_amax:
        lawler_amax = np.max(alist)
laplace_ideg_list = []
laplace_Wdeg_list = []
dflap = pd.read_csv('b005_laplace_bary_jd246e4.csv')
nlap = dflap.shape[0]
for ilap in range(nlap):
    if lawler_amin <= dflap['aau'][ilap] <= lawler_amax:
        laplace_ideg_list.append(dflap['laplace_ideg'][ilap])
        laplace_Wdeg_list.append(dflap['laplace_Wdeg'][ilap])
laplace_irad_list = np.radians(np.array(laplace_ideg_list))
laplace_Wrad_list = np.radians(np.array(laplace_Wdeg_list))
laplace_q_list = np.sin(laplace_irad_list)*np.cos(laplace_Wrad_list)
laplace_p_list = np.sin(laplace_irad_list)*np.sin(laplace_Wrad_list)
#%%
A_vmf,qrel_vmf,prel_vmf,srel_vmf,ireldeg_vmf,Wreldeg_vmf = shift_to_pole(q_array,p_array,s_array,idegmean_vmf,Wdegmean_vmf)
A_vm17,qrel_vm17,prel_vm17,srel_vm17,ireldeg_vm17,Wreldeg_vm17 = shift_to_pole(q_array,p_array,s_array,idegmean_vm17,Wdegmean_vm17)
A_siraj,qrel_siraj,prel_siraj,srel_siraj,ireldeg_siraj,Wreldeg_siraj = shift_to_pole(q_array,p_array,s_array,idegmean_siraj,Wdegmean_siraj)
A_invar,qrel_invar,prel_invar,srel_invar,ireldeg_invar,Wreldeg_invar = shift_to_pole(q_array,p_array,s_array,ideg_invar,Wdeg_invar)
A_neptune,qrel_neptune,prel_neptune,srel_neptune,ireldeg_neptune,Wreldeg_neptune = shift_to_pole(q_array,p_array,s_array,ideg_bary_neptune,Wdeg_bary_neptune)
A_laplace,qrel_laplace,prel_laplace,srel_laplace,ireldeg_laplace,Wreldeg_laplace = shift_to_pole(q_array,p_array,s_array,ideg_laplace,Wdeg_laplace)
#%%
probmass_vmf_wrt_vm17 = probmass_inside_ellipse_thru_point_vm17(q_vm17,p_vm17,qmean_vmf,pmean_vmf)
probmass_siraj_wrt_vm17 = probmass_inside_ellipse_thru_point_vm17(q_vm17,p_vm17,qmean_siraj,pmean_siraj)
probmass_invar_wrt_vm17 = probmass_inside_ellipse_thru_point_vm17(q_vm17,p_vm17,q_invar,p_invar)
probmass_neptune_wrt_vm17 = probmass_inside_ellipse_thru_point_vm17(q_vm17,p_vm17,q_neptune,p_neptune)
probmass_laplace_wrt_vm17 = probmass_inside_ellipse_thru_point_vm17(q_vm17,p_vm17,q_laplace,p_laplace)
probmass_siraj_wrt_vmf = probmass_inside_circle_thru_point_vmf(q_array,p_array,qmean_siraj,pmean_siraj)
probmass_vm17_wrt_vmf = probmass_inside_circle_thru_point_vmf(q_array,p_array,qmean_vm17,pmean_vm17)
probmass_invar_wrt_vmf = probmass_inside_circle_thru_point_vmf(q_array,p_array,q_invar,p_invar)
probmass_neptune_wrt_vmf = probmass_inside_circle_thru_point_vmf(q_array,p_array,q_neptune,p_neptune)
probmass_laplace_wrt_vmf = probmass_inside_circle_thru_point_vmf(q_array,p_array,q_laplace,p_laplace)
probmass_vmf_wrt_siraj = probmass_siraj(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,qmean_vmf,pmean_vmf)
probmass_vm17_wrt_siraj = probmass_siraj(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,qmean_vm17,pmean_vm17)
probmass_invar_wrt_siraj = probmass_siraj(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,q_invar,p_invar)
probmass_neptune_wrt_siraj = probmass_siraj(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,q_neptune,p_neptune)
probmass_laplace_wrt_siraj = probmass_siraj(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,q_laplace,p_laplace)
#%%
deg_distance_vmf_wrt_vm17      = angle_between_points_qp1_qp2(qmean_vm17,pmean_vm17,qmean_vmf,pmean_vmf)
deg_distance_siraj_wrt_vm17    = angle_between_points_qp1_qp2(qmean_vm17,pmean_vm17,qmean_siraj,pmean_siraj)
deg_distance_invar_wrt_vm17    = angle_between_points_qp1_qp2(qmean_vm17,pmean_vm17,q_invar,p_invar)
deg_distance_neptune_wrt_vm17  = angle_between_points_qp1_qp2(qmean_vm17,pmean_vm17,q_neptune,p_neptune)
deg_distance_laplace_wrt_vm17  = angle_between_points_qp1_qp2(qmean_vm17,pmean_vm17,q_laplace,p_laplace)
deg_distance_siraj_wrt_vmf     = angle_between_points_qp1_qp2(qmean_vmf,pmean_vmf,qmean_siraj,pmean_siraj)
deg_distance_vm17_wrt_vmf      = angle_between_points_qp1_qp2(qmean_vmf,pmean_vmf,qmean_vm17,pmean_vm17)
deg_distance_invar_wrt_vmf     = angle_between_points_qp1_qp2(qmean_vmf,pmean_vmf,q_invar,p_invar)
deg_distance_neptune_wrt_vmf   = angle_between_points_qp1_qp2(qmean_vmf,pmean_vmf,q_neptune,p_neptune)
deg_distance_laplace_wrt_vmf   = angle_between_points_qp1_qp2(qmean_vmf,pmean_vmf,q_laplace,p_laplace)
deg_distance_vmf_wrt_siraj     = angle_between_points_qp1_qp2(qmean_siraj,pmean_siraj,qmean_vmf,pmean_vmf)
deg_distance_vm17_wrt_siraj    = angle_between_points_qp1_qp2(qmean_siraj,pmean_siraj,qmean_vm17,pmean_vm17)
deg_distance_invar_wrt_siraj   = angle_between_points_qp1_qp2(qmean_siraj,pmean_siraj,q_invar,p_invar)
deg_distance_neptune_wrt_siraj = angle_between_points_qp1_qp2(qmean_siraj,pmean_siraj,q_neptune,p_neptune)
deg_distance_laplace_wrt_siraj = angle_between_points_qp1_qp2(qmean_siraj,pmean_siraj,q_laplace,p_laplace)
#%%
plt.rcParams['font.size'] = 8
s = 10
fig = plt.figure(figsize=(2.0,2.0))
vm17color = 'brown'
vmfcolor = 'gray'
laplacecolor = 'magenta'
zerozerocolor = 'black'
neptunecolor = 'blue'
invarcolor = 'red'
sirajcolor = 'black'

ax = fig.add_subplot(111)
ax.tick_params(axis='x',direction='in')
ax.tick_params(axis='y',direction='in')
ax.axhline(color=zerozerocolor,linestyle='-',linewidth=0.2,alpha=0.5)
ax.axvline(color=zerozerocolor,linestyle='-',linewidth=0.2,alpha=0.5)
ax.plot(q_ellipse_vm17,p_ellipse_vm17,color=vm17color,linestyle='dashed',linewidth=0.5)
ax.plot(costh*np.sin(np.radians(angledeg_vmf))+qmean_vmf,sinth*np.sin(np.radians(angledeg_vmf))+pmean_vmf,\
        color=vmfcolor,linestyle='dotted',linewidth=0.5) # confidence circle
ax.plot(q_ellipse_siraj,p_ellipse_siraj,color=sirajcolor,linestyle='solid',linewidth=0.5)
ax.plot(laplace_q_list,laplace_p_list,color=laplacecolor,linestyle='-',linewidth=0.5)
ax.scatter(qmean_vm17,pmean_vm17,color=vm17color,s=s,marker='>') # vm17 midplane of plutinos
ax.scatter(qmean_vmf,pmean_vmf,color=vmfcolor,s=s,marker='v') # vmf midplane of plutinos
ax.scatter(qmean_siraj,pmean_siraj,color=sirajcolor,s=s,marker='*') # siraj midplane of plutinos
ax.scatter(q_invar,p_invar,color=invarcolor,s=s,marker='x') # invariable pole of the solar system
ax.scatter(q_neptune,p_neptune,color=neptunecolor,s=s,marker='o') # orbit pole of neptune
ax.scatter(q_laplace,p_laplace,color=laplacecolor,s=s,marker='+') # laplace pole at nominal resonance
ax.set_xlabel('$q=sin(i)cos(\Omega)$')
ax.set_ylabel('$p=sin(i)sin(\Omega)$',labelpad=-5)
ax.text(0.95,0.95,str(probmass*100)+'%',ha='right',va='top',bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5),\
        transform=ax.transAxes)
xmin_ellipse_plot = -0.07
xmax_ellipse_plot = +0.12
ymin_ellipse_plot = -0.05
ymax_ellipse_plot = +0.10
ax.set_xlim([xmin_ellipse_plot,xmax_ellipse_plot])
ax.set_ylim([ymin_ellipse_plot,ymax_ellipse_plot])
ax.set_box_aspect((ymax_ellipse_plot-ymin_ellipse_plot)/(xmax_ellipse_plot-xmin_ellipse_plot))

plt.tight_layout(pad=0.8,w_pad=+0.5,h_pad=0)
plt.savefig('b008_a_plots_ellipses_'+str(probmass)+'_dots_onepanel.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()
#%%
print('')
print('qmean_vmf = ',np.round(qmean_vmf,3))
print('pmean_vmf = ',np.round(pmean_vmf,3))
print('idegmean_vmf = ',np.round(idegmean_vmf,2))
print('Wdegmean_vmf = ',np.round(Wdegmean_vmf,2))
print('angledeg_vmf = ',np.round(angledeg_vmf,2))
print('kappa_vmf = ',np.round(kappa_vmf,2))
print('sigmadegraleigh_vmf = ',np.round(np.degrees(kappa_vmf**-0.5),2))
print('')
print('qmean_siraj = ',np.round(qmean_siraj,3))
print('pmean_siraj = ',np.round(pmean_siraj,3))
print('idegmean_siraj = ',np.round(idegmean_siraj,2))
print('Wdegmean_siraj = ',np.round(Wdegmean_siraj,2))
print('angledeg_siraj = ',np.round(angledeg_siraj,2))
print('kappa_siraj = ',np.round(kappa_siraj,2))
print('sigmadegraleigh_siraj = ',np.round(np.degrees(kappa_siraj**-0.5),2))
print('')
print('qmean_vm17 = ',np.round(qmean_vm17,3))
print('pmean_vm17 = ',np.round(pmean_vm17,3))
print('idegmean_vm17 = ',np.round(idegmean_vm17,2))
print('Wdegmean_vm17 = ',np.round(Wdegmean_vm17,2))
print('angledeg_vm17 = ',np.round(angledeg_vm17,2))
print('')
print('deg_distance_vmf_wrt_vmf = 0')
print('deg_distance_siraj_wrt_vmf = ',np.round(deg_distance_siraj_wrt_vmf,2))
print('deg_distance_vm17_wrt_vmf = ',np.round(deg_distance_vm17_wrt_vmf,2))
print('deg_distance_invar_wrt_vmf = ',np.round(deg_distance_invar_wrt_vmf,2))
print('deg_distance_neptune_wrt_vmf = ',np.round(deg_distance_neptune_wrt_vmf,2))
print('deg_distance_laplace_wrt_vmf = ',np.round(deg_distance_laplace_wrt_vmf,2))
print('')
print('deg_distance_vmf_wrt_siraj = ',np.round(deg_distance_vmf_wrt_siraj,2))
print('deg_distance_siraj_wrt_siraj = 0')
print('deg_distance_vm17_wrt_siraj = ',np.round(deg_distance_vm17_wrt_siraj,2))
print('deg_distance_invar_wrt_siraj = ',np.round(deg_distance_invar_wrt_siraj,2))
print('deg_distance_neptune_wrt_siraj = ',np.round(deg_distance_neptune_wrt_siraj,2))
print('deg_distance_laplace_wrt_siraj = ',np.round(deg_distance_laplace_wrt_siraj,2))
print('')
print('deg_distance_vmf_wrt_vm17 = ',np.round(deg_distance_vmf_wrt_vm17,2))
print('deg_distance_siraj_wrt_vm17 = ',np.round(deg_distance_siraj_wrt_vm17,2))
print('deg_distance_vm17_wrt_vm17 = 0')
print('deg_distance_invar_wrt_vm17 = ',np.round(deg_distance_invar_wrt_vm17,2))
print('deg_distance_neptune_wrt_vm17 = ',np.round(deg_distance_neptune_wrt_vm17,2))
print('deg_distance_laplace_wrt_vm17 = ',np.round(deg_distance_laplace_wrt_vm17,2))
print('')
print('pval_vmf_wrt_vmf = N/A')
print('pval_siraj_wrt_vmf = ',1-probmass_siraj_wrt_vmf)
print('pval_vm17_wrt_vmf = ',1-probmass_vm17_wrt_vmf)
print('pval_invar_wrt_vmf = ',1-probmass_invar_wrt_vmf)
print('pval_neptune_wrt_vmf = ',1-probmass_neptune_wrt_vmf)
print('pval_laplace_wrt_vmf = ',1-probmass_laplace_wrt_vmf)
print('')
print('pval_vmf_wrt_siraj = ',1-probmass_vmf_wrt_siraj)
print('pval_siraj_wrt_siraj = N/A')
print('pval_vm17_wrt_siraj = ',1-probmass_vm17_wrt_siraj)
print('pval_invar_wrt_siraj = ',1-probmass_invar_wrt_siraj)
print('pval_neptune_wrt_siraj = ',1-probmass_neptune_wrt_siraj)
print('pval_laplace_wrt_siraj = ',1-probmass_laplace_wrt_siraj)
print('')
print('pval_vmf_wrt_vm17 = ',1-probmass_vmf_wrt_vm17)
print('pval_siraj_wrt_vm17 = ',1-probmass_siraj_wrt_vm17)
print('pval_vm17_wrt_vm17 = N/A')
print('pval_invar_wrt_vm17 = ',1-probmass_invar_wrt_vm17)
print('pval_neptune_wrt_vm17 = ',1-probmass_neptune_wrt_vm17)
print('pval_laplace_wrt_vm17 = ',1-probmass_laplace_wrt_vm17)
#%%
dictionary = {'probmass':[probmass],\
    'qmean_vmf':[qmean_vmf],'pmean_vmf':[pmean_vmf],'idegmean_vmf':[idegmean_vmf],\
    'Wdegmean_vmf':[Wdegmean_vmf],'angledeg_vmf':[angledeg_vmf],'kappa_vmf':[kappa_vmf],\
    'qmean_siraj':[qmean_siraj],'pmean_siraj':[pmean_siraj],'idegmean_siraj':[idegmean_siraj],\
    'Wdegmean_siraj':[Wdegmean_siraj],'angledeg_siraj':[angledeg_siraj],'kappa_siraj':[kappa_siraj],\
    'qmean_vm17':[qmean_vm17],'pmean_vm17':[pmean_vm17],'idegmean_vm17':[idegmean_vm17],\
    'Wdegmean_vm17':[Wdegmean_vm17],'angledeg_vm17':[angledeg_vm17],\
    'pval_vmf_wrt_vm17':[1-probmass_vmf_wrt_vm17],\
    'pval_siraj_wrt_vm17':[1-probmass_siraj_wrt_vm17],\
    'pval_invar_wrt_vm17':[1-probmass_invar_wrt_vm17],\
    'pval_neptune_wrt_vm17':[1-probmass_neptune_wrt_vm17],\
    'pval_laplace_wrt_vm17':[1-probmass_laplace_wrt_vm17],\
    'pval_siraj_wrt_vmf':[1-probmass_siraj_wrt_vmf],\
    'pval_vm17_wrt_vmf':[1-probmass_vm17_wrt_vmf],\
    'pval_invar_wrt_vmf':[1-probmass_invar_wrt_vmf],\
    'pval_neptune_wrt_vmf':[1-probmass_neptune_wrt_vmf],\
    'pval_laplace_wrt_vmf':[1-probmass_laplace_wrt_vmf],\
    'pval_vmf_wrt_siraj':[1-probmass_vmf_wrt_siraj],\
    'pval_vm17_wrt_siraj':[1-probmass_vm17_wrt_siraj],\
    'pval_invar_wrt_siraj':[1-probmass_invar_wrt_siraj],\
    'pval_neptune_wrt_siraj':[1-probmass_neptune_wrt_siraj],\
    'pval_laplace_wrt_siraj':[1-probmass_laplace_wrt_siraj],\
    'deg_distance_vmf_wrt_vm17':[deg_distance_vmf_wrt_vm17],\
    'deg_distance_siraj_wrt_vm17':[deg_distance_siraj_wrt_vm17],\
    'deg_distance_invar_wrt_vm17':[deg_distance_invar_wrt_vm17],\
    'deg_distance_neptune_wrt_vm17':[deg_distance_neptune_wrt_vm17],\
    'deg_distance_laplace_wrt_vm17':[deg_distance_laplace_wrt_vm17],\
    'deg_distance_siraj_wrt_vmf':[deg_distance_siraj_wrt_vmf],\
    'deg_distance_vm17_wrt_vmf':[deg_distance_vm17_wrt_vmf],\
    'deg_distance_invar_wrt_vmf':[deg_distance_invar_wrt_vmf],\
    'deg_distance_neptune_wrt_vmf':[deg_distance_neptune_wrt_vmf],\
    'deg_distance_laplace_wrt_vmf':[deg_distance_laplace_wrt_vmf],\
    'deg_distance_vmf_wrt_siraj':[deg_distance_vmf_wrt_siraj],\
    'deg_distance_vm17_wrt_siraj':[deg_distance_vm17_wrt_siraj],\
    'deg_distance_invar_wrt_siraj':[deg_distance_invar_wrt_siraj],\
    'deg_distance_neptune_wrt_siraj':[deg_distance_neptune_wrt_siraj],\
    'deg_distance_laplace_wrt_siraj':[deg_distance_laplace_wrt_siraj],\
    }
dfdict = pd.DataFrame.from_dict(dictionary)
dfdict.to_csv('b008_b_table_plutinos_gplusminusnone_vmf_siraj_vm17.csv')