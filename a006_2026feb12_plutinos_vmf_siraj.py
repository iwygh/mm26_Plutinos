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
import numpy as np
# https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
def fit_ellipse(x, y):
    import numpy as np
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].
    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()
def cart_to_pol(coeffs):
    import numpy as np
    """
    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.
    """
    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]
    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')
    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den
    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))
    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap
    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)
    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi
    return x0, y0, ap, bp, e, phi
def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    import numpy as np
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.
    """
    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y
# if __name__ == '__main__':
#     # Test the algorithm with an example elliptical arc.
#     npts = 250
#     tmin, tmax = np.pi/6, 4 * np.pi/3
#     x0, y0 = 4, -3.5
#     ap, bp = 7, 3
#     phi = np.pi / 4
#     # Get some points on the ellipse (no need to specify the eccentricity).
#     x, y = get_ellipse_pts((x0, y0, ap, bp, None, phi), npts, tmin, tmax)
#     noise = 0.1
#     x += noise * np.random.normal(size=npts) 
#     y += noise * np.random.normal(size=npts)
#     coeffs = fit_ellipse(x, y)
#     print('Exact parameters:')
#     print('x0, y0, ap, bp, phi =', x0, y0, ap, bp, phi)
#     print('Fitted parameters:')
#     print('a, b, c, d, e, f =', coeffs)
#     x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
#     print('x0, y0, ap, bp, e, phi = ', x0, y0, ap, bp, e, phi)
#     plt.plot(x, y, 'x')     # given points
#     x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi))
#     plt.plot(x, y)
#     plt.show()
#%%
def siraj_fun(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,probmasses_array,delta,n_q,n_p):
    # from contourpy import contour_generator
    # import time
    import numpy as np
    from matplotlib import pyplot as plt
    qperiau_array = aau_array * (1-e_array)
    res_out_array = maxlogL(aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array,delta,n_q)
    logLmax_siraj = -res_out_array.fun
    gamma_siraj = res_out_array.x[0]
    hx_siraj = res_out_array.x[1]
    hy_siraj = res_out_array.x[2]
    hz_siraj = res_out_array.x[3]
    q_siraj = -hy_siraj
    p_siraj = hx_siraj
    s_siraj = hz_siraj
    irad_siraj = np.arccos(s_siraj)
    ideg_siraj = np.degrees(irad_siraj)
    sini_siraj = np.sin(irad_siraj)
    Wrad_siraj = np.arctan2(p_siraj/sini_siraj,q_siraj/sini_siraj)
    Wdeg_siraj = np.degrees(Wrad_siraj)
    q_min = q_siraj - delta
    q_max = q_siraj + delta
    p_min = p_siraj - delta
    p_max = p_siraj + delta
    qvec = np.linspace(start=q_min,stop=q_max,num=n_q+1,endpoint=True)
    pvec = np.linspace(start=p_min,stop=p_max,num=n_p+1,endpoint=True)
    qmat,pmat = np.meshgrid(qvec,pvec)
    logL_mat_siraj = np.zeros((n_q+1,n_p+1))
    # t0 = time.time()
    for iq in range(n_q+1):
        for ip in range(n_p+1):
            q_iqip = qmat[iq,ip]
            p_iqip = pmat[iq,ip]
            s_iqip = np.sqrt(1-q_iqip**2-p_iqip**2)
            # ss_iqip = np.sqrt(1-q_iqip**2-p_iqip**2)
            # irad_iqip = 2*np.arccos(ss_iqip)
            # sini_iqip = np.sin(irad_iqip)
            # Wrad_iqip = np.arctan2(p_iqip/sini_iqip,q_iqip/sini_iqip)
            # q_iqip = np.sin(irad_iqip)*np.cos(Wrad_iqip)
            # p_iqip = np.sin(irad_iqip)*np.sin(Wrad_iqip)
            # s_iqip = np.cos(irad_iqip)
            hx_iqip = p_iqip
            hy_iqip = -q_iqip
            hz_iqip = s_iqip
            xin = gamma_siraj
            logL_max_iqip = logL_ref(xin,aau_array,qperiau_array,irad_array,\
                    wrad_array,Wrad_array,Mrad_array,hx_iqip,hy_iqip,hz_iqip)
            logL_mat_siraj[iq,ip] = logL_max_iqip
        # t1 = time.time()
        # dt = (t1-t0)/60
        # print(iq+1,n_q+1,dt)
    sigma_mat_siraj = np.sqrt(2*(logLmax_siraj-logL_mat_siraj))
    pval_mat_siraj = np.exp( -1/2 * sigma_mat_siraj**2 )
    # cont_gen = contour_generator(z=pval_mat_siraj)
    angledegs_siraj = []
    nprob = len(probmasses_array)
    xi = qmat
    yi = pmat
    zi = pval_mat_siraj
    params_probmasses = []
    params_x0_list = []
    params_y0_list = []
    params_ec_list = []
    params_phi_list = []
    params_ap_list = []
    params_bp_list = []
    for iprob in range(nprob):
        print('siraj_fun',iprob+1,nprob,probmasses_array[iprob])
        # lines = cont_gen.lines(1-probmasses_array[iprob])
        
        fig = plt.figure(figsize=(10,10))
        plt.rcParams['font.size'] = 12
        ax = fig.add_subplot(111)
        quadset = ax.contour(xi,yi,zi,levels=[1-probmasses_array[iprob]])
        # ax.contour(xi,yi,zi,levels=[1-probmasses_array[iprob]])
        title_str = str(len(aau_array))+' '+str(np.round(1-probmasses_array[iprob],3))
        ax.set_title(title_str)
        plt.show()
        # plt.close()
        segs = quadset.allsegs[0]
        segs_array = np.array(segs)
        segs_flat = np.squeeze(segs_array)
        xsegs = segs_flat[:,0]
        ysegs = segs_flat[:,1]
        # nsegs = len(xsegs)
        ellipse_coeffs_siraj = fit_ellipse(xsegs,ysegs)
        # print('a, b, c, d, e, f =', coeffs)
        params_x0,params_y0,ap,bp,params_ec,params_phi = cart_to_pol(ellipse_coeffs_siraj)
        ellipse_area = np.pi * ap * bp
        circle_radius = np.sqrt(ellipse_area/np.pi)
        circle_angle_radians = np.arcsin(circle_radius)
        angledegs_siraj.append(np.degrees(circle_angle_radians))
        params_probmasses.append(probmasses_array[iprob])
        params_x0_list.append(params_x0)
        params_y0_list.append(params_y0)
        params_ec_list.append(params_ec)
        params_phi_list.append(params_phi)
        params_ap_list.append(ap)
        params_bp_list.append(bp)
        # # print('x0, y0, ap, bp, e, phi = ', x0, y0, ap, bp, e, phi)
        # # tmin, tmax = 0, 2*np.pi
        # # xell, yell = get_ellipse_pts((x0, y0, ap, bp, None, phi), nsegs, tmin, tmax)
        # # plt.scatter(xsegs,ysegs)
        # # plt.scatter(xell,yell)
        # # plt.show()
    ellipse_params_siraj = [params_x0_list,params_y0_list,params_ec_list,params_phi_list,\
                            params_probmasses,params_ap_list,params_bp_list]
    return ideg_siraj,Wdeg_siraj,q_siraj,p_siraj,s_siraj,angledegs_siraj,\
        ellipse_params_siraj,gamma_siraj,logLmax_siraj,logL_mat_siraj,sigma_mat_siraj,pval_mat_siraj,\
        qmat,pmat
    # return ideg_siraj,Wdeg_siraj,q_siraj,p_siraj,s_siraj,angledegs_siraj,\
    #     gamma_siraj,logLmax_siraj,logL_mat_siraj,sigma_mat_siraj,pval_mat_siraj,\
    #     qmat,pmat
    # params_x0 = ellipse_params_siraj[0]
    # params_y0 = ellipse_params_siraj[1]
    # params_ec = ellipse_params_siraj[2]
    # params_phi = ellipse_params_siraj[3]
    # params_probmasses = ellipse_params_siraj[4]
    # params_ap_list = ellipse_params_siraj[5]
    # params_bp_list = ellipse_params_siraj[6]
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
import numpy as np
import pandas as pd
libration = 'gplusminusnone_2026feb12'
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
probmasses_array = np.array([0.68,0.95,0.997])
delta = 0.4
n_q = 400
n_p = 400
ideg_vmf,Wdeg_vmf,q_vmf,p_vmf,s_vmf,angledegs_vmf,sigmahat_vmf,Rbar_vmf,Kout_vmf,kappahat_vmf = \
    vmf_fun(q_array,p_array,s_array,probmasses_array)
print('vmf',ideg_vmf,Wdeg_vmf,angledegs_vmf)
vtx_array,vty_array,vtz_array = sky_velocity_vectors_schaubandjunkins(\
                aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array)
dictionary_vtvec = {'packed_designation':des_list_2,'vtx':vtx_array,'vty':vty_array,'vtz':vtz_array}
df_vtvec = pd.DataFrame.from_dict(dictionary_vtvec)
df_vtvec.to_csv('b006_2026feb12_plutinos_vtvec_'+libration+'.csv',index=False)
dictionary = {'ideg_vmf':ideg_vmf,'Wdeg_vmf':Wdeg_vmf,'q_vmf':q_vmf,'p_vmf':p_vmf,'s_vmf':s_vmf,\
              'angledegs_vmf':angledegs_vmf,'probmasses':probmasses_array,\
              'sigmahat_vmf':sigmahat_vmf,'Rbar_vmf':Rbar_vmf,\
              'Kout_vmf':Kout_vmf,'kappahat_vmf':kappahat_vmf}
df_vmf = pd.DataFrame.from_dict(dictionary)
df_vmf.to_csv('b006_2026feb12_plutinos_vmf_'+libration+'.csv',index=False)
dictionary = {'mpcdes':des_list_2,'aaubary':aau_array,'ebary':e_array,\
              'idegbary':np.degrees(irad_array),'nodedegbary':np.degrees(Wrad_array),\
              'peridegbary':np.degrees(wrad_array),'Mdegbary':np.degrees(Mrad_array)}
df_fortran = pd.DataFrame.from_dict(dictionary)
df_fortran.to_csv('b006_2026feb12_plutinos_fortran_'+libration+'.csv',index=False)
#%%
for libration in ['gplus','gminus','gplusminusnone']:
    print(libration)
    dfind = pd.read_csv('b004_p3q2_'+libration+'_2026feb12_index.csv')
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
    probmasses_array = np.array([0.68,0.95,0.997])
    delta = 0.4
    n_q = 400
    n_p = 400
    ideg_siraj,Wdeg_siraj,q_siraj,p_siraj,s_siraj,angledegs_siraj,\
        ellipse_params_siraj,gamma_siraj,logLmax_siraj,logL_mat_siraj,sigma_mat_siraj,pval_mat_siraj,\
        q_mat_siraj,p_mat_siraj = \
        siraj_fun(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,probmasses_array,delta,n_q,n_p)
    print(ideg_siraj,Wdeg_siraj,angledegs_siraj)
    outfile_siraj = 'b006_2026feb12_plutinos_'+libration+'_siraj_logLmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    np.savetxt(outfile_siraj,logL_mat_siraj,delimiter=',')
    outfile_siraj = 'b006_2026feb12_plutinos_'+libration+'_siraj_pvalmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    np.savetxt(outfile_siraj,pval_mat_siraj,delimiter=',')
    outfile_siraj = 'b006_2026feb12_plutinos_'+libration+'_siraj_qmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    np.savetxt(outfile_siraj,q_mat_siraj,delimiter=',')
    outfile_siraj = 'b006_2026feb12_plutinos_'+libration+'_siraj_pmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    np.savetxt(outfile_siraj,p_mat_siraj,delimiter=',')
    outfile_siraj = 'b006_2026feb12_plutinos_'+libration+'_siraj_sigmamat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    np.savetxt(outfile_siraj,sigma_mat_siraj,delimiter=',')
    x0_siraj = ellipse_params_siraj[0]
    y0_siraj = ellipse_params_siraj[1]
    ec_siraj = ellipse_params_siraj[2]
    phi_siraj = ellipse_params_siraj[3]
    ap_siraj = ellipse_params_siraj[5]
    bp_siraj = ellipse_params_siraj[6]
    dictionary = {'ideg_siraj':ideg_siraj,'Wdeg_siraj':Wdeg_siraj,'q_siraj':q_siraj,'p_siraj':p_siraj,\
                  's_siraj':s_siraj,'angledegs_siraj':angledegs_siraj,'probmasses':probmasses_array,\
                  'x0_siraj':x0_siraj,'y0_siraj':y0_siraj,'ec_siraj':ec_siraj,'phi_siraj':phi_siraj,\
                  'ap_siraj':ap_siraj,'bp_siraj':bp_siraj,'gamma_siraj':gamma_siraj,'logLmax_siraj':logLmax_siraj}
    df_siraj = pd.DataFrame.from_dict(dictionary)
    df_siraj.to_csv('b006_2026feb12_plutinos_'+libration+'_siraj.csv',index=False)