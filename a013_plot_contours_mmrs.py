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
def curve_vec_exact(radians_vec,kappa):
    curve_vec_exact = kappa/(np.exp(kappa)-np.exp(-kappa))*\
        np.exp(kappa*np.cos(radians_vec))*np.sin(radians_vec)
    return curve_vec_exact
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
def maxlogL(aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array):
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
    delta = 0.4
    n_q = 400
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
        # print('starting maxlogL iteration',iteration)
        bnds = ((0,None),(-sini,sini),(-sini,sini),(cosi,1))
        cons = ({'type':'eq','fun':lambda x: x[1]**2+x[2]**2+x[3]**2-1})
        res = minimize(fun,x_in,method='trust-constr',bounds=bnds,constraints=cons)
        # res = minimize(fun,x_in,method='L-BFGS-B',constraints=cons,bounds=bnds)
        x_out = res.x
        diff = (x_out[1]-x_in[1])**2 + (x_out[2]-x_in[2])**2 + (x_out[3]-x_in[3])**2
        # print('end maxlogL iteration',iteration)
        # print('diff = ',diff)
        x_in = x_out
    print(res.x)
    return res
#%%
def probmass_siraj(aau_array,e_array,irad_array,wrad_array,Wrad_array,Mrad_array,xpt,ypt):
    import numpy as np
    qperiau_array = aau_array * (1-e_array)
    res_out_array = maxlogL(aau_array,qperiau_array,irad_array,wrad_array,Wrad_array,Mrad_array)
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
def vmf_fun(xvec,yvec,zvec,probmass):
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
    A = 1 - probmass
    angledeg = np.degrees(np.arcsin(sigmahat*np.sqrt(-np.log(A))))
    return xcc,ycc,zcc,angledeg,sigmahat,Rbar,Kout
#%%
import numpy as np
# from matplotlib import pyplot as plt
# https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
def fit_ellipse(x, y):
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
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
jdstr = 'jd246e4'
delta = 0.2
n_q = 400
n_p = 400
# librations = ['p2q1','p5q3','p5q2','p7q4']
# laplace_indices = [2,4,1,3]
# mmrps = [2,5,5,7]
# mmrqs = [1,3,2,4]
librations = ['p5q3','p7q4','p2q1','p5q2']
laplace_indices = [4,3,2,1]
mmrps = [5,7,2,5]
mmrqs = [3,4,1,2]
nlib = len(mmrps)
colors = ['black','black','black','black']
postitles = ['TOPLEFT','TOPRIGHT','BOTTOMLEFT','BOTTOMRIGHT']
xmins = [-0.15,-0.15,-0.15,-0.15]
xmaxs = [+0.10,+0.10,+0.10,+0.10]
ymins = [-0.10,-0.10,-0.10,-0.10]
ymaxs = [+0.15,+0.15,+0.15,+0.15]
markersize_qcc = 30
marker_qcc = '*'

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
df_laplace = pd.read_csv('b005_laplace_bary_rescenters_'+jdstr+'.csv')
df_laplace_vec = pd.read_csv('b005_laplace_bary_'+jdstr+'.csv')
aau_laplace_vec = np.array(df_laplace_vec['aau'].to_list())
ideg_laplace_vec = np.array(df_laplace_vec['laplace_ideg'].to_list())
Wdeg_laplace_vec = np.array(df_laplace_vec['laplace_Wdeg'].to_list())
q_laplace_vec = np.sin(np.radians(ideg_laplace_vec))*np.cos(np.radians(Wdeg_laplace_vec))
p_laplace_vec = np.sin(np.radians(ideg_laplace_vec))*np.sin(np.radians(Wdeg_laplace_vec))
df_tnos = pd.read_csv('b004_tnos_orbels_'+jdstr+'.csv')
aau_bary_tnos = np.array(df_tnos['aau_bary'].to_list())
e_bary_tnos = np.array(df_tnos['e_bary'].to_list())
ideg_bary_tnos = np.array(df_tnos['ideg_bary'].to_list())
wdeg_bary_tnos = np.array(df_tnos['wdeg_bary'].to_list())
Wdeg_bary_tnos = np.array(df_tnos['Wdeg_bary'].to_list())
Mdeg_bary_tnos = np.array(df_tnos['Mdeg_bary'].to_list())
irad_bary_tnos = np.radians(ideg_bary_tnos)
wrad_bary_tnos = np.radians(wdeg_bary_tnos)
Wrad_bary_tnos = np.radians(Wdeg_bary_tnos)
Mrad_bary_tnos = np.radians(Mdeg_bary_tnos)
q_bary_tnos = np.sin(np.radians(ideg_bary_tnos))*np.cos(np.radians(Wdeg_bary_tnos))
p_bary_tnos = np.sin(np.radians(ideg_bary_tnos))*np.sin(np.radians(Wdeg_bary_tnos))
s_bary_tnos = np.cos(np.radians(ideg_bary_tnos))

# fig_contours = plt.figure(figsize=(6,6))
# plt.rcParams['font.size'] = 12
# axTOPLEFT = fig_contours.add_subplot(221)
# axTOPRIGHT = fig_contours.add_subplot(222)
# axBOTTOMLEFT = fig_contours.add_subplot(223)
# axBOTTOMRIGHT = fig_contours.add_subplot(224)
# axes = [axTOPLEFT,axTOPRIGHT,axBOTTOMLEFT,axBOTTOMRIGHT]

# for ilib in range(nlib):
#     libration = 'p'+str(mmrps[ilib])+'q'+str(mmrqs[ilib])
#     df_tnos_indices = pd.read_csv('b004_'+libration+'_index.csv')
#     tnos_indices = df_tnos_indices['index'].to_list()
#     nobj = len(tnos_indices)
#     aau_bary_here = aau_bary_tnos[tnos_indices]
#     aaumin = np.min(aau_bary_here)
#     aaumax = np.max(aau_bary_here)
#     min_index = 0
#     max_index = 0
#     for iau in range(len(aau_laplace_vec)):
#         if aau_laplace_vec[iau] <= aaumin:
#             min_index = iau
#         if aau_laplace_vec[iau] <= aaumax:
#             max_index = iau
#     q_laplace = df_laplace['laplace_q'][laplace_indices[ilib]]
#     p_laplace = df_laplace['laplace_p'][laplace_indices[ilib]]
#     df_mmr = pd.read_csv('b012_tnos_'+libration+'_siraj.csv')
#     qcc = df_mmr['q_siraj'][0]
#     pcc = df_mmr['p_siraj'][0]
#     infile_pval = 'b012_tnos_'+libration+'_siraj_pvalmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
#     infile_q = 'b012_tnos_'+libration+'_siraj_qmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
#     infile_p = 'b012_tnos_'+libration+'_siraj_pmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
#     pvalmat = np.loadtxt(infile_pval,delimiter=',')
#     qmat = np.loadtxt(infile_q,delimiter=',')
#     pmat = np.loadtxt(infile_p,delimiter=',')
#     quadset = axes[ilib].contour(qmat,pmat,pvalmat,levels=[1-0.997,1-0.95,1-0.68],colors=colors[ilib],linewidths=0.5)
#     axes[ilib].scatter(qcc,pcc,color=colors[ilib],marker=marker_qcc,s=markersize_qcc)
#     axes[ilib].scatter(q_neptune,p_neptune,color='blue',marker='.',s=3*markersize_qcc)
#     axes[ilib].scatter(q_invar,p_invar,color='red',marker='x',s=markersize_qcc)
#     axes[ilib].scatter(q_laplace,p_laplace,color='magenta',marker='+',s=markersize_qcc)
#     axes[ilib].plot(q_laplace_vec[min_index:max_index],p_laplace_vec[min_index:max_index],color='magenta',lw=1)
#     axes[ilib].set_xlim([xmins[ilib],xmaxs[ilib]])
#     axes[ilib].set_ylim([ymins[ilib],ymaxs[ilib]])
#     axes[ilib].tick_params(axis='x',direction='in')
#     axes[ilib].tick_params(axis='y',direction='in')
#     axes[ilib].axhline(color='black',linestyle='-',linewidth=0.2,alpha=0.5)
#     axes[ilib].axvline(color='black',linestyle='-',linewidth=0.2,alpha=0.5)
#     boxstr = str(mmrps[ilib])+':'+str(mmrqs[ilib])+', n ='+str(nobj)
#     axes[ilib].text(0.95,0.95,boxstr,ha='right',va='top',bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5),\
#             transform=axes[ilib].transAxes)
#     axes[ilib].set_box_aspect((ymaxs[ilib]-ymins[ilib])/(xmaxs[ilib]-xmins[ilib]))
# axTOPLEFT.set_xticklabels([])
# axTOPRIGHT.set_xticklabels([])
# axTOPRIGHT.set_yticklabels([])
# axBOTTOMRIGHT.set_yticklabels([])
# xtl = axBOTTOMLEFT.get_xticklabels()
# xtl[0].set_text('')
# axBOTTOMLEFT.set_xticklabels(xtl)
# xtl = axBOTTOMRIGHT.get_xticklabels()
# xtl[0].set_text('')
# axBOTTOMRIGHT.set_xticklabels(xtl)
# ytl = axBOTTOMLEFT.get_yticklabels()
# ytl[0].set_text('')
# axBOTTOMLEFT.set_yticklabels(ytl)
# ytl = axTOPLEFT.get_yticklabels()
# ytl[0].set_text('')
# axTOPLEFT.set_yticklabels(ytl)
# axBOTTOMLEFT.set_xlabel('$q=sin(i)cos(\Omega)$')
# axBOTTOMRIGHT.set_xlabel('$q=sin(i)cos(\Omega)$')
# axBOTTOMLEFT.set_ylabel('$p=sin(i)sin(\Omega)$')
# axTOPLEFT.set_ylabel('$p=sin(i)sin(\Omega)$')
# fig_contours.tight_layout()
# fig_contours.savefig('b013_plot_contours_mmrs_2x2.png',dpi=400)


# fig_contours = plt.figure(figsize=(6.5,2))
# plt.rcParams['font.size'] = 6
# ax1 = fig_contours.add_subplot(141)
# ax2 = fig_contours.add_subplot(142)
# ax3 = fig_contours.add_subplot(143)
# ax4 = fig_contours.add_subplot(144)
# axes = [ax1,ax2,ax3,ax4]

# for ilib in range(nlib):
#     libration = 'p'+str(mmrps[ilib])+'q'+str(mmrqs[ilib])
#     df_tnos_indices = pd.read_csv('b004_'+libration+'_index.csv')
#     tnos_indices = df_tnos_indices['index'].to_list()
#     nobj = len(tnos_indices)
#     aau_bary_here = aau_bary_tnos[tnos_indices]
#     aaumin = np.min(aau_bary_here)
#     aaumax = np.max(aau_bary_here)
#     min_index = 0
#     max_index = 0
#     for iau in range(len(aau_laplace_vec)):
#         if aau_laplace_vec[iau] <= aaumin:
#             min_index = iau
#         if aau_laplace_vec[iau] <= aaumax:
#             max_index = iau
#     q_laplace = df_laplace['laplace_q'][laplace_indices[ilib]]
#     p_laplace = df_laplace['laplace_p'][laplace_indices[ilib]]
#     df_mmr = pd.read_csv('b012_tnos_'+libration+'_siraj.csv')
#     qcc = df_mmr['q_siraj'][0]
#     pcc = df_mmr['p_siraj'][0]
#     infile_pval = 'b012_tnos_'+libration+'_siraj_pvalmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
#     infile_q = 'b012_tnos_'+libration+'_siraj_qmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
#     infile_p = 'b012_tnos_'+libration+'_siraj_pmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
#     pvalmat = np.loadtxt(infile_pval,delimiter=',')
#     qmat = np.loadtxt(infile_q,delimiter=',')
#     pmat = np.loadtxt(infile_p,delimiter=',')
#     quadset = axes[ilib].contour(qmat,pmat,pvalmat,levels=[1-0.997,1-0.95,1-0.68],colors=colors[ilib],linewidths=0.5)
#     axes[ilib].scatter(qcc,pcc,color=colors[ilib],marker=marker_qcc,s=markersize_qcc)
#     axes[ilib].scatter(q_neptune,p_neptune,color='blue',marker='.',s=3*markersize_qcc)
#     axes[ilib].scatter(q_invar,p_invar,color='red',marker='x',s=markersize_qcc)
#     axes[ilib].scatter(q_laplace,p_laplace,color='magenta',marker='+',s=markersize_qcc)
#     axes[ilib].plot(q_laplace_vec[min_index:max_index],p_laplace_vec[min_index:max_index],color='magenta',lw=1)
#     axes[ilib].set_xlim([xmins[ilib],xmaxs[ilib]])
#     axes[ilib].set_ylim([ymins[ilib],ymaxs[ilib]])
#     axes[ilib].tick_params(axis='x',direction='in')
#     axes[ilib].tick_params(axis='y',direction='in')
#     axes[ilib].axhline(color='black',linestyle='-',linewidth=0.2,alpha=0.5)
#     axes[ilib].axvline(color='black',linestyle='-',linewidth=0.2,alpha=0.5)
#     boxstr = str(mmrps[ilib])+':'+str(mmrqs[ilib])+', n ='+str(nobj)
#     axes[ilib].text(0.95,0.95,boxstr,ha='right',va='top',bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5),\
#             transform=axes[ilib].transAxes)
#     axes[ilib].set_box_aspect((ymaxs[ilib]-ymins[ilib])/(xmaxs[ilib]-xmins[ilib]))
# ax2.set_yticklabels([])
# ax3.set_yticklabels([])
# ax4.set_yticklabels([])
# ax1.set_xlabel('$q=sin(i)cos(\Omega)$')
# ax2.set_xlabel('$q=sin(i)cos(\Omega)$')
# ax3.set_xlabel('$q=sin(i)cos(\Omega)$')
# ax4.set_xlabel('$q=sin(i)cos(\Omega)$')
# ax1.set_ylabel('$p=sin(i)sin(\Omega)$')
# xtl = ax1.get_xticklabels()
# xtl[0].set_text('')
# ax1.set_xticklabels(xtl)
# xtl = ax2.get_xticklabels()
# xtl[0].set_text('')
# ax2.set_xticklabels(xtl)
# xtl = ax3.get_xticklabels()
# xtl[0].set_text('')
# ax3.set_xticklabels(xtl)
# xtl = ax4.get_xticklabels()
# xtl[0].set_text('')
# ax4.set_xticklabels(xtl)
# fig_contours.tight_layout()
# fig_contours.savefig('b013_plot_contours_mmrs_1x4.png',dpi=400)

for ilib in range(nlib):
    fig_contours = plt.figure(figsize=(2,2))
    plt.rcParams['font.size'] = 6
    ax = fig_contours.add_subplot(111)
    libration = 'p'+str(mmrps[ilib])+'q'+str(mmrqs[ilib])
    df_tnos_indices = pd.read_csv('b004_'+libration+'_index.csv')
    tnos_indices = df_tnos_indices['index'].to_list()
    nobj = len(tnos_indices)
    aau_bary_here = aau_bary_tnos[tnos_indices]
    aaumin = np.min(aau_bary_here)
    aaumax = np.max(aau_bary_here)
    min_index = 0
    max_index = 0
    for iau in range(len(aau_laplace_vec)):
        if aau_laplace_vec[iau] <= aaumin:
            min_index = iau
        if aau_laplace_vec[iau] <= aaumax:
            max_index = iau
    q_laplace = df_laplace['laplace_q'][laplace_indices[ilib]]
    p_laplace = df_laplace['laplace_p'][laplace_indices[ilib]]
    df_mmr = pd.read_csv('b012_tnos_'+libration+'_siraj.csv')
    qcc = df_mmr['q_siraj'][0]
    pcc = df_mmr['p_siraj'][0]
    infile_pval = 'b012_tnos_'+libration+'_siraj_pvalmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    infile_q = 'b012_tnos_'+libration+'_siraj_qmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    infile_p = 'b012_tnos_'+libration+'_siraj_pmat_delta'+str(delta)+'_nq'+str(n_q)+'_np'+str(n_p)+'.csv'
    pvalmat = np.loadtxt(infile_pval,delimiter=',')
    qmat = np.loadtxt(infile_q,delimiter=',')
    pmat = np.loadtxt(infile_p,delimiter=',')
    quadset = ax.contour(qmat,pmat,pvalmat,levels=[1-0.997,1-0.95,1-0.68],colors=colors[ilib],linewidths=0.5)
    ax.scatter(qcc,pcc,color=colors[ilib],marker=marker_qcc,s=markersize_qcc)
    ax.scatter(q_neptune,p_neptune,color='blue',marker='.',s=3*markersize_qcc)
    ax.scatter(q_invar,p_invar,color='red',marker='x',s=markersize_qcc)
    ax.scatter(q_laplace,p_laplace,color='magenta',marker='+',s=markersize_qcc)
    ax.plot(q_laplace_vec[min_index:max_index],p_laplace_vec[min_index:max_index],color='magenta',lw=1)
    ax.set_xlim([xmins[ilib],xmaxs[ilib]])
    ax.set_ylim([ymins[ilib],ymaxs[ilib]])
    ax.tick_params(axis='x',direction='in')
    ax.tick_params(axis='y',direction='in')
    ax.axhline(color='black',linestyle='-',linewidth=0.2,alpha=0.5)
    ax.axvline(color='black',linestyle='-',linewidth=0.2,alpha=0.5)
    boxstr = str(mmrps[ilib])+':'+str(mmrqs[ilib])+', n ='+str(nobj)
    ax.text(0.95,0.95,boxstr,ha='right',va='top',bbox=dict(boxstyle='round,pad=0.2',facecolor='white',linewidth=0.5),\
            transform=ax.transAxes)
    ax.set_box_aspect((ymaxs[ilib]-ymins[ilib])/(xmaxs[ilib]-xmins[ilib]))
    ax.set_xlabel('$q=sin(i)cos(\Omega)$')
    ax.set_ylabel('$p=sin(i)sin(\Omega)$')
    fig_contours.tight_layout()
    fig_contours.savefig('b013_plot_contours_mmrs_1x1_'+libration+'.png',dpi=400)
#%%
longbins = np.arange(start=0,stop=95,step=5)
bbox_props = dict(fc='white',alpha=1,ec='black',lw=0.5)

# fig_histograms = plt.figure(figsize=(6,6))
# plt.rcParams['font.size'] = 10
# axTOPLEFT = fig_histograms.add_subplot(221)
# axTOPRIGHT = fig_histograms.add_subplot(222)
# axBOTTOMLEFT = fig_histograms.add_subplot(223)
# axBOTTOMRIGHT = fig_histograms.add_subplot(224)
# axes = [axTOPLEFT,axTOPRIGHT,axBOTTOMLEFT,axBOTTOMRIGHT]

# for ilib in range(nlib):
#     ax = axes[ilib]
#     libration = 'p'+str(mmrps[ilib])+'q'+str(mmrqs[ilib])
#     df_tnos_indices = pd.read_csv('b004_'+libration+'_index.csv')
#     tnos_indices = df_tnos_indices['index'].to_list()
#     nobj = len(tnos_indices)
#     q_laplace = df_laplace['laplace_q'][laplace_indices[ilib]]
#     p_laplace = df_laplace['laplace_p'][laplace_indices[ilib]]
#     df_mmr = pd.read_csv('b012_tnos_'+libration+'_siraj.csv')
#     qcc = df_mmr['q_siraj'][0]
#     pcc = df_mmr['p_siraj'][0]
#     scc = df_mmr['s_siraj'][0]
#     kappa_here = df_mmr['gamma_siraj'][0]
#     q_bary_here = q_bary_tnos[tnos_indices]
#     p_bary_here = p_bary_tnos[tnos_indices]
#     s_bary_here = s_bary_tnos[tnos_indices]
#     ireldeg_list = []
#     for iobj in range(nobj):
#         dot = qcc*q_bary_here[iobj] + pcc*p_bary_here[iobj] + scc*s_bary_here[iobj]
#         ireldeg_list.append(np.degrees(np.arccos(dot)))
#     data = ireldeg_list
#     kappa = kappa_here
#     ax.hist(data,bins=longbins,color='lightblue',linewidth=0.5,edgecolor='black')
#     myHist = np.histogram(data,bins=longbins)
#     myHist_maxbinheight = np.max(myHist[0])
#     myHist_dense = np.histogram(data,bins=longbins,density=True)
#     myHist_maxdenseheight = np.max(myHist_dense[0])
#     radians_vec = np.radians(np.linspace(0,90,num=10000,endpoint=True))
#     curve_vec = curve_vec_exact(radians_vec,kappa_here)
#     curve_max = np.max(curve_vec)
#     ax.plot(np.degrees(radians_vec),curve_vec*myHist_maxbinheight/curve_max,color='red',linestyle='solid',lw=0.5)
#     boxstr = str(mmrps[ilib])+':'+str(mmrqs[ilib])+', $\gamma_{est}=$'+str(np.round(kappa,1))
#     ax.text(0.92,0.92,boxstr,horizontalalignment='right',verticalalignment='top',\
#             bbox=bbox_props,transform=ax.transAxes)
#     ax.set_xlabel('Relative inclination (degrees)')
#     ax.set_ylabel('Count')
#     ax.set_xlim([0,90])
# axTOPLEFT.set_xlabel('')
# axTOPRIGHT.set_xlabel('')
# axTOPRIGHT.set_ylabel('')
# axBOTTOMRIGHT.set_ylabel('')
# fig_histograms.tight_layout()
# fig_histograms.savefig('b013_histograms_mmrs_2x2.png',dpi=400)

# fig_histograms = plt.figure(figsize=(6.5,2))
# plt.rcParams['font.size'] = 6
# ax1 = fig_histograms.add_subplot(141)
# ax2 = fig_histograms.add_subplot(142)
# ax3 = fig_histograms.add_subplot(143)
# ax4 = fig_histograms.add_subplot(144)
# axes = [ax1,ax2,ax3,ax4]

# for ilib in range(nlib):
#     ax = axes[ilib]
#     libration = 'p'+str(mmrps[ilib])+'q'+str(mmrqs[ilib])
#     df_tnos_indices = pd.read_csv('b004_'+libration+'_index.csv')
#     tnos_indices = df_tnos_indices['index'].to_list()
#     nobj = len(tnos_indices)
#     q_laplace = df_laplace['laplace_q'][laplace_indices[ilib]]
#     p_laplace = df_laplace['laplace_p'][laplace_indices[ilib]]
#     df_mmr = pd.read_csv('b012_tnos_'+libration+'_siraj.csv')
#     qcc = df_mmr['q_siraj'][0]
#     pcc = df_mmr['p_siraj'][0]
#     scc = df_mmr['s_siraj'][0]
#     kappa_here = df_mmr['gamma_siraj'][0]
#     q_bary_here = q_bary_tnos[tnos_indices]
#     p_bary_here = p_bary_tnos[tnos_indices]
#     s_bary_here = s_bary_tnos[tnos_indices]
#     ireldeg_list = []
#     for iobj in range(nobj):
#         dot = qcc*q_bary_here[iobj] + pcc*p_bary_here[iobj] + scc*s_bary_here[iobj]
#         ireldeg_list.append(np.degrees(np.arccos(dot)))
#     data = ireldeg_list
#     kappa = kappa_here
#     ax.hist(data,bins=longbins,color='lightblue',linewidth=0.5,edgecolor='black')
#     myHist = np.histogram(data,bins=longbins)
#     myHist_maxbinheight = np.max(myHist[0])
#     myHist_dense = np.histogram(data,bins=longbins,density=True)
#     myHist_maxdenseheight = np.max(myHist_dense[0])
#     radians_vec = np.radians(np.linspace(0,90,num=10000,endpoint=True))
#     curve_vec = curve_vec_exact(radians_vec,kappa_here)
#     curve_max = np.max(curve_vec)
#     ax.plot(np.degrees(radians_vec),curve_vec*myHist_maxbinheight/curve_max,color='red',linestyle='solid',lw=0.5)
#     boxstr = str(mmrps[ilib])+':'+str(mmrqs[ilib])+', $\gamma_{est}=$'+str(np.round(kappa,1))
#     ax.text(0.92,0.92,boxstr,horizontalalignment='right',verticalalignment='top',\
#             bbox=bbox_props,transform=ax.transAxes)
#     ax.set_xlabel('Relative inclination (degrees)')
#     ax.set_ylabel('Count')
#     ax.set_xlim([0,90])
# ax2.set_ylabel('')
# ax3.set_ylabel('')
# ax4.set_ylabel('')
# fig_histograms.tight_layout()
# fig_histograms.savefig('b013_histograms_mmrs_1x4.png',dpi=400)

for ilib in range(nlib):
    fig_histograms = plt.figure(figsize=(1.75,1.75))
    plt.rcParams['font.size'] = 6
    ax = fig_histograms.add_subplot(111)
    libration = 'p'+str(mmrps[ilib])+'q'+str(mmrqs[ilib])
    df_tnos_indices = pd.read_csv('b004_'+libration+'_index.csv')
    tnos_indices = df_tnos_indices['index'].to_list()
    nobj = len(tnos_indices)
    q_laplace = df_laplace['laplace_q'][laplace_indices[ilib]]
    p_laplace = df_laplace['laplace_p'][laplace_indices[ilib]]
    df_mmr = pd.read_csv('b012_tnos_'+libration+'_siraj.csv')
    qcc = df_mmr['q_siraj'][0]
    pcc = df_mmr['p_siraj'][0]
    scc = df_mmr['s_siraj'][0]
    kappa_here = df_mmr['gamma_siraj'][0]
    q_bary_here = q_bary_tnos[tnos_indices]
    p_bary_here = p_bary_tnos[tnos_indices]
    s_bary_here = s_bary_tnos[tnos_indices]
    ireldeg_list = []
    for iobj in range(nobj):
        dot = qcc*q_bary_here[iobj] + pcc*p_bary_here[iobj] + scc*s_bary_here[iobj]
        ireldeg_list.append(np.degrees(np.arccos(dot)))
    data = ireldeg_list
    kappa = kappa_here
    ax.hist(data,bins=longbins,color='lightblue',linewidth=0.5,edgecolor='black')
    myHist = np.histogram(data,bins=longbins)
    myHist_maxbinheight = np.max(myHist[0])
    myHist_dense = np.histogram(data,bins=longbins,density=True)
    myHist_maxdenseheight = np.max(myHist_dense[0])
    radians_vec = np.radians(np.linspace(0,90,num=10000,endpoint=True))
    curve_vec = curve_vec_exact(radians_vec,kappa_here)
    curve_max = np.max(curve_vec)
    ax.plot(np.degrees(radians_vec),curve_vec*myHist_maxbinheight/curve_max,color='red',linestyle='solid',lw=0.5)
    boxstr = str(mmrps[ilib])+':'+str(mmrqs[ilib])+', $\gamma_{est}=$'+str(np.round(kappa,1))
    ax.text(0.92,0.92,boxstr,horizontalalignment='right',verticalalignment='top',\
            bbox=bbox_props,transform=ax.transAxes)
    ax.set_xlabel('Relative inclination (degrees)')
    ax.set_ylabel('Count')
    ax.set_xlim([0,90])
    fig_histograms.tight_layout()
    fig_histograms.savefig('b013_histograms_mmrs_1x1_'+libration+'.png',dpi=400)
#%%
for ilib in range(nlib):
    libration = 'p'+str(mmrps[ilib])+'q'+str(mmrqs[ilib])
    df_tnos_indices = pd.read_csv('b004_'+libration+'_index.csv')
    tnos_indices = df_tnos_indices['index'].to_list()
    nobj = len(tnos_indices)
    df_mmr = pd.read_csv('b012_tnos_'+libration+'_siraj.csv')
    qcc = df_mmr['q_siraj'][0]
    pcc = df_mmr['p_siraj'][0]
    ideg = df_mmr['ideg_siraj'][0]
    Wdeg = df_mmr['Wdeg_siraj'][0]
    angledeg95 = df_mmr['angledegs_siraj'][1]
    kappa_here = df_mmr['gamma_siraj'][0]
    q_laplace = df_laplace['laplace_q'][laplace_indices[ilib]]
    p_laplace = df_laplace['laplace_p'][laplace_indices[ilib]]
    scc = np.sqrt(1-qcc**2-pcc**2)
    s_laplace = np.sqrt(1-q_laplace**2-p_laplace**2)
    s_neptune = np.sqrt(1-q_neptune**2-p_neptune**2)
    s_invar = np.sqrt(1-q_invar**2-p_invar**2)
    dot_invar = qcc*q_invar + pcc*p_invar + scc*s_invar
    dot_neptune = qcc*q_neptune + pcc*p_neptune + scc*s_neptune
    dot_laplace = qcc*q_laplace + pcc*p_laplace + scc*s_laplace
    deg_distance_invar = np.degrees(np.arccos(dot_invar))
    deg_distance_neptune = np.degrees(np.arccos(dot_neptune))
    deg_distance_laplace = np.degrees(np.arccos(dot_laplace))
    aau_array_qcc = aau_bary_tnos[tnos_indices]
    e_array_qcc = e_bary_tnos[tnos_indices]
    irad_array_qcc = irad_bary_tnos[tnos_indices]
    wrad_array_qcc = wrad_bary_tnos[tnos_indices]
    Wrad_array_qcc = Wrad_bary_tnos[tnos_indices]
    Mrad_array_qcc = Mrad_bary_tnos[tnos_indices]
    probmass_invar_wrt_qcc = probmass_siraj(aau_array_qcc,e_array_qcc,\
            irad_array_qcc,wrad_array_qcc,Wrad_array_qcc,Mrad_array_qcc,q_invar,p_invar)
    probmass_neptune_wrt_qcc = probmass_siraj(aau_array_qcc,e_array_qcc,\
            irad_array_qcc,wrad_array_qcc,Wrad_array_qcc,Mrad_array_qcc,q_neptune,p_neptune)
    probmass_laplace_wrt_qcc = probmass_siraj(aau_array_qcc,e_array_qcc,\
            irad_array_qcc,wrad_array_qcc,Wrad_array_qcc,Mrad_array_qcc,q_laplace,p_laplace)
    pval_invar = 1 - probmass_invar_wrt_qcc
    pval_neptune = 1 - probmass_neptune_wrt_qcc
    pval_laplace = 1 - probmass_laplace_wrt_qcc
    print(libration)
    print('n',nobj)
    print('q',np.round(qcc,4))
    print('p',np.round(pcc,4))
    print('ideg',np.round(ideg,2))
    print('Wdeg',np.round(Wdeg,2))
    print('thetadeg95',np.round(angledeg95,2))
    print('gamma_est',np.round(kappa_here,2))
    print('sigmadegrayleigh_est',np.round(np.degrees(kappa_here**-0.5),2))
    print('deg_distance_invar',np.round(deg_distance_invar,4))
    print('deg_distance_neptune',np.round(deg_distance_neptune,4))
    print('deg_distance_laplace',np.round(deg_distance_laplace,4))
    print('pval_invar',np.round(pval_invar,4))
    print('pval_neptune',np.round(pval_neptune,4))
    print('pval_laplace',np.round(pval_laplace,4))
    print('')
#%%
# import pandas as pd
# import numpy as np
# # from matplotlib import pyplot as plt
# #%%
# lawler_strs = ['KozaiFlippers','UnstableKozai','StableKozaiPlutinos','StablePlutinos']
# law_shorts = ['KF','UK','SKP','SP']
# nlaw = len(lawler_strs)
# librations = ['gplus_2026feb12','gminus_2026feb12','gnone_2026feb12','gplusminusnone_2026feb12',\
#               'gunstable_2026feb12']
# lib_shorts = ['g+','g-','g0','g+-o','gu']
# nlib = len(librations)
# jd = 2460796.5 # May 1, 2025 00:00:00
# time_yrs  = int(1e8)
# tstep_yrs = int(1e3)
# dt_yrs = 0.5
# tyrs_str = '1e8yr'
# tstepyrs_str = '1e3yr'
# dtyrs_str = '0.5yr'
# yrsstr = tyrs_str + '_' + tstepyrs_str + '_' + dtyrs_str + '_jd' + str(jd)
# Wrad_mat_planets = np.radians(np.loadtxt('b0000_planets_nodedeg_barycentric_'+yrsstr+'.csv',delimiter=','))
# wrad_mat_planets = np.radians(np.loadtxt('b0000_planets_perideg_barycentric_'+yrsstr+'.csv',delimiter=','))
# Mrad_mat_planets = np.radians(np.loadtxt('b0000_planets_Mdeg_barycentric_'+yrsstr+'.csv',delimiter=','))
# irad_mat_planets = np.radians(np.loadtxt('b0000_planets_ideg_barycentric_'+yrsstr+'.csv',delimiter=','))
# e_mat_planets = np.loadtxt('b0000_planets_e_barycentric_'+yrsstr+'.csv',delimiter=',')
# aau_mat_planets = np.loadtxt('b0000_planets_aau_barycentric_'+yrsstr+'.csv',delimiter=',')
# irad_N = irad_mat_planets[3,0]
# Wrad_N = Wrad_mat_planets[3,0]
# qqN = np.sin(irad_N/2)*np.cos(Wrad_N)
# ppN = np.sin(irad_N/2)*np.sin(Wrad_N)
# aN = aau_mat_planets[3,0]
# del aau_mat_planets,e_mat_planets,irad_mat_planets,wrad_mat_planets,Wrad_mat_planets,Mrad_mat_planets
# laplacemat = np.loadtxt('b0004_b_laplaceplane_t0.csv',delimiter=',')
# mmrs = ['32','21','52','74','53']
# colors = ['magenta','tomato','green','cyan','black']
# res_q_list = np.array([3,2,5,7,5])
# res_p_list = np.array([2,1,2,4,3])
# anom_list = (res_q_list/res_p_list)**(2/3) * aN
# # deltas = [0.2,0.25,0.04]
# anom_min = np.min(anom_list)
# anom_max = np.max(anom_list)
# print(anom_min,anom_max) # 39.3987,55.3836
# anom_min2 = np.round(anom_min,0)-2
# anom_max2 = np.round(anom_max,0)+2
# aau_vec = np.arange(start=anom_min2,stop=anom_max2,step=0.01)
# laplace_q = laplacemat[0,:]
# laplace_p = laplacemat[1,:]
# laplace_s = np.sqrt(1-laplace_q**2-laplace_p**2)
# laplace_irad = np.arccos(laplace_s)
# laplace_Wrad = np.arctan2(laplace_p/np.sin(laplace_irad),laplace_q/np.sin(laplace_irad))
# laplace_qq = np.sin(laplace_irad/2)*np.cos(laplace_Wrad)
# laplace_pp = np.sin(laplace_irad/2)*np.sin(laplace_Wrad)
# laplacenom_qq_list = np.interp(anom_list,aau_vec,laplace_qq)
# laplacenom_pp_list = np.interp(anom_list,aau_vec,laplace_pp)
# nmmrs = len(mmrs)
# df_mean = pd.read_csv('b0000_siraj_mmrs_mean.csv')
# ideg_mean_vec = np.array(df_mean['ideg_mean'].to_list())
# Wdeg_mean_vec = np.array(df_mean['Wdeg_mean'].to_list())
# logLmax_mean_vec = np.array(df_mean['logLmax_mean'].to_list())
# qq_mean_vec = np.sin(np.radians(ideg_mean_vec/2))*np.cos(np.radians(Wdeg_mean_vec))
# pp_mean_vec = np.sin(np.radians(ideg_mean_vec/2))*np.sin(np.radians(Wdeg_mean_vec))
# ideg_invar = 1.58
# Wdeg_invar = 107.6
# qq_invar = np.sin(np.radians(ideg_invar/2))*np.cos(np.radians(Wdeg_invar))
# pp_invar = np.sin(np.radians(ideg_invar/2))*np.sin(np.radians(Wdeg_invar))
# #%%
# # fig = plt.figure(figsize=(7,7))
# # plt.rcParams['font.size'] = 12
# # ax = fig.add_subplot(111)
# # plt.axhline(0,color='gray')
# # plt.axvline(0,color='gray')
# # plt.xlabel('qq=sin(i/2)cos(W)')
# # plt.ylabel('pp=sin(i/2)sin(W)')
# # # ax.scatter(0,0,color='gray',marker='+',s=10)
# # ax.scatter(qqN,ppN,color='blue',marker='.',s=20)
# # ax.text(qqN,ppN,'N')
# # ax.scatter(qq_invar,pp_invar,color='red',marker='x',s=20)
# # ax.text(qq_invar,pp_invar,'I')
# # ax.scatter(laplace_qq,laplace_pp,color='black',marker='.',s=0.2,alpha=0.5)
# # for immr in range(nmmrs):
# #     mmr = mmrs[immr]
# #     color = colors[immr]
# #     lqq = laplacenom_qq_list[immr]
# #     lpp = laplacenom_pp_list[immr]
# #     ax.scatter(lqq,lpp,color='black',marker='+',s=5)
# #     ax.text(lqq,lpp,'L'+mmr)
# #     ax.scatter(qq_mean_vec[immr],pp_mean_vec[immr],color=color,marker='x',s=20)
# #     infile = 'b0000_siraj_barycentric_'+mmr+'_logL_mat_200x200_pm0.08.csv'
# #     logLmat = np.loadtxt(infile,delimiter=',')
# #     logLmax_mean = logLmax_mean_vec[immr]
# #     sigma_mat = np.sqrt(2*(logLmax_mean-logLmat))
# #     # delta = deltas[ilibrate]
# #     delta = 0.08
# #     qqmean = qq_mean_vec[immr]
# #     ppmean = pp_mean_vec[immr]
# #     qqmin = qqmean - delta
# #     qqmax = qqmean + delta
# #     ppmin = ppmean - delta
# #     ppmax = ppmean + delta
# #     nqq = 200
# #     npp = 200
# #     dqq = 2*delta/nqq
# #     dpp = 2*delta/npp
# #     qqvec = np.linspace(start=qqmin,stop=qqmax,num=nqq+1,endpoint=True)
# #     ppvec = np.linspace(start=ppmin,stop=ppmax,num=npp+1,endpoint=True)
# #     xi = qqvec
# #     yi = ppvec
# #     p_mat = np.exp(-sigma_mat**2.2)
# #     zi = p_mat
# #     pmin = 1000
# #     pmax = 0
# #     for iqq in range(nqq+1):
# #         for ipp in range(npp+1):
# #             phere = p_mat[iqq,ipp]
# #             if phere < pmin:
# #                 pmin = phere
# #             if phere > pmax:
# #                 pmax = phere
# #     print(pmin,pmax)
# #     zmin = pmin
# #     zmax = pmax
# #     quadset = ax.contour(xi,yi,zi,levels=[0.0001,0.001,0.01,0.05,0.1,0.5],colors=color)
# #     ax.scatter(qqmean,ppmean,marker='.',c=color,s=50)
# #     ax.text(qqmean,ppmean,'M'+mmr)
# # ax.set_xlim([-0.028,0.05])
# # ax.set_ylim([0,0.04])
# # plt.savefig('b0014_plot_contours_allmmrs.png',dpi=400)
# # plt.show()
# #%%
# # print('x0, y0, ap, bp, e, phi = ', x0, y0, ap, bp, e, phi)
# # levels = [0.0001,0.001,0.01,0.05,0.1,1-0.682,0.5]
# # nlevels = len(levels)
# # x0_list = []
# # y0_list = []
# # e_list = []
# # phi_list = []
# # ap_mat = np.zeros((nmmrs,nlevels))
# # bp_mat = np.zeros((nmmrs,nlevels))
# # for immr in range(nmmrs):
# for immr in [0]:
#     fig = plt.figure(figsize=(10,10))
#     plt.rcParams['font.size'] = 12
#     ax = fig.add_subplot(111)
#     plt.axhline(0,color='gray')
#     plt.axvline(0,color='gray')
#     plt.xlabel('qq=sin(i/2)cos(W)')
#     plt.ylabel('pp=sin(i/2)sin(W)')
#     # ax.scatter(0,0,color='gray',marker='+',s=10)
#     ax.scatter(qqN,ppN,color='blue',marker='.',s=20)
#     ax.text(qqN,ppN,'N')
#     ax.scatter(qq_invar,pp_invar,color='red',marker='x',s=20)
#     ax.text(qq_invar,pp_invar,'I')
#     ax.scatter(laplace_qq,laplace_pp,color='black',marker='.',s=0.2,alpha=0.5)
#     mmr = mmrs[immr]
#     color = colors[immr]
#     lqq = laplacenom_qq_list[immr]
#     lpp = laplacenom_pp_list[immr]
#     print(mmr,'lqq =',lqq,', lpp =',lpp)
#     ax.scatter(lqq,lpp,color='black',marker='+',s=5)
#     ax.text(lqq,lpp,'L'+mmr)
#     ax.scatter(qq_mean_vec[immr],pp_mean_vec[immr],color=color,marker='x',s=20)
#     infile = 'b0000_siraj_barycentric_'+mmr+'_logL_mat_200x200_pm0.08.csv'
#     logLmat = np.loadtxt(infile,delimiter=',')
#     logLmax_mean = logLmax_mean_vec[immr]
#     sigma_mat = np.sqrt(2*(logLmax_mean-logLmat))
#     # delta = deltas[ilibrate]
#     delta = 0.08
#     qqmean = qq_mean_vec[immr]
#     ppmean = pp_mean_vec[immr]
#     # x0_list.append(qqmean)
#     # y0_list.append(ppmean)
#     qqmin = qqmean - delta
#     qqmax = qqmean + delta
#     ppmin = ppmean - delta
#     ppmax = ppmean + delta
#     nqq = 200
#     npp = 200
#     dqq = 2*delta/nqq
#     dpp = 2*delta/npp
#     qqvec = np.linspace(start=qqmin,stop=qqmax,num=nqq+1,endpoint=True)
#     ppvec = np.linspace(start=ppmin,stop=ppmax,num=npp+1,endpoint=True)
#     xi = qqvec
#     yi = ppvec
#     p_mat = np.exp(-sigma_mat**2.2)
#     zi = p_mat
#     pmin = 1000
#     pmax = 0
#     for iqq in range(nqq+1):
#         for ipp in range(npp+1):
#             phere = p_mat[iqq,ipp]
#             if phere < pmin:
#                 pmin = phere
#             if phere > pmax:
#                 pmax = phere
#     print(mmr,'pmin = ',pmin,', pmax = ',pmax)
#     zmin = pmin
#     zmax = pmax
#     levels = [1-0.997]
#     quadset = ax.contour(xi,yi,zi,levels=levels,colors=color)
#     ax.clabel(quadset,fontsize=6,inline=False)
#     ax.scatter(qqmean,ppmean,marker='.',c=color,s=50)
#     ax.text(qqmean,ppmean,'M'+mmr)
#     # ax.set_xlim([-0.028,0.05])
#     # ax.set_ylim([0,0.04])
#     i_mmk23_deg = 3.57
#     W_mmk23_deg = 124.38
#     q997_mmk23_deg = 1.68
#     idebiased_mmk23_deg = 2.26
#     Wdebiased_mmk23_deg = 22.69
#     qq_mmk23 = np.sin(np.radians(i_mmk23_deg))*np.cos(np.radians(W_mmk23_deg))
#     pp_mmk23 = np.sin(np.radians(i_mmk23_deg))*np.sin(np.radians(W_mmk23_deg))
#     qq_mmk23_de = np.sin(np.radians(idebiased_mmk23_deg))*np.cos(np.radians(Wdebiased_mmk23_deg))
#     pp_mmk23_de = np.sin(np.radians(idebiased_mmk23_deg))*np.sin(np.radians(Wdebiased_mmk23_deg))
#     theta = np.linspace(start=0,stop=2*np.pi,num=50,endpoint=True)
#     qq_mmk23_circle = qq_mmk23 + np.sin(np.radians(q997_mmk23_deg))*np.cos(theta)
#     pp_mmk23_circle = pp_mmk23 + np.sin(np.radians(q997_mmk23_deg))*np.sin(theta)
#     qq_mmk23_circle_de = qq_mmk23_de + np.sin(np.radians(q997_mmk23_deg))*np.cos(theta)
#     pp_mmk23_circle_de = pp_mmk23_de + np.sin(np.radians(q997_mmk23_deg))*np.sin(theta)
#     ax.scatter(qq_mmk23,pp_mmk23,marker='^',c='blue',s=50)
#     ax.scatter(qq_mmk23_de,pp_mmk23_de,marker='<',c='green',s=50)
#     ax.plot(qq_mmk23_circle,pp_mmk23_circle,color='blue')
#     ax.plot(qq_mmk23_circle_de,pp_mmk23_circle_de,color='green')
#     # xmin0 = np.min(qq_mmk23_circle)
#     # xmax0 = np.max(qq_mmk23_circle_de)
#     # ymin0 = np.min(pp_mmk23_circle_de)
#     # ymax0 = np.max(pp_mmk23_circle)
#     xmin0 = -0.05
#     xmax0 = 0.06
#     ymin0 = -0.005
#     ymax0 = 0.1
#     xmid0 = (xmin0+xmax0)/2
#     ymid0 = (ymin0+ymax0)/2
#     xdiff0 = xmax0-xmin0
#     ydiff0 = ymax0-ymin0
#     maxdiff0 = np.max([xdiff0,ydiff0])
#     ratio = 1.05
#     maxdiff = ratio * maxdiff0
#     xmin = xmid0 - maxdiff/2
#     xmax = xmid0 + maxdiff/2
#     ymin = ymid0 - maxdiff/2
#     ymax = ymid0 + maxdiff/2
#     ax.set_xlim([xmin,xmax])
#     ax.set_ylim([ymin,ymax])
#     ax.text(qq_mmk23,pp_mmk23,'mmk23')
#     ax.text(qq_mmk23_de,pp_mmk23_de,'mmk23_debiased')
#     ax.set_box_aspect(1)
#     for ilaw in range(nlaw):
#     # for immr in [0]:
#         lawler_str = lawler_strs[ilaw]
#         # print('')
#         # print(lawler_str)
#         siraj_file = 'a00000_lawler_'+lawler_str+'.txt'
#         df_siraj = pd.read_csv(siraj_file,sep=r"\s+")
#         nobj = df_siraj.shape[0]
#         # print('nobj',nobj)
#         aau_list = np.array(df_siraj['a'].to_list())
#         e_list = np.array(df_siraj['e'].to_list())
#         irad_list = np.radians(np.array(df_siraj['inc'].to_list()))
#         wrad_list = np.radians(np.array(df_siraj['omega'].to_list()))
#         Wrad_list = np.radians(np.array(df_siraj['Omega'].to_list()))
#         Mrad_list = np.radians(np.array(df_siraj['Manom'].to_list()))
#         qperiau_list = aau_list * (1-e_list)
#         q_list = np.sin(irad_list)*np.cos(Wrad_list)
#         p_list = np.sin(irad_list)*np.sin(Wrad_list)
#         s_list = np.cos(irad_list)
#         hx_list = p_list
#         hy_list = -q_list
#         hz_list = s_list
#         xvec = hx_list
#         yvec = hy_list
#         zvec = hz_list
#         # t0 = time.time()
#         xcc,ycc,zcc,angledeg,sigmahat,Rbar,Kout = vmf_fun(xvec,yvec,zvec,0.997)
#         qcc = -ycc
#         pcc = xcc
#         scc = zcc
#         irad_cc = np.arccos(scc)
#         sini = np.sin(irad_cc)
#         Wrad_cc = np.arctan2(pcc/sini,qcc/sini)
#         ideg_cc = np.degrees(irad_cc)
#         Wdeg_cc = np.degrees(Wrad_cc)
#         qq_cc = np.sin(irad_cc/2)*np.cos(Wrad_cc)
#         pp_cc = np.sin(irad_cc/2)*np.sin(Wrad_cc)
#         qq_cc_circle = qq_cc + np.sin(np.radians(angledeg))*np.cos(theta)
#         pp_cc_circle = pp_cc + np.sin(np.radians(angledeg))*np.sin(theta)
#         ax.scatter(qq_cc,pp_cc,s=50)
#         ax.plot(qq_cc_circle,pp_cc_circle)
#         ax.text(qq_cc,pp_cc,law_shorts[ilaw])
#     for ilib in range(nlib):
#         libstr = librations[ilib]
#         meanfile = 'b0004_siraj_p3q2librations_mean.csv'
#         df = pd.read_csv(meanfile)
#         libsdf = df['librations'].to_list()
#         libind = libsdf.index(libstr)
#         ideg_cc = df['ideg_mean'][libind]
#         Wdeg_cc = df['Wdeg_mean'][libind]
#         logL_cc = df['logLmax_mean'][libind]
#         qq_cc = np.sin(np.radians(ideg_cc/2))*np.cos(np.radians(Wdeg_cc))
#         pp_cc = np.sin(np.radians(ideg_cc/2))*np.sin(np.radians(Wdeg_cc))
#         logLmat = np.loadtxt('b0004_siraj_bary_p3q2_'+libstr+'_logLmat_nqq100npp100_delta0.08.csv',delimiter=',')
#         sigma_mat = np.sqrt(2*(logL_cc-logLmat))
#         p_mat = np.exp(-sigma_mat**2.2)
#         delta = 0.08
#         qqmean = qq_cc
#         ppmean = pp_cc
#         qqmin = qqmean - delta
#         qqmax = qqmean + delta
#         ppmin = ppmean - delta
#         ppmax = ppmean + delta
#         nqq = 100
#         npp = 100
#         dqq = 2*delta/nqq
#         dpp = 2*delta/npp
#         qqvec = np.linspace(start=qqmin,stop=qqmax,num=nqq+1,endpoint=True)
#         ppvec = np.linspace(start=ppmin,stop=ppmax,num=npp+1,endpoint=True)
#         xi = qqvec
#         yi = ppvec
#         zi = p_mat
#         pmin = 1000
#         pmax = 0
#         for iqq in range(nqq+1):
#             for ipp in range(npp+1):
#                 phere = p_mat[iqq,ipp]
#                 if phere < pmin:
#                     pmin = phere
#                 if phere > pmax:
#                     pmax = phere
#         print(pmin,pmax)
#         zmin = pmin
#         zmax = pmax
#         quadset = ax.contour(xi,yi,zi,levels=[1-0.997])
#         ax.scatter(qq_cc,pp_cc,s=50)
#         ax.text(qq_cc,pp_cc,lib_shorts[ilib])
        
    
#     plt.savefig('b0014_plot_contours_'+mmr+'_lotsof997ellipses.png',dpi=400)
#     plt.show()
#     # fig = plt.figure(figsize=(7,7))
#     # plt.rcParams['font.size'] = 12
#     # ax = fig.add_subplot(111)
#     # plt.axhline(0,color='gray')
#     # plt.axvline(0,color='gray')
#     # plt.xlabel('qq=sin(i/2)cos(W)')
#     # plt.ylabel('pp=sin(i/2)sin(W)')
#     # level = quadset.levels[0]
#     # segs = quadset.allsegs[0]
#     # segs_array = np.array(segs)
#     # segs_flat = np.squeeze(segs_array)
#     # xsegs = segs_flat[:,0]
#     # ysegs = segs_flat[:,1]
#     # nsegs = len(xsegs)
#     # coeffs = fit_ellipse(xsegs,ysegs)
#     # print('a, b, c, d, e, f =', coeffs)
#     # x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
#     # print('x0, y0, ap, bp, e, phi = ', x0, y0, ap, bp, e, phi)
#     # tmin, tmax = 0, 2*np.pi
#     # xell, yell = get_ellipse_pts((x0, y0, ap, bp, None, phi), nsegs, tmin, tmax)
#     # plt.scatter(xsegs,ysegs)
#     # plt.scatter(xell,yell)
#     # plt.show()
    
