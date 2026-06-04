#%% laplace plane helper function
def b32_1_fun(alpha):
    import numpy as np
    from scipy import integrate
    I = integrate.quad(b32_1_integrand,0,2*np.pi,args=(alpha))
    b32_1_result = I[0]/np.pi
    error = I[1]
    return b32_1_result, error
#%% laplace plane helper function
def b32_1_integrand(psi,alpha):
    import numpy as np
    integrand = np.cos(psi)/(1-2*alpha*np.cos(psi)+alpha**2)**(3/2)
    return integrand
#%%
def get_GMdict():
    mu_planets = np.loadtxt('b003_m8ss12.csv')
    GM_sun = mu_planets[0]
    GM_mercury = mu_planets[1]
    GM_venus = mu_planets[2]
    GM_earthmoon = mu_planets[3]
    GM_mars = mu_planets[4]
    GM_jupiter = mu_planets[5]
    GM_saturn = mu_planets[6]
    GM_uranus = mu_planets[7]
    GM_neptune = mu_planets[8]
    # GM_sun = 1
    # GM_mercury = 1/6023600
    # GM_venus = 1/408523.71
    # GM_earthmoon = 1/328900.56
    # GM_mars = 1/3098708
    # GM_jupiter = 1/1047.3486
    # GM_saturn = 1/3497.898
    # GM_uranus = 1/22902.98
    # GM_neptune = 1/19412.24
    GMdict = {'sun':GM_sun,'mercury':GM_mercury,'venus':GM_venus,\
              'earth':GM_earthmoon,'mars':GM_mars,'jupiter':GM_jupiter,\
              'saturn':GM_saturn,'uranus':GM_uranus,'neptune':GM_neptune}
    return GMdict
#%%
def GM_outerplanets():
    GMdict = get_GMdict()
    mu_sun = GMdict['sun']
    m_mercury = GMdict['mercury']
    m_venus = GMdict['venus']
    m_earth = GMdict['earth']
    m_mars = GMdict['mars']
    m_rockies = m_mercury + m_venus + m_earth + m_mars
    m_jupiter = GMdict['jupiter']/(mu_sun + m_rockies)
    m_saturn = GMdict['saturn']/(mu_sun + m_rockies)
    m_uranus = GMdict['uranus']/(mu_sun + m_rockies)
    m_neptune = GMdict['neptune']/(mu_sun + m_rockies)
    mu_planets = np.array([m_jupiter,m_saturn,m_uranus,m_neptune])
    return mu_planets
#%% compute laplace plane
def laplace_plane(a_sample,mu_planets,wmat_planets,\
    Wmat_planets,Mmat_planets,imat_planets,emat_planets,amat_planets): # inputs already in au, rad
    import numpy as np
    mu_sun = 1.0
    n_sample = np.sqrt(mu_sun/a_sample**3) # rad/s
    N = len(mu_planets)
    a_planets = amat_planets
    i_planets = imat_planets
    W_planets = Wmat_planets
    n_planets = np.sqrt(mu_sun/a_planets**3)
    B = np.zeros([N,N])
    alpha = np.zeros([N,N])
    alphabar = np.zeros([N,N])
    # eq 7.128, 7.129
    for j in range(N):
        for k in range(N):
           aj = a_planets[j]
           ak = a_planets[k]
           if aj > ak:
               alpha[j,k] = ak/aj
               alphabar[j,k] = 1
           else:
               alpha[j,k] = aj/ak
               alphabar[j,k] = aj/ak
    # eq 7.134, 7.135
    for j in range(N):
        for k in range(N):
            b32_1_result,error = b32_1_fun(alpha[j,k])
            B[j,k] = 1/4 * mu_planets[k]/(mu_sun+mu_planets[j]) * \
                n_planets[j] * alpha[j,k] * alphabar[j,k] * b32_1_result
    for j in range(N):
        B[j,j] = 0
        for k in range(N):
            if k != j:
                b32_1_result,error = b32_1_fun(alpha[j,k])
                B[j,j] = B[j,j] - n_planets[j] * 1/4 * mu_planets[k] / \
                    (mu_sun+mu_planets[j]) * alpha[j,k] * alphabar[j,k] * \
                    b32_1_result
    I_mat = np.zeros([N,N])
    f_list,Ibar = np.linalg.eig(B) # pg 301 below eq 7.138
    q_planets = np.sin(i_planets)*np.cos(W_planets) # eq 7.19
    p_planets = np.sin(i_planets)*np.sin(W_planets) # eq 7.19
    T_cosgamma = np.linalg.solve(Ibar,q_planets) # eq 7.47
    T_singamma = np.linalg.solve(Ibar,p_planets) # eq 7.47
    T = np.sqrt(T_singamma**2+T_cosgamma**2)
    cosgamma = T_cosgamma/T
    singamma = T_singamma/T
    for i in range(N):
        for j in range(N):
            I_mat[j,i] = Ibar[j,i]*T[i] # eq 7.41
    gamma_array = np.mod(np.arctan2(singamma,cosgamma),2*np.pi)
    alpha_list = []
    alphabar_list = []
    # eq 7.128, 7.129
    for i in range(N):
        if a_planets[i] < a_sample:
            alpha_list.append(a_planets[i]/a_sample)
            alphabar_list.append(1)
        else:
            alpha_list.append(a_sample/a_planets[i])
            alphabar_list.append(a_sample/a_planets[i])
    # eq 7.144
    B_list = []
    for i in range(N):
        b32_1_result,error =  b32_1_fun(alpha_list[i])
        B_here = n_sample/4*mu_planets[i]/mu_sun*alpha_list[i]*alphabar_list[i] * \
            b32_1_result;
        B_list.append(B_here)
    # eq 7.143
    B_scalar = -np.sum(B_list);
    # eq 7.76
    mu_list = [] # not mu as in GM, overloaded notation
    for i in range(N):
        mu_here = 0
        for j in range(N):
            mu_here = mu_here + B_list[j]*I_mat[j,i]
        mu_list.append(mu_here)
    q0 = 0
    p0 = 0
    # eq 7.149, 7.150
    for i in range(N):
        qterm = mu_list[i] / (B_scalar-f_list[i])*np.cos(gamma_array[i])
        pterm = mu_list[i] / (B_scalar-f_list[i])*np.sin(gamma_array[i])
        q0 = q0 - qterm
        p0 = p0 - pterm
    i0 = np.arcsin(np.sqrt(q0**2+p0**2))
    W0 = np.arctan2(p0,q0)
    # i0 = np.mod(i0,2*np.pi)
    W0 = np.mod(W0,2*np.pi)
    f1 = f_list[0]
    f2 = f_list[1]
    f3 = f_list[2]
    f4 = f_list[3]
    mu1 = mu_list[0]
    mu2 = mu_list[1]
    mu3 = mu_list[2]
    mu4 = mu_list[3]
    gm1 = gamma_array[0]
    gm2 = gamma_array[1]
    gm3 = gamma_array[2]
    gm4 = gamma_array[3]
    return q0,p0,i0,W0,B_scalar,f1,f2,f3,f4,mu1,mu2,mu3,mu4,gm1,gm2,gm3,gm4
#%%
def laplace_lawler(a_sample):
    import numpy as np
    import pandas as pd
    mu_planets_reduced = np.loadtxt('b001_lawler_masses_reduced.csv')
    # df = pd.read_csv('a000_lawler_plEndState_edited_HE.csv')
    # mu_planets = mu_planets_reduced[1:]
    # aau = np.array(df['aau'].to_list())
    # e = np.array(df['e'].to_list())
    # irad = np.radians(np.array(df['ideg'].to_list()))
    # wrad = np.radians(np.array(df['wdeg'].to_list()))
    # Wrad = np.radians(np.array(df['Wdeg'].to_list()))
    # Mrad = np.radians(np.array(df['Mdeg'].to_list()))
    df = pd.read_csv('b001_lawler_plEndState_edited_BE.csv')
    mu_planets = mu_planets_reduced[1:]
    aau = np.array(df['aau'][1:].to_list())
    e = np.array(df['e'][1:].to_list())
    irad = np.radians(np.array(df['ideg'][1:].to_list()))
    wrad = np.radians(np.array(df['wdeg'][1:].to_list()))
    Wrad = np.radians(np.array(df['Wdeg'][1:].to_list()))
    Mrad = np.radians(np.array(df['Mdeg'][1:].to_list()))
    q0,p0,i0,W0,B_scalar,f1,f2,f3,f4,mu1,mu2,mu3,mu4,gm1,gm2,gm3,gm4 = \
        laplace_plane(a_sample,mu_planets,wrad,Wrad,Mrad,irad,e,aau) # inputs already in au, rad
    q_laplace_lawler = q0
    p_laplace_lawler = p0
    qNN = np.sin(irad[3])*np.cos(Wrad[3])
    pNN = np.sin(irad[3])*np.sin(Wrad[3])
    qJJ = np.sin(irad[0])*np.cos(Wrad[0])
    pJJ = np.sin(irad[0])*np.sin(Wrad[0])
    return q_laplace_lawler,p_laplace_lawler,qNN,pNN,qJJ,pJJ
    # import numpy as np
    # amat_planets = np.array([5.202119341346830,9.547430702901838,19.196065808579732,30.066551174919987])
    # emat_planets = np.array([0.052512849446157,0.047388561326068,0.036409121383009,0.011282673729328])
    # imat_planets = np.radians(np.array([1.456878225971503,2.269534903594406,1.877819857147376,1.038907776633367]))
    # Wmat_planets = np.radians(np.array([116.067189585728926,85.975600390121031,142.433145480612609,121.363731782446763]))
    # wmat_planets = np.radians(np.array([88.333147267124843,132.054570256651658,13.306112111733851,203.691377654794167]))
    # Mmat_planets = np.radians(np.array([268.356764527837981,245.443525085863200,234.812606300749962,95.102145814212406]))
    # mu_planets = GM_outerplanets()
    # q0,p0,i0,W0,B_scalar,f1,f2,f3,f4,mu1,mu2,mu3,mu4,gm1,gm2,gm3,gm4 = \
    #     laplace_plane(a_sample,mu_planets,wmat_planets,\
    #         Wmat_planets,Mmat_planets,imat_planets,emat_planets,amat_planets) # inputs already in au, rad
    # q_laplace_lawler = q0
    # p_laplace_lawler = p0
    # qNN = np.sin(imat_planets[3])*np.cos(Wmat_planets[3])
    # pNN = np.sin(imat_planets[3])*np.sin(Wmat_planets[3])
    # return q_laplace_lawler,p_laplace_lawler,qNN,pNN
#%%
import numpy as np
import pandas as pd
jdstr = '246e4'
df_planets = pd.read_csv('b003_planets_orbels_jd'+jdstr+'_HEHIBEBI.csv')
aau_bary_planets = np.array(df_planets['aau_BE'].to_list())
e_bary_planets = np.array(df_planets['e_BE'].to_list())
irad_bary_planets = np.radians(np.array(df_planets['ideg_BE'].to_list()))
wrad_bary_planets = np.radians(np.array(df_planets['wdeg_BE'].to_list()))
Wrad_bary_planets = np.radians(np.array(df_planets['Wdeg_BE'].to_list()))
Mrad_bary_planets = np.radians(np.array(df_planets['Mdeg_BE'].to_list()))
# mu_planets = GM_outerplanets()
mu_planets = np.loadtxt('b003_m8ss12.csv')
aau_vec = np.arange(start=30,stop=150,step=0.01)
na = len(aau_vec)
lidegvec = []
lWdegvec = []
for ia in range(na):
    aau0 = aau_vec[ia]
    q0,p0,i0,W0,B_scalar,f1,f2,f3,f4,mu1,mu2,mu3,mu4,gm1,gm2,gm3,gm4 = \
                laplace_plane(aau0,mu_planets,wrad_bary_planets,Wrad_bary_planets,\
                              Mrad_bary_planets,irad_bary_planets,e_bary_planets,aau_bary_planets)
    lidegvec.append(np.degrees(i0))
    lWdegvec.append(np.degrees(W0))
    print(ia,na)
dictionary = {'aau':aau_vec,'laplace_ideg':lidegvec,'laplace_Wdeg':lWdegvec}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b005_laplace_bary_jd'+jdstr+'.csv',index=False)
#%%
res_p_list = np.array([3,5,2,7,5])
res_q_list = np.array([2,2,1,4,3])
anom_list = (res_p_list/res_q_list)**(2/3) * aau_bary_planets[8]
print(anom_list)
lidegvec = []
lWdegvec = []
lqvec = []
lpvec = []
for ia in range(len(res_q_list)):
    aau0 = anom_list[ia]
    q0,p0,i0,W0,B_scalar,f1,f2,f3,f4,mu1,mu2,mu3,mu4,gm1,gm2,gm3,gm4 = \
                laplace_plane(aau0,mu_planets,wrad_bary_planets,Wrad_bary_planets,\
                              Mrad_bary_planets,irad_bary_planets,e_bary_planets,aau_bary_planets)
    lidegvec.append(np.degrees(i0))
    lWdegvec.append(np.degrees(W0))
    lqvec.append(q0)
    lpvec.append(p0)
    # print(ia,len(res_q_list),res_p_list[ia],res_q_list[ia],aau0,np.degrees(i0),np.degrees(W0))
dictionary = {'res_q':res_q_list,'res_p':res_p_list,'aau':anom_list,\
              'laplace_ideg':lidegvec,'laplace_Wdeg':lWdegvec,'laplace_q':lqvec,'laplace_p':lpvec}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b005_laplace_bary_rescenters_jd'+jdstr+'.csv',index=False)
dfout.to_csv('b005_laplace_BE_rescenters_jd'+jdstr+'.csv',index=False)
lidegvec = []
lWdegvec = []
lqvec = []
lpvec = []
for ia in range(len(res_q_list)):
    aau0 = anom_list[ia]
    q0,p0,qNN,pNN,qJJ,pJJ = laplace_lawler(aau0)
    s0 = np.sqrt(1-q0**2-p0**2)
    i0 = np.arccos(s0)
    sini = np.sin(i0)
    W0 = np.arctan2(p0/sini,q0/sini)
    lidegvec.append(np.degrees(i0))
    lWdegvec.append(np.degrees(W0))
    lqvec.append(q0)
    lpvec.append(p0)
    # print(ia,len(res_q_list),res_p_list[ia],res_q_list[ia],aau0,np.degrees(i0),np.degrees(W0))
dictionary = {'res_q':res_q_list,'res_p':res_p_list,'aau':anom_list,\
              'laplace_ideg':lidegvec,'laplace_Wdeg':lWdegvec,'laplace_q':lqvec,'laplace_p':lpvec}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b005_laplace_lawler_rescenters.csv',index=False)
#%%
df_tnos = pd.read_csv('b004_tnos_orbels_jd'+jdstr+'.csv')
aau_bary_tnos = np.array(df_tnos['aau_bary'].to_list())
librations = ['gplus_2026feb12','gminus_2026feb12','gnone_2026feb12','gunstable_2026feb12','gplusminusnone_2026feb12']
amean_vec = []
lidegvec2 = []
lWdegvec2 = []
lq2 = []
lp2 = []
plusminusnone_index = []
for ilib in range(len(librations)):
    lib = librations[ilib]
    dfind = pd.read_csv('b004_p3q2_'+lib+'_index.csv')
    indices = dfind['index'].to_list()
    aau_lib = aau_bary_tnos[indices]
    aau0 = np.mean(aau_lib)
    amean_vec.append(aau0)
    q0,p0,qNN,pNN,qJJ,pJJ = laplace_lawler(aau0)
    s0 = np.sqrt(1-q0**2-p0**2)
    i0 = np.arccos(s0)
    sini = np.sin(i0)
    W0 = np.arctan2(p0/sini,q0/sini)
    lidegvec2.append(np.degrees(i0))
    lWdegvec2.append(np.degrees(W0))
    lq2.append(q0)
    lp2.append(p0)
    print(ilib,len(librations))
dictionary = {'librations':librations,'aau':amean_vec,\
              'laplace_ideg':lidegvec2,'laplace_Wdeg':lWdegvec2,'laplace_q':lq2,'laplace_p':lp2}
dfout = pd.DataFrame.from_dict(dictionary)
dfout.to_csv('b005_laplace_lawler_p3q2librations_jd'+jdstr+'.csv',index=False)
#%%
laplace_q_list_end = []
laplace_p_list_end = []
for ia in range(na):
    aau0 = aau_vec[ia]
    q_laplace_lawler,p_laplace_lawler,qNN,pNN,qJJ,pJJ = laplace_lawler(aau0)
    laplace_q_list_end.append(q_laplace_lawler)
    laplace_p_list_end.append(p_laplace_lawler)
    print(ia,na,'end')
dictionary = {'aau':aau_vec,'lq':laplace_q_list_end,'lp':laplace_p_list_end}
dflapend = pd.DataFrame.from_dict(dictionary)
dflapend.to_csv('b005_laplace_lawler.csv')



    