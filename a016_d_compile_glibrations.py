#%%
import numpy as np
import pandas as pd
jdstr = 'jd246e4'
tyrsmax = '1e7'
tstepyrs = '5e2'
dtdays = '5'
tyrsstr = jdstr+'_'+tyrsmax+'yr_'+tstepyrs+'yr_'+dtdays+'d'
mass = 'm8ss12'
mmrstrs = ['p2q1','p5q2','p5q3','p7q4']
nobj_list = [105,56,68,103]
nmmrs = len(mmrstrs)
for immr in range(nmmrs):
    mmrstr = mmrstrs[immr]
    print(mmrstr)
    g0_list = []
    g90_list = []
    g180_list = []
    g270_list = []
    gnone_list = []
    gmulti_list = []
    gall_list = []
    nobj = nobj_list[immr]
    for iobj in range(nobj):
        df = pd.read_csv('b015_glibration_classification_tnos_'+mmrstr+'_i'+str(iobj)+\
            '_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv')
        g0_list.append(df['g0'][0])
        g90_list.append(df['g90'][0])
        g180_list.append(df['g180'][0])
        g270_list.append(df['g270'][0])
        gnone_list.append(df['gnone'][0])
        gmulti_list.append(df['gmulti'][0])
        gall_list.append(df['gall'][0])
    print('nobj',nobj)
    print('g0',np.sum(g0_list))
    print('g90',np.sum(g90_list))
    print('g180',np.sum(g180_list))
    print('g270',np.sum(g270_list))
    print('gnone',np.sum(gnone_list))
    print('gmulti',np.sum(gmulti_list))
    print('gall',np.sum(gall_list))
    print('')
# masses = ['m4ss12','m8ss12']
# #%%
# df_combo_lengths = pd.read_csv('b002_pq_combo_lengths.csv')
# n_combos = df_combo_lengths.shape[0]
# n_min = 10
# for icombo in range(n_combos):
#     nobj = df_combo_lengths['pq_combo_lengths'][icombo]
#     mmrp = df_combo_lengths['res_p_bignumber'][icombo]
#     mmrq = df_combo_lengths['res_q_smallnumber'][icombo]
#     mmrstr = 'p'+str(mmrp)+'q'+str(mmrq)
#     if nobj >= n_min:
#         for mass in masses:
#             g0_list = []
#             g90_list = []
#             g180_list = []
#             g270_list = []
#             gnone_list = []
#             gmulti_list = []
#             gall_list = []
#             for iobj in range(nobj):
#                 infile = 'b003_glibration_classification_'+mmrstr+'_i'+str(iobj)+'_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv'
#                 df_infile = pd.read_csv(infile)
#                 g0_list.append(df_infile['g0'][0])
#                 g90_list.append(df_infile['g90'][0])
#                 g180_list.append(df_infile['g180'][0])
#                 g270_list.append(df_infile['g270'][0])
#                 gnone_list.append(df_infile['gnone'][0])
#                 gmulti_list.append(df_infile['gmulti'][0])
#                 gall_list.append(df_infile['gall'][0])
#             dictionary = {'g0':g0_list,'g90':g90_list,'g180':g180_list,'g270':g270_list,\
#                           'gnone':gnone_list,'gmulti':gmulti_list,'gall':gall_list}
#             df_out = pd.DataFrame.from_dict(dictionary)
#             outfile = 'b004_glibration_classification_'+mmrstr+'_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv'
#             df_out.to_csv(outfile,index=False)
#             infile = 'b003_wdeg_'+mmrstr+'_i'+str(iobj)+'_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv'
#             wdeg = np.loadtxt(infile,delimiter=',')
#             nt = len(wdeg)
#             wdeg_mat = np.zeros((nobj,nt))
#             for iobj in range(nobj):
#                 infile = 'b003_wdeg_'+mmrstr+'_i'+str(iobj)+'_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv'
#                 wdeg = np.loadtxt(infile,delimiter=',')
#                 wdeg_mat[iobj,:] = wdeg
#             outfile = 'b004_wdeg_'+mmrstr+'_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv'
#             np.savetxt(outfile,wdeg_mat,delimiter=',')
#             try:
#                 infile = 'b003_sigmapqdeg_'+mmrstr+'_i'+str(iobj)+'_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv'
#                 sigmapqdeg = np.loadtxt(infile,delimiter=',')
#                 nt = len(sigmapqdeg)
#                 sigmapqdeg_mat = np.zeros((nobj,nt))
#                 for iobj in range(nobj):
#                     infile = 'b003_sigmapqdeg_'+mmrstr+'_i'+str(iobj)+'_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv'
#                     sigmapqdeg = np.loadtxt(infile,delimiter=',')
#                     sigmapqdeg_mat[iobj,:] = sigmapqdeg
#                 outfile = 'b004_sigmapqdeg_'+mmrstr+'_n'+str(nobj)+'_'+tyrsstr+'_'+mass+'_BI.csv'
#                 np.savetxt(outfile,sigmapqdeg_mat,delimiter=',')
#             except:
#                 print(mmrstr,'no sigmapqdeg for this')
# # b003_sigmapqdeg_p3q2_i423_n453_jd246e4_1e7yr_5e2yr_5d_m8ss12_BI