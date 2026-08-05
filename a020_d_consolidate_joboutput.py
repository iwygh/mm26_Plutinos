#%%
import pandas as pd
import numpy as np
njobs = 1000
nreps = 40
i_mid_deg_list = []
node_mid_deg_list = []
diff_deg_list = []
for ijob in range(njobs):
    infile = 'b020_fortran_mean_planes_vm17_2026feb12_gplus_ijob'+str(ijob+1)+\
        '_njobs'+str(njobs)+'_nreps'+str(nreps)+'.txt'
    line_number = 0
    # irep, nreps, i_mid_deg, node_mid_deg, diff_deg, elapsed_minutes
    with open(infile,'r') as file:
        for line in file:
            line_number = line_number + 1
            if line_number > 2:
                numbers = [float(number_str) for number_str in line.strip().split()]
                i_mid_deg_list.append(numbers[2])
                node_mid_deg_list.append(numbers[3])
                diff_deg_list.append(numbers[4])
nreps_total = len(diff_deg_list)
print(nreps_total)
i_mid_deg_list = np.array(i_mid_deg_list)
node_mid_deg_list = np.array(node_mid_deg_list)
irad = np.radians(i_mid_deg_list)
Wrad = np.radians(node_mid_deg_list)
q_mid_list = np.sin(irad)*np.cos(Wrad)
p_mid_list = np.sin(irad)*np.sin(Wrad)
dictionary = {'i_mid_deg':i_mid_deg_list,'node_mid_deg':node_mid_deg_list,\
              'diff_mid_deg':diff_deg_list,'q_mid':q_mid_list,'p_mid':p_mid_list}
df = pd.DataFrame.from_dict(dictionary)
outfile = 'b020_d_consolidated_mean_planes_vm17_2026feb12_plutinos_gplus_njobs'+\
    str(njobs)+'_nreps'+str(nreps)+'.csv'
df.to_csv(outfile,index=False)
#%%
i_mid_deg_list = []
node_mid_deg_list = []
diff_deg_list = []
for ijob in range(njobs):
    infile = 'b020_fortran_mean_planes_vm17_2026feb12_gminus_ijob'+str(ijob+1)+\
        '_njobs'+str(njobs)+'_nreps'+str(nreps)+'.txt'
    line_number = 0
    # irep, nreps, i_mid_deg, node_mid_deg, diff_deg, elapsed_minutes
    with open(infile,'r') as file:
        for line in file:
            line_number = line_number + 1
            if line_number > 2:
                numbers = [float(number_str) for number_str in line.strip().split()]
                i_mid_deg_list.append(numbers[2])
                node_mid_deg_list.append(numbers[3])
                diff_deg_list.append(numbers[4])
nreps_total = len(diff_deg_list)
print(nreps_total)
i_mid_deg_list = np.array(i_mid_deg_list)
node_mid_deg_list = np.array(node_mid_deg_list)
irad = np.radians(i_mid_deg_list)
Wrad = np.radians(node_mid_deg_list)
q_mid_list = np.sin(irad)*np.cos(Wrad)
p_mid_list = np.sin(irad)*np.sin(Wrad)
dictionary = {'i_mid_deg':i_mid_deg_list,'node_mid_deg':node_mid_deg_list,\
              'diff_mid_deg':diff_deg_list,'q_mid':q_mid_list,'p_mid':p_mid_list}
df = pd.DataFrame.from_dict(dictionary)
outfile = 'b020_d_consolidated_mean_planes_vm17_2026feb12_plutinos_gminus_njobs'+\
    str(njobs)+'_nreps'+str(nreps)+'.csv'
df.to_csv(outfile,index=False)
#%%
i_mid_deg_list = []
node_mid_deg_list = []
diff_deg_list = []
for ijob in range(njobs):
    infile = 'b020_fortran_mean_planes_vm17_2026feb12_gplusminusnone_ijob'+str(ijob+1)+\
        '_njobs'+str(njobs)+'_nreps'+str(nreps)+'.txt'
    line_number = 0
    # irep, nreps, i_mid_deg, node_mid_deg, diff_deg, elapsed_minutes
    with open(infile,'r') as file:
        for line in file:
            line_number = line_number + 1
            if line_number > 2:
                numbers = [float(number_str) for number_str in line.strip().split()]
                i_mid_deg_list.append(numbers[2])
                node_mid_deg_list.append(numbers[3])
                diff_deg_list.append(numbers[4])
nreps_total = len(diff_deg_list)
print(nreps_total)
i_mid_deg_list = np.array(i_mid_deg_list)
node_mid_deg_list = np.array(node_mid_deg_list)
irad = np.radians(i_mid_deg_list)
Wrad = np.radians(node_mid_deg_list)
q_mid_list = np.sin(irad)*np.cos(Wrad)
p_mid_list = np.sin(irad)*np.sin(Wrad)
dictionary = {'i_mid_deg':i_mid_deg_list,'node_mid_deg':node_mid_deg_list,\
              'diff_mid_deg':diff_deg_list,'q_mid':q_mid_list,'p_mid':p_mid_list}
df = pd.DataFrame.from_dict(dictionary)
outfile = 'b020_d_consolidated_mean_planes_vm17_2026feb12_plutinos_gplusminusnone_njobs'+\
    str(njobs)+'_nreps'+str(nreps)+'.csv'
df.to_csv(outfile,index=False)