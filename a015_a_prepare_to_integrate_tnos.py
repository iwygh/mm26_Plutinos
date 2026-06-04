#%%
import pandas as pd
import numpy as np
maxjobs = 1000
jdstr = 'jd246e4'
slurm_template = 'a015_c_slurm_integrate_tnos_template.slurm'
python_template = 'a015_b_integrate_tnos_template.py'
#%%
mmrstrs = ['p5q3','p5q2','p7q4','p2q1']
nmmrs = len(mmrstrs)
for immr in range(nmmrs):
    mmrstr = mmrstrs[immr]
    infile = 'b004_horizons_orbels_'+mmrstr+'_jd246e4_HEHIBEBI.csv'
    df = pd.read_csv(infile)
    nobj = df.shape[0]
    with open(python_template,'r') as file:
        python_template_data = file.read()        
        for iobj in range(nobj):
            python_combo_data = python_template_data
            python_combo_data = python_combo_data.replace('iobj = 0','iobj = '+str(iobj))
            python_combo_data = python_combo_data.replace('"p3q2"','"'+mmrstr+'"')
            outfile = 'd015_tnos_'+mmrstr+'_i'+str(iobj)+'_n'+str(nobj)+'.py'
            with open(outfile,'w') as file:
                file.write(python_combo_data)
        with open(slurm_template,'r') as file:
            slurm_template_data = file.read()
        slurm_data = slurm_template_data
        slurm_data = slurm_data.replace('p3q2',mmrstr)
        slurm_data = slurm_data.replace('nNOBJ','n'+str(nobj))
        if nobj > maxjobs:
            starts = np.arange(start=0,stop=nobj,step=maxjobs)
            ends = starts+maxjobs-1
            if ends[-1] != nobj-1:
                ends[-1] = nobj-1
            nstarts = len(starts)
            for istart in range(nstarts):
                start = str(starts[istart])
                end = str(ends[istart])
                slurm_data = slurm_data.replace('1-NJOBS',start+'-'+end)
                outfile = 'slurm_tnos_'+mmrstr+'_'+str(istart+1)+'_n'+str(nobj)+'.slurm'
                with open(outfile,'w') as file:
                    file.write(slurm_data)
                slurm_data = slurm_data.replace(start+'-'+end,'1-NJOBS')
        else:
            slurm_data = slurm_data.replace('1-NJOBS','0-'+str(nobj-1))
            outfile = 'slurm_tnos_'+mmrstr+'_n'+str(nobj)+'.slurm'
            with open(outfile,'w') as file:
                file.write(slurm_data)
