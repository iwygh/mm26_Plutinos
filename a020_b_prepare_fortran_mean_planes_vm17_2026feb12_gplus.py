#%% prepare fortran_mean_planes template, make copies for an array job
njobs = 1000
nreps = 40
#%%
template_file = 'a020_a_fortran_mean_planes_vm17_2026feb12_gplus.f95'
with open(template_file,'r') as file:
    template_data = file.read()
    template_data = template_data.replace('nreps = 3','nreps = '+str(nreps))
    for ijob in range(njobs):
        outfile_str = 'b020_fortran_mean_planes_vm17_2026feb12_gplus_ijob'+str(ijob+1)+'_njobs'+str(njobs)+\
            '_nreps'+str(nreps)+'.txt'
        filedata = template_data
        filedata = filedata.replace('b020_fortran_mean_planes_vm17_2026feb12_gplus_ijobNNN.txt',outfile_str)
        outfile = 'c020_a_gplus_ijob'+str(ijob+1)+'.f95'
        with open(outfile,'w') as file:
            file.write(filedata)
#%%
template_file = 'a020_a_fortran_mean_planes_vm17_2026feb12_gminus.f95'
with open(template_file,'r') as file:
    template_data = file.read()
    template_data = template_data.replace('nreps = 3','nreps = '+str(nreps))
    for ijob in range(njobs):
        outfile_str = 'b020_fortran_mean_planes_vm17_2026feb12_gminus_ijob'+str(ijob+1)+'_njobs'+str(njobs)+\
            '_nreps'+str(nreps)+'.txt'
        filedata = template_data
        filedata = filedata.replace('b020_fortran_mean_planes_vm17_2026feb12_gminus_ijobNNN.txt',outfile_str)
        outfile = 'c020_a_gminus_ijob'+str(ijob+1)+'.f95'
        with open(outfile,'w') as file:
            file.write(filedata)
#%%
template_file = 'a020_a_fortran_mean_planes_vm17_2026feb12_gplusminusnone.f95'
with open(template_file,'r') as file:
    template_data = file.read()
    template_data = template_data.replace('nreps = 3','nreps = '+str(nreps))
    for ijob in range(njobs):
        outfile_str = 'b020_fortran_mean_planes_vm17_2026feb12_gplusminusnone_ijob'+str(ijob+1)+'_njobs'+str(njobs)+\
            '_nreps'+str(nreps)+'.txt'
        filedata = template_data
        filedata = filedata.replace('b020_fortran_mean_planes_vm17_2026feb12_gplusminusnone_ijobNNN.txt',outfile_str)
        outfile = 'c020_a_gplusminusnone_ijob'+str(ijob+1)+'.f95'
        with open(outfile,'w') as file:
            file.write(filedata)
#%%
template_file = 'a020_c_slurm_fortran_mean_planes_vm17_2026feb12_gplus_template.slurm'
with open(template_file,'r') as file:
    template_data = file.read()
filedata = template_data
filedata = filedata.replace('NJOBS',str(njobs))
outfile = 'c020_c_slurm.slurm'
with open(outfile,'w') as file:
    file.write(filedata)
