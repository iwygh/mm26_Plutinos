#%% prepare fortran_mean_planes template, make copies for an array job
njobs = 1000
nreps = 40
template_file = 'a007_a_fortran_mean_planes_vm17_2026feb12_plutinos.f95'
with open(template_file,'r') as file:
    template_data = file.read()
    template_data = template_data.replace('nreps = 3','nreps = '+str(nreps))
    for ijob in range(njobs):
        outfile_str = 'b007_fortran_mean_planes_vm17_2026feb12_plutinos_ijob'+str(ijob+1)+'_njobs'+str(njobs)+\
            '_nreps'+str(nreps)+'.txt'
        filedata = template_data
        filedata = filedata.replace('b007_fortran_mean_planes_vm17_2026feb12_plutinos_ijobNNN.txt',outfile_str)
        outfile = 'c007_a_ijob'+str(ijob+1)+'.f95'
        with open(outfile,'w') as file:
            file.write(filedata)
#%% prepare array job submission script
template_file = 'a007_c_slurm_fortran_mean_planes_vm17_2026feb12_plutinos_template.slurm'
with open(template_file,'r') as file:
    template_data = file.read()
filedata = template_data
filedata = filedata.replace('NJOBS',str(njobs))
outfile = 'c007_c_slurm.slurm'
with open(outfile,'w') as file:
    file.write(filedata)
