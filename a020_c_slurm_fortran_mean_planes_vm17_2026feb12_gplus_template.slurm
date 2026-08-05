#!/bin/bash

### Optional. Set the job name
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
### SBATCH --output=%x-%j.out
### REQUIRED. Specify the PI group for this job
### Optional. Request email when job begins and ends
### Optional. Specify email address to use for notification
### REQUIRED. Set the partition for your job.
### REQUIRED. Set the number of cores that will be used for this job.
### REQUIRED. Set the memory required for this job.
### REQUIRED. Specify the time required for this job, hhh:mm:ss

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ianmatheson@arizona.edu
#SBATCH --account=malhotra
#SBATCH --job-name=fortran_mean_planes_vm17_2026feb12_gplus_gminus_gplusminusnone
#SBATCH --output=%u-%x-%A-%a.out
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --array 1-NJOBS

gfortran c020_a_gplus_ijob${SLURM_ARRAY_TASK_ID}.f95 -o c020_d_gplus_ijob${SLURM_ARRAY_TASK_ID}
chmod +x c020_d_gplus_ijob${SLURM_ARRAY_TASK_ID}
./c020_d_gplus_ijob${SLURM_ARRAY_TASK_ID}

gfortran c020_a_gminus_ijob${SLURM_ARRAY_TASK_ID}.f95 -o c020_d_gminus_ijob${SLURM_ARRAY_TASK_ID}
chmod +x c020_d_gminus_ijob${SLURM_ARRAY_TASK_ID}
./c020_d_gminus_ijob${SLURM_ARRAY_TASK_ID}

gfortran c020_a_gplusminusnone_ijob${SLURM_ARRAY_TASK_ID}.f95 -o c020_d_gplusminusnone_ijob${SLURM_ARRAY_TASK_ID}
chmod +x c020_d_gplusminusnone_ijob${SLURM_ARRAY_TASK_ID}
./c020_d_gplusminusnone_ijob${SLURM_ARRAY_TASK_ID}
