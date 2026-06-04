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
#SBATCH --job-name=a015_integrate_tnos_p3q2
#SBATCH --output=%u-%x-%A-%a.out
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=00:45:00
#SBATCH --array 1-NJOBS

module load python/3.11/3.11.4
source /home/u26/ianmatheson/py311/bin/activate
python3 d015_tnos_p3q2_i${SLURM_ARRAY_TASK_ID}_nNOBJ.py
