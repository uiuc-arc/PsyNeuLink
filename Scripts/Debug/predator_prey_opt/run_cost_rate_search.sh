#!/bin/bash
#SBATCH --job-name=pred_prey_search
#SBATCH --output=logs/%A.out
#SBATCH --time=03:00:00
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=2

module load anaconda intel-mpi/gcc
conda activate psyneulink2

WORKDIR=/scratch/gpfs/dmturner/${SLURM_JOB_ID}

srun python predator_prey_dmt.py
