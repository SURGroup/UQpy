#!/bin/bash 
#PBS -A ARLAP00581800
#PBS -q standard
#PBS -l select=1:ncpus=40:mpiprocs=40
#PBS -l place=scatter:excl
#PBS -l walltime=20:00:00
#PBS -l lsdyna=40
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M michael.shields@jhu.edu

cd /p/home/shields/LS-DYNA_model/f01_vy10_vx5
source /usr/cta/unsupported/hms/ken/hms/env.sh
tail -n+3 ${PBS_NODEFILE} > machinefile_lower
export PYTHONPATH=/p/home/shields/LS-DYNA_model/f01_vy10_vx5:${PYTHONPATH}
module load ls-dyna/971_10.1.0
# export OMP_NUM_THREADS=16
python ARL_model.py