#!/bin/bash -l

#SBATCH
#SBATCH --job-name=Opensees
#SBATCH --time=00-00:05:00
#SBATCH --nodes=1
#SBATCH	--ntasks-per-node=5
#SBATCH --partition=express
#SBATCH -o out_file.txt
#SBATCH -e err_file.txt
#SBATCH	--mail-type=end
#SBATCH --mail-user=michael.shields@jhu.edu

module load python   
module load opensees/3.2.0
module load parallel

python run_opensees_UQpy.py       

