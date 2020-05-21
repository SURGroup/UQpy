#!/bin/bash -l

#SBATCH
#SBATCH --job-name=UQpy_LSDyna_Test_Parallel
#SBATCH --time=00-2:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --partition=parallel
#SBATCH --mail-type=end
#SBATCH --mail-user=michael.shields@jhu.edu

module load ls-dyna/10.1.0
module load python
module load parallel

#python run_LSDyna_python.py
python dyna_model.py



#echo ">>>Begin LS-Dyna test Shields..."
#ls-dyna i=Shields_5.k memory=300000000
#echo ">>>Finish LS-Dyna test!"
