#!/bin/bash -l
#-------------------------------------------
# run_SFE_example.sh
#
# Script to submit SFE job
#
#-------------------------------------------

# Job name to be reported by qstat
#-------------------------------------------
#SBATCH --job-name=SFE_Beam

# Prescribe the output and error files
#-------------------------------------------
#SBATCH -o out_file.txt
#SBATCH -e err_file.txt

# Wall time requested
#-------------------------------------------
#SBATCH --time=00-2:00:00

# Request number of nodes, processes per node, partition and memory 
#-------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#------------------------

#------------------------
#SBATCH --partition=parallel
#SBATCH --mem-per-cpu=4800MB
#------------------------

# Send an email upon completion
#-------------------------------------------
#SBATCH --mail-type=end
#SBATCH --mail-user=michael.shields@jhu.edu


# Useful information for the log file:
#-------------------------------------------
echo Master process running on `hostname`
echo Directory is `pwd`


# Put in a timestamp
#-------------------------------------------
echo Starting executation at `date`


# Load the required modules (python3, ABAQUS) and run the job
#------------------------------------------
module load abaqus
module load python
module load parallel
python abaqus_example.py

# Print the date again -- when finished
#-------------------------------------------
echo Finished at `date`
