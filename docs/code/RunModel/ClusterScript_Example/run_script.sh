#!/bin/bash                                                                           

# NOTE: The job configuration etc. would be in the batch script that launches
# your python script that uses UQpy. This script would then utilize those
# resources by using the appropriate commands here to launch parallel jobs.  For
# example, TACC uses slurm and ibrun, so you would launch your python script in
# the slurm batch script and then use ibrun here to tile parallel runs.

# This function is where you can define all the parts of a single
taskFunction(){
    coresPerProc=$1
    runNumber=$2
    host=$3   

    let offset=$coresPerProc*$runNumber # Sometimes, this might be necessary to pass as an argument to launch jobs. Not used here.

    cd run_$runNumber
    # Here, we launch a parallel job. The example uses multiple cores to add numbers,
    # which is somewhat pointless. This is just to illustrate the process for how tiled
    # parallel jobs are launched and where MPI-capable applications would be initiated
    mkdir -p ./OutputFiles
    mpirun -n $coresPerProc --host $host:$coresPerProc python3 ../add_numbers.py ./InputFiles/inputRealization_$runNumber.json ./OutputFiles/qoiFile_$runNumber.txt
    cd ..
}

# Get list of hosts
echo $SLURM_NODELIST > hostfile

# Split by comma
IFS="," read -ra HOSTS < hostfile

# This is the loop that launches taskFunction in parallel
coresPerProcess=$1
numberOfJobs=$2
# This number will vary depending on the number of cores per node. In this case, it is 32.
N=32

echo
echo "Starting parallel job launch"

declare -i index=0

for i in $(seq 0 $((numberOfJobs-1)))
do
    # Launch task function and put into the background
    echo "Launching job number ${i} on ${HOSTS[$index]}"
    taskFunction $coresPerProcess $i ${HOSTS[$index]}&

    # Increment host when all nodes allocated on current node
    if !((${i}%N)) && [ $i -ne 0 ]
    then
    	index=${index}+1
    fi
done

wait # This wait call is necessary so that loop above completes before script returns
echo "Analyses done!"
