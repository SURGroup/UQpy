#! /bin/sh

export PATH=$PATH:/Applications/MATLAB_R2016b.app/bin/
matlab -nodisplay -nodesktop -r "run matlab_model($1); exit" 

# python python_model.py $1




