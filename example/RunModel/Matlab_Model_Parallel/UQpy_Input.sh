#!/bin/bash

filename="modelInput_$1.txt"


touch "$filename"
if [ -e UQpy_run_$1.txt ]
then
mv "UQpy_run_$1.txt" "$filename"
fi



