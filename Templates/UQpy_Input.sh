#!/bin/bash

# Chnage only the part of filename  "modelInput_"
filename="modelInput_$1.txt"


# Don not modify
touch "$filename"
if [ -e TEMP_val_$1.txt ]
then
<<<<<<< HEAD
mv "TEMP_val_$1.txt" "$filename"
=======
mv "UQpy_run_$1.txt" "$filename"
>>>>>>> master
fi



