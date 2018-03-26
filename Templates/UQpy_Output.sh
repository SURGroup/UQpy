#!/bin/bash

# Don not modify
<<<<<<< HEAD
filename="UQpyInput_$1.txt"
=======
filename="UQpy_eval_$1.txt"
>>>>>>> master
touch "$filename"


# Chnage ONLY the name "solution_" according to the name of the model's output
cat "solution_$1.txt" >> "$filename"