#!/bin/bash

# Don not modify
filename="UQpyInput_$1.txt"
touch "$filename"


# Chnage ONLY the name "solution_" according to the name of the model's output
cat "solution_$1.txt" >> "$filename"