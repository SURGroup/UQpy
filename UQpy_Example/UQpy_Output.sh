#!/bin/bash

filename="UQpyInput_$1.txt"
touch "$filename"

cat "solution_$1.txt" >> "$filename"