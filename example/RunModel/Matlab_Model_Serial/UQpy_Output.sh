#!/bin/bash

filename="UQpy_eval_$1.txt"
touch "$filename"

cat "solution_$1.txt" >> "$filename"