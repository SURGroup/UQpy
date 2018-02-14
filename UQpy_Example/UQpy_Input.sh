#!/bin/bash

filename="modelInput_$1.txt"


touch "$filename"
if [ -e TEMP_val_$1.txt ]
then
mv "TEMP_val_$1.txt" "$filename"
fi



