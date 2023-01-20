import sys
import os
import json
import numpy as np

def addNumbers():
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]

    # Open JSON file
    with open(inputPath, "r") as jsonFile:
        data = json.load(jsonFile)

    # Read generated numbers 
    number1 = data["number1"]
    number2 = data["number2"]

    randomAddition = number1 + number2
    
    # Write addition to file
    with open(outputPath, 'w') as outputFile:
        outputFile.write('{}\n'.format(randomAddition))                

if __name__ == '__main__':
    addNumbers()
