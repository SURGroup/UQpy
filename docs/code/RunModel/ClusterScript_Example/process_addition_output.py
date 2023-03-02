import numpy as np
from pathlib import Path


class OutputProcessor:

    def __init__(self, index):
        filePath = Path("./OutputFiles/qoiFile_" + str(index) + ".txt")
        self.numberOfColumns = 0
        self.numberOfLines = 0
        addedNumbers = []

        # Check if file exists
        if filePath.is_file():
            # Now, open and read data
            with open(filePath) as f:
                for line in f:
                    currentLine = line.split()

                    if len(currentLine) != 0:
                        addedNumbers.append(currentLine[:])

        if not addedNumbers:
            self.qoi = np.empty(shape=(0, 0))
        else:
            self.qoi = np.vstack(addedNumbers)
