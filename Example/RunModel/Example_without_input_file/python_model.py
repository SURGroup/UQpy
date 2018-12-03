import fire
import numpy as np


class CalculatePi:
    def __init__(self, samples=None):
        self.qoi = self.model(samples)

    def model(self, inputs=None):
        inside = 0
        if inputs is not None:
            x = inputs[0][0]
            y = inputs[0][1]
            if x ** 2 + y ** 2 < 1:
                inside = 1

        return inside


def main():
    fire.Fire(CalculatePi)


if __name__ == '__main__':
    main()
