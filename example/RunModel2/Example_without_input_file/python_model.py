import fire
import numpy as np


class CalculatePi:
    def __init__(self, samples=None):
        self.qoi = self.model(samples)
        # self.model_output = self.model(samples)
        # self.qoi = self.get_qoi(self.model_output)

    def model(self, inputs=None):
        # if inputs is not None and np.shape(inputs)[1] == 2:
        # ratio = None
        inside = 0
        if inputs is not None:
            nvar = np.shape(inputs)[0]

            for i in range(nvar):
                x = inputs[0]
                y = inputs[1]

                if x ** 2 + y ** 2 < 1:
                    inside = 1

        return inside

    # def model(self, input=None):
    #     # if input is not None and np.shape(input)[1] == 2:
    #     ratio = None
    #     if input is not None:
    #         nsim = np.shape(input)[0]
    #         inside = 0
    #
    #         for i in range(nsim):
    #             x = input[i][0]
    #             y = input[i][1]
    #
    #             if x**2 + y**2 < 1:
    #                 inside += 1
    #
    #         ratio = inside/nsim
    #
    #     return ratio
    #
    # def get_qoi(self, output):
    #     qoi = output*4
    #     return qoi


def main():
    fire.Fire(CalculatePi)


if __name__ == '__main__':
    main()
