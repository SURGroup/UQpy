# import fire
import numpy as np

# class CalculatePi:
#     def __init__(self, samples=None):
#         self.model_output = self.model(samples)
#         self.qoi = self.get_qoi(self.model_output)


def model(inputs=None):
    # if inputs is not None and np.shape(inputs)[1] == 2:
    # ratio = None
    # print(inputs)
    # print(type(inputs))
    if inputs is not None:
        # nvar = np.shape(inputs)[0]
        #
        # for i in range(nvar):

        x = inputs[0]
        y = inputs[1]
        inside = _check_if_inside(x, y)


        # ratio = inside/nvar
        # qoi = ratio * 4

    return inside

    # def get_qoi(self, output):
    #     qoi = output*4
    #     return qoi


def _check_if_inside(x, y):
    if x ** 2 + y ** 2 < 1:
        inside = 1
    else:
        inside = 0
    return inside

# def main():
#     fire.Fire(model)
#
#
# if __name__ == '__main__':
#     main()
