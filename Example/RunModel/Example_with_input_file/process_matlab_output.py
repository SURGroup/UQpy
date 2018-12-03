import numpy as np
import fire


def read_output(index):
    x = np.loadtxt("./OutputFiles/oupt_%d.out" % (index + 1))
    # print(x)
    return x


def main():
    fire.Fire(read_output)


if __name__ == '__main__':
    main()
