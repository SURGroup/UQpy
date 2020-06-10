class RunPythonModel:

    def __init__(self, samples=None):

        self.samples = samples
        self.qoi = [0]*self.samples.shape[0]

        for i in range(self.samples.shape[0]):
            self.qoi[i] = 120 - self.samples[i][1] - 3*self.samples[i][0]