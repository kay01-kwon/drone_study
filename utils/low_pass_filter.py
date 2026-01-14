import numpy as np

class LowPassFilter:
    def __init__(self, cut_off_freq, dim = 3):

        # Set cutoff frequency
        self.cut_off_freq = cut_off_freq

        # Initialize low pass filter
        self.output = np.zeros(dim)

        self.dim = dim

    def do_filter(self, input, dt):
        alpha = 1 - np.exp(-dt * 2.0 * np.pi * self.cut_off_freq)
        for i in range(self.dim):
            self.output[i] = (1-alpha)*self.output[i] + alpha * input[i]
        return self.output