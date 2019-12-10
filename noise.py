import numpy as np


class GaussianNoise:
    """ Class dedicated to return noise """

    def __init__(self, size, factor):
        self.size = size
        self.factor = factor
        self.seed = np.random.seed(0)

    def reset(self):
        pass

    def sample(self):
        return np.random.standard_normal(self.size) * self.factor


class NoiseReducer:
    """ Class dedicated to progressively reduce the noise """

    def __init__(self, factor_reduction, min_factor, rate_reduction):
        """Initialize a NoiseReducer object.

        Params
        ======
            factor_reduction (float): Initial factor to apply
            min_factor (float): minimum value for the factor
            rate_reduction (float): reduction apply at each update
        """
        self.factor_reduction = factor_reduction
        self.min_factor = min_factor
        self.rate_reduction = rate_reduction

    def reduce_noise(self, noise):
        """Initialize a NoiseReducer object.

        Params
        ======
            noise: The noise to reduce
        """
        return noise * self.factor_reduction

    def update_factor(self):
        """
        Decay the factor by the rate.
        """
        self.factor_reduction = max(self.factor_reduction * self.rate_reduction, self.min_factor)
