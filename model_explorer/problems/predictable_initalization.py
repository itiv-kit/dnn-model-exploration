import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util.normalization import denormalize


def random_by_bounds(n_var, xl, xu, n_samples=1):
    np.random.seed(0)
    val = np.random.random((n_samples, n_var))
    return denormalize(val, xl, xu)


def random(problem, n_samples=1):
    return random_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples=n_samples)


class FloatRandomSamplingPredictable(Sampling):
    """Same as Pymoo's Random sampling method but with a fixed np.random.seed of 0
    """
    def _do(self, problem, n_samples, **kwargs):
        return random(problem, n_samples=n_samples)


class IntegerRandomSamplingPredictable(FloatRandomSamplingPredictable):
    """Same as Pymoo's Random sampling method but with a fixed np.random.seed of 0
    """
    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        return np.around(X).astype(int)

