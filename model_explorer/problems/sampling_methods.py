import numpy as np
from pymoo.operators.sampling.rnd import FloatRandomSampling


class FloatRandomSamplingWithDefinedIndividual(FloatRandomSampling):
    """A modified version of FloatRandomSampling, which is able to add predefined individuals to the population
    """

    def __init__(self, var_type=np.float64, predefined: list = []) -> None:
        super().__init__()
        np.random.seed(None)
        self.var_type = var_type
        self.predefined = predefined

    def _do(self, problem, n_samples, **kwargs):
        pop = super()._do(problem, n_samples, **kwargs)

        # Predefined individuals are added to the population!
        for predef in self.predefined:
            predef_ind = np.full((1, problem.n_var), predef)
            pop = np.concatenate((predef_ind, pop))
        return pop

