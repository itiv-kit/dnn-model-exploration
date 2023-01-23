import numpy as np
from tqdm import tqdm
from model_explorer.utils.logger import logger

from pymoo.core.problem import ElementwiseEvaluationFunction, LoopedElementwiseEvaluation
from pymoo.algorithms.moo.nsga2 import NSGA2


class ElementwiseEvaluationFunctionWithIndex(ElementwiseEvaluationFunction):
    def __init__(self, problem, args, kwargs) -> None:
        super().__init__(problem, args, kwargs)

    def __call__(self, i, x):
        out = dict()
        self.problem._evaluate(i, x, out, *self.args, **self.kwargs)
        return out


class LoopedElementwiseEvaluationWithIndex(LoopedElementwiseEvaluation):
    def __call__(self, f, X):
        algorithm: NSGA2 = f.kwargs.get('algorithm')
        pbar = tqdm(total=len(X), position=1, desc="Generation {}".format(algorithm.n_iter))
        results = []
        for i, x in enumerate(X):
            results.append(f(i, x))
            pbar.update(1)
        pbar.close()

        # do some info generation
        accs = []
        for result in results:
            accs.append(-result['F'][0])
        acc_str = ", ".join(format(x, ".3f") for x in accs)
        logger.info("Finished Generation {} \n Accs: [{}]".format(algorithm.n_iter, acc_str))

        return results


