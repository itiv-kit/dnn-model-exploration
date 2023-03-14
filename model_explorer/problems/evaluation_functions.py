from tqdm import tqdm
from model_explorer.utils.logger import logger

from pymoo.core.problem import ElementwiseEvaluationFunction, LoopedElementwiseEvaluation
from pymoo.algorithms.moo.nsga2 import NSGA2


class ElementwiseEvaluationFunctionWithIndex(ElementwiseEvaluationFunction):
    """This class adds some extra information during the elementwise evaluation to show a nice progress bar
    """
    def __init__(self, problem, args, kwargs) -> None:
        super().__init__(problem, args, kwargs)

    def __call__(self, i, x):
        out = dict()
        self.problem._evaluate(i, x, out, *self.args, **self.kwargs)
        return out


class LoopedElementwiseEvaluationWithIndex(LoopedElementwiseEvaluation):
    """This class will eventually evaluate the individuals, in the meanwhile it
    renders a progress bar and returns some information to the logger.
    """
    def __call__(self, f, X):
        algorithm: NSGA2 = f.kwargs.get('algorithm')
        progress_bar = tqdm(total=len(X), position=1, ascii=True, desc="Generation {}".format(algorithm.n_iter))
        results = []
        for i, x in enumerate(X):
            results.append(f(i, x))
            progress_bar.update(1)
        progress_bar.close()

        # do some info generation
        accuracy_string = ", ".join(format(-result['G'][0], ".3f") for result in results)
        logger.info("Finished Generation {} \n Accuracies: [{}]".format(algorithm.n_iter, accuracy_string))

        return results


