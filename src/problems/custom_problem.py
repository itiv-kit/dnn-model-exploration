from src.problems.evaluation_functions import ElementwiseEvaluationFunctionWithIndex, LoopedElementwiseEvaluationWithIndex

from pymoo.core.problem import ElementwiseProblem


class CustomExplorationProblem(ElementwiseProblem):

    def __init__(self,
                 accuracy_function: callable,
                 progress: bool = True,
                 min_accuracy: float = 0.3,
                 elementwise: bool = True,
                 **kwargs
                 ):
        super().__init__(elementwise,
                         elementwise_func=ElementwiseEvaluationFunctionWithIndex,
                         elementwise_runner=LoopedElementwiseEvaluationWithIndex(),
                         **kwargs)
        self.progress = progress
        self.min_accuracy = min_accuracy
        self.accuracy_function = accuracy_function
