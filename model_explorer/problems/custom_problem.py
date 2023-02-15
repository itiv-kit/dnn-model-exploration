from model_explorer.problems.evaluation_functions import ElementwiseEvaluationFunctionWithIndex, \
        LoopedElementwiseEvaluationWithIndex
from model_explorer.models.custom_model import CustomModel

from pymoo.core.problem import ElementwiseProblem


class CustomExplorationProblem(ElementwiseProblem):
    """This class inherits the ElementwiseProblem from pymoo and holds 4 global
    parameters, which should apply to all exploration problems. They are
    required for easy extension of the framework: a new kind of problem can
    simply inherit from this class and can be explored seamlessly.
    """

    def __init__(self, model: CustomModel, accuracy_function: callable,
                 progress: bool, min_accuracy: float, elementwise: bool = True,
                 **kwargs: dict):
        super().__init__(
            elementwise,
            elementwise_func=ElementwiseEvaluationFunctionWithIndex,
            elementwise_runner=LoopedElementwiseEvaluationWithIndex(),
            **kwargs)

        self.model = model
        self.progress = progress
        self.min_accuracy = min_accuracy
        self.accuracy_function = accuracy_function
