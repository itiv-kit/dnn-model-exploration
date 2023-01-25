from model_explorer.utils.pickeling import CPUUnpickler
from model_explorer.result_handling.result_entry import ResultEntry
from pymoo.core.result import Result
import pymoo.algorithms.moo.nsga2
import pandas as pd


class ResultsCollection():

    def __init__(self, pickle_file=None) -> None:
        self.accuracy_limit = 0.0
        self.explorable_module_names = []
        self.individuals = []
        if pickle_file:
            self._load(pickle_file)

    def _load(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            d: Result = CPUUnpickler(f).load()

        self.accuracy_limit = d.problem.min_accuracy
        self.explorable_module_names = d.problem.model.explorable_module_names

        for generation_idx, h in enumerate(d.history):
            assert isinstance(h, pymoo.algorithms.moo.nsga2.NSGA2)
            for individual_idx, ind in enumerate(h.pop):
                self.individuals.append(
                    ResultEntry(accuracy=-(ind.get("F")[0]),
                                further_objectives=ind.get("F")[1:],
                                parameter=ind.get("X").tolist(),
                                generation=generation_idx,
                                individual_idx=individual_idx,
                                pymoo_mating=h.mating))

    def merge(self, other):
        assert self.accuracy_limit == other.accuracy_limit
        assert self.explorable_module_names == other.quantizer_names
        self.individuals.extend(other.individuals)

    def drop_duplicate_parameters(self):
        # in place
        seen_lists = []
        new_individuals = []
        for ind in self.individuals:
            if ind.parameter not in seen_lists:
                seen_lists.append(ind.parameter)
                new_individuals.append(ind)
        self.individuals = new_individuals

    def get_weighted_params_sorted_individuals(self, index=0):
        return sorted(self.individuals, key=lambda ind: ind.further_objectives[index])

    def get_accuracy_sorted_individuals(self):
        return sorted(self.individuals,
                      reverse=True,
                      key=lambda ind: ind.accuracy)

    def get_better_than_individuals(self, acc_threshold):
        return [
            ind for ind in self.individuals if ind.accuracy >= acc_threshold
        ]

    def to_dataframe(self):
        return pd.DataFrame.from_records(
            [r.to_dict(self.explorable_module_names) for r in self.individuals])

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self) -> iter:
        yield next(self.individuals)
