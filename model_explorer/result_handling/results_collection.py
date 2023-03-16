from __future__ import annotations

from model_explorer.utils.pickeling import CPUUnpickler
from model_explorer.result_handling.result_entry import ResultEntry
from pymoo.core.result import Result

import pymoo.algorithms.moo.nsga2
import pandas as pd


class ResultsCollection():
    """Holds result entrys in a list and adds some supportive function to, e.g., drop duplicates
    """

    def __init__(self, pickle_file: str = None) -> None:
        self.accuracy_limit = None
        self.explorable_module_names = []
        self.individuals = []
        if pickle_file:
            self._load(pickle_file)

    def _load(self, pickle_file: str):
        with open(pickle_file, 'rb') as f:
            d: Result = CPUUnpickler(f).load()

        if d is None:
            # No feasible solutions found
            raise ValueError(f"Result collection at {pickle_file} has no solutions")

        self.accuracy_limit = d.problem.min_accuracy
        self.explorable_module_names = d.problem.model.explorable_module_names

        for generation_idx, h in enumerate(d.history):
            assert isinstance(h, pymoo.algorithms.moo.nsga2.NSGA2)
            for individual_idx, ind in enumerate(h.pop):
                further_args = {}
                if hasattr(d.problem.model, '_block_size'):
                    further_args['block_size'] = d.problem.model._block_size

                # Compute Constraint - Acc Limit for each available Accuracy
                accuracies = None
                if isinstance(self.accuracy_limit, list):
                    accuracies = [acc - limit for acc, limit in zip(ind.get("G"), self.accuracy_limit)]
                elif isinstance(self.accuracy_limit, float):
                    accuracies = [ind.get("G") - self.accuracy_limit]

                self.individuals.append(
                    ResultEntry(accuracies=accuracies,
                                further_objectives=ind.get("F"),
                                parameter=ind.get("X").tolist(),
                                generation=generation_idx,
                                individual_idx=individual_idx,
                                pymoo_mating=h.mating,
                                further_args=further_args))

    def merge(self, other: ResultsCollection):
        assert self.accuracy_limit == other.accuracy_limit
        assert self.explorable_module_names == other.explorable_module_names
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

    def get_weighted_params_sorted_individuals(self, index: int = 0) -> list:
        """Get all individuals sorted by a given objective index

        Args:
            index (int, optional): index of the objective to be sorted. Defaults to 0.

        Returns:
            list: List of individuals, not a ResultCollection
        """
        return sorted(self.individuals, key=lambda ind: ind.further_objectives[index])

    def get_accuracy_sorted_individuals(self) -> list:
        return sorted(self.individuals,
                      reverse=True,
                      key=lambda ind: ind.accuracy)

    def get_better_than_individuals(self, acc_threshold: float) -> list:
        return [ind for ind in self.individuals if ind.accuracy >= acc_threshold]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [r.to_dict() for r in self.individuals])

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self) -> iter:
        yield next(self.individuals)
