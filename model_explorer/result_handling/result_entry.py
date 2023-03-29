from typing import List
from dataclasses import dataclass

import pymoo.core.mating
import pymoo.operators.mutation.pm
import pymoo.operators.crossover.sbx
import pymoo.operators.selection.tournament


@dataclass
class ResultEntry:
    """A dataclass to store results found during the exploration of an explorable model
    """

    accuracies: list
    parameter: list
    generation: int
    individual_idx: int
    further_objectives: List[int]
    pymoo_mating: pymoo.core.mating.Mating
    further_args: dict

    def parameters_sum(self) -> float:
        return sum(self.parameter)

    def to_dict_without_parameters(self) -> dict:
        rdict = {
            "generation": self.generation,
            "individual": self.individual_idx,
            "accuracies": self.accuracies,
        }
        if isinstance(self.pymoo_mating.mutation, pymoo.operators.mutation.pm.PolynomialMutation):
            rdict['mutation_eta'] = self.pymoo_mating.mutation.eta.value
            rdict['mutation_prob'] = self.pymoo_mating.mutation.prob.value
        if isinstance(self.pymoo_mating.crossover, pymoo.operators.crossover.sbx.SBX):
            rdict['crossover_eta'] = self.pymoo_mating.crossover.eta.value
            rdict['crossover_prob'] = self.pymoo_mating.crossover.prob.value
        if isinstance(self.pymoo_mating.selection, pymoo.operators.selection.tournament.TournamentSelection):
            rdict['selection_press'] = self.pymoo_mating.selection.pressure

        for i, v in enumerate(self.further_objectives):
            rdict[f'F_{i}'] = v
        rdict.update(self.further_args)

        return rdict

    def to_dict(self) -> dict:
        rdict = self.to_dict_without_parameters()

        rdict['parameters'] = self.parameter

        return rdict
