from typing import List
from dataclasses import dataclass
import pymoo.core.mating
import pymoo.operators.mutation.pm
import pymoo.operators.crossover.sbx
import pymoo.operators.selection.tournament


@dataclass
class ResultEntry:
    accuracy: float
    parameter: List[int]
    generation: int
    individual_idx: int
    further_objectives: List[int]
    pymoo_mating: pymoo.core.mating.Mating

    def parameters_sum(self):
        return sum(self.parameter)

    def to_dict_without_parameters(self):
        rdict = {
            "generation": self.generation,
            "individual": self.individual_idx,
            "accuracy": self.accuracy,
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

        return rdict

    def to_dict(self, layer_names=[]):
        rdict = self.to_dict_without_parameters()

        if layer_names == []:
            for idx, param in enumerate(self.parameter):
                rdict['param_{}'.format(idx)] = param
        else:
            for name, param in zip(layer_names, self.parameter):
                rdict[name] = param

        return rdict

