from typing import List
from dataclasses import dataclass
import pymoo.core.mating
import pymoo.operators.mutation.pm
import pymoo.operators.crossover.sbx
import pymoo.operators.selection.tournament


@dataclass
class ResultEntry:
    accuracy: float
    parameters: List[int]
    generation: int
    individual_idx: int
    weighted_parameters: int
    pymoo_mating: pymoo.core.mating.Mating

    def parameters_sum(self):
        return sum(self.parameters)

    def to_dict_without_parameters(self):
        rdict = {
            "generation": self.generation,
            "individual": self.individual_idx,
            "accuracy": self.accuracy,
            "weighted_parameters": self.weighted_parameters
        }
        if isinstance(self.pymoo_mating.mutation, pymoo.operators.mutation.pm.PolynomialMutation):
            rdict['mutation_eta'] = self.pymoo_mating.mutation.eta.value
            rdict['mutation_prob'] = self.pymoo_mating.mutation.prob.value
        if isinstance(self.pymoo_mating.crossover, pymoo.operators.crossover.sbx.SBX):
            rdict['crossover_eta'] = self.pymoo_mating.crossover.eta.value
            rdict['crossover_prob'] = self.pymoo_mating.crossover.prob.value
        if isinstance(self.pymoo_mating.selection, pymoo.operators.selection.tournament.TournamentSelection):
            rdict['selection_press'] = self.pymoo_mating.selection.pressure

        return rdict

    def to_dict(self, layer_names=[]):
        rdict = self.to_dict_without_parameters()

        if layer_names == []:
            for idx, param in enumerate(self.parameters):
                rdict['param_{}'.format(idx)] = param
        else:
            for name, param in zip(layer_names, self.parameters):
                rdict[name] = param

        return rdict

