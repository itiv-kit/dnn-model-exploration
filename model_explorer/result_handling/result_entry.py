from typing import List
from dataclasses import dataclass
import pymoo.core.mating
import pymoo.operators.mutation.pm
import pymoo.operators.crossover.sbx
import pymoo.operators.selection.tournament


@dataclass
class ResultEntry:
    accuracy: float
    bits: List[int]
    generation: int
    individual_idx: int
    weighted_bits: int
    pymoo_mating: pymoo.core.mating.Mating

    def bit_sum(self):
        return sum(self.bits)

    def to_dict_without_bits(self):
        rdict = {
            "generation": self.generation,
            "individual": self.individual_idx,
            "accuracy": self.accuracy,
            "weighted_bits": self.weighted_bits
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
        rdict = self.to_dict_without_bits()

        if layer_names == []:
            for idx, bit in enumerate(self.bits):
                rdict['bits_{}'.format(idx)] = bit
        else:
            for name, bit in zip(layer_names, self.bits):
                rdict[name] = bit

        return rdict

