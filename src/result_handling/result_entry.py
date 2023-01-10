from typing import List
from dataclasses import dataclass

@dataclass
class ResultEntry:
    accuracy: float
    bits: List[int]
    generation: int
    individual_idx: int
    weighted_bits: int
    
    def bit_sum(self):
        return sum(self.bits)

    def to_dict(self, layer_names=[]):
        rdict = {
            "generation" : self.generation,
            "individual" : self.individual_idx,
            "accuracy": self.accuracy,
            "weighted_bits": self.weighted_bits
        }
        if layer_names == []:
            for idx, bit in enumerate(self.bits):
                rdict['bits_{}'.format(idx)] = bit
        else:
            for name, bit in zip(layer_names, self.bits):
                rdict[name] = bit
        return rdict
    