from src.utils.pickeling import CPUUnpickler
from src.result_handling.result_entry import ResultEntry
from pymoo.core.result import Result
import pymoo.algorithms.moo.nsga2
import pandas as pd


class ResultsLoader():
    
    def __init__(self, pickle_file) -> None:
        with open(pickle_file, 'rb') as f:
            d:Result = CPUUnpickler(f).load()
        
        self.accuracy_limit = d.problem.min_accuracy
        self.individuals = []
        self.quantizer_names = d.problem.qmodel.quantizer_names
        
        for generation_idx, h in enumerate(d.history):
            assert isinstance(h, pymoo.algorithms.moo.nsga2.NSGA2)
            for individual_idx, ind in enumerate(h.pop):
                self.individuals.append(
                    ResultEntry(
                        accuracy=-(ind.get("F")[0]),
                        weighted_bits=ind.get("F")[1],
                        bits=ind.get("X").tolist(),
                        generation=generation_idx,
                        individual_idx=individual_idx
                        ))

    def drop_duplicate_bits(self):
        # in place
        seen_lists = []
        new_individuals = []
        for ind in self.individuals:
            if not ind.bits in seen_lists:
                seen_lists.append(ind.bits)
                new_individuals.append(ind)
        self.individuals = new_individuals

    def get_bit_sorted_individuals(self):
        return sorted(self.individuals, key=lambda ind:ind.weighted_bits)

    def get_accuracy_sorted_individuals(self):
        return sorted(self.individuals, reverse=True, key=lambda ind:ind.accuracy)
        
    def get_better_than_individuals(self, acc_threshold):
        return [ind for ind in self.individuals if ind.accuracy >= acc_threshold]

    def to_dataframe(self):
        return pd.DataFrame.from_records([r.to_dict(self.quantizer_names) for r in self.individuals])