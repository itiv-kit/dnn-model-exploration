from datetime import datetime
import pickle
import torch
import io
import argparse
import pymoo
import os

from src.visualize.exploration_visualizer import ExplorationVisualizer


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
         

def render_results(pickle_file, output_dir):
    # WARNING: this is not backwards compatible

    print("Loading pickle file ... ")
    with open(pickle_file, 'rb') as f:
        res = CPUUnpickler(f).load()
    if not isinstance(res, pymoo.core.result.Result):
        raise TypeError("Failed to read pickle file .. not correct output format")
    
    print("... Done ... Plotting now ...")
    exp_vis = ExplorationVisualizer(
        output_dir,
        res)

    exp_vis.print_first_feasable_solution()
    exp_vis.plot_constraint_violation_over_n_gen()
    exp_vis.plot_2d_pareto()
    exp_vis.plot_2d_pareto(acc_bound=res.problem.min_accuracy)
    exp_vis.plot_layer_bit_resolution()
    exp_vis.plot_objective_space()

    print("... Done. Look at {}".format(os.path.abspath(output_dir)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("results_file")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose information.")
    parser.add_argument(
        "-o",
        "--output_dir",
        help="override predefined output dir")
    opt = parser.parse_args()

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = opt.output_dir if opt.output_dir else './results/viz_{}/'.format(date_str)
    
    render_results(opt.results_file, output_dir)

