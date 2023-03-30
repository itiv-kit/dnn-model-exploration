import socket
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import pymoo.core.result

from model_explorer.utils.logger import logger
from model_explorer.utils.setup import build_dataloader_generators, setup_torch_device, \
        setup_workload, get_prepare_exploration_function
from model_explorer.utils.workload import Workload



def explore_model(workload: Workload,
                  skip_baseline: bool,
                  progress: bool) -> pymoo.core.result.Result:
    """Function to explore the influence of model parameter to the accuracy. It
    instanciates an NSGA algorithm to automatically explore different model
    parameter sets.

    Args:
        workload (Workload): Workload description
        skip_baseline (bool): Skip the initial base line accuracy computation?
        progress (bool): Show evaluation progress?

    Returns:
        pymoo.core.result.Result: pymoo result object with the found
        configurations, can be evaluated with the result_collection tools
    """
    dataloaders = build_dataloader_generators(workload['exploration']['datasets'])
    model, accuracy_function = setup_workload(workload['model'])
    device = setup_torch_device()

    if not skip_baseline:
        # collect model basline information
        baseline_dataloader = dataloaders['baseline']
        logger.info("Collecting baseline...")
        baseline = accuracy_function(model, baseline_dataloader, title="Baseline Generation")
        if isinstance(baseline, list):
            acc_str = ", ".join([f"{x:.3f}" for x in baseline])
            logger.info(f"Done. Baseline accuracy: {acc_str}")
        elif isinstance(baseline, float):
            logger.info(f"Done. Baseline accuracy: {baseline:.3f}")

    prepare_function, repair_method, sampling_method = \
        get_prepare_exploration_function(workload['problem']['problem_function'])
    kwargs: dict = workload['exploration']['extra_args']

    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    if 'energy_evaluation' in workload['exploration']:
        kwargs['dram_analysis_file'] = workload['exploration']['energy_evaluation']['dram_analysis_file']
    min_accuracy = workload['exploration']['minimum_accuracy']
    problem = prepare_function(model, device, dataloaders['exploration'],
                               accuracy_function, min_accuracy, progress, **kwargs)

    # Setup algorithm
    crossover = SBX(prob_var=workload['exploration']['nsga']['crossover_prob'],
                    eta=workload['exploration']['nsga']['crossover_eta'],
                    repair=repair_method,
                    vtype=float)
    mutation = PolynomialMutation(prob=workload['exploration']['nsga']['mutation_prob'],
                                  eta=workload['exploration']['nsga']['mutation_eta'],
                                  repair=repair_method)

    algorithm = NSGA2(
        pop_size=workload['exploration']['nsga']['pop_size'],
        n_offsprings=workload['exploration']['nsga']['offsprings'],
        sampling=sampling_method,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", workload['exploration']['nsga']['generations'])
    # termination = get_termination("moo")

    logger.info("Prepared Run, run infomation:")
    logger.info(f"\tComputer name: {socket.gethostname()}")
    logger.info(f"\tNSGA crossover eta: {workload['exploration']['nsga']['crossover_eta']} " +
                f"prob: {workload['exploration']['nsga']['crossover_prob']}")
    logger.info(f"\tNSGA mutation eta: {workload['exploration']['nsga']['mutation_eta']} " +
                f"prob: {workload['exploration']['nsga']['mutation_prob']}")
    logger.info(f"\tNSGA gens: {workload['exploration']['nsga']['generations']}")
    logger.info(f"\tNSGA pop size: {workload['exploration']['nsga']['pop_size']} " +
                f"offsprings: {workload['exploration']['nsga']['offsprings']}")

    logger.info("Starting problem minimization.")

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True
    )

    logger.info("Finished problem minimization.")

    if res.F is None:
        logger.warning("No solutions found for the given constraints.")
        return

    # since we inverted our objective functions we have to invert the result back
    res.F = np.abs(res.F)

    return res



