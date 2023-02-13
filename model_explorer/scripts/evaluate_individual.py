import os
import argparse
import sys
from datetime import datetime

from model_explorer.utils.logger import logger
from model_explorer.utils.setup import build_dataloader_generators, setup_torch_device, setup_workload
from model_explorer.utils.workload import Workload
from model_explorer.utils.setup import get_model_init_function, get_model_update_function
from model_explorer.result_handling.collect_results import collect_results
from model_explorer.result_handling.save_results import save_results_df_to_csv

RESULTS_DIR = "./results"



slurm_id_settings = [
    ["results/different_block_szies/expl_sparsity_problem_resnet50_imagenet_2023-02-02_15-13_slurmid_1.pkl",[16,16]],
    ["results/different_block_szies/expl_sparsity_problem_resnet50_imagenet_2023-02-02_15-17_slurmid_0.pkl",[8,8]],
    ["results/different_block_szies/expl_sparsity_problem_resnet50_imagenet_2023-02-02_15-23_slurmid_2.pkl",[1,16]]
]


def reevaluate_individuals(workload: Workload, results_path: str,
                           count: int, progress: bool,
                           verbose: bool):

    dataloaders = build_dataloader_generators(
        workload['reevaluation']['datasets'])
    reevaluate_dataloader = dataloaders['reevaluate']
    model, accuracy_function = setup_workload(workload['model'])
    device = setup_torch_device()

    model_init_func = get_model_init_function(workload['problem']['problem_function'])
    model_update_func = get_model_update_function(workload['problem']['problem_function'])
    kwargs: dict = workload['exploration']['extra_args']
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        kwargs['block_size'] = slurm_id_settings[int(os.environ['SLURM_ARRAY_TASK_ID'])][1]
        results_path = slurm_id_settings[int(os.environ['SLURM_ARRAY_TASK_ID'])][0]
    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    explorable_model = model_init_func(model, device, verbose, **kwargs)

    # Load the specified results file and pick n individuals
    results_collection = collect_results(results_path)

    results_collection.drop_duplicate_parameters()
    logger.debug("Loaded in total {} individuals".format(
        len(results_collection.individuals)))

    # select individuals with limit and cost function
    ind_df = results_collection.to_dataframe()
    ind_df['F_0'] = -ind_df['F_0'] * 1_000_000
    ind_df['norm_f0'] = ind_df['F_0'] / ind_df['F_0'].max()
    ind_df['norm_acc'] = ind_df['accuracy'] / ind_df['accuracy'].max()
    ind_df['weighted'] = ind_df['norm_f0'] * ind_df['norm_acc']
    # ind_filtered = ind_df[ind_df['weighted'] > 0.65]
    ind_filtered = ind_df.sort_values(by=['weighted'], ascending=False)
    ind_filtered = ind_filtered.head(count)
    print(ind_filtered[['weighted', 'F_0', 'accuracy']])

    ind_results = ind_filtered.copy(deep=True)
    ind_results['full_accuracy'] = -1

    # results = pd.DataFrame(columns=[
    #     'generation', 'individual', 'accuracy', 'acc_full', 'F_0',
    #     'mutation_eta', 'mutation_prob', 'crossover_eta', 'crossover_prob',
    #     'selection_press'
    # ])

    logger.info(
        "Selecting {} individual(s) for reevaluation with the full dataset.".
        format(len(ind_filtered)))

    for i, row in ind_filtered.iterrows():
        # print(row[10:63])
        # sys.exit(0)
        logger.debug(
            "Evaluating {} / {} models with optimization accuracy: {}".format(
                i + 1, len(ind_filtered), row['accuracy']))
        thresholds = row[10:63].tolist()
        model_update_func(explorable_model, thresholds)
        full_accuracy = accuracy_function(explorable_model.base_model,
                                          reevaluate_dataloader,
                                          progress=progress,
                                          title="Reevaluating {}/{}".format(
                                              i + 1,
                                              len(ind_filtered)))

        logger.info(
            "Done with ind {} / {}, accuracy is {:.4f}, was before {:.4f}, fo={}"
            .format(i + 1, len(ind_filtered), full_accuracy,
                    row['accuracy'], row['F_0']))

        # loc_dict['full_accuracy'] = full_accuracy.item()
        # loc_dict['bits'] = individual.bits
        # loc_dict['cost'] = cost
        ind_results.loc[i, 'full_accuracy'] = full_accuracy.item()

    return ind_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("workload", help="The path to the workload yaml file.")
    parser.add_argument(
        "results_path",
        help="Path to the results file or folder to be evaluated")
    parser.add_argument('-n',
                        "--top_elements",
                        help="Select n individuals with the lowest bits",
                        type=int)
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Show verbose information.")
    parser.add_argument("-p",
                        "--progress",
                        action="store_true",
                        help="Show the current inference progress.")
    opt = parser.parse_args()

    logger.info("Reevaluation of individuals started")

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        results = reevaluate_individuals(workload,
                                         opt.results_path, opt.top_elements,
                                         opt.progress, opt.verbose)
        save_results_df_to_csv(
            'reeval', results,
            workload['problem']['problem_function'], workload['model']['type'],
            workload['reevaluation']['datasets']['reevaluate']['type']
        )

    else:
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    logger.info("Reevaluation of individuals finished")
