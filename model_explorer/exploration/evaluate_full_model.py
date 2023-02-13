import pandas as pd

from model_explorer.utils.logger import logger
from model_explorer.utils.setup import build_dataloader_generators, setup_torch_device, setup_workload
from model_explorer.utils.workload import Workload
from model_explorer.utils.setup import get_model_init_function, get_model_update_function



def evaluate_full_model(workload: Workload, model_configurations: pd.DataFrame,
                        progress: bool, verbose: bool):

    dataloaders = build_dataloader_generators(
        workload['reevaluation']['datasets'])
    reevaluate_dataloader = dataloaders['reevaluate']
    model, accuracy_function = setup_workload(workload['model'])
    device = setup_torch_device()

    model_init_func = get_model_init_function(workload['problem']['problem_function'])
    model_update_func = get_model_update_function(workload['problem']['problem_function'])
    kwargs: dict = workload['exploration']['extra_args']
    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    explorable_model = model_init_func(model, device, verbose, **kwargs)

    # Copy individuals and add full_accuracy column
    evaluated_configs = model_configurations.copy(deep=True)
    evaluated_configs['full_accuracy'] = -1

    tot_eval = len(model_configurations)
    logger.info(f"Starting to evaluate {tot_eval} individuals")

    for i, row in model_configurations.iterrows():
        logger.debug(f"Evaluating {i+1} / {tot_eval} models with optimization accuracy: {row['accuracy']}")

        thresholds = row[10:63].tolist()
        model_update_func(explorable_model, thresholds)
        full_accuracy = accuracy_function(explorable_model.base_model,
                                          reevaluate_dataloader,
                                          progress=progress,
                                          title=f"Reevaluating {i+1}/{tot_eval}")

        logger.info(f"Done with ind {i+1} / {tot_eval}, accuracy is {full_accuracy:.4f}, \
                    was before {row['accuracy']:.4f}, fo={row['F_0']}")

        # add full accuary to dataframe
        evaluated_configs.loc[i, 'full_accuracy'] = full_accuracy.item()

    return evaluated_configs
