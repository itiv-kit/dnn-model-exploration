import os
import pandas as pd

from model_explorer.utils.logger import logger
from model_explorer.utils.setup import setup_workload, setup_torch_device, build_dataloader_generators
from model_explorer.utils.workload import Workload
from model_explorer.utils.setup import get_model_init_function, get_model_update_function


def retrain_model(workload: Workload, model_configurations: pd.DataFrame,
                  result_dir: str, progress: bool) -> pd.DataFrame:
    """Function that start the retraining of models with the parameters set in the workload description

    Args:
        workload (Workload): Workload description
        model_configurations (pd.DataFrame): Dataframe with the selected model configurations for retraining
        result_dir (str): output directory for the retrained models
        progress (bool): Show retrain progress?

    Returns:
        pd.DataFrame: Dataframe with the accuracies of the retrained models
    """

    train_dataloaders = build_dataloader_generators(workload['retraining']['datasets'])
    reeval_dataloader = build_dataloader_generators(workload['reevaluation']['datasets'])
    reevaluate_dataloader = reeval_dataloader['reevaluate']
    model, accuracy_function = setup_workload(workload['model'])
    device = setup_torch_device()

    model_init_func = get_model_init_function(workload['problem']['problem_function'])
    model_update_func = get_model_update_function(workload['problem']['problem_function'])
    kwargs: dict = workload['exploration']['extra_args']
    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    explorable_model = model_init_func(model, device, **kwargs)

    # Copy individuals and add full_accuracy column
    configs_to_evaluate = model_configurations.copy(deep=True)
    configs_to_evaluate['accuracy_after_training'] = -1
    configs_to_evaluate['accuracies_over_epochs'] = [-1] * len(configs_to_evaluate)
    configs_to_evaluate['accuracies_over_epochs'] = configs_to_evaluate['accuracies_over_epochs'].astype('object')

    tot_eval = len(model_configurations)
    logger.info(f"Starting to retrain {tot_eval} individuals")

    index = 1
    for i, row in model_configurations.iterrows():
        logger.debug(f"Retraining model {i} ({index} / {tot_eval}) with accuracy: {row['accuracy']:.4f} and F_0 at {row['F_0']}")

        parameters = row['parameters']
        model_update_func(explorable_model, parameters)

        epoch_accs = explorable_model.retrain(
            train_dataloader_generator=train_dataloaders['train'],
            test_dataloader_generator=train_dataloaders['validation'],
            accuracy_function=accuracy_function,
            num_epochs=workload['retraining']['epochs'],
            progress=progress)
        explorable_model.save_parameters(os.path.join(result_dir, f'retrained_model_{i}.pkl'))

        acc_after_training = accuracy_function(explorable_model.base_model,
                                               reevaluate_dataloader,
                                               progress=progress,
                                               title=f"Reevaluating {i}/{tot_eval}")

        logger.info(f"Retrained model {i} ({index} / {tot_eval}), accuracy is {acc_after_training:.4f}")

        # add full accuracy to dataframe
        configs_to_evaluate.loc[i, 'accuracy_after_training'] = acc_after_training.item()
        configs_to_evaluate.at[i, 'accuracies_over_epochs'] = epoch_accs

        index += 1

    return configs_to_evaluate

