import os
import argparse

from model_explorer.utils.setup import build_dataloader_generators, setup_torch_device, setup_workload
from model_explorer.utils.workload import Workload
from model_explorer.utils.setup import get_model_init_function, get_model_update_function



def compute_already_present_blocks(workload):
    dataloaders = build_dataloader_generators(
        workload['reevaluation']['datasets'])
    reevaluate_dataloader = dataloaders['reevaluate']
    model, accuracy_function = setup_workload(workload['model'])
    device = setup_torch_device(workload['problem'].get('gpu_id', -1))

    model_init_func = get_model_init_function(workload['problem']['problem_function'])
    model_update_func = get_model_update_function(workload['problem']['problem_function'])
    kwargs: dict = workload['exploration']['extra_args']
    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    explorable_model = model_init_func(model, device, **kwargs)

    thresholds = [0] * len(explorable_model.explorable_module_names)
    model_update_func(explorable_model, thresholds)
    _ = accuracy_function(explorable_model.base_model, reevaluate_dataloader,
                          progress=True)
    already_present_sparse = explorable_model.get_total_present_sparse_blocks() / len(reevaluate_dataloader)
    print(f"Model has {already_present_sparse} sparse blocks present")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "workload",
        help="The path to the workload yaml file.")
    opt = parser.parse_args()

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        results = compute_already_present_blocks(workload)
    else:
        raise Exception(f"No file {opt.workload} found.")


