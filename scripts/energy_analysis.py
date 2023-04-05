import pandas as pd
import os
import sys
import argparse
import subprocess
import yaml
import logging

from pytorch_quantization.nn import QuantConv2d
from torchinfo.layer_info import LayerInfo
from torchinfo import summary

from typing import List
from tqdm import tqdm

from model_explorer.third_party.timeloop.scripts.parse_timeloop_output import parse_timeloop_stats
from model_explorer.utils.setup import setup_workload, get_model_init_function, setup_torch_device
from model_explorer.utils.workload import Workload
from model_explorer.utils.logger import logger, set_console_logger_level


# FIXME?
sys.path.append(os.path.join(os.path.dirname(__file__), '../model_explorer/third_party/yolop_det_seg'))

# Fixed bitwidth in timeloop
TIMELOOP_BITS = 16


def compute_memory_saving(workload: Workload, progress: bool = True):
    """Collection information about the DRAM energy of an individual layer.
    Therefore this function executes all quantized layers using Timeloop to get
    estimates with respect to the underlying architecture. 
    FIXME: This has to be extended for sparse layers or more general problems as
    well...

    Args:
        workload (Workload): workload description file
        progress (bool, optional): show progress? Defaults to True.

    Raises:
        FileNotFoundError: When Timeloop is not build, make sure to do that
        beforehand
    """
    timeloop_binary = os.path.abspath(os.path.join(os.path.dirname(
        __file__), '../model_explorer/third_party/timeloop/bin/timeloop-mapper'))
    timeloop_lib_path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), '../model_explorer/third_party/timeloop/lib'))

    if not os.path.exists(timeloop_binary):
        raise FileNotFoundError("Please build timeloop first (see model_explorer/third_party/timeloop)")

    model, _ = setup_workload(workload['model'])
    device = setup_torch_device()
    model_init_func = get_model_init_function(workload['problem']['problem_function'])

    # Build model with custom modules
    kwargs: dict = workload['exploration']['extra_args']
    if 'calibration' in workload.yaml_data:
        kwargs['calibration_file'] = workload['calibration']['file']
    explorable_model = model_init_func(model, device, **kwargs)

    input_shape = workload['exploration']['energy_evaluation']['input_shape']
    assert len(input_shape) == 4, "Input shape has to be N, C, W, H"

    layer_list = summary(explorable_model.base_model, input_size=input_shape, verbose=0, depth=100)

    timeloop_layers: List[LayerInfo] = []
    for layer_info in layer_list.summary_list:
        if isinstance(layer_info.module, QuantConv2d):
            timeloop_layers.append(layer_info)

    print(f"working on {len(timeloop_layers)} layers with timeloop")

    layer: LayerInfo
    results = {}
    total_inference_energy = 0.0

    for i, layer in enumerate(tqdm(timeloop_layers, ascii=True, disable=not progress)):
        # Gather information about layer dimension
        w = layer.input_size[2]
        h = layer.input_size[3]
        s = layer.kernel_size[0]
        r = layer.kernel_size[1]
        w_pad = layer.module.padding[0]
        h_pad = layer.module.padding[1]
        w_stride = layer.module.stride[0]
        h_stride = layer.module.stride[1]

        q = int((w - s + 2 * w_pad) / w_stride) + 1
        p = int((h - r + 2 * h_pad) / h_stride) + 1

        # Setup Paths
        config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../timeloop_eval/workload.yaml'))
        timeloop_base_wd = os.path.abspath(os.path.dirname(config_file))

        timeloop_wd = os.path.join(timeloop_base_wd, f'layers/{i}')
        if not os.path.exists(timeloop_wd):
            os.makedirs(timeloop_wd)

        altered_config_file = os.path.join(timeloop_wd, f"layer.yaml")
        projected_results_file = os.path.join(timeloop_wd, "timeloop-mapper.map+stats.xml")
        timeloop_log_file = os.path.join(timeloop_wd, "timeloop.log")

        # Alter config
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        config['problem']['R'] = layer.kernel_size[1]
        config['problem']['S'] = layer.kernel_size[0]
        config['problem']['P'] = p
        config['problem']['Q'] = q
        config['problem']['C'] = layer.input_size[1]
        config['problem']['K'] = layer.output_size[1]
        config['problem']['N'] = layer.input_size[0]
        config['problem']['Wstride'] = w_stride
        config['problem']['Hstride'] = h_stride
        config['problem']['Wdilation'] = 1
        config['problem']['Hdilation'] = 1
        # set bitwidth to a fixed bit value
        if 'storage' in config['arch']:
            for hierarchy_level in config['arch']['storage']:
                if hierarchy_level['name'] == 'DRAM':
                    hierarchy_level['word-bits'] = TIMELOOP_BITS

        with open(altered_config_file, "w") as f:
            f.write(yaml.dump(config))

        # run Timeloop
        timeloop_env = os.environ.copy()
        timeloop_env["LD_LIBRARY_PATH"] = timeloop_lib_path

        with open(timeloop_log_file, "w") as outfile:
            subprocess.call([timeloop_binary, altered_config_file],
                            cwd=timeloop_wd,
                            stdout=outfile,
                            stderr=outfile,
                            env=timeloop_env)

        if not os.path.exists(projected_results_file):
            logger.warn(
                f"Skipping Layer {i}, most likely no mapping was found, check log file at : {timeloop_log_file}")

        else:
            timeloop_result = parse_timeloop_stats(projected_results_file)

            # Store results
            results[i] = {'layer_idx': i,
                          'layer_type': layer.class_name,
                          'w_reads': timeloop_result['energy_breakdown_pJ']['DRAM']['reads_per_instance'][0],
                          'i_reads': timeloop_result['energy_breakdown_pJ']['DRAM']['reads_per_instance'][1],
                          'o_reads': timeloop_result['energy_breakdown_pJ']['DRAM']['reads_per_instance'][2],
                          'w_updates': timeloop_result['energy_breakdown_pJ']['DRAM']['updates_per_instance'][0],
                          'i_updates': timeloop_result['energy_breakdown_pJ']['DRAM']['updates_per_instance'][1],
                          'o_updates': timeloop_result['energy_breakdown_pJ']['DRAM']['updates_per_instance'][2],
                          'w_fills': timeloop_result['energy_breakdown_pJ']['DRAM']['fills_per_instance'][0],
                          'i_fills': timeloop_result['energy_breakdown_pJ']['DRAM']['fills_per_instance'][1],
                          'o_fills': timeloop_result['energy_breakdown_pJ']['DRAM']['fills_per_instance'][2],
                          'w_energy': timeloop_result['energy_breakdown_pJ']['DRAM']['energy_per_access_per_instance'][0],
                          'i_energy': timeloop_result['energy_breakdown_pJ']['DRAM']['energy_per_access_per_instance'][1],
                          'o_energy': timeloop_result['energy_breakdown_pJ']['DRAM']['energy_per_access_per_instance'][2],
                          'total_dram_energy': timeloop_result['energy_breakdown_pJ']['DRAM']['energy'],
                          'bitwidth': TIMELOOP_BITS}
            total_inference_energy += timeloop_result['energy_pJ']

    # Save results as CSV to the target file
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(workload['exploration']['energy_evaluation']['dram_analysis_file'])

    # Finally print total energy ...
    print(f"Total Energy per Inference: {total_inference_energy/1_000_000_000:.4f}mJ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("workload_file")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Show verbose information.")
    parser.add_argument("-p",
                        "--progress",
                        action="store_true",
                        help="Show the current inference progress.")
    opt = parser.parse_args()

    if opt.verbose:
        set_console_logger_level(level=logging.DEBUG)

    logger.info("Computation of energy for memory accesses started")

    workload_file = opt.workload_file

    if not os.path.isfile(workload_file):
        logger.warning("Declared workload file could not be found.")
        raise Exception(f"No file {opt.workload} found.")

    workload = Workload(workload_file)
    compute_memory_saving(workload, opt.progress)

    logger.info("Energy computation finised")
