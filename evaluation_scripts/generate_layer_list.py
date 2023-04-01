import os
import sys
import argparse
import torch

# FIXME?
sys.path.append(os.path.join(os.path.dirname(__file__), '../model_explorer/third_party/yolop_det_seg'))

import pandas as pd

from torchinfo import summary
from torchinfo.layer_info import LayerInfo
from typing import List

from model_explorer.utils.workload import Workload
from model_explorer.utils.setup import setup_workload, get_model_init_function, setup_torch_device

from pytorch_quantization.nn import QuantConv2d


def compute_memory_saving(workload: Workload):
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

    for i, layer in enumerate(timeloop_layers):
        # Gather information about layer
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


        bias = layer.module.bias == None
        if bias:
            bias = torch.zeros(1)
        else:
            bias = layer.module.bias

        # Store results
        results[i] = {'layer_idx': i,
                      'layer_type': layer.class_name,
                      'w': w,
                      'h': h,
                      's': s,
                      'r': r,
                      'q': q,
                      'p': p,
                      'k': layer.output_size[1],
                      'n': layer.input_size[0],
                      'c': layer.input_size[1],
                      'macs': layer.macs,
                      'num_params': layer.num_params,
                      'max_w': torch.max(layer.module.weight).item(),
                      'min_w': torch.min(layer.module.weight).item(),
                      'max_b': torch.max(bias).item(),
                      'min_b': torch.min(bias).item(),
                      'param_bytes': layer.param_bytes}

    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('layer_list.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("workload_file")
    opt = parser.parse_args()

    workload_file = opt.workload_file

    if not os.path.isfile(workload_file):
        raise Exception(f"No file {opt.workload} found.")

    workload = Workload(workload_file)

    compute_memory_saving(workload)


