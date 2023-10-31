import os
import functools
import argparse
import torch

from torch import nn

from model_explorer.utils.workload import Workload
from model_explorer.utils.setup import setup_workload


class DummyConv2d(nn.Conv2d):
    def __init__(self,
                 old_module: nn.Conv2d,
                 block_size: list):
        super(DummyConv2d, self).__init__(in_channels=old_module.in_channels,
                                          out_channels=old_module.out_channels,
                                          kernel_size=old_module.kernel_size,
                                          stride=old_module.stride,
                                          padding=old_module.padding,
                                          dilation=old_module.dilation)

        # Keep weights and biases
        self.kernel = old_module.weight
        self.bias = old_module.bias

        self.block_width = block_size[0]
        self.block_height = block_size[1]

        self.block_count = 0

    def forward(self, inp):
        batch_size = inp.shape[0]
        # 0 is batch size and 1 is channel size
        h_in, w_in = inp.shape[2], inp.shape[3]

        # calculate output height and width, also see: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        h_out = (h_in + 2 * self.padding[0] - self.dilation[0]
                 * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.dilation[1]
                 * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        h_out, w_out = int(h_out), int(w_out)

        # modified convolution equivalent code from:
        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
        # Convolution <=> Unfold + Matrix Multiplication + Fold (or view to output shape)

        inp_unf = nn.functional.unfold(inp, self.kernel_size, self.dilation, self.padding, self.stride)

        inp_transp = inp_unf.transpose(1, 2)

        # apply sparsity filter or any other custom transformation
        inp_transformed = self.transform_transposed_unfolded_input(inp_transp)
        inp_transformed = inp_transformed.to('cuda')

        out_unf = inp_transformed.matmul(self.kernel.view(self.kernel.size(0), -1).t()).transpose(1, 2)

        out = out_unf.view(batch_size, self.out_channels, h_out, w_out)

        if self.bias is not None:
            bias_size = self.bias.size(0)
            out = out + self.bias.view(1, bias_size, 1, 1).expand_as(out)

        return out

    def transform_transposed_unfolded_input(self, inp_unf):
        original_width = inp_unf.size(2)
        original_height = inp_unf.size(1)

        # add padding to make the input divisible by the block size
        padding_right = original_width % self.block_width
        if (padding_right):
            padding_right = self.block_width - padding_right

        padding_bottom = original_height % self.block_height
        if (padding_bottom):
            padding_bottom = self.block_height - padding_bottom

        # padded_width = original_width + padding_right
        # padded_height = original_height + padding_bottom

        # add padding with nan so we can ignore nan values when calculating the mean
        # padding allows vectorization
        inp_unf = torch.nn.functional.pad(
            inp_unf, (0, padding_right, 0, padding_bottom), "constant",
            torch.nan)

        # perform sparcity analysis for every batch
        # FIXME: is this the most efficient way?
        for batch in range(inp_unf.size(0)):
            im2col_2dtensor = inp_unf[batch]
            blocks_h = im2col_2dtensor.shape[1] / self.block_height
            blocks_w = im2col_2dtensor.shape[0] / self.block_width
            # print(im2col_2dtensor.shape)
            self.block_count = blocks_h * blocks_w

        # remove padding
        output_unpadded = inp_unf[:, :original_height, :original_width]

        return output_unpadded





def compute_total_blocks(workload):
    model, _ = setup_workload(workload['model'])
    model = model.to('cuda')
    altered_modules = []
    block_size = workload['exploration']['extra_args']['block_size']

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            sparse_conv = DummyConv2d(module, block_size)
            altered_modules.append(sparse_conv)

            # Replace actual conv2d with sparse_conv2d
            # FIXME: is this save for all networks?
            module_name = name.split('.')[-1]
            module_path = name.split('.')[:-1]
            module_parent = functools.reduce(getattr, [model] + module_path)
            setattr(module_parent, module_name, sparse_conv)

    testinput = torch.randn( (1, 3, 640, 480) )
    testinput = testinput.to('cuda')
    _ = model(testinput)

    total_blocks = 0
    for i, module in enumerate(altered_modules):
        print(f'{i:<10} : {module.block_count:<10}, {module}')
        total_blocks += module.block_count

    print(f"Total blocks: {total_blocks}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "workload",
        help="The path to the workload yaml file.")
    opt = parser.parse_args()

    workload_file = opt.workload
    if os.path.isfile(workload_file):
        workload = Workload(workload_file)
        results = compute_total_blocks(workload)
    else:
        raise Exception(f"No file {opt.workload} found.")


