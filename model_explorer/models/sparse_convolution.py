import os
import torch

from torch import nn 

from utils.sparsity_metrics import add_forward_data, add_layer_information
from utils.custom_timeit import timeit

from .custom_convolution import CustomConvModule

from visualizations.tensor2d import tensor2d_to_heatmap_comparison


class SparseConv(nn.Conv2d):
    """Sparse convolution module that splits the im2col representation of the forwarded input
    into blocks of the provided block sizes. The mean of each block is then compared to a
    pre-set threshold. If the mean is smaller than the threshold the block is set to zero.
    Otherwise the block will remain unchanged.
    """

    def __init__(self,
                 old_module: nn.Conv2d,
                 node_name: str,
                 threshold: float,
                 block_size: int):
        """Initilizes a sparse convolution module.

        Args:
            old_module (nn.Conv2d): The convolution module that this sparse convolution should be based off.
            node_name (str): The name of the replaced module inside its parent model.
            threshold (double): The threshold that determines if a block should be set to zero.
            block_size (list): A list or tuple of (width, height) for the block size.
            collect_details (bool, optional): Wether to collect metrics on the sparse forward method. Defaults to True.
            visualise (bool, optional): Wether to visualize the applied sparsity. Defaults to False.
        """
        super(SparseConv, self).__init__(old_module)

        self.threshold = threshold
        self.block_width = block_size[0]
        self.block_height = block_size[1]
        self.node_name = node_name

        add_layer_information(node_name, threshold, block_size[0])

    @timeit
    def _tensor_to_blocks(self, tensor):

        return tensor.unfold(0, self.block_height,
                             self.block_height).unfold(1, self.block_width,
                                                       self.block_width)

    @timeit
    def _blocks_to_tensor(self, blocks, padded_width, padded_height):

        return blocks.permute(0, 2, 1, 3).view(padded_height, padded_width)

    @timeit
    def _apply_sparsity(self, blocks):

        num_sparse_before = 0
        num_sparse_produced = 0

        # torch.nanmean(blocks, 2) -> each block replaced with mean
        # use this to speed up

        mean_blocks = torch.abs(torch.nanmean(blocks, (2, 3)))
        if self.collect_details:
            mean_of_block_means = torch.nanmean(mean_blocks)
            min_of_blocks = torch.min(blocks)
            max_of_blocks = torch.max(blocks)
        else:
            mean_of_block_means = 0
            min_of_blocks = 0
            max_of_blocks = 0

        # count how many sparse blocks are already present
        num_sparse_before = (mean_blocks == 0).sum()

        if torch.cuda.is_available():
            blocks[mean_blocks < self.threshold,
                   ...] = torch.cuda.FloatTensor(self.block_height,
                                                 self.block_width).fill_(0)
        else:
            blocks[mean_blocks < self.threshold, ...] = torch.zeros(
                (self.block_height, self.block_width))

        mean_blocks = torch.abs(torch.nanmean(blocks, (2, 3)))

        # count how many sparse blocks are present after sparsity operation
        num_sparse_produced = (mean_blocks == 0).sum() - num_sparse_before

        # add data from new forward pass
        add_forward_data(self.node_name, min_of_blocks.item(),
                         max_of_blocks.item(),
                         blocks.shape[0], blocks.shape[1],
                         num_sparse_before.item(), num_sparse_produced.item(),
                         mean_of_block_means.item())

        return blocks

    @timeit
    def transform_transposed_unfolded_input(self, inp_unf):
        """Splits the im2col representation of the forwarded input
    into blocks of the modules pre-set block sizes. The mean of each block is then compared to a
    pre-set threshold. If the mean is smaller than the threshold the block is set to zero.
    Otherwise the block will remain unchanged.

        Args:
            inp_unf (torch.Tensor): The unfolded and transposed input of the module foward method.

        Returns:
            torch.Tensor: A Tensor with the same dimensions as the original tensor,
            but with all blocks replaced where the mean was smaller than the pre-set threshold.
        """
        original_width = inp_unf.size(2)
        original_height = inp_unf.size(1)

        # add padding to make the input divisible by the block size
        padding_right = original_width % self.block_width

        if (padding_right):
            padding_right = self.block_width - padding_right

        padding_bottom = original_height % self.block_height

        if (padding_bottom):
            padding_bottom = self.block_height - padding_bottom

        padded_width = original_width + padding_right
        padded_height = original_height + padding_bottom

        # add padding with nan so we can ignore nan values when calculating the mean
        # padding allows vectorization
        inp_unf = torch.nn.functional.pad(
            inp_unf, (0, padding_right, 0, padding_bottom), "constant",
            torch.nan)

        # perform sparcity analysis for every batch
        for batch in range(inp_unf.size(0)):

            im2col_2dtensor = inp_unf[batch]

            blocks = self._tensor_to_blocks(im2col_2dtensor)

            sparse_blocks = self._apply_sparsity(blocks)

            inp_unf[batch] = self._blocks_to_tensor(sparse_blocks,
                                                    padded_width,
                                                    padded_height)

            if self.visualize:
                filename = os.path.join(
                    self.visualization_folder,
                    '{:05d}.png'.format(self.sample_counter))
                tensor2d_to_heatmap_comparison(im2col_2dtensor.to('cpu'),
                                               inp_unf[batch].to('cpu'),
                                               filename, "")
                self.sample_counter += 1

        # remove padding
        output_unpadded = inp_unf[:, :original_height, :original_width]

        return output_unpadded
