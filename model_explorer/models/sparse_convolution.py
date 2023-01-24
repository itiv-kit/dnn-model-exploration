import os
import torch

from torch import nn


class SparseConv2d(nn.Conv2d):
    """Sparse convolution module that splits the im2col representation of the forwarded input
    into blocks of the provided block sizes. The mean of each block is then compared to a
    pre-set threshold. If the mean is smaller than the threshold the block is set to zero.
    Otherwise the block will remain unchanged.
    """

    def __init__(self,
                 old_module: nn.Conv2d,
                 block_size: list):
        super(SparseConv2d, self).__init__(in_channels=old_module.in_channels,
                                           out_channels=old_module.out_channels,
                                           kernel_size=old_module.kernel_size,
                                           stride=old_module.stride,
                                           padding=old_module.padding,
                                           dilation=old_module.dilation)

        # Keep weights and biases
        self.kernel = old_module.weight
        self.bias = old_module.bias

        self._threshold = None
        self.block_width = block_size[0]
        self.block_height = block_size[1]

        self.collect_details = True

        # Metrics and evaluation stats
        self.sparse_present: torch.Tensor = 0.0
        self.sparse_created: torch.Tensor = 0.0
        self.min_of_blocks: torch.Tensor = -torch.inf
        self.max_of_blocks: torch.Tensor = torch.inf
        self.number_of_blocks_w: int = 0
        self.number_of_blocks_h: int = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold: float):
        self._threshold = new_threshold

    def reset_stats(self):
        self.sparse_present: float = 0.0
        self.sparse_created: float = 0.0
        self.min_of_blocks: float = -torch.inf
        self.max_of_blocks: float = torch.inf
        self.number_of_blocks_w: int = 0
        self.number_of_blocks_h: int = 0

    def extra_repr(self):
        s = super().extra_repr()
        s += ", block_size={block_width}x{block_height}, threshold={_threshold}"
        return s.format(**self.__dict__)

    def forward(self, x):
        return self._custom_conv(x)

    def _tensor_to_blocks(self, tensor):
        return tensor.unfold(0, self.block_height,
                             self.block_height).unfold(1, self.block_width,
                                                       self.block_width)

    def _blocks_to_tensor(self, blocks, padded_width, padded_height):
        return blocks.permute(0, 2, 1, 3).view(padded_height, padded_width)

    def _apply_sparsity(self, blocks):
        num_sparse_before = 0
        num_sparse_produced = 0

        # torch.nanmean(blocks, 2) -> each block replaced with mean
        # use this to speed up

        mean_blocks = torch.abs(torch.nanmean(blocks, (2, 3)))
        if self.collect_details:
            # mean_of_block_means = torch.nanmean(mean_blocks)
            min_of_blocks = torch.min(blocks)
            max_of_blocks = torch.max(blocks)
        else:
            # mean_of_block_means = 0
            min_of_blocks = 0
            max_of_blocks = 0

        # count how many sparse blocks are already present
        num_sparse_before = (mean_blocks == 0).sum()

        if torch.cuda.is_available():
            blocks[mean_blocks < self._threshold,
                   ...] = torch.cuda.FloatTensor(self.block_height,
                                                 self.block_width).fill_(0)
        else:
            blocks[mean_blocks < self._threshold, ...] = torch.zeros(
                (self.block_height, self.block_width))

        mean_blocks = torch.abs(torch.nanmean(blocks, (2, 3)))

        # count how many sparse blocks are present after sparsity operation
        num_sparse_produced = (mean_blocks == 0).sum() - num_sparse_before

        # gather statistics for later evaluation
        self.sparse_created += num_sparse_produced.item()
        self.sparse_present += num_sparse_before.item()
        self.max_of_blocks = max(self.max_of_blocks, max_of_blocks)
        self.min_of_blocks = min(self.min_of_blocks, min_of_blocks)

        return blocks

    def _custom_conv(self, inp):

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

        out_unf = inp_transformed.matmul(self.kernel.view(self.kernel.size(0), -1).t()).transpose(1, 2)

        out = out_unf.view(batch_size, self.out_channels, h_out, w_out)

        if self.bias is not None:
            bias_size = self.bias.size(0)
            out = out + self.bias.view(1, bias_size, 1, 1).expand_as(out)

        return out

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
        # FIXME: is this the most efficient way?
        for batch in range(inp_unf.size(0)):

            im2col_2dtensor = inp_unf[batch]

            blocks = self._tensor_to_blocks(im2col_2dtensor)

            sparse_blocks = self._apply_sparsity(blocks)

            inp_unf[batch] = self._blocks_to_tensor(sparse_blocks,
                                                    padded_width,
                                                    padded_height)

        # remove padding
        output_unpadded = inp_unf[:, :original_height, :original_width]

        return output_unpadded
