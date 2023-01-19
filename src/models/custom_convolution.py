from torch import nn
from utils.custom_timeit import timeit


class CustomConvModule(nn.Module):
    """Convolution module that allows to insert a transformation to the im2col representation
    of the input by inherenting from it and implementing the transform_transposed_unfolded_input function.
    """

    def __init__(self, old_module):
        """Initializes the internal Module state based of a provided Conv2d module.

        Args:
            old_module (nn.Conv2d): The module this custom convolution is based of.
        """
        super(CustomConvModule, self).__init__()

        assert isinstance(
            old_module, nn.Conv2d), "Trying to init a CustomConvModule with a non Conv2d module."

        self.old_module = old_module

        self.in_channels = old_module.in_channels
        self.out_channels = old_module.out_channels
        self.kernel_size = old_module.kernel_size
        self.stride = old_module.stride
        self.padding = old_module.padding
        self.dilation = old_module.dilation

        self.kernel = old_module.weight
        self.bias = old_module.bias

    @timeit
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

        # modified convolution equivalent code from: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
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
        """
        This method is applied to the unfolded and transposed input of the convolution.
        By default this transformation does nothing.
        To implement your own transformation overwrite this function in a subclass.

        Args:
            inp_unf (Tensor): The unfolded input tensor of the convolution.

        Returns:
            Tensor: The convolution result.
        """
        return inp_unf

    def forward(self, x):

        return self._custom_conv(x)
