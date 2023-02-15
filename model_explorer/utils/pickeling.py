import pickle
import torch
import io


# https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
class CPUUnpickler(pickle.Unpickler):
    """Helper class to unpickle torch generated data on CPU only devices
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
