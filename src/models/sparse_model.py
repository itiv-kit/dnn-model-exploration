import logging
import copy
import functools

from modules.sparse_convolution import SparseConv


class SparseModel():
    """The base model for our custom sparse models.
    """

    def __init__(self, model, accuracy_function) -> None:
        """Initilizes a sparse model with the provided arguments.

        Args:
            model (nn.Model): The model where layers are to be replaced by our custom sparse layers.
            accuracy_function (callable): A function to evaluate the accuracy of the new sparse model.
        """
        super().__init__()
        self.model = model
        self.accuracy_function = accuracy_function

    def get_sparse_model(self):
        """Placeholder method to be overwritten by subclasses.
        Provides a sparse model.
        """
        pass

    def sparse_conv2d_module_generator(self, old_module, new_module_dict):
        """Provides a sparse convolution module initilized with the given arguments.

        Args:
            old_module (nn.Conv2d): The old convolution module that is to be replaced.
            new_module_dict (dict): The arguments for initilizing the sparse convolution module.

        Returns:
            SparseConv: The newly created sparse convolution module.
        """
        return SparseConv(old_module,
                          new_module_dict["name"],
                          new_module_dict["threshold"],
                          new_module_dict["block_size"],
                          visualise=new_module_dict["visualize"])

    def compute_accuracy(self, dataloader, progress=True):
        """Computes the accuracy metric for an instance of this sparse model class.

        Args:
            dataloader (DataLoader): The data loader that provides the validation data.
            progress (bool, optional): Wether to show a progress bar while evaluating. Defaults to True.

        Returns:
            object: The accuracy metric returned by the accuracy function.
        """
        return self.accuracy_function(self, dataloader, progress)


class NodeSparseModel(SparseModel):

    def __init__(self, model, layers, block_size, accuracy_function) -> None:
        super().__init__(model, accuracy_function)

        self.layers = layers
        self.block_size = block_size

    # Must implements
    def update_thresholds(self, thresholds: list):
        pass


# FIXME: rename
class NodeSparseCopyModel(NodeSparseModel):
    """This model implements a sparse model, which takes any torch model and
    lets you add the nodes which have to be replaces by a certain threshold and block
    size.
    """

    def __init__(self, model, layers, block_size, accuracy_function) -> None:
        super().__init__(model, layers, block_size, accuracy_function)
        self.model = model
        self.sparse_nodes = []

    def update_thresholds(self, thresholds: list):
        """Update SparseModel with recompiliation"""
        self.remove_sparse_nodes()

        for idx, layername in enumerate(self.layers):
            self.add_explicit_sparse_node(layername, thresholds[idx],
                                          self.block_size, False)

    def add_explicit_sparse_node(self,
                                 node_name,
                                 threshold,
                                 block_size,
                                 visualize=False):

        self.sparse_nodes.append({
            'name': node_name,
            'threshold': threshold,
            'block_size': block_size,
            'visualize': visualize
        })

    def remove_sparse_nodes(self):
        self.sparse_nodes.clear()

    def get_sparse_model(self):

        sparse_model = copy.deepcopy(self.model)

        for sparse_node in self.sparse_nodes:

            module_path = sparse_node["name"].split('.')[:-1]
            module_name = sparse_node["name"].split('.')[-1]

            old_module = functools.reduce(getattr, [sparse_model] +
                                          sparse_node["name"].split('.'))

            replacement_module = self.sparse_conv2d_module_generator(
                old_module, sparse_node)

            module_parent = functools.reduce(getattr,
                                             [sparse_model] + module_path)

            # replace the old module with the new one
            setattr(module_parent, module_name, replacement_module)

            logging.getLogger('sparsity-analysis').info(
                "Replaced node {}, with threshold: {}, block_size: {}".format(
                    sparse_node['name'], sparse_node['threshold'],
                    sparse_node['block_size']))

        return sparse_model

    def __str__(self) -> str:
        return "Sparse Model, with {} replaced nodes".format(
            len(self.sparse_nodes))

    def __repr__(self) -> str:
        return "Sparse Model, with {} replaced Nodes:\n\t{}".\
            format(len(self.sparse_nodes), ";\n\t".join(["{}, thres:{}, bs:{}".format(
                x['name'], x['threshold'], x['block_size'])
                for x in self.sparse_nodes]))


class NodeSparseRefModel(NodeSparseCopyModel):
    """This model implements a sparse model, which takes any torch model and
    lets you add the nodes which have to be replaces by a certain threshold and block
    size. The values can then be edited without creating a new instance.
    """

    def __init__(self, model, layers, block_size, accuracy_function) -> None:
        super().__init__(model, layers, block_size, accuracy_function)

        self.model = model

        self.sparse_model = None
        self.sparse_conv_nodes = {}

    def update_thresholds(self, thresholds: list):
        """Update the thresholds of the sparse model without creating a new
        instance or create one no instance exists yet.

        Args:
            thresholds (list): The list of thresholds.
        """

        assert len(thresholds) == len(
            self.layers), "Length missmatch for thresholds and layers list."

        if self.sparse_model is None:
            self.sparse_model = self.get_sparse_model(thresholds)
            return

        # update the thresholds of the layers without having to create new ones
        for i, threshold in enumerate(thresholds):

            self.sparse_conv_nodes[self.layers[i]].threshold = threshold
            logging.getLogger('sparsity-analysis').info(
                "Update threshold of {} to: {:.5f}".format(
                    self.layers[i], threshold))

    def sparse_conv2d_module_generator(self, old_module, new_module_dict):

        # the module is only given as a reference to allow changing of its attributes
        self.sparse_conv_nodes[
            new_module_dict["name"]] = super().sparse_conv2d_module_generator(
                old_module, new_module_dict)

        return self.sparse_conv_nodes[new_module_dict["name"]]

    def get_sparse_model(self, thresholds=None):
        """This method provides a sparse model if one exists already or respective thresholds have been given.

        Args:
            thresholds (list, optional): The list of thresholds for the sparse model. Defaults to None.

        Returns:
            model: The sparse model. If thresholds were not provided the original model is returned.
        """

        if self.sparse_model is not None:
            return self.sparse_model

        if thresholds is None:
            return self.model

        # FIXME: is this redunant now?
        for i, threshold in enumerate(thresholds):
            self.add_explicit_sparse_node(self.layers[i], threshold,
                                          self.block_size)

        # FIXME: also this super() might break now ...
        self.sparse_model = super().get_sparse_model()
        return self.sparse_model
