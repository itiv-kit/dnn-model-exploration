import torch
from torch.utils.data import DataLoader, Subset


class DataLoaderGenerator:
    """Generator for different dataloaders depending on the sample limit."""

    def __init__(
        self, base_dataset, collate_fn: callable, batch_size: int = 32
    ) -> None:
        """Inits a dataloader generator with the given parameters and configures the batch size for all generated data loaders.

        Args:
            base_dataset (Dataset):
                The base dataset to generate dta loader of.
            collate_fn (callable):
                Merges a list of samples to form a mini-batch of Tensor(s).
                Used when using batched loading from a map-style dataset.
            batch_size (int, optional):
                The batch size used by all generated data loaders. Defaults to 32.
        """
        assert base_dataset is not None, "A dataset has to be provided."

        self.collate_fn = collate_fn
        self.base_dataset = base_dataset
        self._batch_size = batch_size

    def get_data_loader(self, limit: int = None, batch_size: int = None):
        """This method returns a data loader which returns only batches up to max_n_batches.

        Args:
            limit (int):
                The maximum of samples taken from the original dataset.
            batch_size (int, optional):
                The batch size used for the data loaders.
                If None the batch size set in the constructor is used.
                Defaults to None.
        """

        dataset = self.base_dataset

        if limit is not None:
            dataset = Subset(dataset, indices=list(range(limit)))

        if batch_size is None:
            batch_size = self._batch_size

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
        )
