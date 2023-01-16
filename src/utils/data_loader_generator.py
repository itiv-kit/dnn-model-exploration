import torch
from torch.utils.data import DataLoader, Subset, RandomSampler
import webdataset as wds


class DataLoaderGenerator:
    """Generator for different dataloaders depending on the sample limit."""

    def __init__(
        self, dataset, collate_fn: callable, batch_size: int = 32, 
        items: int = None, limit: int = None, fixed_random: bool = False
    ) -> None:
        """Inits a dataloader generator with the given parameters and configures
        the batch size for all generated data loaders.

        Args:
            base_dataset (Dataset):
                The base dataset to generate dta loader of.
            collate_fn (callable):
                Merges a list of samples to form a mini-batch of Tensor(s).
                Used when using batched loading from a map-style dataset.
            batch_size (int, optional):
                The batch size used by all generated data loaders. Defaults to 32.
            limit (int):
                Limit the amount of total samples
            fixed_random (bool):
                If set to true, a fixed selection will be created and then used
                for each generated dataset, otherwise a new sample will be
                created on each call
        """
        assert dataset is not None, "A dataset has to be provided."

        if isinstance(dataset, wds.WebDataset):
            assert items is not None, "dataset has no length attribute, hence, we need the items options with the total amount of elements in the dataset"
            assert limit is None, "webdataset types do not support limits, as shuffling is not working across shards"
            assert fixed_random is False, "webdatasets do nor support random selection"
            self.kind = 'wds'
        elif isinstance(dataset, torch.utils.data.dataset.Dataset):
            self.kind = 'torch_ds'
        else:
            raise ValueError("Only supporting Webdataset or Torch Datasets")

        self.collate_fn = collate_fn
        self.dataset = dataset
        self.batch_size = batch_size
        self.limit = limit
        self.dataloader = None
        self.fixed_random = fixed_random

        self._create_data_loader()
        
        if self.kind == 'wds':
            self.length = items
            self.n_batches = (self.length // batch_size) + 1
        else:
            self.length = len(self.dataset)
            if self.limit:
                self.length = self.limit
            self.n_batches = len(self.dataloader) // batch_size
        
    def __len__(self) -> int:
        return self.length

    def get_dataloader(self):
        return self.dataloader

    def _create_data_loader(self):
        """This method returns a data loader which returns only batches up to max_n_batches.
        """

        dataset = self.dataset
        sampler = None

        if self.limit is not None:
            dataset = Subset(dataset, indices=list(range(self.limit)))

        if self.fixed_random:
            sampler = RandomSampler(dataset)

        self.dataloader = DataLoader(
            dataset=dataset,
            num_workers=8,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler
        )
