import torch
from torch.utils.data import DataLoader, Subset, RandomSampler
import webdataset as wds


class DataLoaderGenerator:
    """Generator for different dataloaders depending on the sample limit."""

    def __init__(self,
                 dataset,
                 collate_fn: callable,
                 batch_size: int = 32,
                 items: int = None,
                 limit: int = None,
                 randomize: bool = False) -> None:
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
            randomize (bool):
                If set to true, the dataset will be shuffled
        """
        assert dataset is not None, "A dataset has to be provided."

        if isinstance(dataset, wds.WebDataset):
            assert items is not None, "dataset has no length attribute, hence, we need the items \
                    options with the total amount of elements in the dataset"
            assert limit is None, "webdataset types do not support limits, as shuffling is not working across shards"
            assert randomize is False, "webdatasets do not support random selection"
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
        self.randomize = randomize

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

    def get_dataloader(self) -> DataLoader:
        return self.dataloader

    def _create_data_loader(self):
        """This method creates the internal dataloader with a given sample limit
        """

        dataset = self.dataset
        sampler = None

        if self.randomize:
            sampler = RandomSampler(dataset, num_samples=self.limit)
        else:
            if self.limit is not None:
                # FIXME: add random range instead of range
                dataset = Subset(dataset, indices=list(range(self.limit)))

        self.dataloader = DataLoader(dataset=dataset,
                                     num_workers=4,
                                     batch_size=self.batch_size,
                                     collate_fn=self.collate_fn,
                                     pin_memory=torch.cuda.is_available(),
                                     sampler=sampler)
