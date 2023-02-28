from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import torch
from joblib.externals.loky.backend.context import get_context
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.transforms import transforms


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_class: str = "torchvision.datasets.MNIST",
        dataset_class_arguments: Dict = {},
        num_classes: int = 10,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        dataset_module, dataset_class = dataset_class.rsplit(".", 1)
        self.dataset_class = getattr(import_module(dataset_module), dataset_class)
        self.dataset_class_arguments = dataset_class_arguments

        # data transformations
        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        if num_workers > 0:
            self.multiprocessing_context = get_context("loky")
        else:
            self.multiprocessing_context = None

    @property
    def num_classes(self):
        return self.hparams.num_classes

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        self.dataset_class(
            self.hparams.data_dir, train=True, download=True, **self.dataset_class_arguments
        )
        self.dataset_class(
            self.hparams.data_dir, train=False, download=True, **self.dataset_class_arguments
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = self.dataset_class(
                self.hparams.data_dir,
                train=True,
                transform=self.transforms,
                **self.dataset_class_arguments,
            )
            testset = self.dataset_class(
                self.hparams.data_dir,
                train=False,
                transform=self.transforms,
                **self.dataset_class_arguments,
            )
            dataset = ConcatDataset(datasets=[trainset, testset])

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            multiprocessing_context=self.multiprocessing_context,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
        )

    def test_dataloader_by_class(self, num_samples_per_class):
        subsets = {
            target: Subset(
                self.data_test, [i for i, (x, y) in enumerate(self.data_test) if y == target]
            )
            for target in range(self.num_classes)
        }
        return {
            target: DataLoader(subset, batch_size=num_samples_per_class, shuffle=True)
            for target, subset in subsets.items()
        }

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = ImageDataModule()
