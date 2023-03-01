from abc import abstractmethod

from pytorch_lightning import LightningModule
from surgeon_pytorch import Inspect


class InspectableModule(LightningModule):
    @abstractmethod
    def initialize(self, num_train_last_layers: int = -1):
        raise NotImplementedError("Must override initialize")

    @abstractmethod
    def change_num_classes(self, num_classes: int):
        raise NotImplementedError("Must override change_num_classes")

    @abstractmethod
    def inspect(self) -> Inspect:
        raise NotImplementedError("Must override inspect")
