from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import src.figures as fig
from src import utils
from src.comparators.comparator import Comparator
from src.models.image_classification import ImageClassification

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def compare(cfg: DictConfig) -> Tuple[dict, dict]:
    """Compares multiple given models on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.models.path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating comparator <{cfg.comparator._target_}>")
    comparator: Comparator = hydra.utils.instantiate(cfg.comparator)

    root_path = Path(cfg.models.path)

    path_template = str(root_path)
    path_structure: Dict[str, Set] = {}

    for level, folder_type in enumerate(cfg.models.folder_levels, 1):

        path_template += "/{" + folder_type + "}"

        path_structure[folder_type] = set()

        log.info("*/" * level)
        for folder in root_path.glob("*/" * level):
            if folder.is_dir():
                path_structure[folder_type].add(folder.parts[-1])

    log.info(path_structure)
    log.info(path_template)

    if "model" not in path_structure:
        log.error("No model folder level specified. A model folder level is always needed.")
        exit(1)

    if "group" not in path_structure:
        log.info("No comparison group folder level specified, assuming a single comparison.")
        model_instances = instantiate_models(
            path_template, path_structure["model"], path_structure["seed"]
        )
        comparator.compare(model_instances, datamodule, Path(cfg.paths.output_dir))
    else:
        for group in path_structure["group"]:
            model_instances = instantiate_models(
                path_template.replace("{group}", group),
                path_structure["model"],
                path_structure["seed"],
            )
            comparator.compare(model_instances, datamodule, Path(cfg.paths.output_dir) / group)

    return {}, {}


def instantiate_models(
    path_template: str, models: List[str], seeds: List[str]
) -> Dict[str, List[LightningModule]]:

    model_instances = {}

    for model_name in models:

        seed_model_instances = []

        for seed in seeds:
            model_path = (
                Path(path_template.replace("{model}", model_name).replace("{seed}", seed))
                / "checkpoints/last.ckpt"
            )

            model = ImageClassification.load_from_checkpoint(model_path)

            # TODO: Figure out why this is needed to load the weights (probably because of submodule net inside)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["state_dict"])

            seed_model_instances.append(model)

        model_instances[model_name] = seed_model_instances

    return model_instances


@hydra.main(version_base="1.3", config_path="../configs", config_name="compare.yaml")
def main(cfg: DictConfig) -> None:
    compare(cfg)


if __name__ == "__main__":
    main()
