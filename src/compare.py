import glob
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import hydra
import pyrootutils
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import LightningDataModule

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
from src.models.inspectable import InspectableModule

log = utils.get_pylogger(__name__)


def get_parameter_tag(config: Union[DictConfig, ListConfig], parameters: List[str]):
    parameter_values = []
    for parameter in parameters:
        parameter_value = OmegaConf.select(config, parameter)
        parameter_values.append(f"{parameter}={parameter_value}")
    parameter_tag = ",".join(parameter_values)
    return parameter_tag if parameter_tag != "" else "default"


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

    assert len(cfg.models.search_paths) > 0

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating comparator <{cfg.comparator._target_}>")
    comparator: Comparator = hydra.utils.instantiate(cfg.comparator)

    models: Dict[
        str, Dict[str, Dict[str, tuple[Union[DictConfig, ListConfig], InspectableModule]]]
    ] = {}

    for search_path in cfg.models.search_paths:
        for checkpoint_path in glob.glob(search_path):
            try:
                model_config_path = Path(checkpoint_path).parent.parent / ".hydra" / "config.yaml"
                model_config = OmegaConf.load(model_config_path)
            except Exception as exception:
                log.warning(
                    f"Skipping model {checkpoint_path}: Config file {model_config_path} is invalid or could not be loaded.",
                    exc_info=exception,
                )
                continue

            group_tag = get_parameter_tag(model_config, cfg.models.group)
            compare_tag = get_parameter_tag(model_config, cfg.models.compare)
            combine_tag = get_parameter_tag(model_config, cfg.models.combine)

            if group_tag not in models:
                models[group_tag] = {}

            if compare_tag not in models[group_tag]:
                models[group_tag][compare_tag] = {}

            if combine_tag not in models[group_tag][compare_tag]:
                try:
                    model = ImageClassification.load_from_checkpoint(checkpoint_path)
                    models[group_tag][compare_tag][combine_tag] = (model_config, model)

                except Exception as exception:
                    log.warning(
                        f"Skipping model {checkpoint_path}: Checkpoint file is invalid or could not be loaded.",
                        exc_info=exception,
                    )
                    continue

            else:
                log.warning(
                    f"Skipping model: {checkpoint_path}: Model with the same parameter combination is already loaded."
                )
                continue

    if len(models) > 0:
        for group in models.keys():
            comparator.compare(models[group], datamodule, Path(cfg.paths.output_dir) / group)
    else:
        log.warning("No models found -> No comparisons made.")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="compare.yaml")
def main(cfg: DictConfig) -> None:
    compare(cfg)


if __name__ == "__main__":
    main()
