import argparse
import copy
from functools import partial
from pathlib import Path

import pandas as pd
import yaml
from ssl_library.src.datasets.downstream_tasks.ddi_dataset import DDILabel
from ssl_library.src.datasets.downstream_tasks.derma_compass_dataset import (
    DermaCompassLabel,
)
from ssl_library.src.datasets.downstream_tasks.fitzpatrick17_dataset import (
    FitzpatrickLabel,
)
from ssl_library.src.datasets.downstream_tasks.vindr_bodypart_xr import (
    VinDrBodyPartXRDataset,
)
from ssl_library.src.datasets.helper import DatasetName, get_dataset
from ssl_library.src.utils.loader import Loader
from ssl_library.src.utils.logging import set_log_level
from ssl_library.src.utils.utils import cleanup, fix_random_seeds, init_distributed_mode
from torchvision import transforms

from .evaluate_synthetic import get_data_path
from .evaluator.evaluator import Evaluator, PretrainingType
from .perturbed_datasets.artefact_augmentation_dataset import (
    ArtefactAugmentationDataset,
)
from .perturbed_datasets.blur_augment_dataset import BlurAugmentDataset
from .perturbed_datasets.combined_dataset import CombinedDataset
from .perturbed_datasets.label_error_dataset import LabelErrorDataset
from .perturbed_datasets.mixed_augmentation_dataset import MixedAugmentationDataset
from .perturbed_datasets.nd_augment_dataset import NDAugmentDataset

my_parser = argparse.ArgumentParser(description="Evaluates SelfClean on given dataset.")
my_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml for evaluation.",
)
my_parser.add_argument(
    "--additional_arch_info",
    type=str,
    default="",
    help="Path to the config yaml for evaluation.",
)
args = my_parser.parse_args()

if __name__ == "__main__":
    # load config yaml
    args.config_path = Path(args.config_path)
    if not args.config_path.exists():
        raise ValueError(f"Unable to find config yaml file: {args.config_path}")
    config = yaml.load(open(args.config_path, "r"), Loader=Loader)

    set_log_level(min_log_level=config["log_level"])
    init_distributed_mode()
    fix_random_seeds(config["seed"])

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Global configurations
    synth_setup = "XR+AUG+LBLC"
    l_cont = [0.01, 0.03, 0.1, 0.3, 0.5]
    dataset_name = config.pop("dataset_name")
    dataset_name = DatasetName(dataset_name)
    l_seeds = [1, 42, 111]
    df_results = pd.DataFrame()

    # load the "seed" dataset
    label_col = None
    if dataset_name == DatasetName.FITZPATRICK17K:
        label_col = FitzpatrickLabel.MID
    elif dataset_name == DatasetName.DDI:
        label_col = DDILabel.MALIGNANT
    else:
        label_col = DermaCompassLabel.SECONDARY

    data_path, data_kwargs = get_data_path(config=config, dataset_name=dataset_name)
    seed_dataset = get_dataset(
        dataset_name=dataset_name,
        dataset_path=Path(data_path),
        transform=transform,
        label_col=label_col,
        return_loader=False,
        **data_kwargs,
    )

    # create the super augment dataset (worst performance)
    # [0] : off-topic samples
    # [1] : near-duplicates
    # [2] : label errors
    synth_setup = synth_setup.split("+")
    datasets = []
    for cont in l_cont:
        for seed in l_seeds:
            fix_random_seeds(seed)
            dataset = copy.deepcopy(seed_dataset)
            # 1. off-topic samples
            if "XR" == synth_setup[0]:
                _dataset_path, _ = get_data_path(
                    config=config,
                    dataset_name=DatasetName.VINDR_BODY_PART_XR,
                )
                other_dataset = VinDrBodyPartXRDataset(
                    _dataset_path,
                    return_path=True,
                )
                # select only those X-Rays that are also showing a hand
                # i.e. are different than CheXpert for example.
                other_dataset.meta_data = other_dataset.meta_data[
                    other_dataset.meta_data["diagnosis"] == "others"
                ]
                irrelevant_dataset_cls = partial(
                    CombinedDataset,
                    dataset_aug=other_dataset,
                )
            elif "BLUR" == synth_setup[0]:
                irrelevant_dataset_cls = BlurAugmentDataset
            else:
                raise ValueError("Unknown off-topic contamination.")

            # 2. near-duplicates
            if "AUG" == synth_setup[1]:
                duplicate_dataset_cls = NDAugmentDataset
            elif "ARTE" == synth_setup[1]:
                duplicate_dataset_cls = partial(
                    ArtefactAugmentationDataset,
                    **config["artefact_augmentation_dataset"],
                )
            else:
                raise ValueError("Unknown near-duplicate contamination.")

            # 3. label errors
            if "LBL" == synth_setup[2]:
                label_error_dataset_cls = partial(
                    LabelErrorDataset,
                    change_for_every_label=False,
                )
            elif "LBLC" == synth_setup[2]:
                label_error_dataset_cls = partial(
                    LabelErrorDataset,
                    change_for_every_label=True,
                )
            else:
                raise ValueError("Unknown label error contamination.")

            super_aug_dataset = MixedAugmentationDataset(
                dataset=dataset,
                duplicate_dataset_cls=duplicate_dataset_cls,
                irrelevant_dataset_cls=irrelevant_dataset_cls,
                label_error_dataset_cls=label_error_dataset_cls,
                frac_error=cont,
                transform=transform,
                random_state=seed,
                name=f"{dataset_name.value}+MixedAug({'+'.join(synth_setup)})/S:{seed}/C:{cont}",
            )
            datasets.append(super_aug_dataset)
    del seed_dataset

    pretraining_type = PretrainingType[config["pretraining_type"]]
    evaluator = Evaluator(
        super_aug_datasets=datasets,
        pretraining_type=pretraining_type,
        config=config,
        config_path=args.config_path,
        additional_arch_info=args.additional_arch_info,
    )
    evaluator.evaluate_all()
    cleanup()
