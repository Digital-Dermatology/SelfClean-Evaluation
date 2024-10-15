import argparse
import copy
import itertools
from pathlib import Path

import yaml
from ssl_library.src.datasets.downstream_tasks.chexpert_dataset import CheXpertLabel
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

from .evaluator.evaluator import Evaluator, PretrainingType
from .perturbed_datasets.artefact_augmentation_dataset import (
    ArtefactAugmentationDataset,
)
from .perturbed_datasets.blur_augment_dataset import BlurAugmentDataset
from .perturbed_datasets.combined_dataset import CombinedDataset
from .perturbed_datasets.label_error_dataset import LabelErrorDataset
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


def get_data_path(config: dict, dataset_name: DatasetName):
    data_config = copy.deepcopy(config["dataset"])
    data_path = "data/"
    data_kwargs = {}
    if dataset_name.value in data_config.keys():
        data_path = data_config[dataset_name.value].pop("path")
        data_kwargs = data_config[dataset_name.value]
    return data_path, data_kwargs


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
    l_cont = [0.05, 0.1]
    l_datasets = [DatasetName.STL_10, DatasetName.VINDR_BODY_PART_XR, DatasetName.DDI]

    # Near duplicates
    # --------
    dup_datasets = []
    for cont, dataset_name in itertools.product(l_cont, l_datasets):
        label_col = None
        if dataset_name == DatasetName.FITZPATRICK17K:
            label_col = FitzpatrickLabel.MID
        elif dataset_name == DatasetName.DDI:
            label_col = DDILabel.MALIGNANT
        elif dataset_name == DatasetName.CHEXPERT:
            label_col = CheXpertLabel.ATELECTASIS
        else:
            label_col = DermaCompassLabel.SECONDARY

        dataset_path, _ = get_data_path(config=config, dataset_name=dataset_name)
        dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(dataset_path),
            transform=transform,
            label_col=label_col,
            return_fitzpatrick=False,
            return_loader=False,
        )
        med_dup = NDAugmentDataset(
            dataset=dataset,
            frac_error=cont,
            name=f"{dataset_name.value}+AUG/C:{cont}",
            random_state=config["seed"],
        )
        arte_dup = ArtefactAugmentationDataset(
            dataset=dataset,
            frac_error=cont,
            name=f"{dataset_name.value}+ARTE/C:{cont}",
            **config["artefact_augmentation_dataset"],
            random_state=config["seed"],
            grayscale=True if dataset_name == DatasetName.VINDR_BODY_PART_XR else False,
        )
        dup_datasets.append(arte_dup)
        dup_datasets.append(med_dup)

    # Label errors
    # --------
    lbl_datasets = []
    for cont, dataset_name in itertools.product(l_cont, l_datasets):
        label_col = None
        if dataset_name == DatasetName.FITZPATRICK17K:
            label_col = FitzpatrickLabel.MID
        elif dataset_name == DatasetName.DDI:
            label_col = DDILabel.MALIGNANT
        elif dataset_name == DatasetName.CHEXPERT:
            label_col = CheXpertLabel.ATELECTASIS
        else:
            label_col = DermaCompassLabel.SECONDARY

        dataset_path, _ = get_data_path(config=config, dataset_name=dataset_name)
        dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(dataset_path),
            transform=transform,
            label_col=label_col,
            return_fitzpatrick=False,
            return_loader=False,
        )
        trivial_lbl = LabelErrorDataset(
            dataset=dataset,
            frac_error=cont,
            change_for_every_label=False,
            name=f"{dataset_name.value}+LBL/C:{cont}",
            random_state=config["seed"],
        )
        realistic_lbl = LabelErrorDataset(
            dataset=dataset,
            frac_error=cont,
            change_for_every_label=True,
            name=f"{dataset_name.value}+LBLC/C:{cont}",
            random_state=config["seed"],
        )
        lbl_datasets.append(trivial_lbl)
        lbl_datasets.append(realistic_lbl)

    # Irrelevant Sample Evaluation
    # --------
    irrelevant_datasets = []
    l_datasets = [
        "STL+XR",
        "STL+BLUR",
        "VDR+BLUR",
        "VDR+XR",
        "DDI+XR",
        "DDI+BLUR",
    ]

    for cont, error_name in itertools.product(l_cont, l_datasets):
        d_name = error_name.split("+")[0]
        e_name = error_name.split("+")[1]

        # Seed Dataset
        if d_name == "DDI":
            dataset_name = DatasetName.DDI
        elif d_name == "STL":
            dataset_name = DatasetName.STL_10
        elif d_name == "VDR":
            dataset_name = DatasetName.VINDR_BODY_PART_XR
        else:
            raise ValueError(f"Unknown dataset type: {d_name}")

        label_col = None
        if dataset_name == DatasetName.FITZPATRICK17K:
            label_col = FitzpatrickLabel.MID
        elif dataset_name == DatasetName.DDI:
            label_col = DDILabel.MALIGNANT
        elif dataset_name == DatasetName.CHEXPERT:
            label_col = CheXpertLabel.ATELECTASIS
        else:
            label_col = DermaCompassLabel.SECONDARY

        dataset_path, _ = get_data_path(config=config, dataset_name=dataset_name)
        dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(dataset_path),
            transform=transform,
            label_col=label_col,
            return_fitzpatrick=False,
            return_loader=False,
        )

        # Contamination Dataset
        if e_name == "XR":
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
        elif e_name == "BLUR":
            blur_dataset = BlurAugmentDataset(
                dataset=dataset,
                frac_error=cont,
                name=f"{dataset_name.value}+BLUR/C:{cont}",
                random_state=config["seed"],
            )
            irrelevant_datasets.append(blur_dataset)
            continue
        else:
            raise ValueError(f"Unknown error type: {e_name}")

        aug_dataset = CombinedDataset(
            dataset=dataset,
            dataset_aug=other_dataset,
            transform=transform,
            frac_error=cont,
            name=f"{error_name}/C:{cont}",
            random_state=config["seed"],
        )
        irrelevant_datasets.append(aug_dataset)

    pretraining_type = PretrainingType[config["pretraining_type"]]
    evaluator = Evaluator(
        duplicate_datasets=dup_datasets,
        irrelevant_datasets=irrelevant_datasets,
        label_error_datasets=lbl_datasets,
        pretraining_type=pretraining_type,
        config=config,
        config_path=args.config_path,
        additional_arch_info=args.additional_arch_info,
        eval_competitors=False,
    )
    evaluator.evaluate_irrelevant_samples()
    evaluator.evaluate_near_duplicates()
    evaluator.evaluate_lbl_errors()
    cleanup()
