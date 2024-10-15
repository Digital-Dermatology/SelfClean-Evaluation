import argparse
import copy
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import yaml
from selfclean.src.cleaner.selfclean_cleaner import SelfCleanCleaner
from ssl_library.src.datasets.helper import DatasetName, get_dataset
from ssl_library.src.pkg import Embedder, embed_dataset
from ssl_library.src.utils.loader import Loader
from ssl_library.src.utils.logging import calculate_scores_from_ranking
from torchvision import transforms
from tqdm import tqdm

from src.detectors.duplicates.phash import pHASHDetector
from src.detectors.duplicates.ssim import SSIMDetector
from src.detectors.label_errors.confident_learning import ConfidentLearningDetector
from src.detectors.label_errors.fastdup import FastDupDetector
from src.detectors.label_errors.noise_rank import NoiseRankDetector

my_parser = argparse.ArgumentParser(description="Compares the rankings with metadata.")
my_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml.",
)
my_parser.add_argument(
    "--datasets",
    nargs="+",
    default=[
        "pad_ufes_20",
        "ham10000",
        "ISIC_2019",
        "CheXpert",
        "CelebA",
        "ImageNet-1k",
        "FOOD_101",
    ],
    help="Name of the datasets to evaluate on.",
)
my_parser.add_argument(
    "--print_results_only",
    action="store_true",
    help="If the results should only be printed and no training is performed.",
)
my_parser.add_argument(
    "--append_results",
    action="store_true",
    help="If the results should be appended to the existing df (needs to be used with care!)",
)
args = my_parser.parse_args()

if __name__ == "__main__":
    # load config yaml
    args.config_path = Path(args.config_path)
    if not args.config_path.exists():
        raise ValueError(f"Unable to find config yaml file: {args.config_path}")
    config = yaml.load(open(args.config_path, "r"), Loader=Loader)
    # define the models for each dataset
    model_dict = {
        DatasetName.PAD_UFES_20: "models/pad_ufes_20/DINO/checkpoint-epoch500.pth",
        DatasetName.HAM10000: "models/HAM10000/DINO/checkpoint-epoch500.pth",
        DatasetName.ISIC_2019: "models/ISIC_2019/DINO/checkpoint-epoch500.pth",
        DatasetName.CHEXPERT: "models/CheXpert/DINO/checkpoint-epoch500.pth",
        DatasetName.CELEB_A: "models/CelebA/DINO/checkpoint-epoch500.pth",
        DatasetName.IMAGENET_1K: "models/ImageNet-1k/DINO/checkpoint-epoch500.pth",
        DatasetName.FOOD_101: "models/Food101N/DINO/checkpoint-epoch500.pth",
    }
    # loop over all given datasets
    for dataset_name in args.datasets:
        dataset_name = DatasetName(dataset_name)
        # load the correct model to use as initialization
        model, _, _ = Embedder.load_dino(
            ckp_path=model_dict.get(dataset_name),
            return_info=True,
            n_head_layers=0,
        )
        model = model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        # load the dataset to evaluate on
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        data_config = copy.deepcopy(config["dataset"])
        data_path = "data/"
        if dataset_name.value in data_config.keys():
            data_path = data_config[dataset_name.value].pop("path")
        dataset, torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(data_path),
            batch_size=128,
            transform=transform,
            **data_config.get(dataset_name.value, {}),
        )

        # only select the verified errors
        if dataset_name == DatasetName.IMAGENET_1K:
            torch_dataset.dataset.meta_data = torch_dataset.dataset.meta_data[
                torch_dataset.dataset.meta_data["mturk"].notna()
            ]
            torch_dataset.dataset.meta_data.reset_index(drop=False, inplace=True)
            dataset.meta_data = torch_dataset.dataset.meta_data
        elif dataset_name == DatasetName.FOOD_101:
            torch_dataset.dataset.meta_data = torch_dataset.dataset.meta_data[
                torch_dataset.dataset.meta_data["verification_label"].notna()
            ]
            torch_dataset.dataset.meta_data.reset_index(drop=False, inplace=True)
            dataset.meta_data = torch_dataset.dataset.meta_data

        emb_space, labels, _, _ = embed_dataset(
            torch_dataset=torch_dataset,
            model=model,
            n_layers=1,
            memmap=False,
            normalize=True,
        )
        del _

        cleaner = SelfCleanCleaner(
            memmap=False,
            global_leaves=False,
            auto_cleaning=False,
        )
        cleaner = cleaner.fit(
            emb_space=emb_space,
            labels=labels,
            class_labels=dataset.classes,
        )
        issues = cleaner.predict()
        if "near_duplicates" in issues.issue_dict.keys():
            pred_dups_scores = issues["near_duplicates"]["scores"]
            pred_dups_indices = issues["near_duplicates"]["indices"]

        if "irrelevants" in issues.issue_dict.keys():
            pred_irr_scores = issues["irrelevants"]["scores"]
            pred_irr_indices = issues["irrelevants"]["indices"]

        if "label_errors" in issues.issue_dict.keys():
            pred_lbl_scores = issues["label_errors"]["scores"]
            pred_lbl_indices = issues["label_errors"]["indices"]

        # --- Near Duplicates ---
        # create ground truth metadata
        dup_col = None
        if dataset_name in [
            DatasetName.PAD_UFES_20,
            DatasetName.HAM10000,
            DatasetName.ISIC_2019,
        ]:
            dup_col = "lesion_id"
        elif dataset_name == DatasetName.CHEXPERT:
            dup_col = "patient_id"
        elif dataset_name == DatasetName.CELEB_A:
            dup_col = "celeb_id"

        if dup_col is not None:
            meta = dataset.meta_data.reset_index(drop=True)
            l_indices = []
            lesion_ids = list(
                meta[dup_col].value_counts()[meta[dup_col].value_counts() > 1].index
            )
            for lesion_id in tqdm(lesion_ids):
                dup_indices = list(meta[meta[dup_col] == lesion_id].index)
                dup_indices_perm = list(permutations(dup_indices, 2))
                l_indices += dup_indices_perm

            # calculate the % of duplicates in the dataset
            print(
                f"% of duplicates: {(len(l_indices) / (len(dataset) * ((len(dataset) - 1) / 2))) * 100}"
            )

            l_truths = []
            # eval SelfClean
            pred_dups_list = list(map(tuple, pred_dups_indices))
            truth = [
                1 if (x[0], x[1]) in l_indices else 0 for x in tqdm(pred_dups_list)
            ]
            truth = np.asarray(truth)
            l_truths.append(("SelfClean", truth))

            # eval competitors
            dataset.transform = None
            for dup_detector in [SSIMDetector, pHASHDetector]:
                l_ranking = dup_detector.get_ranking(
                    dataset=dataset,
                    hash_size=8,
                    n_processes=24 * 2,
                )
                truth = [1 if (x[0], x[1]) in l_indices else 0 for x in tqdm(l_ranking)]
                truth = np.asarray(truth)
                l_truths.append((dup_detector.__class__.__name__, truth))
            dataset.transform = transform

            for name, truth in l_truths:
                calculate_scores_from_ranking(
                    ranking=truth,
                    log_wandb=False,
                    show_plots=False,
                    show_scores=True,
                )

        # --- Label Errors ---
        # create ground truth metadata
        lbl_col = None
        if dataset_name == DatasetName.IMAGENET_1K:
            lbl_col = "mturk"
        elif dataset_name == DatasetName.FOOD_101:
            lbl_col = "verification_label"

        if lbl_col is not None:
            avail_meta = list(
                dataset.meta_data[dataset.meta_data[lbl_col].notna()].index
            )
            if dataset_name == DatasetName.IMAGENET_1K:
                lbl_errs = list(
                    dataset.meta_data[dataset.meta_data["label_error"].notna()].index
                )
            elif dataset_name == DatasetName.FOOD_101:
                lbl_errs = list(
                    dataset.meta_data[
                        dataset.meta_data["verification_label"] == 0.0
                    ].index
                )

            # calculate the % of label errors in the dataset
            print(f"% of label errors: {(len(lbl_errs) / len(dataset)) * 100}")

            l_truths = []
            # eval SelfClean
            truth = [1 if int(x) in lbl_errs else 0 for x in pred_lbl_indices]
            truth = np.asarray(truth)
            l_truths.append(("SelfClean", truth))

            # eval competitors
            for lbl_detector in [
                NoiseRankDetector,
                FastDupDetector,
                ConfidentLearningDetector,
            ]:
                ranking = lbl_detector.get_ranking(
                    dataset=copy.deepcopy(dataset),
                )
                if type(ranking[0]) is tuple:
                    truth = [1 if int(x[1]) in lbl_errs else 0 for x in ranking]
                else:
                    truth = [1 if int(x) in lbl_errs else 0 for x in ranking]
                truth = np.asarray(truth)
                l_truths.append((lbl_detector.__class__.__name__, truth))

            for name, truth in l_truths:
                scores = calculate_scores_from_ranking(
                    ranking=truth,
                    log_wandb=False,
                    show_plots=False,
                    show_scores=True,
                )
