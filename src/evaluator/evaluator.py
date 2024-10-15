import copy
import gc
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import wandb
from loguru import logger
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from selfclean.src.cleaner.selfclean_cleaner import SelfCleanCleaner
from ssl_library.src.augmentations.multi_crop import MultiCropAugmentation
from ssl_library.src.augmentations.simclr import (
    MultiCropSimCLRAugmentation,
    SimCLRDataAugmentation,
)
from ssl_library.src.datasets.base_dataset import BaseDataset
from ssl_library.src.pkg import Embedder, embed_dataset
from ssl_library.src.utils.logging import calculate_scores_from_ranking
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from ..detectors.duplicates.phash import pHASHDetector
from ..detectors.duplicates.ssim import SSIMDetector
from ..detectors.label_errors.confident_learning import ConfidentLearningDetector
from ..detectors.label_errors.fastdup import FastDupDetector
from ..detectors.label_errors.noise_rank import NoiseRankDetector
from ..perturbed_datasets.base_contamination_dataset import BaseContaminationDataset
from ..perturbed_datasets.mixed_augmentation_dataset import MixedAugmentationDataset
from ..trainers.online_cleaning_mixin import OnlineEvalType
from ..trainers.selfclean_trainers import (
    create_byol_trainer_class,
    create_dino_trainer_class,
    create_mae_trainer_class,
    create_simclr_trainer_class,
)


class PretrainingType(Enum):
    IMAGENET = "imagenet"
    IMAGENET_VIT = "imagenet_vit"
    IMAGENET_DINO = "imagenet_dino"
    SIMCLR = "simclr"
    DINO = "dino"
    MAE = "mae"
    BYOL = "byol"


class Evaluator:
    def __init__(
        self,
        pretraining_type: PretrainingType,
        config: dict,
        config_path: Union[str, Path],
        duplicate_datasets: List[BaseContaminationDataset] = [],
        irrelevant_datasets: List[BaseContaminationDataset] = [],
        label_error_datasets: List[BaseContaminationDataset] = [],
        super_aug_datasets: List[MixedAugmentationDataset] = [],
        additional_arch_info: str = "",
        eval_competitors: bool = False,
    ):
        self.config = config
        self.config_path = config_path
        self.pretraining_type = pretraining_type
        self.additional_arch_info = additional_arch_info
        # datasets to evaluate
        self.duplicate_datasets = duplicate_datasets
        self.irrelevant_datasets = irrelevant_datasets
        self.label_error_datasets = label_error_datasets
        self.super_aug_datasets = super_aug_datasets
        # transformations
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        # detector baselines
        self.irrelevant_baselines = []
        self.duplicate_baselines = []
        self.label_error_baselines = []
        if eval_competitors:
            self.irrelevant_baselines = [
                ("ECOD", ECOD),
                ("IForest", IForest),
                ("KNN", KNN),
                ("HBOS", HBOS),
            ]
            self.duplicate_baselines = [
                ("SSIM", SSIMDetector),
                ("pHASH", pHASHDetector),
            ]
            self.label_error_baselines = [
                ("NoiseRank", NoiseRankDetector),
                ("FastDup", FastDupDetector),
                ("ConfidentLearning", ConfidentLearningDetector),
            ]

    def train(
        self,
        dataset: BaseDataset,
        dataset_name: Optional[str] = None,
        online_eval_type: Optional[OnlineEvalType] = None,
    ):
        name = dataset_name if dataset_name is not None else dataset.name
        if self.pretraining_type is PretrainingType.DINO:
            # train using SSL on dataset
            model_config = self.config["model_config"]
            # load the train dataset
            dino_transform = MultiCropAugmentation(
                **model_config["dataset"]["augmentations"]
            )
            dataset.transform = dino_transform
            if online_eval_type is OnlineEvalType.ALL:
                dataset.irrelevant_dataset.transform = dino_transform
                dataset.duplicate_dataset.transform = dino_transform
                dataset.label_error_dataset.transform = dino_transform
            sampler = DistributedSampler(dataset, shuffle=True)
            train_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=model_config["batch_size"],
                **model_config["dataset"]["loader"],
            )
            # copy a dataset for online evaluation
            error_dataset = copy.deepcopy(dataset)
            error_dataset.transform = self.base_transform
            if online_eval_type is OnlineEvalType.ALL:
                error_dataset.irrelevant_dataset.transform = self.base_transform
                error_dataset.duplicate_dataset.transform = self.base_transform
                error_dataset.label_error_dataset.transform = self.base_transform
            error_dataset_loader = DataLoader(
                error_dataset,
                batch_size=model_config["batch_size"],
                **model_config["dataset"]["val_loader"],
            )
            cleaner = SelfCleanCleaner(memmap=False)
            trainer = create_dino_trainer_class()(
                train_dataset=train_loader,
                val_dataset=error_dataset_loader,
                config=model_config,
                config_path=self.config_path,
                additional_run_info=name,
                additional_arch_info=self.additional_arch_info,
                cleaner_cls=cleaner,
                online_eval_type=online_eval_type,
                irrelevant_baselines=self.irrelevant_baselines,
                duplicate_baselines=self.duplicate_baselines,
                label_error_baselines=self.label_error_baselines,
            )
            model = trainer.fit()
            del trainer
            gc.collect()
            return model

        elif self.pretraining_type is PretrainingType.MAE:
            # train using SSL on dataset
            model_config = self.config["model_config"]
            # load the train dataset
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        model_config["dataset"]["augmentations"]["input_size"],
                        scale=(0.2, 1.0),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            dataset.transform = train_transform
            if online_eval_type is OnlineEvalType.ALL:
                dataset.irrelevant_dataset.transform = train_transform
                dataset.duplicate_dataset.transform = train_transform
                dataset.label_error_dataset.transform = train_transform
            sampler = DistributedSampler(dataset, shuffle=True)
            train_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=model_config["batch_size"],
                **model_config["dataset"]["loader"],
            )
            # copy a dataset for online evaluation
            error_dataset = copy.deepcopy(dataset)
            error_dataset.transform = self.base_transform
            if online_eval_type is OnlineEvalType.ALL:
                error_dataset.irrelevant_dataset.transform = self.base_transform
                error_dataset.duplicate_dataset.transform = self.base_transform
                error_dataset.label_error_dataset.transform = self.base_transform
            error_dataset_loader = DataLoader(
                error_dataset,
                batch_size=model_config["batch_size"],
                **model_config["dataset"]["val_loader"],
            )
            cleaner = SelfCleanCleaner(memmap=False)
            trainer = create_mae_trainer_class()(
                train_dataset=train_loader,
                val_dataset=error_dataset_loader,
                config=model_config,
                config_path=self.config_path,
                additional_run_info=name,
                additional_arch_info=self.additional_arch_info,
                cleaner_cls=cleaner,
                online_eval_type=online_eval_type,
            )
            model = trainer.fit()
            del trainer
            gc.collect()
            return model

        elif self.pretraining_type is PretrainingType.SIMCLR:
            # train using SSL on dataset
            model_config = self.config["model_config"]
            # load the train dataset
            multi_crop = model_config["dataset"].get("multi_crop", False)
            if multi_crop:
                simclr_transform = MultiCropSimCLRAugmentation(
                    **model_config["dataset"]["augmentations"]
                )
            else:
                simclr_transform = SimCLRDataAugmentation(
                    **model_config["dataset"]["augmentations"]
                )
            dataset.transform = simclr_transform
            if online_eval_type is OnlineEvalType.ALL:
                dataset.irrelevant_dataset.transform = simclr_transform
                dataset.duplicate_dataset.transform = simclr_transform
                dataset.label_error_dataset.transform = simclr_transform
            sampler = DistributedSampler(dataset, shuffle=True)
            train_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=model_config["batch_size"],
                **model_config["dataset"]["loader"],
            )
            # copy a dataset for online evaluation
            error_dataset = copy.deepcopy(dataset)
            error_dataset.transform = self.base_transform
            if online_eval_type is OnlineEvalType.ALL:
                error_dataset.irrelevant_dataset.transform = self.base_transform
                error_dataset.duplicate_dataset.transform = self.base_transform
                error_dataset.label_error_dataset.transform = self.base_transform
            error_dataset_loader = DataLoader(
                error_dataset,
                batch_size=model_config["batch_size"],
                drop_last=False,
                shuffle=False,
            )
            cleaner = SelfCleanCleaner(memmap=False)
            trainer = create_simclr_trainer_class()(
                train_dataset=train_loader,
                val_dataset=error_dataset_loader,
                config=model_config,
                config_path=self.config_path,
                additional_run_info=name,
                additional_arch_info=self.additional_arch_info,
                cleaner_cls=cleaner,
                online_eval_type=online_eval_type,
            )
            model = trainer.fit()
            del trainer
            gc.collect()
            return model

        elif self.pretraining_type is PretrainingType.BYOL:
            # train using SSL on dataset
            model_config = self.config["model_config"]
            simclr_transform = SimCLRDataAugmentation(
                **model_config["dataset"]["augmentations"]
            )
            dataset.transform = simclr_transform
            if online_eval_type is OnlineEvalType.ALL:
                dataset.irrelevant_dataset.transform = simclr_transform
                dataset.duplicate_dataset.transform = simclr_transform
                dataset.label_error_dataset.transform = simclr_transform
            sampler = DistributedSampler(dataset, shuffle=True)
            train_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=model_config["batch_size"],
                **model_config["dataset"]["loader"],
            )
            # copy a dataset for online evaluation
            error_dataset = copy.deepcopy(dataset)
            error_dataset.transform = self.base_transform
            if online_eval_type is OnlineEvalType.ALL:
                error_dataset.irrelevant_dataset.transform = self.base_transform
                error_dataset.duplicate_dataset.transform = self.base_transform
                error_dataset.label_error_dataset.transform = self.base_transform
            error_dataset_loader = DataLoader(
                error_dataset,
                batch_size=model_config["batch_size"],
                drop_last=False,
                shuffle=False,
            )
            cleaner = SelfCleanCleaner(memmap=False)
            trainer = create_byol_trainer_class()(
                train_dataset=train_loader,
                val_dataset=error_dataset_loader,
                config=model_config,
                config_path=self.config_path,
                additional_run_info=name,
                additional_arch_info=self.additional_arch_info,
                cleaner_cls=cleaner,
                online_eval_type=online_eval_type,
            )
            model = trainer.fit()
            del trainer
            gc.collect()
            return model

        elif self.pretraining_type is PretrainingType.IMAGENET_DINO:
            if wandb.run is None:
                wandb.init(
                    config=self.config,
                    project="SelfClean",
                    group=self.pretraining_type.value,
                )
                run_name = f"{self.pretraining_type.value}-{wandb.run.name}"
                # update the name of the run
                wandb.run.name = run_name
                wandb.run.save()
            model = Embedder.load_dino(
                ckp_path="models/ImageNet-1k/DINO/checkpoint-epoch500.pth",
                n_head_layers=0,
            )
            return model

        elif (
            not self.pretraining_type is PretrainingType.IMAGENET
            and not self.pretraining_type is PretrainingType.IMAGENET_VIT
        ):
            if wandb.run is None:
                wandb.init(
                    config=self.config,
                    project="SelfClean",
                    group=self.pretraining_type.value,
                )
                run_name = f"{self.pretraining_type.value}-{wandb.run.name}"
                # update the name of the run
                wandb.run.name = run_name
                wandb.run.save()
            model = Embedder.load_pretrained(self.pretraining_type.value)
            return model

    def evaluate_all(self):
        for dataset in self.super_aug_datasets:
            cleaner = self.get_cleaner(
                dataset=dataset,
                online_eval_type=OnlineEvalType.ALL,
            )
            issues = cleaner.predict()
            # Near-Duplicates
            pred_dups_indices = issues.get_issues("near_duplicates")["indices"]
            true_dups = dataset.get_duplicate_errors()[0]
            ranking_target = [
                1 if (int(x[0]), int(x[1])) in true_dups else 0
                for x in tqdm(pred_dups_indices)
            ]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                wandb_cat="NearDuplicates/",
                show_plots=False,
            )

            # Label Errors
            pred_lbl_errs = issues.get_issues("label_errors")["indices"]
            true_lbl_errs = dataset.get_label_errors()[0]
            ranking_target = [int(true_lbl_errs[i]) for i in pred_lbl_errs]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                wandb_cat="LabelErrors/",
                show_plots=False,
            )

            # Irrelevant Samples
            pred_ood = issues.get_issues("irrelevants")["indices"]
            true_ood = dataset.get_irrelevant_errors()[0]
            ranking_target = [1 if int(x) in true_ood else 0 for x in pred_ood]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                wandb_cat="IrrelevantSamples/",
                show_plots=False,
            )
            # cleanup
            dataset.cleanup()
            wandb.run.finish()
            wandb.finish()

    def simulate_missing_errors(
        self,
        dataset: MixedAugmentationDataset,
        indices,
        true_indices,
        size_dataset: int,
        error_name: str,
        n_annotators=3,
    ):
        # Simulate the missed errors
        l_simu_anno = []
        for _ in range(n_annotators):
            simulated_anno = [
                np.random.choice([0, 1], p=[1 - dataset.frac_error, dataset.frac_error])
                for _ in range(size_dataset)
            ]
            l_simu_anno.append(simulated_anno)
        l_simu_anno = np.asarray(l_simu_anno)
        df_simu_anno = pd.DataFrame(l_simu_anno.T)
        df_simu_anno.columns = [f"anno_{x}" for x in df_simu_anno.columns]

        p_plus = 0.05
        p_chance = 0.05
        n_repeats = 10
        l_results = []
        for _ in range(n_repeats):
            _df = self.sensitivity_of_stopping_critera(
                df=df_simu_anno, p_plus=p_plus, p_chance=p_chance
            )
            simu_indices = np.where(_df["all_equal_vote"])[0]
            simu_indices = indices[simu_indices]
            if type(true_indices[0]) is tuple:
                simu_indices = [tuple(x) for x in simu_indices]
            non_detected_errors = [x for x in true_indices if x not in simu_indices]
            l_results.append(len(non_detected_errors))
        wandb.log(
            {
                f"SimulatedNonDetected_{error_name}/p_plus:{p_plus}+p_chance:{p_chance}": np.asarray(
                    l_results
                )
            }
        )

    def sensitivity_of_stopping_critera(
        self,
        df: pd.DataFrame,
        p_plus=0.05,
        p_chance=0.05,
        label_cols=["anno_0", "anno_1", "anno_2"],
        debug: bool = False,
    ):
        # $n_{\text{clean}} = \lfloor \ln(p_\text{chance})/\ln(1 - p_+) \rfloor$

        NO_CLEAN_IN_SEQUENCE = int(np.log(p_chance) / np.log(1 - p_plus))
        if debug:
            logger.info(f"N.o. clean samples needed in a row: {NO_CLEAN_IN_SEQUENCE}")

        _df = df.copy()

        for col_label in label_cols:
            last_idx = _df[_df[col_label].notna()].index[-1]
            stopping_point = last_idx
            if debug:
                logger.info(f"\n({col_label}) Length of the sequence: {last_idx}")

            # if they skip some samples they are not considered
            _df[col_label].iloc[: last_idx + 1].fillna("Yes", inplace=True)
            for i in range(last_idx + 1):
                anno_sequence = _df[col_label].values
                anno_sequence = anno_sequence[: i + 1]

                last_issue_found = np.argwhere(np.asarray(anno_sequence) == 1)
                if len(last_issue_found) > 0:
                    last_issue_found = last_issue_found[-1][0]
                else:
                    last_issue_found = -1

                no_sequential_non_issues = len(anno_sequence) - (last_issue_found + 1)
                if no_sequential_non_issues >= NO_CLEAN_IN_SEQUENCE:
                    if debug:
                        logger.info(
                            f"({col_label}) Stopping reached at: {i}, Length of the sequence: {last_idx}"
                        )
                    stopping_point = i
                    break
            _df.loc[stopping_point + 1 :, col_label] = np.nan

        _df["all_equal_vote"] = _df[label_cols].apply(
            lambda x: x.values[0] if np.all(x.values == x.values[0]) else 0,
            axis=1,
        )
        return _df

    def evaluate_near_duplicates(self):
        for dataset in self.duplicate_datasets:
            cleaner = self.get_cleaner(
                dataset=dataset,
                online_eval_type=OnlineEvalType.DUPLICATES,
            )
            issues = cleaner.predict()
            pred_dups_indices = issues.get_issues("near_duplicates")["indices"]
            true_dups = dataset.get_errors()[0]
            ranking_target = [
                1 if (int(x[0]), int(x[1])) in true_dups else 0
                for x in tqdm(pred_dups_indices)
            ]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                wandb_cat="NearDuplicates/",
                show_plots=False,
            )
            dataset.cleanup()
            wandb.run.finish()
            wandb.finish()

    def evaluate_irrelevant_samples(self):
        for dataset in self.irrelevant_datasets:
            cleaner = self.get_cleaner(
                dataset=dataset,
                online_eval_type=OnlineEvalType.IRRELEVANT,
            )
            issues = cleaner.predict()
            pred_ood = issues.get_issues("irrelevants")["indices"]
            true_ood = dataset.get_errors()[0]
            ranking_target = [1 if int(x) in true_ood else 0 for x in pred_ood]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                wandb_cat="IrrelevantSamples/",
                show_plots=False,
            )
            dataset.cleanup()
            wandb.run.finish()
            wandb.finish()

    def evaluate_lbl_errors(self):
        for dataset in self.label_error_datasets:
            cleaner = self.get_cleaner(
                dataset=dataset,
                online_eval_type=OnlineEvalType.LABELS,
            )
            issues = cleaner.predict()
            pred_lbl_errs = issues.get_issues("label_errors")["indices"]
            true_lbl_errs = dataset.get_errors()[0]
            ranking_target = [int(true_lbl_errs[i]) for i in pred_lbl_errs]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                wandb_cat="LabelErrors/",
                show_plots=False,
            )
            dataset.cleanup()
            wandb.run.finish()
            wandb.finish()

    def get_cleaner(self, dataset, online_eval_type: OnlineEvalType):
        logger.info("-" * 20 + f" Evaluating dataset: {dataset.name} " + "-" * 20)
        dataset.info()
        model = self.train(
            dataset=dataset,
            online_eval_type=online_eval_type,
        )
        model.eval()
        dataset.transform = self.base_transform
        if online_eval_type is OnlineEvalType.ALL:
            dataset.irrelevant_dataset.transform = self.base_transform
            dataset.duplicate_dataset.transform = self.base_transform
            dataset.label_error_dataset.transform = self.base_transform
        torch_dataset = DataLoader(
            dataset,
            batch_size=128,
            drop_last=False,
            shuffle=False,
        )
        emb_space, labels, images, paths = embed_dataset(
            torch_dataset=torch_dataset,
            model=model,
            n_layers=self.config["n_layers"],
            normalize=self.config["apply_l2_norm"],
            memmap=False,
        )

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner = cleaner.fit(
            emb_space=emb_space,
            labels=labels,
            paths=paths,
            dataset=copy.deepcopy(dataset),
        )
        return cleaner
