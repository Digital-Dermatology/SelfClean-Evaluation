import copy
import inspect
import os
from enum import Enum
from itertools import product
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from selfclean.src.cleaner.base_cleaner import BaseCleaner
from ssl_library.src.utils.logging import (
    calculate_scores_from_ranking,
    visualize_worst_duplicate_ranking,
    visualize_worst_label_error_ranking,
)
from torch.utils.data import Dataset
from tqdm import tqdm

from src.detectors.irrelevant_samples.pyod_wrapper import PyODWrapper
from src.perturbed_datasets.base_contamination_dataset import BaseContaminationDataset
from src.perturbed_datasets.mixed_augmentation_dataset import MixedAugmentationDataset
from src.utils.utils import (
    irrelevant_samples_plot,
    label_errors_plot,
    near_duplicate_plot,
)


class OnlineEvalType(Enum):
    IRRELEVANT = "irrelevant"
    DUPLICATES = "duplicates"
    LABELS = "labels"
    ALL = "all"
    INSPECT = "inspect"


class OnlineCleaningMixin(object):
    def __init__(
        self,
        cleaner_cls: Optional[BaseCleaner] = None,
        online_eval_type: Optional[OnlineEvalType] = None,
        duplicate_baselines: list = [],
        irrelevant_baselines: list = [],
        label_error_baselines: list = [],
        wandb_logging: bool = True,
    ):
        self.cleaner_cls = cleaner_cls
        self.online_eval_type = online_eval_type
        self.wandb_logging = wandb_logging
        # define baselines, which also get measured
        self.duplicate_baselines = duplicate_baselines
        self.irrelevant_baselines = irrelevant_baselines
        self.label_error_baselines = label_error_baselines
        # define the metrics we want to maximize for better tracking
        if self.wandb_logging:
            for error_type, metric_name in product(
                ["IrrelevantSamples", "NearDuplicates", "LabelErrors"],
                ["AUROC", "AP", "AUPRG", "Recall@20", "Precision@20"],
            ):
                wandb.define_metric(
                    f"{error_type}/evaluation/{metric_name}", summary="max"
                )

    def online_cleaning_evaluation(
        self,
        val_dataset: Union[Dataset, BaseContaminationDataset, MixedAugmentationDataset],
        embeddings: torch.Tensor,
        imgs: torch.Tensor,
        lbls: torch.Tensor,
        paths: np.ndarray,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        # online evaluation
        if self.cleaner_cls is not None and self.online_eval_type is not None:
            cleaner = self.cleaner_cls.fit(
                emb_space=embeddings.numpy(),
                labels=(
                    lbls.numpy()
                    if self.online_eval_type is OnlineEvalType.LABELS
                    or self.online_eval_type is OnlineEvalType.ALL
                    else None
                ),
                dataset=copy.copy(val_dataset),
            )
            issues = cleaner.predict()

            fig = near_duplicate_plot(
                issues=issues,
                dataset=val_dataset,
                plot_top_N=7,
                return_fig=True,
            )
            wandb.log({f"SelfClean/near_duplicates": fig})
            fig.clf()
            plt.clf()
            del fig

            if type(val_dataset) is torch.utils.data.ConcatDataset:
                class_labels = None
            else:
                class_labels = val_dataset.classes
            fig = irrelevant_samples_plot(
                issues=issues,
                dataset=val_dataset,
                lbls=lbls,
                class_labels=class_labels,
                plot_top_N=7,
                return_fig=True,
            )
            wandb.log({f"SelfClean/irrelevant_samples": fig})
            fig.clf()
            plt.clf()
            del fig

            if issues.get_issues("label_errors")["indices"] is not None:
                if (
                    self.online_eval_type is OnlineEvalType.LABELS
                    or self.online_eval_type is OnlineEvalType.ALL
                ):
                    error_func = val_dataset.get_errors
                    if self.online_eval_type is OnlineEvalType.ALL:
                        error_func = val_dataset.get_label_errors
                    true_errors = error_func()[0]

                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(np.where(true_errors)[0], bins=len(true_errors))
                    wandb.log({"LabelErrors/ranking_hist": wandb.Image(fig)})
                    plt.show()
                    plt.clf()
                else:
                    true_errors = None
                fig = label_errors_plot(
                    issues=issues,
                    dataset=val_dataset,
                    lbls=lbls,
                    class_labels=val_dataset.classes,
                    plot_top_N=7,
                    errors=true_errors,
                    return_fig=True,
                )
                wandb.log({"SelfClean/label_errors": fig})
                fig.clf()
                plt.clf()
                del fig
            if (
                self.online_eval_type is OnlineEvalType.IRRELEVANT
                or self.online_eval_type is OnlineEvalType.ALL
            ):
                pred_ood = issues.get_issues("irrelevants")["indices"]
                error_func = val_dataset.get_errors
                if self.online_eval_type is OnlineEvalType.ALL:
                    error_func = val_dataset.get_irrelevant_errors
                true_errors = error_func()[0]
                ranking_target = [1 if int(x) in true_errors else 0 for x in pred_ood]
                logger.info("Irrelevant Samples:")
                calculate_scores_from_ranking(
                    ranking=ranking_target,
                    log_wandb=self.wandb_logging,
                    wandb_cat="IrrelevantSamples/",
                    show_plots=False,
                    **kwargs,
                )
                self.run_irrelevant_baselines(
                    val_dataset=val_dataset,
                    embeddings=embeddings,
                    true_errors=true_errors,
                    **kwargs,
                )
            if (
                self.online_eval_type is OnlineEvalType.DUPLICATES
                or self.online_eval_type is OnlineEvalType.ALL
            ):
                pred_dups_indices = issues.get_issues("near_duplicates")["indices"]
                pred_dups_scores = issues.get_issues("near_duplicates")["scores"]
                error_func = val_dataset.get_errors
                if self.online_eval_type is OnlineEvalType.ALL:
                    error_func = val_dataset.get_duplicate_errors
                true_errors = error_func()[0]
                ranking_target = [
                    1 if (int(x[0]), int(x[1])) in true_errors else 0
                    for x in tqdm(pred_dups_indices)
                ]
                visualize_worst_duplicate_ranking(
                    ranking_target=ranking_target,
                    pred_dups_indices=pred_dups_indices,
                    pred_dups_scores=pred_dups_scores,
                    images=imgs,
                    paths=paths,
                    imgs_to_visualize=5,
                )
                logger.info("Near Duplicates:")
                calculate_scores_from_ranking(
                    ranking=ranking_target,
                    log_wandb=self.wandb_logging,
                    wandb_cat="NearDuplicates/",
                    show_plots=False,
                    **kwargs,
                )
                self.run_duplicate_baselines(
                    val_dataset=val_dataset,
                    true_errors=true_errors,
                    **kwargs,
                )
            if (
                self.online_eval_type is OnlineEvalType.LABELS
                or self.online_eval_type is OnlineEvalType.ALL
            ):
                pred_lbl_errs = issues.get_issues("label_errors")["indices"]
                error_func = val_dataset.get_errors
                if self.online_eval_type is OnlineEvalType.ALL:
                    error_func = val_dataset.get_label_errors
                true_errors = error_func()[0]
                ranking_target = [int(true_errors[i]) for i in pred_lbl_errs]
                visualize_worst_label_error_ranking(
                    ranking_target=ranking_target,
                    pred_le_indices=pred_lbl_errs,
                    lbls=lbls,
                    class_labels=val_dataset.classes,
                    images=imgs,
                    imgs_to_visualize=5,
                )
                logger.info("Label Errors:")
                calculate_scores_from_ranking(
                    ranking=ranking_target,
                    log_wandb=self.wandb_logging,
                    wandb_cat="LabelErrors/",
                    show_plots=False,
                    **kwargs,
                )
                self.run_label_error_baselines(
                    val_dataset=val_dataset,
                    embeddings=embeddings,
                    lbls=lbls,
                    true_errors=true_errors,
                    **kwargs,
                )

    def run_irrelevant_baselines(
        self,
        val_dataset: Union[Dataset, BaseContaminationDataset, MixedAugmentationDataset],
        embeddings: torch.Tensor,
        true_errors: Union[np.ndarray, list],
        **kwargs,
    ):
        for name_detector, irr_detector in self.irrelevant_baselines:
            # run for INet features
            ranking = PyODWrapper.get_ranking(
                irrelevant_detector=irr_detector,
                dataset=val_dataset,
                ssl_model="imagenet_vit_tiny",
            )
            ranking_target = [1 if x in true_errors else 0 for x in ranking]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                show_scores=False,
                show_plots=False,
                wandb_cat=f"IrrelevantSamples/{name_detector}-INet/",
                **kwargs,
            )
            # run for INetSSL features
            ranking = PyODWrapper.get_ranking(
                irrelevant_detector=irr_detector,
                dataset=val_dataset,
                ssl_model="imagenet_dino",
            )
            ranking_target = [1 if x in true_errors else 0 for x in ranking]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                show_scores=False,
                show_plots=False,
                wandb_cat=f"IrrelevantSamples/{name_detector}-INetSSL/",
                **kwargs,
            )
            # run for SSL features
            ranking = PyODWrapper.get_ranking(
                irrelevant_detector=irr_detector,
                dataset=val_dataset,
                emb_space=embeddings,
            )
            ranking_target = [1 if x in true_errors else 0 for x in ranking]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                show_scores=False,
                show_plots=False,
                wandb_cat=f"IrrelevantSamples/{name_detector}-SSL/",
                **kwargs,
            )

    def run_duplicate_baselines(
        self,
        val_dataset: Union[Dataset, BaseContaminationDataset, MixedAugmentationDataset],
        true_errors: Union[np.ndarray, list],
        **kwargs,
    ):
        for name_detector, dup_detector in self.duplicate_baselines:
            _dataset = copy.deepcopy(val_dataset)
            _dataset.transform = None
            if isinstance(_dataset, MixedAugmentationDataset):
                _dataset.irrelevant_dataset.transform = None
                _dataset.duplicate_dataset.transform = None
                _dataset.label_error_dataset.transform = None
            l_ranking = dup_detector.get_ranking(
                dataset=_dataset,
                hash_size=8,
                n_processes=os.cpu_count() * 2,
            )
            ranking_target = [
                1 if (int(x[0]), int(x[1])) in true_errors else 0 for x in l_ranking
            ]
            calculate_scores_from_ranking(
                ranking=ranking_target,
                log_wandb=True,
                show_scores=False,
                show_plots=False,
                wandb_cat=f"NearDuplicates/{name_detector}/",
                **kwargs,
            )

    def run_label_error_baselines(
        self,
        val_dataset: Union[Dataset, BaseContaminationDataset, MixedAugmentationDataset],
        embeddings: torch.Tensor,
        lbls: torch.Tensor,
        true_errors: Union[np.ndarray, list],
        **kwargs,
    ):
        for name_detector, lbl_detector in self.label_error_baselines:
            _dataset = copy.deepcopy(val_dataset)
            _dataset.transform = None
            if isinstance(_dataset, MixedAugmentationDataset):
                _dataset.irrelevant_dataset.transform = None
                _dataset.duplicate_dataset.transform = None
                _dataset.label_error_dataset.transform = None
                label_classes = _dataset.label_error_classes
            else:
                label_classes = _dataset.classes

            def _run_ranking(name_detector, ranking_kwargs, score_kwargs):
                ranking = lbl_detector.get_ranking(**ranking_kwargs)
                if type(ranking[0]) is tuple:
                    ranking = [x[1] for x in ranking]
                ranking_target = [int(true_errors[i]) for i in ranking]
                calculate_scores_from_ranking(
                    ranking=ranking_target,
                    log_wandb=True,
                    show_scores=False,
                    show_plots=False,
                    wandb_cat=f"LabelErrors/{name_detector}/",
                    **score_kwargs,
                )

            # check the signature and do either (INet or SSL)
            parameters = inspect.signature(lbl_detector.get_ranking).parameters.keys()
            if (
                "dataset" in parameters
                and "emb_space" in parameters
                and "labels" in parameters
            ):
                # representation-based
                # here we need to run with INet and SSL representations
                _run_ranking(
                    f"{name_detector}-SSL",
                    ranking_kwargs={
                        "dataset": _dataset,
                        "emb_space": embeddings.numpy(),
                        "labels": lbls.numpy(),
                        "label_classes": label_classes,
                    },
                    score_kwargs=kwargs,
                )
                _run_ranking(
                    f"{name_detector}-INet",
                    ranking_kwargs={
                        "dataset": val_dataset,
                        "label_classes": label_classes,
                        "ssl_model": "imagenet_vit_tiny",
                    },
                    score_kwargs=kwargs,
                )
                _run_ranking(
                    f"{name_detector}-INetSSL",
                    ranking_kwargs={
                        "dataset": val_dataset,
                        "label_classes": label_classes,
                        "ssl_model": "imagenet_dino",
                    },
                    score_kwargs=kwargs,
                )
            elif "dataset" in parameters:
                # image-based
                # here we need to run only once (image domain)
                _run_ranking(
                    name_detector,
                    ranking_kwargs={
                        "dataset": _dataset,
                        "emb_space": embeddings.numpy(),
                        "labels": lbls.numpy(),
                        "label_classes": label_classes,
                    },
                    score_kwargs=kwargs,
                )
