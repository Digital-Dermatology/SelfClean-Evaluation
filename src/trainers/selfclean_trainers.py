from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from selfclean.src.cleaner.base_cleaner import BaseCleaner
from ssl_library.src.trainers.byol_trainer import BYOLTrainer
from ssl_library.src.trainers.dino_trainer import DINOTrainer
from ssl_library.src.trainers.mae_trainer import MAETrainer
from ssl_library.src.trainers.simclr_trainer import SimCLRTrainer
from torch.utils.data import DataLoader

from src.trainers.online_cleaning_mixin import OnlineCleaningMixin, OnlineEvalType


def create_custom_trainer_class(trainer_class):
    class BaseSelfCleanTrainer(trainer_class, OnlineCleaningMixin):
        def __init__(
            self,
            train_dataset: DataLoader,
            config: dict,
            val_dataset: Optional[DataLoader] = None,
            config_path: Optional[Union[str, Path]] = None,
            additional_run_info: str = "",
            additional_arch_info: str = "",
            print_model_summary: bool = False,
            wandb_logging: bool = True,
            wandb_project_name="SelfClean",
            # online evaluation mixin
            cleaner_cls: Optional[BaseCleaner] = None,
            online_eval_type: Optional[OnlineEvalType] = None,
            # baselines
            duplicate_baselines: list = [],
            irrelevant_baselines: list = [],
            label_error_baselines: list = [],
        ):
            trainer_class.__init__(
                self,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=config,
                config_path=config_path,
                additional_run_info=additional_run_info,
                additional_arch_info=additional_arch_info,
                print_model_summary=print_model_summary,
                wandb_logging=wandb_logging,
                wandb_project_name=wandb_project_name,
            )
            OnlineCleaningMixin.__init__(
                self,
                cleaner_cls=cleaner_cls,
                online_eval_type=online_eval_type,
                duplicate_baselines=duplicate_baselines,
                irrelevant_baselines=irrelevant_baselines,
                label_error_baselines=label_error_baselines,
                wandb_logging=wandb_logging,
            )

        def _log_embeddings(
            self,
            model: torch.nn.Module,
            patch_size: Optional[int] = None,
            n_items: Optional[int] = 3_000,
            log_self_attention: bool = False,
            log_mae: bool = False,
            return_embedding: bool = False,
            **kwargs,
        ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
            ret = super()._log_embeddings(
                model=model,
                patch_size=patch_size,
                n_items=n_items,
                log_self_attention=log_self_attention,
                log_mae=log_mae,
                return_embedding=True,
            )
            if ret is not None:
                embeddings, imgs, lbls, paths = ret
                self.online_cleaning_evaluation(
                    val_dataset=self.val_dataset.dataset,
                    embeddings=embeddings,
                    imgs=imgs,
                    lbls=lbls,
                    paths=paths,
                )

    return BaseSelfCleanTrainer


def create_dino_trainer_class():
    return create_custom_trainer_class(trainer_class=DINOTrainer)


def create_simclr_trainer_class():
    return create_custom_trainer_class(trainer_class=SimCLRTrainer)


def create_mae_trainer_class():
    return create_custom_trainer_class(trainer_class=MAETrainer)


def create_byol_trainer_class():
    return create_custom_trainer_class(trainer_class=BYOLTrainer)
