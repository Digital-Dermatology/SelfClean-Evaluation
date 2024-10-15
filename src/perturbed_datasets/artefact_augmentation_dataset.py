import copy
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from ssl_library.src.datasets.base_dataset import BaseDataset
from torchvision import transforms

from ..perturbed_datasets.base_contamination_dataset import BaseContaminationDataset
from ..utils.utils_pil import get_concat_h, get_concat_v


class ArtefactAugmentationDataset(BaseContaminationDataset):
    def __init__(
        self,
        dataset: BaseDataset,
        frac_error: float = 0.1,
        n_errors: Optional[int] = None,
        transform=None,
        add_watermark: bool = True,
        add_colorbar: bool = True,
        add_mosaic: bool = True,
        watermark_prob: float = 0.5,
        colorbar_prob: float = 0.5,
        mosaic_prob: float = 0.5,
        watermark_path: str = "../assets/uni-basel-logo.png",
        watermark_text: str = "CONFIDENTIAL",
        watermark_frac: float = 0.5,
        colorbar_path: str = "../assets/colorbar.png",
        mosaic_scale: float = 0.5,
        grayscale: bool = False,
        name: Optional[str] = None,
        deterministic: bool = True,
        reference_image_size: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.add_watermark = add_watermark
        self.add_colorbar = add_colorbar
        self.add_mosaic = add_mosaic
        self.watermark_prob = watermark_prob
        self.colorbar_prob = colorbar_prob
        self.mosaic_prob = mosaic_prob
        self.watermark_path = watermark_path
        self.watermark_text = watermark_text
        self.watermark_frac = watermark_frac
        self.colorbar_path = colorbar_path
        self.mosaic_scale = mosaic_scale
        self.grayscale = grayscale
        self.name = name if name is not None else type(self).__name__
        self.dataset_orig = copy.deepcopy(dataset)
        self.dataset_aug = copy.deepcopy(dataset)
        # by setting the transform to `None` we obtain the raw PIL image
        self.dataset_orig.transform = None
        self.dataset_aug.transform = None
        self.transform = transform
        self.reference_image_size = reference_image_size
        # random sample to augment
        if n_errors is not None:
            n_samples = n_errors
        else:
            n_samples = math.ceil(frac_error * len(self.dataset_orig))
        idx_range = np.arange(0, len(self.dataset_orig))
        self.list_aug_idx = np.random.choice(idx_range, size=n_samples, replace=False)
        # transform the random sample
        if deterministic:
            self.tmp_aug_path = Path(tempfile.mkdtemp())
            self.aug_samples = []
            for idx in self.list_aug_idx:
                rets = self.dataset_aug[idx]
                image = rets[0]
                image = self._apply_augmentation(image)
                img_path = self.tmp_aug_path / f"{idx}.jpg"
                image.save(img_path)
                rets = (img_path,) + rets[1:]
                self.aug_samples.append(rets)
        # global configs
        self.deterministic = deterministic
        self.classes = self.dataset_orig.classes
        self.n_classes = self.dataset_orig.n_classes

    def __len__(self):
        return len(self.dataset_orig) + len(self.list_aug_idx)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < len(self.dataset_orig):
            rets = self.dataset_orig[idx]
            image = rets[0]
        else:
            idx = idx - len(self.dataset_orig)
            if self.deterministic:
                rets = self.aug_samples[idx]
                image = Image.open(rets[0])
                image = image.convert("RGB")
            else:
                idx = self.list_aug_idx[idx]
                rets = self.dataset_aug[idx]
                image = rets[0]
                image = self._apply_augmentation(image)
        if self.transform is not None:
            image = self.transform(image)
        rets = (image,) + rets[1:]
        return rets

    def cleanup(self):
        if self.deterministic:
            shutil.rmtree(self.tmp_aug_path)

    def info(self):
        logger.info(f"Name of dataset: {self.name}")
        logger.info(f"Original datset size: {len(self.dataset_orig)}")
        logger.info(f"Augmented datset size: {len(self.list_aug_idx)}")
        logger.info(f"Combined datset size: {len(self)}")

    def add_watermark_to_image(self, image: Image.Image) -> Image.Image:
        # add watermark image
        watermark_img = self._load_watermark(watermark_path=self.watermark_path)
        watermark_size = (
            int(watermark_img.size[0] * self.watermark_frac),
            int(watermark_img.size[1] * self.watermark_frac),
        )
        watermark_img = watermark_img.resize(watermark_size)
        rand_x = image.size[0] - watermark_img.size[0]
        rand_y = image.size[1] - watermark_img.size[1]
        if rand_x < 0:
            rand_x = 0
        if rand_y < 0:
            rand_y = 0
        x = random.randint(0, rand_x)
        y = random.randint(0, rand_y)
        image.paste(watermark_img, (x, y), watermark_img)
        # add watermark text
        draw = ImageDraw.Draw(image)
        font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
        font_size = ArtefactAugmentationDataset.find_font_size(
            self.watermark_text,
            font_path,
            image,
            0.8,
        )
        font = ImageFont.truetype(font_path, font_size)
        observed_width, observed_height = ArtefactAugmentationDataset.get_text_size(
            self.watermark_text,
            image,
            font,
        )
        x = random.randint(0, int(image.size[0] - observed_width))
        y = random.randint(0, int(image.size[1] - observed_height))
        draw.text((x, y), self.watermark_text, font=font)
        return image

    def add_colorbar_to_image(self, image: Image.Image) -> Image.Image:
        colorbar_img = Image.open(self.colorbar_path)
        ratio_resize = image.size[-1] / colorbar_img.size[1]
        new_size = (
            int(colorbar_img.size[0] * ratio_resize),
            int(colorbar_img.size[1] * ratio_resize),
        )
        colorbar_img = colorbar_img.resize(new_size)
        image = get_concat_h(image, colorbar_img)
        return image

    def add_mosaic_to_image(self, image: Image.Image) -> Image.Image:
        rand_idx = random.randint(0, len(self.dataset_aug) - 1)
        rand_img = self.dataset_aug[rand_idx][0]
        rand_img = transforms.Resize((256, 256))(rand_img)
        rand_scale = (
            int(rand_img.size[0] * self.mosaic_scale),
            int(rand_img.size[1] * self.mosaic_scale),
        )
        rand_img = rand_img.resize(rand_scale)
        mosaic_img = get_concat_v(image, rand_img, spacing=10)
        mosaic_img = get_concat_h(mosaic_img, rand_img, spacing=10)
        return mosaic_img

    def _load_watermark(self, watermark_path):
        watermark = Image.open(watermark_path)
        # replace all black pixels with white ones
        data = np.array(watermark)
        red, green, blue, alpha = data.T
        black_areas = (red == 0) & (blue == 0) & (green == 0) & (alpha > 0)
        data[..., :-1][black_areas.T] = (255, 255, 255)
        watermark = Image.fromarray(data)
        return watermark

    def _apply_augmentation(self, image):
        image = transforms.Resize(size=self.reference_image_size)(image)
        if self.add_watermark:
            random_apply = np.random.uniform(0.0, 1.0)
            if random_apply <= self.watermark_prob:
                image = self.add_watermark_to_image(image=image)
        if self.add_colorbar:
            random_apply = np.random.uniform(0.0, 1.0)
            if random_apply <= self.watermark_prob:
                image = self.add_colorbar_to_image(image=image)
        if self.add_mosaic:
            random_apply = np.random.uniform(0.0, 1.0)
            if random_apply <= self.mosaic_prob:
                image = self.add_mosaic_to_image(image=image)
        if self.grayscale:
            image = transforms.Grayscale(num_output_channels=3)(image)
        return image

    def get_errors(self):
        l_errors = []
        for idx in range(len(self)):
            if idx >= len(self.dataset_orig):
                dup_idx = idx
                idx = idx - len(self.dataset_orig)
                orig_idx = self.list_aug_idx[idx]
                l_errors.append((orig_idx, dup_idx))
        return set(l_errors), ["original", "duplicate"]

    @staticmethod
    def find_font_size(text, font, image, target_width_ratio):
        tested_font_size = 100
        tested_font = ImageFont.truetype(font, tested_font_size)
        observed_width, observed_height = ArtefactAugmentationDataset.get_text_size(
            text, image, tested_font
        )
        estimated_font_size = (
            tested_font_size / (observed_width / image.width) * target_width_ratio
        )
        return round(estimated_font_size)

    @staticmethod
    def get_text_size(text, image, font):
        im = Image.new("RGB", (image.width, image.height))
        draw = ImageDraw.Draw(im)
        return draw.textsize(text, font)
