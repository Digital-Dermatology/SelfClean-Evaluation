import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from selfclean.src.cleaner.issue_manager import IssueManager
from ssl_library.src.utils.logging import denormalize_image
from torch.utils.data import Dataset
from torchvision import transforms


def calc_row_idx(k, n):
    return int(
        math.ceil(
            (1 / 2.0) * (-((-8 * k + 4 * n**2 - 4 * n - 7) ** 0.5) + 2 * n - 1) - 1
        )
    )


def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) // 2


def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k, n, progress_bar=None):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    if progress_bar is not None:
        progress_bar.update(1)
    return i, j


def actual_indices(idx, n):
    n_row_elems = np.cumsum(np.arange(1, n)[::-1])
    ii = (n_row_elems[:, None] - 1 < idx[None, :]).sum(axis=0)
    shifts = np.concatenate([[0], n_row_elems])
    jj = np.arange(1, n)[ii] + idx - shifts[ii]
    if np.sum(ii < 0) > 0 or np.sum(jj < 0) > 0:
        print(f"Negative indices")
    return ii, jj


def has_same_label(arr):
    arr = np.array(arr)
    result = arr[:, None] == arr
    return result


def near_duplicate_plot(
    issues: IssueManager,
    dataset: Dataset,
    plot_top_N: int,
    return_fig: bool = False,
):
    fig, ax = plt.subplots(2, plot_top_N, figsize=(10, 3))
    nd_indices = issues.get_issues("near_duplicates")["indices"][:plot_top_N]
    nd_scores = issues.get_issues("near_duplicates")["scores"][:plot_top_N]
    for i, ((idx1, idx2), score) in enumerate(zip(nd_indices, nd_scores)):
        ax[0, i].imshow(
            transforms.ToPILImage()(denormalize_image(dataset[int(idx1)][0]))
        )
        ax[1, i].imshow(
            transforms.ToPILImage()(denormalize_image(dataset[int(idx2)][0]))
        )
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[0, i].set_title(f"Pair: ({int(idx1)}, {int(idx2)})", fontsize=10)
        ax[1, i].set_title(f"Score: {score:.3f}", fontsize=10)
    fig.tight_layout()
    if return_fig:
        return fig
    plt.show()
    del fig


def irrelevant_samples_plot(
    issues: IssueManager,
    dataset: Dataset,
    lbls: torch.Tensor,
    plot_top_N: int,
    class_labels: Optional[list] = None,
    return_fig: bool = False,
):
    fig, ax = plt.subplots(1, plot_top_N, figsize=(10, 3))
    for i, idx in enumerate(issues.get_issues("irrelevants")["indices"][:plot_top_N]):
        ax[i].imshow(transforms.ToPILImage()(denormalize_image(dataset[idx][0])))
        if class_labels is not None:
            ax[i].set_title(
                f"label: {class_labels[lbls[idx]]}\nidx: {idx}",
                fontsize=10,
            )
        else:
            ax[i].set_title(f"idx: {idx}", fontsize=10)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    fig.tight_layout()
    if return_fig:
        return fig
    plt.show()
    del fig


def label_errors_plot(
    issues: IssueManager,
    lbls: torch.Tensor,
    class_labels: list,
    dataset: Dataset,
    plot_top_N: int,
    errors: Optional[list] = None,
    return_fig: bool = False,
):
    fig, ax = plt.subplots(1, plot_top_N, figsize=(10, 3))
    for i, idx in enumerate(issues.get_issues("label_errors")["indices"][:plot_top_N]):
        ax[i].imshow(transforms.ToPILImage()(denormalize_image(dataset[idx][0])))
        title = f"label: {class_labels[lbls[idx]]}\nidx: {idx}"
        if errors is not None:
            title += f"\nerror: {bool(errors[idx])}"
        ax[i].set_title(title)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    fig.tight_layout()
    if return_fig:
        return fig
    plt.show()
    del fig
