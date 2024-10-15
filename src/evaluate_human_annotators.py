import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import scipy

sys.path.append("../")

from ssl_library.src.utils.utils import p_value_stars


def mwu_test(X, Y):
    mw_test = scipy.stats.mannwhitneyu(
        X, Y, alternative="greater", use_continuity=False
    )
    U = mw_test.statistic
    r_b = 2 * U / (len(X) * len(Y))

    print(mw_test, mw_test.pvalue, p_value_stars(mw_test.pvalue), r_b)


def stat_test(max_bias_samples, min_bias_samples):
    # MWU test (Max. vs Min. bias)
    X = [1 if x == "Yes" else 0 for x in list(max_bias_samples["label"])]
    Y = [1 if x == "Yes" else 0 for x in list(min_bias_samples["label"])]
    print(f"Low 50: {round((np.sum(X) / len(X)) * 100, 1)}")
    print(f"Random: {round((np.sum(Y) / len(Y)) * 100, 1)}")
    mwu_test(X, Y)
    print("-" * 50)

    # MWU test (Max 0-25 vs. 26-50)
    max_X = [1 if x == "Yes" else 0 for x in list(max_bias_samples["label"])]
    print(f"Low 0-25: {round((np.sum(max_X[:25]) / len(max_X[:25])) * 100, 1)}")
    print(f"Low 26-50: {round((np.sum(max_X[25:]) / len(max_X[25:])) * 100, 1)}")
    mwu_test(max_X[:25], max_X[25:])
    return X, Y


def visualize_anno(l_vis_errors, l_vis_names, dataset_name):
    rename_dict = {
        "IrrelevantSamples": "Off-topic Samples",
        "NearDuplicates": "Near Duplicates",
        "LabelErrors": "Label Errors",
    }

    with plt.style.context(["science", "std-colors", "grid"]):
        fig, ax = plt.subplots(figsize=(4, 3))
        for i in range(len(l_vis_errors)):
            plt.plot(
                l_vis_errors[i][:, 0],
                l_vis_errors[i][:, 1],
                label=rename_dict.get(l_vis_names[i]),
            )
        plt.ylim(-5.0, 105.0)
        plt.xlabel("Rolling Window Center")
        plt.ylabel("Percentage of Errors (\%)")
        plt.legend()
        # ax.set_xticks(ticks=l_vis_errors[0][:, 0][::10], labels=tick_labels)
        plt.savefig(
            save_fig_path / f"{dataset_name}_HumanValidation.pdf",
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    save_fig_path = Path("assets/notebook_outputs/")

    annotation_dict = {
        "DDI": {
            "IrrelevantSamples": "DDI - Irrelevant Samples",
            "NearDuplicates": "DDI - Near Duplicates",
            "LabelErrors": "DDI - Label Errors",
        },
        "Fitzpatrick17k": {
            "IrrelevantSamples": "fitzpatrick17k - Irrelevant Samples",
            "NearDuplicates": "fitzpatrick17k - Near Duplicates",
            "LabelErrors": "fitzpatrick17k - Label Errors",
        },
    }

    base_anno_path = Path("assets/annotations/SelfClean")
    for dataset_name, values in annotation_dict.items():
        print(dataset_name)

        l_vis_errors = []
        l_vis_names = []

        for error_type, file_stem in values.items():
            df = pd.read_csv(base_anno_path / file_stem / "data_sampled_new.csv")
            label_cols = [
                "XXX_label",
                "XXX_label",
                "XXX_label",
                "XXX_label",
                "XXX_label",
            ]

            df["label"] = df[label_cols].apply(
                lambda row: (
                    row.value_counts().index[0]
                    if len(row.value_counts()) > 0 and row.notna().sum() > 0
                    else np.nan
                ),
                axis=1,
            )

            print("*" * 20 + f" {error_type} ({len(df['label'])}) " + "*" * 20)

            # get the minimum and maximum bias
            max_bias_samples = df[
                (df["sampling_type"] == "maximum_bias") & df["label"].notna()
            ].sort_values(by="Ranking")[:50]
            min_bias_samples = df[
                (df["sampling_type"] == "minimum_bias") & df["label"].notna()
            ].sort_values(by="Ranking")[:50]

            # calculate the tests
            X, Y = stat_test(max_bias_samples, min_bias_samples)
            print()

            # results for visualisation
            XY = X + Y
            l_errors = []
            for i in range(11, len(XY) + 1):
                l_errors.append(
                    [i, (np.sum(XY[i - 11 : i]) / len(XY[i - 11 : i])) * 100]
                )
            l_errors = np.asarray(l_errors)
            l_errors[:, 0] = np.arange(5, 5 + len(l_errors))
            l_vis_errors.append(l_errors)
            l_vis_names.append(error_type)

        # Visualize the annotation
        visualize_anno(l_vis_errors, l_vis_names, dataset_name)

    annotation_dict = {
        "ImageNet-1k": {
            "IrrelevantSamples": "order_441105___completed_data.csv",
            "NearDuplicates": "order_440803___all_data.csv",
            "LabelErrors": "order_441241___completed_data.csv",
        },
        "Food101N": {
            "IrrelevantSamples": "order_441505___all_data.csv",
            "NearDuplicates": "order_441501___all_data.csv",
            "LabelErrors": "order_441499___all_data.csv",
        },
    }

    base_anno_path = Path("assets/annotations/SelfClean")
    base_sample_path = Path("../med_clean_verification_tool/data/")
    for dataset_name, values in annotation_dict.items():
        print(dataset_name)

        l_vis_errors = []
        l_vis_names = []

        for error_type, file_stem in values.items():
            anno_path = base_anno_path / dataset_name / error_type / file_stem
            sample_path = (
                base_sample_path
                / f"{dataset_name}_{error_type}"
                / "data_sampled_new.csv"
            )

            df_sampled = pd.read_csv(anno_path)
            df_sampled.set_index("ID", inplace=True)

            df = pd.read_csv(sample_path)
            df = df.merge(df_sampled, left_index=True, right_index=True)
            df.rename(
                columns={"Does the image reflect a label error?": "label"}, inplace=True
            )
            df.rename(
                columns={"Do the images reflect an irrelevant sample?": "label"},
                inplace=True,
            )
            df.rename(
                columns={"Do the images reflect a near-duplicate?": "label"},
                inplace=True,
            )

            print("*" * 20 + f" {error_type} ({len(df['label'])}) " + "*" * 20)

            # get the minimum and maximum bias
            max_bias_samples = df[
                (df["sampling_type"] == "maximum_bias") & df["label"].notna()
            ].sort_values(by="Ranking")[:50]
            min_bias_samples = df[
                (df["sampling_type"] == "minimum_bias") & df["label"].notna()
            ].sort_values(by="Ranking")[:50]

            # calculate the tests
            X, Y = stat_test(max_bias_samples, min_bias_samples)
            print()

            # results for visualisation
            XY = X + Y
            l_errors = []
            for i in range(11, len(XY) + 1):
                l_errors.append(
                    [i, (np.sum(XY[i - 11 : i]) / len(XY[i - 11 : i])) * 100]
                )
            l_errors = np.asarray(l_errors)
            l_errors[:, 0] = np.arange(5, 5 + len(l_errors))
            l_vis_errors.append(l_errors)
            l_vis_names.append(error_type)

        # Visualize the annotation
        visualize_anno(l_vis_errors, l_vis_names, dataset_name)
