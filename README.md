# SelfClean - Evaluation
This repository contains the code to reproduce all evaluations in the paper "Intrinsic Self-Supervision for Data Quality Audits".
It builds on the SelfClean package by including evaluation scenarios and competing methods.

Context-aware self-supervised learning combined with distance-based indicators is very effective to identify data quality issues in computer-vision datasets.

## Usage
Run `make` for a list of possible targets.

## Installation
Run this command for installation
`make install`

## Reproducibility of the Paper
To reproduce our experiments, we list the detailed comments needed for replicating each experiment below.
Note that our experiments were run on a DGX Workstation 1.
If less computational power is available, this would require adaptations of the configuration file.

### Experiment 5.1: Synthetic contamination

Comparison on data quality issues (i.e. Table 1 and Table 10):
> python -m src.evaluate_synthetic --config configs/evaluation.yaml

Influence of contamination (i.e. Figure 3):
> python -m src.evaluate_mixed_contamination --config configs/evaluation.yaml

### Experiment 5.2: Natural contamination

Comparison with metadata (i.e. Table 11):
> python -m src.evaluate_metadata --config configs/metadata_comparison.yaml

Comparison with human annotators (i.e. Table 15 and Figure 9):
> python -m src.evaluate_human_annotators

### Experiment 5.3: Ablation studies

Influence of contamination (i.e. Table 2, 6, 7, 8, 9 and Figure 4, 5):
> python -m src.evaluate_mixed_contamination --config configs/evaluation.yaml

*Note:* With changes to the `configs/evaluation.yaml`, namely `pretraining_type` and `model_config`.

### Discussion 6

Influence of dataset cleaning (i.e. Table 3, 13):
> python -m src.cleaning_performance --config configs/cleaning_performance.yaml

## Code and test conventions
- `black` for code style
- `isort` for import sorting
- docstring style: `sphinx`
- `pytest` for running tests

### Development installation and configurations
To set up your dev environment run:
```bash
pip install -r requirements.txt
# Install pre-commit hook
pre-commit install
```
To run all the linters on all files:
```bash
pre-commit run --all-files
```
