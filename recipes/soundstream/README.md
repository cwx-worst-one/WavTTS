<<<<<<< HEAD
# Template Project

This is a template project for `samantha` repository. This template is for
projects that won't be deployed via `sami_engine`, so we don't need `BaseStage` and
`ModlePipeline` to implement our models.

When starting a new project:

1. Copy this project and rename it to the name of your project.
2. Update the `README.md` file with the description of your project.
3. Implement and test your datasets in `dataset/` folder.
4. Implement and test your models in `models/` folder.
5. Implement `LightningDataModule` and `LightningModule` in `modules/` folder.
6. Implement and test `conf/default.yaml` with your classes and configurations.
7. Implement customized preparation scripts in `bootstrap.sh`. (optional)
=======
# Soundstream

TBD
>>>>>>> 9934dae25f0f4870b67e5be86ffed69ff69e5cdb

## Introduction

<!-- Briefly describe your project. -->

## Prerequisites

<!-- List the prerequisites for your project and installation instructions. -->

## Running trials

<!-- Describe how to run trials for your project, e.g. the command line. -->

## Benchmarks

This part is a changelog of the benchmarks of your project. You can use the following template to add a new benchmark.

It's recommended to record your benchmarks every time you make a significant change to your project, e.g. a new model, a new dataset, a new feature, etc.

### Model Benchmark

<!-- Record any metrics you want to track. -->

| | Train Loss | Val Loss | Val Accuracy | Epochs |
|-|------|------|------|-----|
| First verion | | | | |

### Resource Benchmark

<!-- Record the resource usage of your project. -->

| | GPUs | GPU Memory | Epoch Time | Step Time | Total Epochs |
|-|------|------|------|------|-----|
| First version | | | | | |
