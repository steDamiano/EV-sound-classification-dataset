# Data Generation for Emergency Sound Identification Using a Small-Scale Convolutional Neural Network
This repository contains the code for the data generation procedure proposed in the paper
>S. Damiano, A. Guntoro and T. van Waterschoot, "Emergency Sound Identification for Automotive Applications Using a Small-Scale Convolutional Neural Network", 2023 (under review)

## Folder structure
To correctly use the script, follow the pre-defined folder structure of the `Data/` directory.

### Input files
Input audio clips can be divided into `traindev` and `test` folders to create two independent datasets:
- `traindev` is used for training and tuning the CNN. Typically, it can be then further split into training and validation sets
- `test` is used to evaluate the ability of the network to generalize to unseen horn/siren sounds.

The name of the files contained in the `siren` and `horn` subdirectories should contain the corresponding label (`horn` for the horn class, `wail`, `yelp` or `hilo` for the siren class).

### Simulated dataset
The generated samples will be stored according to the folder structure defined in `Data/dataset`. The `traindev` samples will be generated using the `traindev` input files, and the `test` samples using the `test` input files.

## Usage
To generate data store the samples to be used as input for the simulations in the `Data/input_files` directory and run the `simulator.py` script. The script accepts the following parameters:
- `batch_num`: number of samples batch to be generated (to generate sample batches in parallel)
- `num_samples`: number of samples to generate
- `dataset_split`: can be `traindev` or `test`
- `sample_duration`: duration in seconds of each simulated datapoint (default: 1s)
- `sample_rate`: sample rate used in the simulations (default: 16000Hz)
