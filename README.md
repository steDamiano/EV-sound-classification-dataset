# Data Generation for Emergency Sound Identification Using a Small-Scale Convolutional Neural Network
This repository contains the code for the data generation procedure proposed in the paper
>S. Damiano, A. Guntoro and T. van Waterschoot, "Emergency Sound Identification for Automotive Applications Using a Small-Scale Convolutional Neural Network", 2023 (under review)


## Description
This respository contains the code to generate a dataset containing audio clips of emergency vehicle sounds for audio classification purpose. The synthetic data generation procedure is build on top of the *pyroadacoustics* [1,2] simulator, that is used to simulate acoustic propagation in a traffic scenario for both static and dynamic sound sources.

The acoustic scene, defined in `simulator.py`, contains a mixture of static and dynamic noise sources randomply located (and moving) within a circle with radius 100m, and one sound source emitting an emergency signal (i.e. either a car honking or a siren) moving on a random trajectory within the same area. The emergency sound and the background noise are summed with an SNR in the range $[-30, 0]\,\mathrm{dB}$.

The repository contains three scripts:
- `create_folder_structure.py` contains the code to generate the required folder in which the input and output files will be stores
- `trajectory_generator.py` contains the code to generate rectilinear or Bezier trajectories used to simulate moving sources
- `simulator.py` contains the code to build the acoustic scene and run the simulations to obtain the audio samples, and is the main access point to the package.

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
- `batch_num`: number of samples batch to be generated (to generate sample batches in parallel, default: 0)
- `num_samples`: number of samples to generate (default: 1)
- `dataset_split`: can be `traindev` or `test` (default: traindev)
- `event_class`: class of samples generated in current batch. Can be noise, horn, or siren (default: noise)
- `sample_duration`: duration in seconds of each simulated datapoint (default: 1s)
- `sample_rate`: sample rate used in the simulations (default: 16000Hz)

## References
>[1] S. Damiano and T. van Waterschoot, “Pyroadacoustics: a Road Acoustics Simulator Based on Variable Length Delay Lines,” in *Proc. 25th Int. Conf. Digital Audio Effects (DAFx20in22)*, Vienna, Austria, Sept. 2022, pp. 216–223.

>[2] https://github.com/steDamiano/pyroadacoustics 
