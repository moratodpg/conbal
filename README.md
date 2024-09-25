
# ConBatch-BAL: Batch Bayesian Active Learning under Budget Constraints

This repository contains the code for running **active learning** experiments under **batch budget constraints**. 

In ConBatch-BAL, you can:

- Reproduce the results presented in the paper.
- Test ConBatch BAL strategies on a the available real-world building datasets or a dataset of your choice.
- Develop a new active learning strategy and test it against ConBatch BAL strategies or the implemented random selection baseline.

## Installation
The code has been implemented and tested on Python 3.9. Please install the requirements file to include the required dependencies.

```bash
pip install -r requirements.txt
```

## Usage
You can run the code by executing:

```bash
python main.py <config_file> <custom_name> --use_wandb
```

Example:

```bash
python main.py config_file custom_name
```
Parameters:
- `config_file`: This parameters points to the input yaml configuration file associated with the experiment.
- `custom_name`: Define a name for the experiment.
- `--use_wandb`: Optional flag to activate experiment tracking via wandb.

## File Structure
```
├── main.py                 # Main script for running experiments
├── models/                 # Model architectures and trainer
├── datasets/               # Dataset files
├── config/                 # Input configuration files
├── active_learning/        # Acquisition functions
├── results/                # Output results and logs
├── LICENSE                 # License
└── README.md               # This file!
```

## Results
Results will be saved in the `results/` folder. Each subfolder here corresponds to an experiment set: dataset_configuration (e.g. build6k_distance => dataset build6k, configuration distance).

Inside each experiment set, directories are named based on the strategy_budget_seed_date. For instance, greedy_01_1_2024_07_25_16_32_00 corresponds to constrained greedy batch-BAL under 100 m batch constraint, seed 1, and experiment date.

Within each specific experiment, a yaml config file is included with all parameters needed to run again the experiment and an json output file that contains the results. 

The json output file contains a dictionary categorized by the following keys and values:
- `accuracy_test`: accuracy achieved on the test set over active learning iterations.
- `cost`: acquisition cost incurred over active learning iterations.
- `idx_train`: Nested list where each sublist contains the indices of the samples acquired at each active learning iteration.
- `idx_test`: Indices of the samples on the test set.
- `idx_pool`: Indices of the samples that remained on the pool set after the experiment concluded.

## How to reproduce the results from the paper?

1. Download the dataset(s) torch files and the stored results of the experiment(s) you would like to reproduce

You can manually download the datasets and stored experiments via the following anonymous link: 

[https://zenodo.org/records/13839235?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImVkOTJhOTAxLWQ2NTAtNGZiYy05ZmFiLWFlYTBjOWRiZTU1YyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZDhiNzI1MTBmMTlmZmI4NTEwNDA4YWNiYjljMWI2ZSJ9.82fnsTZRRD1SRzcdMKlMQ7WrLexUUeaitZkDf5H0zKte1YMP7rx4rQ8UH_j6niXBtlNW4w6g2qCKkA27yiz2Lg](https://zenodo.org/records/13839235?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImVkOTJhOTAxLWQ2NTAtNGZiYy05ZmFiLWFlYTBjOWRiZTU1YyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZDhiNzI1MTBmMTlmZmI4NTEwNDA4YWNiYjljMWI2ZSJ9.82fnsTZRRD1SRzcdMKlMQ7WrLexUUeaitZkDf5H0zKte1YMP7rx4rQ8UH_j6niXBtlNW4w6g2qCKkA27yiz2Lg)

Or, alternatively, you can download it through wget:

2. Add the dataset(s) 

Include the dataset torch files contained in the folder `constrained_batchBAL_datasets/benchmark_torch_files` to the directory [datasets/](datasets/).

3. Add the config file(s) corresponding to the experiment(s) you would like to reproduce

Include the input config files associated with the experiments you would like to reproduce. All results are included in the folder `constrained_batchBAL_datasets/benchmark_results`. Add the specific config file(s) to the directory [config/](config/). Feel free to modify the name of the config files to avoid duplicates.

> Please note that the config file already contains the random seed used for the experiment.

4. Run the experiment(s) 

Following the instructions described in the [Usage](#usage) section.

Example:

```bash
python main.py config custom_name
```

5. Check the results

The json file containing the results will be stored in the folder [results/json/](results/json). Note this folder will be automatically created once you start running the experiment(s).

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.