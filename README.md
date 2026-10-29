# Low Variance Importance Sampling for SDE

This repository contains the code for the paper "Low Variance Importance Sampling for Discretely Observed Stochastic Differential Equations" by authors Karl Hammar, Moritz Schauer, Adam Andersson and Benjamin Svedung Wettervik.

# Installation

Create a Python environment with the usual scientific stack (torch, numpy, scipy, matplotlib...). Install dependencies as needed for your system.

# Configuration layout

Configs live in `configs/` and are composed as follows:

1. `configs/nn_configs.json`  
   Neural network architecture blocks used by methods (e.g., `multi_default`).

2. `configs/method_configs.json`  
   Method hyperparameters only (optimizer, schedules, training iterations, etc.).  

3. `configs/ssm_configs.json`  
   State-space model bundle containing:
   - `sde_config` key into `configs/sde_configs.json`
   - `meas_config` key into `configs/meas_configs.json`
   - SSM overhead: `dt`, `deltaT_meas`, `num_meas`

4. `configs/master_configs.json`  
   The top-level config that references:
   - `nn_config`
   - `method`
   - `ssm_config`


# Training

Default training uses the `default` master config:

```bash
python train.py --master_config default
```

Override components on the command line:

```bash
python train.py --master_config default --method_config WLogVar
python train.py --master_config default --ssm_config default
python train.py --master_config default --sde_config ou_1d --meas_config normal_1d_std0p125
```

Output is written to `models/<name>/` and includes the run parameters in `models/<name>/params.json`.

# Testing

Testing loads a trained model by name:

```bash
python test.py --model-name <run_name>
```

Compare multiple trained models:

```bash
python test.py --model-name <run_name> --extra-model-names <run_name_2> <run_name_3>
```

Choose which test to run:

```bash
python test.py --model-name <run_name> --test-type smoothing
python test.py --model-name <run_name> --test-type twisting
```

If `--test-name` is provided, results are written to `results/<test-type>/<test-name>/`. Otherwise, single-model runs keep writing to `models/<name>/` and multi-model comparisons go to `results/`.

# Experiment scripts

Portable Bash scripts containing the experiment commands are in
[experiments/](experiments/). For example:

```bash
bash experiments/jobscript_17_von_mises.sh
```
