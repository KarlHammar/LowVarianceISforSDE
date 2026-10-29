import numpy as np
import torch
torch._dynamo.config.recompile_limit = 64

import json
import argparse
import os
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from collections import OrderedDict
from controller import multiNnController
from training import create_train_method

def _parse_override_value(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if raw == "True":
            return True
        if raw == "False":
            return False
        if raw == "None":
            return None
        return raw

def _parse_override_path(path):
    parts = []
    for chunk in path.split("."):
        if not chunk:
            raise ValueError(f"Invalid override path: '{path}'")
        buf = ""
        i = 0
        while i < len(chunk):
            if chunk[i] == "[":
                if buf:
                    parts.append(buf)
                    buf = ""
                j = chunk.find("]", i)
                if j == -1:
                    raise ValueError(f"Invalid override path: '{path}'")
                idx = chunk[i + 1 : j]
                if not idx.isdigit():
                    raise ValueError(f"Invalid list index in override path: '{path}'")
                parts.append(int(idx))
                i = j + 1
            else:
                buf += chunk[i]
                i += 1
        if buf:
            parts.append(buf)
    return parts

def _set_nested_value(root, path_parts, value):
    cur = root
    for key in path_parts[:-1]:
        if isinstance(key, int):
            if not isinstance(cur, list):
                raise KeyError(f"Expected list for index {key}, got {type(cur)}")
            if key >= len(cur):
                raise IndexError(f"List index out of range: {key}")
            cur = cur[key]
        else:
            if not isinstance(cur, dict):
                raise KeyError(f"Expected dict for key '{key}', got {type(cur)}")
            if key not in cur:
                raise KeyError(f"Key not found in config: '{key}'")
            cur = cur[key]

    last = path_parts[-1]
    if isinstance(last, int):
        if not isinstance(cur, list):
            raise KeyError(f"Expected list for index {last}, got {type(cur)}")
        if last >= len(cur):
            raise IndexError(f"List index out of range: {last}")
        cur[last] = value
    else:
        if not isinstance(cur, dict):
            raise KeyError(f"Expected dict for key '{last}', got {type(cur)}")
        if last not in cur:
            raise KeyError(f"Key not found in config: '{last}'")
        cur[last] = value

def override_specific_params_from_args(args, master_param):
    overrides = getattr(args, "override", None)
    if not overrides:
        return master_param

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected format: path.to.key=value")
        path, raw = item.split("=", 1)
        path_parts = _parse_override_path(path.strip())
        value = _parse_override_value(raw.strip())
        _set_nested_value(master_param, path_parts, value)

    return master_param

def load_nn_config(nn_config_name):
    with open('configs/nn_configs.json', 'r') as f:
        configs = json.load(f)
    return configs[nn_config_name]

def load_method_config(method_name):
    with open('configs/method_configs.json', 'r') as f:
        configs = json.load(f)
    return configs[method_name]

def load_sde_config(sde_name):
    with open('configs/sde_configs.json', 'r') as f:
        configs = json.load(f)
    return configs[sde_name]

def load_meas_config(meas_name):
    with open('configs/meas_configs.json', 'r') as f:
        configs = json.load(f)
    return configs[meas_name]

def load_ssm_config(ssm_name):
    with open('configs/ssm_configs.json', 'r') as f:
        configs = json.load(f)
    ssm_params = configs[ssm_name]
    
    # resolve nested configs"
    sde_params = load_sde_config(ssm_params['sde_config'])
    meas_params = load_meas_config(ssm_params['meas_config'])
    ssm_params['sde_config'] = sde_params
    ssm_params['meas_config'] = meas_params

    return ssm_params

def load_master_config(master_name, config_path='configs/master_configs.json'):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    master_params = configs[master_name]

    # resolve nested configs
    method_params = load_method_config(master_params['method_config'])
    nn_params = load_nn_config(master_params['nn_config'])
    ssm_config = load_ssm_config(master_params['ssm_config'])

    master_params['method_config'] = method_params
    master_params['nn_config'] = nn_params
    master_params['ssm_config'] = ssm_config

    return master_params

def load_model_params(model_name):
    with open(os.path.join("models", model_name, "params.json"), "r") as handle:
        params = json.load(handle)
    params = dict(params)
    params["path"] = os.path.join("models", model_name)
    return params

def setup_runtime_params(run_name, master_param, seed=None):
    master_param_runtime = {
        'name': run_name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': torch.randint(0, 5000, (1,)).item() if seed is None else seed,
    }
    master_param.update(master_param_runtime)
    master_param['path'] = os.path.join('models', master_param['name'])
    return master_param

def log_run_parameters(params):
    if not os.path.exists(params['path']):
        print('Run directory does not exist, creating it now!')
        os.makedirs(params['path'])
    elif params['name'] == 'test':
        print('Doing test run, will overwrite directory.')
    else:
        print('########################################################')
        print(f" RUN DIRECTORY WILL BE OVERWRITTEN FOR RUN: {params['name']} ")
        print('########################################################')

    files = os.listdir(params['path'])
    for file_name in files:
        file_path = os.path.join(params['path'], file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    slurm_job_id_raw = os.getenv("SLURM_JOB_ID", "").strip()
    slurm_job_id = int(slurm_job_id_raw) if slurm_job_id_raw else -1


    ordered_params = OrderedDict()
    ordered_params["name"] = params["name"]
    ordered_params["date"] = params["date"]
    ordered_params["seed"] = params["seed"]
    ordered_params["slurm_job_id"] = slurm_job_id
    ordered_params["path"] = params["path"]
    ordered_params["method_config"] = params["method_config"]
    ordered_params["ssm_config"] = params["ssm_config"]
    ordered_params["nn_config"] = params["nn_config"]

    with open(f"{params['path']}/params.json", "w") as handle:
        json.dump(ordered_params, handle, indent=4)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    diff = subprocess.check_output(["git", "diff"]).decode()

    with open(f"{params['path']}/commit.txt", "w") as handle:
        handle.write(f"{commit}\n")

    with open(f"{params['path']}/patch.diff", "w") as handle:
        handle.write(diff)

    with open(f"{params['path']}/env.txt", "w") as handle:
        handle.write(f"Python: {platform.python_version()}\n")
        for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
            handle.write(f"{dist.metadata['Name']}=={dist.version}\n")

def run_training(master_param_dict, compile_rollouts=True):

    log_run_parameters(master_param_dict)

    torch.manual_seed(master_param_dict["seed"])
    np.random.seed(master_param_dict["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    controller = multiNnController(
        master_param_dict["ssm_config"],
        master_param_dict["nn_config"],
        device,
        name=master_param_dict["name"],
        path=master_param_dict["path"],
        compile_rollouts=compile_rollouts,
    )
    train_method = create_train_method(controller, master_param_dict["method_config"])
    train_method.train()

if __name__ == "__main__":
    # handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='Name for training run')
    parser.add_argument('--seed', type=int, default=None, help='Optional reproducible training seed')
    parser.add_argument('--master_config', type=str, default='default', help='Master config key')
    parser.add_argument('--method_config', type=str, default=None, help='Method config key')
    parser.add_argument('--nn_config', type=str, default=None, help='NN config key')
    parser.add_argument('--ssm_config', type=str, default=None, help='SSM config key')
    parser.add_argument('--sde_config', type=str, default=None, help='SDE config key')
    parser.add_argument('--meas_config', type=str, default=None, help='Measurement config key')
    parser.add_argument('--override', action='append', default=[], help='Override config params with path=value (e.g., method_config.N_epochs=1000)')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch compilation for quick tests')
    args = parser.parse_args()

    # load base configurations
    master_param = load_master_config(args.master_config)

    # override method config
    if args.method_config is not None:
        method_params = load_method_config(args.method_config)
        master_param['method_config'] = method_params

    # override nn config
    if args.nn_config is not None:
        nn_params = load_nn_config(args.nn_config)
        master_param['nn_config'] = nn_params

    # override ssm config
    if args.ssm_config is not None:
        ssm_params = load_ssm_config(args.ssm_config)
        master_param['ssm_config'] = ssm_params
    
    # override parts of ssm config
    if args.sde_config is not None:
        sde_params = load_sde_config(args.sde_config)
        master_param['ssm_config']['sde_config'] = sde_params
    if args.meas_config is not None:
        meas_params = load_meas_config(args.meas_config)
        master_param['ssm_config']['meas_config'] = meas_params

    # override specific parameters from command line (if needed, can be added here)
    master_param = override_specific_params_from_args(args, master_param)
    
    master_param_dict = setup_runtime_params(args.name, master_param, seed=args.seed)

    # start training
    run_training(master_param_dict, compile_rollouts=not args.no_compile)
