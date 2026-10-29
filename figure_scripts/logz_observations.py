"""Read observation-wise smoothing outputs without pooling different log-Z targets."""
import csv
from pathlib import Path

import numpy as np


def read_rows(path):
    with open(path, newline='') as stream:
        yield from csv.DictReader(stream)


def load_errors(folder):
    """Return estimate-minus-reference errors and failed observation IDs per controller."""
    folder = Path(folder)
    references = {}
    for row in read_rows(folder / 'smoothing_observation_references.csv'):
        observation = int(row['observation_id'])
        reference = float(row['log_z_hat'])
        if observation in references:
            raise ValueError(f'{folder}: duplicate reference {observation}')
        if not np.isfinite(reference):
            raise ValueError(f'{folder}: invalid reference for observation {observation}')
        references[observation] = reference
    if not references:
        raise ValueError(f'{folder}: no observation references')
    errors_by_controller = {}
    for row in read_rows(folder / 'smoothing_observation_estimates.csv'):
        controller = row['controller']
        observation = int(row['observation_id'])
        if controller not in errors_by_controller:
            errors_by_controller[controller] = {}
        if observation in errors_by_controller[controller]:
            raise ValueError(f'{folder}: duplicate estimate for {controller}, observation {observation}')
        estimate = float(row['log_z_hat'])
        errors_by_controller[controller][observation] = (estimate - references[observation]
                               if row['status'] == 'ok' and np.isfinite(estimate) else np.nan)

    output = {}
    for controller, errors in sorted(errors_by_controller.items()):
        if controller == 'Correct' or controller.startswith('optimal('):
            continue
        values = np.array([errors.get(observation, np.nan) for observation in sorted(references)])
        valid = np.isfinite(values)
        failed = [observation for observation, ok in zip(sorted(references), valid) if not ok]
        output[controller] = dict(errors=values[valid], failed=failed)
    return output
