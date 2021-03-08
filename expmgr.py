""" Experiment manager which helps saving experiments.

"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import psutil
import wandb
import yaml

EXP_BASE_DIR = './results'
FORMAT = "%Y%m%d_%H%M%S"
DATETIME = datetime.now()


project_name = None
exp_name = None

_verbose = 1
_version = 1


def version():
    return _version


def init(project, name, config):
    global project_name, exp_name, _verbose, DATETIME
    project_name = project
    exp_name = name

    global _version
    _version = getattr(config, 'version', 1)

    if getattr(config, 'quiet', False):
        _verbose = 0
    if not hasattr(config, 'jax_enable_x64'):
        config.jax_enable_x64 = False
    if config.jax_enable_x64:
        print('Enabling jax x64 option')
        jax.config.update("jax_enable_x64", True)
    if not config.time_tag:
        print('Omit the time information from experiment tag')
        DATETIME = None
        if getattr(config, 'resume', False):
            config.checkpoint_path = str(get_result_path('checkpoint_last.pkl'))

    wandb.init(
        project=project_name,
        name=exp_name,
        dir=str(get_result_dir())
    )
    wandb.config.update(config)
    save_config(config)


def get_result_dir():
    tag = f'{project_name}_{exp_name}'
    if DATETIME is not None:
        tag = f'{DATETIME.strftime(FORMAT)}_{tag}'
    exp_dir = Path(EXP_BASE_DIR) / tag
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)
    return exp_dir


def get_result_path(filename):
    """ Get file path under result directory.

    Args.
        tag: str, experiment tag name.
        filename: str, file name.
    Returns
        a file path.
    """
    exp_dir = get_result_dir()
    filepath = exp_dir / filename
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    return filepath


def load_config(filepath):
    """ Load experiment configuration from a file path.

    Args:
        filepath: str, configuration file path

    Returns:
        dict, experiment configuration.
    """
    config_path = Path(filepath)
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, upload_to_wandb=True):
    """ Save experiment configuration as a yaml file

    NOTE:
      The configuration file name has changed into `hparams.yaml`
      to avoid the collision with wandb's `config.yaml` file.

    Args:
        config: Namespace or dict, experiment configuration.
        upload_to_wandb: bool, whether to upload wandb server.
    """
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    config_path = get_result_path('hparams.yaml')
    with config_path.open('w') as f:
        yaml.safe_dump(config, f)
    if upload_to_wandb:
        safe_wandb_save(config_path)


def log(step, logging_output):
    logging_str = ' | '.join('='.join([k, str(v)]) for k, v in logging_output.items())
    if _verbose > 0:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Step[{step:d}]: {logging_str}')
    wandb.log(logging_output, step=step)


def save_array(filename, arr, upload_to_wandb=True):
    """ Save jax array. """
    filepath = str(get_result_path(filename))
    jnp.save(filepath, arr)
    if upload_to_wandb:
        safe_wandb_save(filepath)


def save_history(filename, history, upload_to_wandb=True):
    """ Save jax array zip. """
    filepath = str(get_result_path(filename))
    jnp.savez(filepath, **history)
    if upload_to_wandb:
        safe_wandb_save(filepath)


def safe_wandb_save(filepath):
    filepath = str(filepath)
    try:
        wandb.save(filepath)
    except FileNotFoundError as e:
        print(f'[WARNING] Fail to upload {filepath} to wandb server')
        print(str(e))


def log_array(**kwargs):
    for k, v in kwargs.items():
        print(k, str(v))
        wandb.config.__setattr__(k, str(v))


def print_memory_usage(tag=None, unit='mb'):
    unit = unit.upper()
    unit_val = dict(KB=1024., MB=1024*1024.)[unit]
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / unit_val
    str_header = '[Memory Usage]'
    if tag:
        str_header += f' {tag}'
    print(f'{str_header} {mem_usage:.2f} {unit}')


if __name__ == '__main__':
    fp = get_result_path('nnnmm/asdf.png')
    print(fp)
