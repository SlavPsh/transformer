import os
import shutil
import toml
import logging
from datetime import datetime


def get_total_grad_norm(model, norm_type=2):
    total = None
    for p in model.parameters():
        if p.grad is not None:
            # use float() if gradients might be half-precision
            grad = p.grad.float()
            if total is None:
                total = grad.pow(norm_type).sum()
            else:
                total += grad.pow(norm_type).sum()
    return (total ** (1.0 / norm_type)).item() if total is not None else 0.0
def load_config(config_path):
    """
    Load the TOML configuration file and return a dictionary.
    """
    with open(config_path, 'r') as config_file:
        config = toml.load(config_file)
    return config

def setup_logging(config, output_dir, job='training'):
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, f'{job}.log')
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def get_file_path(dir, filename):
    return os.path.join(dir, filename)

def unique_output_dir(config, run_name='run'):
    """
    Generate a unique directory name using the current timestamp
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['output']['base_path'], f"{timestamp}_{run_name}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def copy_config_to_output(config_path, output_dir):
    """
    Copies the configuration file to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_path, output_dir)