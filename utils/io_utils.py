import os
import shutil
import toml
import logging
from datetime import datetime


def get_model_grad_average(model):
    """
    Prints the average absolute gradient magnitude across all parameters of 'model'.
    Call this after  'loss.backward()' (and before 'optimizer.step()').
    """
    total_grad_sum = 0.0
    total_num_params = 0

    for p in model.parameters():
        if p.grad is not None:
            # sum of absolute values of gradients
            grad_sum = p.grad.abs().sum().item()
            total_grad_sum += grad_sum
            # number of elements in this parameter's gradient
            total_num_params += p.grad.numel()

    if total_num_params > 0:
        avg_grad = total_grad_sum / total_num_params
        return avg_grad
    else:
        return 0.0

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