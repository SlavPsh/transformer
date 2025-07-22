import wandb
import os
import psutil
import torch
import matplotlib.pyplot as plt
import numpy as np
from  scipy.stats import beta
import pandas as pd
import pickle

def get_system_memory_stats():
    # GPU memory stats (in MB)
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        gpu_memory_allocated = 0
        gpu_memory_reserved = 0

    # CPU memory stats (in MB)
    # Get the current process
    process = psutil.Process()

    # Get memory information for the process
    memory_info = process.memory_info()

    # memory_info.rss gives the Resident Set Size, the memory actually used by the process in bytes
    cpu_used_memory = memory_info.rss  # Memory used by this process in bytes
    cpu_used_memory_mb = cpu_used_memory / (1024 ** 2)  # Convert to MB

    return {
        'gpu_memory_allocated_mb': gpu_memory_allocated,
        'gpu_memory_reserved_mb': gpu_memory_reserved,
        'cpu_memory_used_mb': cpu_used_memory_mb
    }

class WandbLogger:
    def __init__(
        self,
        config,
        output_dir=None,
        run_name='run',
        job_type="training",
    ):
        self.config = config
        #self.entity = config["entity"]
        #self.project_name = config["project_name"]
        self.run_name = run_name
        self.output_dir = output_dir
        self.initialized = False
        self.job_type = job_type
        self.sweep_id = None

    def get_output_dir(self):
        if self.output_dir is None:
            raise ValueError("Output directory is not set.")
        if not os.path.exists( self.output_dir):
            os.makedirs( self.output_dir, exist_ok=True)
        return self.output_dir
    
    def get_run_name(self):
        return self.run_name

    def initialize(self):
        if not self.initialized:
            self.run = wandb.init(
                name=self.run_name,
                config=self.config,
                dir=self.output_dir,
                job_type=self.job_type,
            )

            self.initialized = True
    
    def get_config(self):
        return wandb.config
    
    def log(self, data, step=None):
        if not self.initialized:
            self.initialize()
        wandb.log(data, step=step)
    
    def alert(self, ttl, txt):
        if not self.initialized:
            self.initialize()
        self.run.alert(title=ttl, text=txt)


    def log_gradient_norm(self, model):
        if not self.initialized:
            self.initialize()
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
        total_norm = total_norm ** 0.5

        self.log({'gradient_norm': total_norm})

    def save_model(self, model, model_name, optimizer, scheduler, epoch, output_dir):
        if not self.initialized:
            self.initialize()
        file_path = os.path.join(output_dir, model_name)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, file_path)

        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def run_sweep(self, func, count=2):
        if self.sweep_id is None:
            self.initialize_sweep()
        wandb.agent(self.sweep_id, function=func, count=count)

    def finish(self):
        if self.initialized:
            wandb.finish()
            self.initialized = False

    def save_binned_scores(self, combined_bin_scores):
        
        # save the combined_bin_scores to a pickle file
        with open(os.path.join(self.output_dir, 'combined_bin_scores.pkl'), 'wb') as f:
            pickle.dump(combined_bin_scores, f)

