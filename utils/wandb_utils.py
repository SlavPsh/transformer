import wandb
import os
import psutil
import torch
import matplotlib.pyplot as plt
import numpy as np
from  scipy.stats import beta
import pandas as pd
import pickle


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
    
    def log(self, data):
        if not self.initialized:
            self.initialize()
        wandb.log(data)
    
    def alert(self, ttl, txt):
        if not self.initialized:
            self.initialize()
        self.run.alert(title=ttl, text=txt)

    def get_system_memory_stats(self):
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

    def plot_binned_scores(self, combined_bin_scores, total_average_score):
        
        # save the combined_bin_scores to a pickle file
        with open(os.path.join(self.output_dir, 'combined_bin_scores.pkl'), 'wb') as f:
            pickle.dump(combined_bin_scores, f)

        # Aggregate the bin scores across all events for each parameter
        aggregated_bin_scores = {}
        for param in combined_bin_scores.keys():
            all_bin_scores_df = pd.concat(combined_bin_scores[param])
            aggregated_bin_scores[param] = all_bin_scores_df.groupby(f'{param}_bin', observed=False).agg({
                    'good_predicted_count': 'sum',  
                    'total_true_count': 'sum', 
                    'total_predicted_count' : 'sum',
                    'good_major_weight': 'sum',
                    'total_true_weight': 'sum',
                    'event_efficiency': ['mean', 'std'],  
                    'event_fake_rate': ['mean', 'std']  
                }).reset_index()



        # Plot the percentage of good_major_weight over total_major_weight per bin and log to wandb
        for param, df in aggregated_bin_scores.items():

            # Get left and right edges of each interval
            left_edges = [interval.left for interval in df[f'{param}_bin']]
            right_edges = [interval.right for interval in df[f'{param}_bin']]
            midpoints = [(interval.left + interval.right) / 2 for interval in df[f'{param}_bin']]
            horizontal_error = [(right - left) / 2 for left, right in zip(left_edges, right_edges)]

            # Calculate track efficiency mode 
            k = df['good_predicted_count']['sum']
            n = df['total_true_count']['sum']
            y = (k / n) * 100

            # Calculate track efficiency mean. Note, not the same as the mode. 
            # Also note we calculate efficiency here from total counts, not from event efficiency
            y_mean = ((k + 1)/ (n + 2)) * 100
            bayes_error = np.sqrt( ((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1)**2) / ((n + 2)**2) )*100
            norm_error = np.sqrt((k/n) * (1 - k/n) / n) * 100
            
            # Calculate track efficiency error as Clopper–Pearson interval
            alpha =  0.05
            # Calculate lower and upper bounds of the confidence interval
            e_lower = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0)
            e_upper = np.where(k < n , beta.ppf(1 - alpha / 2, k + 1, n - k), 1)
            lower_error = y - 100*e_lower      # Lower error
            upper_error = 100*e_upper - y  # Upper error
            lower_error = np.maximum(lower_error, 0)
            upper_error = np.maximum(upper_error, 0)
            
            y_errors = [lower_error, upper_error]

            # Plot track_efficiency
            plt.figure(figsize=(8, 5))
            
            # Add error bars
            plt.errorbar(midpoints, y, xerr=horizontal_error, fmt='o', color='black', capsize=0, capthick=1, elinewidth=1)
            plt.errorbar(midpoints, y, yerr=y_errors, label="EncReg", fmt='o', color='black', capsize=3, capthick=1, elinewidth=1)
            plt.ylim(max(y.min() - 10, 0), 100)
            plt.title(f'Track Efficiency for {param}')
            # Labels and legend
            plt.ylabel("Efficiency")           
            plt.xlabel(f'Particle {param}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.grid(True, linestyle=':', color='gray', alpha=0.7)
            self.log({f'{param}_track_efficiency': wandb.Image(plt)})
            plt.close()

            # Plot track_efficiency from mean and std across events, in percentage
            y = 100*df['event_efficiency']['mean']
            y_errors = 100*df['event_efficiency']['std']
            plt.figure(figsize=(8, 5))
            
            # Add error bars
            plt.errorbar(midpoints, y, xerr=horizontal_error, fmt='o', color='black', capsize=0, capthick=1, elinewidth=1)
            plt.errorbar(midpoints, y, yerr=y_errors, label="EncReg", fmt='o', color='black', capsize=3, capthick=1, elinewidth=1)
            plt.ylim(max(y.min() - 10, 0), min(y.max() + 10, 100))
            plt.title(f'Track Efficiency avg. for {param}')
            # Labels and legend
            plt.ylabel("Efficiency")           
            plt.xlabel(f'Particle {param}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.grid(True, linestyle=':', color='gray', alpha=0.7)
            self.log({f'{param}_track_efficiency_avg': wandb.Image(plt)})
            plt.close()

            # Plot track_fake_rate
            k_fake = df['total_predicted_count']['sum'] - df['good_predicted_count']['sum']
            n_fake = df['total_predicted_count']['sum']
            y = (k_fake / n_fake) * 100
            # Calculate lower and upper bounds of the confidence interval
            e_lower = np.where(k_fake > 0, beta.ppf(alpha / 2, k_fake, n_fake - k_fake + 1), 0)
            e_upper = np.where(k_fake < n_fake , beta.ppf(1 - alpha / 2, k_fake + 1, n_fake - k_fake), 1)
            lower_error = y - 100*e_lower      # Lower error
            upper_error = 100*e_upper - y  # Upper error
            lower_error = np.maximum(lower_error, 0)
            upper_error = np.maximum(upper_error, 0)
            y_errors = [lower_error, upper_error]

            plt.figure(figsize=(8, 5))

            plt.errorbar(midpoints, y, xerr=horizontal_error, fmt='o', color='black', capsize=0, capthick=1, elinewidth=1)
            plt.errorbar(midpoints, y, yerr=y_errors, label="EncReg", fmt='o', color='black', capsize=3, capthick=1, elinewidth=1)
            plt.ylim(max(y.min() - 10, 0), min(y.max() + 10, 100))
            plt.title(f'Track Fake Rate for {param}')
            plt.xlabel(f'Particle {param}')
            plt.ylabel('Fake Rate (%)')
            plt.tight_layout()
            plt.grid(True, linestyle=':', color='gray', alpha=0.7)
            self.log({f'{param}_track_fake_rate': wandb.Image(plt)})
            plt.close()

            # Plot track_fake_rate from mean and std across events
            y = 100*df['event_fake_rate']['mean']
            y_errors = 100*df['event_fake_rate']['std']
            plt.figure(figsize=(8, 5))

            plt.errorbar(midpoints, y, xerr=horizontal_error, fmt='o', color='black', capsize=0, capthick=1, elinewidth=1)
            plt.errorbar(midpoints, y, yerr=y_errors, label="EncReg", fmt='o', color='black', capsize=3, capthick=1, elinewidth=1)
            plt.ylim(max(y.min() - 10, 0), min(y.max() + 10, 100))
            plt.title(f'Track Fake Rate avg. for {param}')
            plt.xlabel(f'Particle {param}')
            plt.ylabel('Fake Rate (%)')
            plt.tight_layout()
            plt.grid(True, linestyle=':', color='gray', alpha=0.7)
            self.log({f'{param}_track_fake_rate_avg': wandb.Image(plt)})
            plt.close()

            
            # Plot percentage_good_major_weight
            plt.figure(figsize=(8, 5))
            y = (df['good_major_weight']['sum'] / df['total_true_weight']['sum']) * 100
            plt.plot(midpoints, y, marker='o', color='black')

            plt.ylim(max(y.min() - 10, 0), min(y.max() + 10, 100))  # Set y-axis range for better resolution
            plt.title(f'Good Tracks Weight vs True Weight for {param}. Avg Score: {total_average_score*100:.1f}')
            plt.xlabel(f'Particle {param}')
            plt.ylabel('Percentage')
            plt.tight_layout()
            plt.grid(True, linestyle=':', color='gray', alpha=0.7)
            self.log({f'aggregated_bin_scores_{param}': wandb.Image(plt)})
            plt.close()

