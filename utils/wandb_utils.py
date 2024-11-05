import wandb
import os
import torch
import matplotlib.pyplot as plt


class WandbLogger:
    def __init__(
        self,
        config,
        output_dir=None,
        run_name='run',
        job_type="training",
    ):
        self.config = config
        self.entity = config["entity"]
        self.project_name = config["project_name"]
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
                entity=self.entity,
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                dir=self.output_dir,
                job_type=self.job_type,
            )

            self.initialized = True

    def initialize_sweep(self):
        if not self.initialized:
            self.initialize()
            # Extract sweep configuration
            sweep_config = {
                'method': self.config['sweep']['method'],
                'metric': {
                    'name': self.config['sweep']['metric_name'],
                    'goal': self.config['sweep']['metric_goal']
                },
                'parameters': self.config['sweep']['parameters']
            }
            self.sweep_id = wandb.sweep(sweep_config, project=self.project_name)
            print(f'Sweep ID: {self.sweep_id}')
    
    def get_config(self):
        return wandb.config
    
    def log(self, data):
        if not self.initialized:
            self.initialize()
        wandb.log(data)

    def log_gradient_norm(self, model):
        if not self.initialized:
            self.initialize()
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
        total_norm = total_norm ** 0.5

        self.log({'gradient_norm': total_norm})

    def save_model(self, model, model_name, optimizer, epoch, output_dir):
        if not self.initialized:
            self.initialize()
        file_path = os.path.join(output_dir, model_name)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'att_mask_used': model.att_mask_used
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

    def plot_binned_scores(self, aggregated_bin_scores, total_average_score):
        # Plot the percentage of good_major_weight over total_major_weight per bin and log to wandb
        for param, df in aggregated_bin_scores.items():
            df['track_efficiency'] = (df['good_predicted_count'] / df['total_true_count']) * 100
            df['track_fake_rate'] = ((df['total_predicted_count'] - df['good_predicted_count']) / df['total_predicted_count']) * 100
            df['percentage_good_major_weight'] = (df['good_major_weight'] / df['total_true_weight']) * 100
            
            x = df[f'{param}_bin'].astype(str)
            # Plot track_efficiency
            plt.figure()
            y = df['track_efficiency']
            plt.plot(x, y, marker='o', color='black')
            plt.fill_between(x, y, 0, where=(y >= 0), facecolor='blue', alpha=0.8)
            plt.fill_between(x, y, 100, where=(y >= 0), facecolor='red', alpha=0.3)
            plt.ylim(max(y.min() - 10, 0), 100)
            plt.title(f'Track Efficiency for {param}')
            plt.xlabel(f'{param} Bins')
            plt.ylabel('Efficiency (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.log({f'{param}_track_efficiency': wandb.Image(plt)})
            plt.close()

            # Plot track_fake_rate
            plt.figure()
            y = df['track_fake_rate']
            plt.plot(x, y, marker='o', color='black')
            plt.fill_between(x, y, 0, where=(y >= 0), facecolor='red', alpha=0.8)
            plt.fill_between(x, y, 100, where=(y >= 0), facecolor='green', alpha=0.3)
            plt.ylim(max(y.min() - 10, 0), min(y.max() + 20, 100))
            plt.title(f'Track Fake Rate for {param}')
            plt.xlabel(f'{param} Bins')
            plt.ylabel('Fake Rate (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.log({f'{param}_track_fake_rate': wandb.Image(plt)})
            plt.close()
            
            # Plot percentage_good_major_weight
            plt.figure()
            y = df['percentage_good_major_weight']
            plt.plot(x, y, marker='o', color='black')
            plt.fill_between(x, y, 0, where=(y >= 0), facecolor='blue', alpha=0.8)
            plt.fill_between(x, y, 100, where=(y >= 0), facecolor='red', alpha=0.3)
            plt.ylim(max(y.min() - 10, 0) , 100)  # Set y-axis range for better resolution
            plt.title(f'Good Tracks Weight vs True Weight for {param}. Avg Score: {total_average_score*100:.1f}')
            plt.xlabel(f'{param} Bins')
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            self.log({f'aggregated_bin_scores_{param}': wandb.Image(plt)})
            plt.close()

