import wandb
import os
import torch


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

