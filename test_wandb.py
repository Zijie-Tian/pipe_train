import wandb
import time
import numpy as np

wandb.init(project="test", entity="thu-storage")
# define our custom x axis metric
wandb.define_metric("custom_step")
# define which metrics will be plotted against it
wandb.define_metric("validation_loss", step_metric="custom_step")

for i in range(10):
    log_dict = {
        "train_loss": 1/(i+1),
        "custom_step": i**2,
        "validation_loss": 1/(i+1)   
    }
    wandb.log(log_dict)
    time.sleep(0.2)