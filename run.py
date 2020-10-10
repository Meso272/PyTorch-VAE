import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from pytorch_lightning.callbacks import ModelCheckpoint


trainer = Trainer()
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=config['logging_params']['ckpt_save_dir'],
    save_top_k=-1,
    verbose=True,
    #monitor='val_loss',
    mode='min',
    prefix='',
    period=20
)

runner = Trainer(min_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100,
                 #train_percent_check=1.,
                 #val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 checkpoint_callback=checkpoint_callback,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
runner.save_checkpoint(config['logging_params']['ckpt_save_dir']+"/last.ckpt")