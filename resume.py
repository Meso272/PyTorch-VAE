import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from pytorch_lightning.callbacks import ModelCheckpoint
    
if __name__=='__main__':
    trainer = Trainer()
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
    parser.add_argument('--checkpoint',  '-k',
            
                    metavar='FILE',
                    help =  'path to the checkpoint file',
                    )
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    

# For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[config['model_params']['name']](**config['model_params'])
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #USE_CUDA = torch.cuda.is_available()
    #print(USE_CUDA)
    #device = torch.device("cuda:0" if USE_CUDA else "cpu")
    
   
    #model = nn.DataParallel(model)
    #model= model.to(device)
    experiment = VAEXperiment(model,
                              config['exp_params'])

# DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=config['logging_params']['ckpt_save_dir'],
        save_top_k=-1,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix='',
        period=20
    )
    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )
    runner = Trainer(resume_from_checkpoint=args.checkpoint,min_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100000,
                 #train_percent_check=1.,
                 #val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 checkpoint_callback=checkpoint_callback,
                 distributed_backend='ddp',
                 **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    runner.save_checkpoint(config['logging_params']['ckpt_save_dir']+"/last.ckpt")