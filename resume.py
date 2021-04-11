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
    

    runner = Trainer(resume_from_checkpoint=args.checkpoint)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    runner.save_checkpoint(config['logging_params']['ckpt_save_dir']+"/last.ckpt")