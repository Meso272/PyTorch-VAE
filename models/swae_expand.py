import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
from .types_ import *
from .swae import *

class SWAE_EXPAND(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 pretrained_layers:str,
                 hidden_dims: List = None,
                 reg_weight: int = 100,
                 wasserstein_deg: float= 2.,
                 num_projections: int = 50,
                 projection_dist: str = 'normal',
                    **kwargs) -> None:
        super(SWAE_EXPAND, self).__init__()
        self.in_channels=in_channels
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.p = wasserstein_deg
        self.num_projections = num_projections
        self.proj_dist = projection_dist
        checkpoint = torch.load(pretrained_layers, map_location=lambda storage, loc: storage)
        self.model=SWAE(in_channels,
                 latent_dim,
                 hidden_dims,
                 reg_weight,
                 wasserstein_deg,
                 num_projections,
                 projection_dist).load_state_dict(state_dict=checkpoint['state_dict'])


    

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        a = self.modeldel(input)
        return  a,input
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
       

        

        recons_loss_l2 = F.mse_loss(recons, input)
      

      

        loss = recons_loss_l2 
        return {'loss': loss}

    
