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
                 full_train:int=1,
                    **kwargs) -> None:
        super(SWAE_EXPAND, self).__init__()
        self.in_channels=in_channels
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.p = wasserstein_deg
        self.num_projections = num_projections
        self.proj_dist = projection_dist
        self.full_train=full_train

        checkpoint = torch.load(pretrained_layers, map_location=lambda storage, loc: storage)
        dct=checkpoint['state_dict']
        params=list(dct.keys())
        for param in params:
            a=dct[param]
            dct[param[6:]]=a
            del dct[param]
        self.model=SWAE(in_channels,
                 latent_dim,
                 hidden_dims,
                 reg_weight,
                 wasserstein_deg,
                 num_projections,
                 projection_dist)
        self.model.load_state_dict(state_dict=dct)
        for para in self.model.parameters():
            para.requires_grad = (self.full_train==1)


        modules=[]
        modules.append(
          nn.Sequential(
            #nn.Conv2d(self.in_channels, out_channels= 64,kernel_size= 3, padding= 1),#doubleout
            nn.Conv2d(32, out_channels= 64,kernel_size= 3, padding= 1),#singleout
            nn.LeakyReLU())
          )
        for i in range(5):
          modules.append(
          nn.Sequential(
            nn.Conv2d(64, out_channels= 64,
                                      kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
          )
        modules.append(
          nn.Sequential(
            nn.Conv2d(64, out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1),
            nn.Tanh())
          )
        self.final_layer_3=nn.Sequential(*modules)


    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        return self.model.encode(input)

    def decode(self, z: Tensor) -> Tensor:
        #doubleout
        '''
        result=self.model.decode(z)
        decode=self.final_layer_3(result)
        '''
        #singleout
        result = self.model.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.model.decoder(result)
        result = self.model.final_layer_1(result)
        result = self.final_layer_3(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        #doubleout
        '''
        decode,inp,z = self.model(input)
        decode=self.final_layer_3(decode)
        '''
        #singleout
        z=self.encode(input)
        
        return  [self.decode(z),input,z]
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
       

        

        recons_loss_l2 = F.mse_loss(recons, input)
      

      

        loss = recons_loss_l2 
        return {'loss': loss}

    
