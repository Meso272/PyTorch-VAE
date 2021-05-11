import torch

from models import BaseVAE
from torch import nn,optim
from torch.nn import functional as F
from torch import distributions as dist
from .types_ import *
from .gdn import *
from .quants import *
from .resnet import *
from functools import partial
norm_map={"bn":nn.BatchNorm2d,"gdn":GDN,"no":nn.Identity,"igdn":partial(GDN,inverse=True)}
actv_map={"relu":nn.ReLU,"leakyrelu":nn.LeakyReLU,"prelu":nn.PReLU,"no":nn.Identity}
class SWAE_NEW(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size:int = 64,
                 hidden_dims: List = None,
                 reg_weight: int = 100,
                 wasserstein_deg: float= 2.,
                 num_projections: int = 50,
                 projection_dist: str = 'normal',
                 encoder_final_layer='fc',
                 actv=None,
                 norm=None,
                 actv1='leakyrelu',
                 norm1='bn',
                 actv2='leakyrelu',
                 norm2='bn',
                 group_num=32,
                 struct='new',#deprecated
                 quant_mode=0,
                 resblock_num=[2,2,2,2],
                 first_channel=16,
                 resnet_pooling=True,
                 resnet_fc=True,
                    **kwargs) -> None:
        super(SWAE_NEW, self).__init__()
        self.struct=struct
        self.in_channels=in_channels
        self.encoder_final_layer=encoder_final_layer
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.p = wasserstein_deg
        self.num_projections = num_projections
        self.proj_dist = projection_dist
        self.quant_mode=quant_mode
        if actv!=None:
            actv1=actv
            actv2=actv
        if norm!=None:
            norm1=norm
            norm2=norm
        

        modules = []
        if hidden_dims is None:
            hidden_dims = [16,32,64,128]
        self.last_fm_nums=hidden_dims[-1]
        if self.quant_mode==1:
            self.rounder=Round_1
        
        self.last_resnet_size=input_size
        for i in range(len(hidden_dims)):
          self.last_resnet_size=(self.last_resnet_size+1)//2
        self.last_resnet_size=int(self.last_resnet_size)

        if struct=='resnet' and resnet_pooling:
            self.last_fm_size=1
        else:
            self.last_fm_size=self.last_resnet_size
        self.resnet_pooling=resnet_pooling
        self.resnet_fc=resnet_fc
        
       # Build Encoder
        if struct=="resnet":
            fc_out=latent_dim if resnet_fc else 0
            
            encoder=[ResNet_Encoder(BasicBlock,num_block=resblock_num,channel_list=hidden_dims,default_conv1=True,in_channels=first_channel,avg_pooling=resnet_pooling,fc_out=fc_out,norm=norm_map[norm],actv=actv_map[actv])]
            if not resnet_fc:
                if encoder_final_layer=='conv':
                    encoder.append( nn.Conv2d(hidden_dims[-1], out_channels=latent_dim//(self.last_fm_size**2),
                                  kernel_size= 1, stride= 1, padding  = 0) )
                    self.decoder_input = nn.ConvTranspose2d(latent_dim//(self.last_fm_size**2), out_channels=hidden_dims[-1],
                                  kernel_size= 1, stride= 1, padding  = 0,output_padding=0)
                elif encoder_final_layer=='fc':
                    self.fc_z=nn.Linear(hidden_dims[-1]*self.last_resnet_size*self.last_resnet_size, latent_dim)
                    self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.last_resnet_size*self.last_resnet_size)
            

            hidden_dims.reverse()
            if actv=='gdn':
              actv='igdn'
            self.encoder=nn.Sequential(*encoder)
            self.decoder=ResNet_Decoder(BasicBlock_Decode,num_block=resblock_num,channel_list=hidden_dims,fc_in=fc_out,up_sampling=resnet_pooling,first_size=self.last_resnet_size,last_channel=first_channel,default_convout=True,norm=norm_map[norm],actv=actv_map[actv])
            return 

        print(norm1)
        print(norm2)
        print(actv1)
        print(actv2)
        for h_dim in hidden_dims:
            if norm1=='bn':
                norm1=nn.BatchNorm2d(in_channels)
            elif norm1=='gn':
                norm1=nn.GroupNorm(group_num,in_channels) 
            else:
                norm1=nn.Identity()

            if norm2=='bn':
                norm2=nn.BatchNorm2d(h_dim)
            elif norm2=='gn':
                norm2=nn.GroupNorm(group_num,h_dim) 
            else:
                norm2=nn.Identity()
               
            if actv1=='leakyrelu':
                actv1=nn.LeakyReLU()
            elif actv1=='prelu':
                actv1=nn.PReLU()
            elif actv1=='gdn':
                actv1=GDN(in_channels) 
            else:
                actv1=nn.Identity()
            
            if actv2=='leakyrelu':
                actv2=nn.LeakyReLU()
            elif actv2=='prelu':
                actv2=nn.PReLU()
            elif actv2=='gdn':
                actv2=GDN(h_dim) 
            else:
                actv2=nn.Identity()
           
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=in_channels,
                        kernel_size= 3, stride= 1, padding  = 1),###added layer
                    norm1,
                    actv1,
                    nn.Conv2d(in_channels, out_channels=h_dim,
                        kernel_size= 3, stride= 2, padding  = 1),
                    norm2,
                    actv2
                    
                )    
            )

              
        
            in_channels = h_dim
        if self.encoder_final_layer=='conv':
            modules.append(nn.Sequential( nn.Conv2d(hidden_dims[-1], out_channels=latent_dim//(self.last_fm_size**2),
                                  kernel_size= 1, stride= 1, padding  = 0) ) )
        self.encoder = nn.Sequential(*modules)
        if self.encoder_final_layer=='fc':
            self.fc_z = nn.Linear(hidden_dims[-1]*self.last_fm_size*self.last_fm_size, latent_dim)
        


        # Build Decoder
        if self.encoder_final_layer=='fc':
            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.last_fm_size*self.last_fm_size)
        elif self.encoder_final_layer=='conv':
            self.decoder_input = nn.ConvTranspose2d(latent_dim//(self.last_fm_size**2), out_channels=hidden_dims[-1],
                                  kernel_size= 1, stride= 1, padding  = 0,output_padding=0)
        modules = []
        
        hidden_dims.reverse()
   
        for i in range(len(hidden_dims) - 1):
            if norm1=='bn':
                norm1=nn.BatchNorm2d(hidden_dims[i])
            elif norm1=='gn':
                norm1=nn.GroupNorm(group_num,hidden_dims[i]) 
            else:
                norm1=nn.Identity()

            if norm2=='bn':
                norm2=nn.BatchNorm2d(hidden_dims[i+1])
            elif norm2=='gn':
                norm2=nn.GroupNorm(group_num,hidden_dims[i+1]) 
            else:
                norm2=nn.Identity()
               
            if actv1=='leakyrelu':
                actv1=nn.LeakyReLU()
            elif actv1=='prelu':
                actv1=nn.PReLU()
            elif actv1=='gdn':
                actv1=GDN(hidden_dims[i]) 
            else:
                actv1=nn.Identity()
            
            if actv2=='leakyrelu':
                actv2=nn.LeakyReLU()
            elif actv2=='prelu':
                actv2=nn.PReLU()
            elif actv2=='gdn':
                actv2=GDN(hidden_dims[i+1]) 
            else:
                actv2=nn.Identity()
        


            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i],
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1,
                                        output_padding=0),##added layer
                    norm1,
                    actv1,
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                    norm2,
                    actv2
                  )
            )

            

        for name,parameter in self.encoder.named_parameters():
            print(name)
        self.decoder = nn.Sequential(*modules)
        
        



        
        modules=[]
        if norm1=='bn':
            norm1=nn.BatchNorm2d(hidden_dims[-1])
        elif norm1=='gn':
            norm1=nn.GroupNorm(group_num,hidden_dims[-1]) 
        else:
            norm1=nn.Identity()

        if norm2=='bn':
            norm2=nn.BatchNorm2d(hidden_dims[-1])
        elif norm2=='gn':
            norm2=nn.GroupNorm(group_num,hidden_dims[-1]) 
        else:
            norm2=nn.Identity()
               
        if actv1=='leakyrelu':
            actv1=nn.LeakyReLU()
        elif actv1=='prelu':
            actv1=nn.PReLU()
        elif actv1=='gdn':
            actv1=GDN(hidden_dims[-1]) 
        else:
            actv1=nn.Identity()
            
        if actv2=='leakyrelu':
            actv2=nn.LeakyReLU()
        elif actv2=='prelu':
            actv2=nn.PReLU()
        elif actv2=='gdn':
            actv2=GDN(hidden_dims[-1]) 
        else:
            actv2=nn.Identity()
        modules.append ( nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                        hidden_dims[-1],
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1,
                                        output_padding=0),##added layer
                            norm1,
                            actv1,
                            nn.ConvTranspose2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                            norm2,
                            actv2
                            
                            ) )
            

        self.final_layer_1=nn.Sequential(*modules)

        self.final_layer_2=nn.Sequential(nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        

    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        if self.encoder_final_layer=='fc' and (self.struct!='resnet' or not self.resnet_fc):
            z = self.fc_z(result)
            #del result
        else:
            z= result

        if self.quant_mode==1:
          z=self.rounder.apply(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        if self.encoder_final_layer=='fc':
            if(self.struct!='resnet' or not self.resnet_fc):
                result = self.decoder_input(z)
                result = result.view(-1, self.last_fm_nums, self.last_fm_size, self.last_fm_size)
            else:
                result=z
        else:
            result= z
            result = result.view(-1, self.latent_dim//(self.last_fm_size**2), self.last_fm_size, self.last_fm_size)
            if self.encoder_final_layer=='conv':
                result = self.decoder_input(result)
        result = self.decoder(result)
        if self.struct!='resnet':
            result = self.final_layer_1(result)
            result= self.final_layer_2(result)
        
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        return  [self.decode(z), input, z]
    def get_features(self, input: Tensor, **kwargs)-> Tensor:##need correction
        z=self.encode(input)
        result = self.decoder_input(z)
        result = result.view(-1, self.last_fm_nums, self.last_fm_size, self.last_fm_size)
        result = self.decoder(result)
        result = self.final_layer_1(result)
        return result
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]

        batch_size = input.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        recons_loss_l2 = F.mse_loss(recons, input)
        recons_loss_l1 = F.l1_loss(recons, input)

        swd_loss = self.compute_swd(z, self.p, reg_weight)

        loss = recons_loss_l2 + recons_loss_l1 + swd_loss
        return {'loss': loss, 'Reconstruction_Loss':(recons_loss_l2 + recons_loss_l1), 'SWD': swd_loss}

    def get_random_projections(self, latent_dim: int, num_samples: int) -> Tensor:
        """
        Returns random samples from latent distribution's (Gaussian)
        unit sphere for projecting the encoded samples and the
        distribution samples.

        :param latent_dim: (Int) Dimensionality of the latent space (D)
        :param num_samples: (Int) Number of samples required (S)
        :return: Random projections from the latent unit sphere
        """
        if self.proj_dist == 'normal':
            rand_samples = torch.randn(num_samples, latent_dim)
        elif self.proj_dist == 'cauchy':
            rand_samples = dist.Cauchy(torch.tensor([0.0]),
                                       torch.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
        else:
            raise ValueError('Unknown projection distribution.')

        rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1,1)
        return rand_proj # [S x D]


    def compute_swd(self,
                    z: Tensor,
                    p: float,
                    reg_weight: float) -> Tensor:
        """
        Computes the Sliced Wasserstein Distance (SWD) - which consists of
        randomly projecting the encoded and prior vectors and computing
        their Wasserstein distance along those projections.

        :param z: Latent samples # [N  x D]
        :param p: Value for the p^th Wasserstein distance
        :param reg_weight:
        :return:
        """
        prior_z = torch.randn_like(z) # [N x D]
        device = z.device

        proj_matrix = self.get_random_projections(self.latent_dim,
                                                  num_samples=self.num_projections).transpose(0,1).to(device)

        latent_projections = z.matmul(proj_matrix) # [N x S]
        prior_projections = prior_z.matmul(proj_matrix) # [N x S]

        # The Wasserstein distance is computed by sorting the two projections
        # across the batches and computing their element-wise l2 distance
        w_dist = torch.sort(latent_projections.t(), dim=1)[0] - \
                 torch.sort(prior_projections.t(), dim=1)[0]
        w_dist = w_dist.pow(p)
        return reg_weight * w_dist.mean()

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
