from .base import *
from .vanilla_vae import *
from .gamma_vae import *
from .beta_vae import *
from .wae_mmd import *
from .cvae import *
from .hvae import *
from .vampvae import *
from .iwae import *
from .dfcvae import *
from .mssim_vae import MSSIMVAE
from .fvae import *
from .cat_vae import *
from .joint_vae import *
from .info_vae import *
# from .twostage_vae import *
from .lvae import LVAE
from .logcosh_vae import *
from .swae import *
from .miwae import *
from .vq_vae import *
from .vanilla_ae import *
from .betatc_vae import *
from .dip_vae import *
from .wae_mmd_1d import *
from .swae_1d import *
from .swae_3d import *
from .swae_3d_new import *
from .swae_expand import * 
from .swae_plus import *
from .swae_new import *
# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {'HVAE':HVAE,
              'LVAE':LVAE,
              'IWAE':IWAE,
              'SWAE':SWAE,
              'SWAE_NEW':SWAE_NEW,
              'SWAE_PLUS':SWAE_PLUS,
              'SWAE_EXPAND':SWAE_EXPAND,
              'SWAE_1D':SWAE_1D,
              'SWAE_3D':SWAE_3D,
              'SWAE_3D_NEW':SWAE_3D_NEW,
              'MIWAE':MIWAE,
              'VQVAE':VQVAE,
              'DFCVAE':DFCVAE,
              'DIPVAE':DIPVAE,
              'BetaVAE':BetaVAE,
              'InfoVAE':InfoVAE,
              'WAE_MMD':WAE_MMD,
              'WAE_MMD_1D':WAE_MMD_1D,
              'VampVAE': VampVAE,
              'GammaVAE':GammaVAE,
              'MSSIMVAE':MSSIMVAE,
              'JointVAE':JointVAE,
              'BetaTCVAE':BetaTCVAE,
              'FactorVAE':FactorVAE,
              'SWAE_EXPAND':SWAE_EXPAND,
              'LogCoshVAE':LogCoshVAE,
              'VanillaVAE':VanillaVAE,
              'VanillaAE':VanillaAE,
              'ConditionalVAE':ConditionalVAE,
              'CategoricalVAE':CategoricalVAE}
