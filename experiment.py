import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from cesm import *
from Hurricane import *
from exaalt import EXAALT
from aramco import ARAMCO
from exafel import *
from nyx import *
class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        if 'epsilon' not in self.params.keys():
            self.params['epsilon']=-1


        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0,scalar=False):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        try:
            self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        except:
            pass
        if scalar:
            return train_loss['loss']
        else:
            return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def sample_images(self):
        if self.params['dataset'] != 'celeba':
            return
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass


        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()
        self.params['epsilon']=float(self.params['epsilon'])
        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=True)
        elif self.params['dataset'] == 'cesm':
            dataset=CLDHGH(path=self.params['data_path'],start=0,end=50,size=self.params['img_size'],normalize=True,epsilon=self.params['epsilon'])
        elif self.params['dataset'] =='cesm_new':

            dataset=CESM(path=self.params['data_path'],start=0,end=50,size=self.params['img_size'],field=self.params['field'],global_max=self.params['max'],global_min=self.params['min'],epsilon=self.params['epsilon'])
        elif self.params['dataset'] =='nyx':

            dataset=NYX(path=self.params['data_path'],start=self.params['start'],end=self.params['end'],size=self.params['img_size'],field=self.params['field'],log=self.params['log'],global_max=self.params['max'],global_min=self.params['min'],epsilon=self.params['epsilon'])
        elif self.params['dataset'] =='exafel':
            dataset=EXAFEL(path=self.params['data_path'],start=0,end=300,size=self.params['img_size'],global_max=self.params['max'],global_min=self.params['min'],epsilon=self.params['epsilon'])
        elif self.params['dataset'] =='hurricane':
            dataset=Hurricane(path=self.params['data_path'],start=1,end=41,size=self.params['img_size'],field=self.params['field'],global_max=self.params['max'],global_min=self.params['min'],epsilon=self.params['epsilon'])
        elif self.params['dataset'] == 'exaalt':
            dataset=EXAALT(path=self.params['data_path'],start=0,end=4000)
        elif self.params['dataset'] == 'aramco':
            dataset=ARAMCO(path=self.params['data_path'],start=self.params['start'],end=self.params['end'],size=self.params['img_size'],global_max=0.0386,global_min=-0.0512,cache_size=self.params['cache_size'],epsilon=self.params['epsilon'])
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True,num_workers=0)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()
        self.params['epsilon']=float(self.params['epsilon'])

        if self.params['dataset'] == 'celeba':
            celeba=CelebA(root = self.params['data_path'],split = "test",transform=transform,download=True)
            self.sample_dataloader =  DataLoader(celeba,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'cesm':
            dataset=CLDHGH(path=self.params['data_path'],start=50,end=52,size=self.params['img_size'],normalize=True)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='cesm_new':
            dataset=CESM(path=self.params['data_path'],start=50,end=52,size=self.params['img_size'],field=self.params['field'],global_max=self.params['max'],global_min=self.params['min'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)

            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='nyx':

            dataset=NYX(path=self.params['data_path'],start=3,end=4,size=self.params['img_size'],field=self.params['field'],log=self.params['log'],global_max=self.params['max'],global_min=self.params['min'],epsilon=self.params['epsilon'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)

            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='exafel':
            dataset=EXAFEL(path=self.params['data_path'],start=300,end=310,size=self.params['img_size'],global_max=self.params['max'],global_min=self.params['min'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)

            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='hurricane':
            dataset=Hurricane(path=self.params['data_path'],start=41,end=42,size=self.params['img_size'],field=self.params['field'],global_max=self.params['max'],global_min=self.params['min'])  
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)     
        elif self.params['dataset'] == 'exaalt':
            dataset=EXAALT(path=self.params['data_path'],start=4000,end=4400)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'aramco':
            dataset=ARAMCO(path=self.params['data_path'],start=1500,end=1503,size=self.params['img_size'],global_max=0.0386,global_min=-0.0512,cache_size=self.params['cache_size'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            transform =  SetRange
            #raise ValueError('Undefined dataset type')
        return transform

