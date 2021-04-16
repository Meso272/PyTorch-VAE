import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,norm=nn.BatchNorm2d,actv=nn.ReLU):
        super().__init__()
        self.norm=norm
        self.actv=actv(inplace=True)
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            norm(out_channels),
            actv(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            norm(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                norm(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return self.actv(self.residual_function(x) + self.shortcut(x))



class BasicBlock_Decode(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,norm=nn.BatchNorm2d,actv=nn.ReLU):
        super().__init__()
        self.norm=norm
        self.actv=actv(inplace=True)
        #residual function
        self.residual_function = nn.Sequential(
            nn.ConvTranspose2d(in_channels*BasicBlock_Decode.expansion, in_channels , kernel_size=3, padding=1, output_padding=0,bias=False),

            norm(in_channels),
            actv(inplace=True),
            
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride-1, bias=False),
            norm(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels *BasicBlock_Decode.expansion !=  out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels*BasicBlock_Decode.expansion, out_channels , kernel_size=1, stride=stride, padding=0,output_padding=1,bias=False),
                norm(out_channels)
            )

    def forward(self, x):
        return self.actv(self.residual_function(x) + self.shortcut(x))
'''
class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


'''

class ResNet_Encoder(nn.Module):
    #can both including conv1 or not

    def __init__(self, block, num_block,channel_list=[64,128,256,512],default_conv1=True,conv1=None,in_channels=64,avg_pooling=True,fc_out=0,norm=nn.BatchNorm2d,actv=nn.ReLU):
        super().__init__()
        ##??
        if default_conv1:
            self.in_channels = in_channels
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
                norm(in_channels),
                actv())
        else:
            self.conv1=conv1
            self.in_channels=in_channels
        self.conv2_x = self._make_layer(block, channel_list[0], num_block[0], 2,norm,actv)
        self.conv3_x = self._make_layer(block, channel_list[1], num_block[1], 2,norm,actv)
        self.conv4_x = self._make_layer(block, channel_list[2], num_block[2], 2,norm,actv)
        self.conv5_x = self._make_layer(block, channel_list[3], num_block[3], 2,norm,actv)
        if avg_pooling or fc_out>0:#auto pooling when fc layer included
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool=None
        if fc_out>0:
            self.fc = nn.Linear(channel_list[-1] * block.expansion, fc_out)
        else:
            self.fc=None

    def _make_layer(self, block, out_channels, num_blocks, stride,norm,actv):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride,norm=norm,actv=actv))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output=x
        #del x
        #print("i")
        #print(output.shape)
        if self.conv1!=None:
            output = self.conv1(output)
            #print("conv1")
            #print(output.shape)
        output = self.conv2_x(output)
        #print("conv2")
        #print(output.shape)
        output = self.conv3_x(output)
        #print("conv3")
        #print(output.shape)
        output = self.conv4_x(output)
        #print("conv4")

        #print(output.shape)
        #print("conv5")

        output = self.conv5_x(output)
        #print("conv3")
       

        #print(output.shape)
        if self.avg_pool!=None:
            output = self.avg_pool(output)
            #print("pool")

            #print(output.shape)
        if self.fc!=None:
            output = output.view(output.size(0), -1)
            output = self.fc(output)
            #print("fc")

            #print(output.shape)
        return output



class ResNet_Decoder(nn.Module):
    #optional convlast
    

    def __init__(self, block, num_block,channel_list=[512,256,128,64],fc_in=0,up_sampling=True,first_size=1,last_channel=64,default_convout=True,convout=None,norm=nn.BatchNorm2d,actv=nn.ReLU):
        super().__init__()
        
        self.in_channels=channel_list[0]*block.expansion
        self.first_size=first_size
        if fc_in>0:#when fc_in>0,  automatically up_sampling 
            self.fc = nn.Linear(fc_in, self.in_channels)
        else:
            self.fc=None
        if fc_in>0 or up_sampling:
            self.up_sampling = nn.Upsample(size=first_size)
        else:
            self.up_sampling=None

       


        self.out_channels = channel_list[1]*block.expansion
        

        self.deconv1_x = self._make_layer(block, channel_list[0], num_block[0], 2,norm,actv)
        self.out_channels = channel_list[2]*block.expansion
        self.deconv2_x = self._make_layer(block, channel_list[1], num_block[1], 2,norm,actv)
        self.out_channels = channel_list[3]*block.expansion
        self.deconv3_x = self._make_layer(block, channel_list[2], num_block[2], 2,norm,actv)
        self.out_channels = last_channel
        self.deconv4_x = self._make_layer(block, channel_list[3], num_block[3], 2,norm,actv)

        if default_convout:
            self.convout=nn.Sequential(nn.Conv2d(last_channel, 1, kernel_size=3, padding=1, bias=False),nn.Tanh())
        else:
            self.convout=convout
        
        

    def _make_layer(self, block, in_channels, num_blocks, stride,norm,actv):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        
        layers = []
        for i in range(num_blocks-1):
            
            layers.append(block(in_channels, in_channels, 1,norm=norm,actv=actv))
        layers.append(block(in_channels, self.out_channels, stride,norm=norm,actv=actv))
            

        return nn.Sequential(*layers)

    def forward(self, x):
        output=x
        #del x
        #print("decoder")

        #print(output.shape)

        if self.fc!=None:
            output = self.fc(output)
            output= output.view(output.size(0), self.in_channels,1,1)
            #print("dfc")

            #print(output.shape)
        if self.up_sampling!=None:
            output=self.up_sampling(output)
            #print("umsampling")

            #print(output.shape)
        output = self.deconv1_x(output)
        #print("deconv1")

        #print(output.shape)
        output = self.deconv2_x(output)
        #print("deconv2")
        #print(output.shape)
        output = self.deconv3_x(output)
        #print("deconv3")
        #print(output.shape)
        output = self.deconv4_x(output)
        #print("deconv4")
        #print(output.shape)
        
        if self.convout!=None:
            output=self.convout(output)
            #print("out")
            #print(output.shape)
        return output
'''
def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


'''
