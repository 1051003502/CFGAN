import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
class discriminator(nn.Module):
    def __init__(self,itemCount):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(itemCount*2,200),

            nn.ReLU(True),
            nn.Linear(200,50),
            nn.ReLU(True),

            nn.Linear(50,1),
            nn.Sigmoid()
        )
    def forward(self,data,condition):

        result=self.dis( torch.cat((data,condition),1) )
        return result
class generator(nn.Module):
    def __init__(self,itemCount):
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(itemCount,200),

            nn.ReLU(True),
            nn.Linear(200,200),
            nn.ReLU(True),

            nn.Linear(200, itemCount),
            nn.Sigmoid()
        )
    def forward(self,x):
        result=self.gen(x)
        return result