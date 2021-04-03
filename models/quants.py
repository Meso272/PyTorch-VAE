import torch
from torch.autograd import Variable

class Round_1(torch.autograd.Function):

    def forward(self, input_):
        
        #self.save_for_backward(input_)         # 
        return torch.round(input_)

    def backward(self, grad_output):
        
       
        grad_input = grad_output.clone()
                
        return grad_input