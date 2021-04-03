import torch
from torch.autograd import Variable

class Round_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        
        #self.save_for_backward(input_)         # 
        return torch.round(input_)
    @staticmethod
    def backward(ctx, grad_output):
        
       
        grad_input = grad_output.clone()
                
        return grad_input