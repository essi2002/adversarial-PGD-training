from model.model import *
import torch.nn.functional as F
epsilon = 0.03
iterations = 6
alpha = 0.007
class LinfPGD(object):
    def __init__(self,model):
        self.model = model
    def perturb(self,x_natural,y):
        x = x_natural.detach()
        x += torch.zeros_like(x).uniform_(-epsilon,epsilon)
        for i in range(iterations):
            x.requires_grad_()
            with torch.enable_grad():
                output = self.model(x)
                loss = F.cross_entropy(output,y)
            gradient = torch.autograd.grad(loss,[x])[0]
            adv_image = x.detach()+ alpha * torch.sign(gradient.detach())
            eta = torch.clamp(adv_image - x_natural,min=-epsilon,max=epsilon)
            x = torch.clamp(x_natural + eta,min=0,max = 1)
        return x


           


            

                






        
