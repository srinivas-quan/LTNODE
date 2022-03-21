import torch
import torch.nn as nn
from torch.autograd import Variable
#from ..attack import Attack
"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
    
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 0.007)
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)
        
"""

#old
def FGSM(images,labels,model,mean,std,eps=0.007):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss = nn.NLLLoss()#in generla we want to minimize,  so w(t+1) = w(t) - lr*loss,  but here we wwant to maximize w(t+1) = w(t) + lr*loss
    images.requires_grad = True
    
    act_vec = model.model.forward_test(images,regenerate_graph=True)
    outputs = model.prob_model.get_activations(act_vec,softmax=True)#.detach().cpu().numpy()
    outputs = (outputs/outputs.shape[0]).sum(0)
   
    cost = loss(outputs, labels)
    model.model.zero_grad()
    grad = torch.autograd.grad(cost, images,retain_graph=False, create_graph=False)[0]

    adv_images = images + eps*grad.sign()
    for j in range(images.shape[1]):
        adv_images[:,j,:,:] = torch.clamp(adv_images[:,j,:,:], -mean[j]/std[j], (1-mean[j])/std[j]).detach()
    

    return adv_images
'''
def FGSM(images,labels,model,mean,std,eps=0.007):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss = nn.NLLLoss()#in generla we want to minimize,  so w(t+1) = w(t) - lr*loss,  but here we wwant to maximize w(t+1) = w(t) + lr*loss
    images.requires_grad = True
    outputs = model.predict_with_grad(images)
   

    cost = loss(outputs, labels)
    model.model.zero_grad()

    print(cost)
    grad = torch.autograd.grad(cost, images,retain_graph=False, create_graph=False)[0]

    adv_images = images + eps*grad.sign()
    for j in range(images.shape[1]):
        adv_images[:,j,:,:] = torch.clamp(adv_images[:,j,:,:], -mean[j]/std[j], (1-mean[j])/std[j]).detach()
    

    return adv_images
'''