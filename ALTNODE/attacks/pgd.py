import torch
import torch.nn as nn

"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 0.3)
        alpha (float): step size. (DEFAULT: 2/255)
        steps (int): number of steps. (DEFAULT: 40)
        random_start (bool): using random initialization of delta. (DEFAULT: False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
"""


def PGD(images,labels,model,mean,std,eps=0.007,steps=40,alpha=2/255):
    images = images.clone().detach()
    labels = labels.clone().detach()
   
   #loss = nn.CrossEntropyLoss()
    loss = nn.NLLLoss()#in generla we want to minimize,  so w(t+1) = w(t) - lr*loss,  but here we wwant to maximize w(t+1) = w(t) + lr*loss
    adv_images = images.clone().detach()
    for i in range(steps):
        adv_images.requires_grad = True
        act_vec = model.model.forward_test(adv_images,regenerate_graph=True)
        outputs = model.prob_model.get_activations(act_vec,softmax=True)#.detach().cpu().numpy()
        outputs = (outputs/outputs.shape[0]).sum(0)
   
        cost = loss(outputs, labels)
        model.model.zero_grad()
        grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        for j in range(images.shape[1]):
            adv_images = torch.clamp(images + delta,-mean[j]/std[j], (1-mean[j])/std[j]).detach()

    return adv_images
    
