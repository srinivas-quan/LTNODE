import argparse
import os
import shutil
import time
import glob

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from attacks.fgsm import FGSM
from attacks.pgd import PGD

from src.datasets.image_loaders import get_image_loader
from src.utils import mkdir, save_object, cprint, load_object, to_variable
from src.probability import depth_gamma_VI
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_img_resnets import resnet18, resnet34, resnet50, resnet101, simple
from OOD_utils import *
from test_methods import class_brier,class_ll,class_ECE,class_err
import random
import calculate_log as callog
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


parser.add_argument('--dataset', type=str, default='MNIST',
                    choices=["CIFAR10", "CIFAR100", "SVHN", "MNIST", "Fashion"],
                    help='dataset to train (default: MNIST)')
parser.add_argument('--data_dir', type=str, default='../data/',
                    help='directory where datasets are saved (default: ../data/)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=None, type=int,
                    help='number of total epochs to run (if None, use dataset default)')
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--savedir', default='./results/', type=str,
                    help='path where to save checkpoints (default: ./results/)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use. (default: 0)')
parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size to use. (default: 256)')
parser.add_argument('--model', type=str, default='simple',
                    choices=["resnet18", "resnet32", "resnet50", "resnet101"],
                    help='model to train (default: resnet50)')
#parser.add_argument('--start_depth', default=1, type=int,
#                    help='first layer to be uncertain about (default: 1)')
#parser.add_argument('--end_depth', default=7, type=int,
#                    help='last layer to be uncertain about + 1 (default: 13)')
parser.add_argument('--q_nograd_its', default=0, type=int,
                    help='number of warmup epochs (where q is not learnt) (default: 0)')
parser.add_argument('--attack', type=str, choices=['FGSM', 'PGD'], default='FGSM')
parser.add_argument('--seed', type=float, default=0)

best_err1 = 1
lr = 0.1
momentum = 0.9


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dataset = args.dataset
    workers = args.workers
    epochs = args.epochs
    weight_decay = args.weight_decay
    resume = args.resume
    savedir = args.savedir
    gpu = args.gpu
    q_nograd_its = args.q_nograd_its
    batch_size = args.batch_size
    data_dir = args.data_dir
    #start_depth = args.start_depth
    #end_depth = args.end_depth
    model = args.model

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    savedir += "/" + "_".join([dataset, model, "DUN", f"warm{q_nograd_its}"]) 
    savedir += "_wd" if weight_decay != 0 else "_nowd"
    #num = len(glob.glob(savedir + "*"))
    num = args.seed
    savedir += f"_{num}"

    epoch_dict = {
        'Imagenet': 90,
        'SmallImagenet': 90,
        'CIFAR10': 300,
        'CIFAR100': 300,
        'SVHN': 90,
        'Fashion': 90,
        'MNIST': 90
    }

    milestone_dict = {
        'Imagenet': [30, 60],  # This is pytorch default
        'SmallImagenet': [30, 60],
        'CIFAR10': [150, 225],
        'CIFAR100': [150, 225],
        'SVHN': [50, 70],
        'Fashion': [40, 70],
        'MNIST': [40, 70]
    }
    target_datasets = {
    "MNIST": ["Fashion"],
    "Fashion": ["MNIST", "KMNIST"],
    "CIFAR10": ["SVHN"],
    "CIFAR100": ["SVHN"],
    "SVHN": ["CIFAR10"]
    }
    if epochs is None:
        epochs = epoch_dict[dataset]
    milestones = milestone_dict[dataset]

    initial_conv = '3x3' if dataset in ['Imagenet', 'SmallImagenet'] else '1x3'
    input_chanels = 1 if dataset in ['MNIST', 'Fashion'] else 3
    if dataset in ['Imagenet', 'SmallImagenet']:
        num_classes = 1000
    elif dataset in ['CIFAR100']:
        num_classes = 100
    else:
        num_classes = 10

    if model == 'resnet18':
        model_class = resnet18
    elif model == 'resnet18':
        model_class = resnet34
    elif model == 'resnet50':
        model_class = resnet50
    elif model == 'resnet101':
        model_class = resnet101
    elif model == 'simple':
        model_class = simple
    else:
        raise Exception('requested model not implemented')

    cuda = torch.cuda.is_available()
    print('cuda', cuda)
    assert cuda

    n_samples = 10

    prior_probs = [2.,0.5]#alpha,beta #instead 2,0.01
    prob_model = depth_gamma_VI(prior_probs, cuda=cuda)

    model = model_class(prob_model,num_classes=num_classes,n_samples = n_samples,input_chanels=input_chanels)
    print("No. of parameters",sum(p.numel() for p in model.parameters() if p.requires_grad))

    N_train = 0


    net = DUN_VI(model, prob_model, N_train, lr=lr, momentum=momentum, weight_decay=weight_decay, cuda=cuda,
                 schedule=milestones, regression=False, pred_sig=None)
    net.prob_model.Train = False
    model_path = savedir + '/checkpoint.pth.tar'
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        start_epoch, best_err1 = net.load(model_path)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    print(model)
    
    #test_loop(net, dname=dataset, data_dir=data_dir, epochs=epochs, workers=workers, resume=resume, savedir=savedir,
    #           q_nograd_its=q_nograd_its, batch_size=batch_size,num_classes=num_classes,n_samples=n_samples)
    
     
    
    #OOD_rej(net, target_dataset = target_datasets[dataset][0],dname=dataset, data_dir=data_dir, epochs=epochs, workers=workers, resume=resume, savedir=savedir,
    #           q_nograd_its=q_nograd_its, batch_size=batch_size,num_classes=num_classes,n_samples=n_samples)
           
    adver(net, args.attack,dname=dataset, data_dir=data_dir, epochs=epochs, workers=workers, resume=resume, savedir=savedir,
               q_nograd_its=q_nograd_its, batch_size=256,num_classes=num_classes,n_samples=n_samples)
    #adver(net, 'FGSM',dname=dataset, data_dir=data_dir, epochs=epochs, workers=workers, resume=resume, savedir=savedir,
    #           q_nograd_its=q_nograd_its, batch_size=batch_size,num_classes=num_classes,n_samples=n_samples)
    
def adver(net, attack, dname, data_dir, epochs=90, workers=4, resume='', savedir='./',
               save_all_epochs=False, q_nograd_its=0, batch_size=256,num_classes=10,n_samples=10):
    print("testing under the adversarial attack :",attack)
    if dname == 'MNIST':
        mean = [0.1307]
        std = [0.3081]
    elif dname == 'Fashion':
        mean = [0.2860]
        std = [0.3530]
    elif dname == "SVHN":
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
    elif dname == "CIFAR10":
        mean = [0.4914, 0.4822, 0.4465]
        std =  [0.2470, 0.2435, 0.2616]
    
    
    net.model.n_samples= n_samples

    source_loader = rotate_load_dataset(dname,0,batch_size=batch_size)
 
    nb_samples = 0
    
    #samples = net.prob_model.get_samples(n_samples)
    for x, y in source_loader:
        nb_samples += x.shape[0]
    
    Entropy_count = []
    Confidence_count = []

    if attack == 'FGSM':
        steps = [0.0,0.05,0.1,0.15,0.20,0.25,0.3]
    elif attack == 'PGD':
        steps = np.arange(0,25,4)

    Error = []
    Prob_vec = []
    for step in steps:
        #print("Current step ",step," ",attack)
        i = 0 
        target = []
        #print(len(source_loader.dataset))
        Probs = np.zeros((n_samples,nb_samples,num_classes))
        for x, y in source_loader:
            #net.model.eval()
            #print("labels ",y.shape)
            x, y = to_variable(var=(x, y), cuda=True)
            if attack == 'FGSM':
                adv_images = FGSM(x,y,net,mean,std,eps=step)
            elif attack == 'PGD':
                adv_images = PGD(x,y,net,mean,std,eps=0.2,steps=step)
            #print(net.prob_model.samples.shape,adv_images.shape)
            target.append(y)
            #Probs[i:i+x.shape[0],:] = net.predict(adv_images,samples=net.prob_model.samples).detach().cpu().numpy()
            
            with torch.no_grad():
                act_vec = net.model.forward_test(adv_images)#,samples=net.prob_model.samples)
                Probs[:,i:i+x.shape[0],:] = net.prob_model.get_activations(act_vec,softmax=True).detach().cpu().numpy()
            
            
            i = i + x.shape[0]
     
           
        pred_mu_source = torch.Tensor((Probs/n_samples).sum(0))
        Prob_vec.append(pred_mu_source.numpy())
        
        pred_source = pred_mu_source.max(dim=1, keepdim=False)[1]#gets index of the class with max probability
        source_entropy = -(pred_mu_source * pred_mu_source.clamp(min=1e-35).log()).sum(dim=1).cpu().numpy()
        pred_prob_source = pred_mu_source.max(dim=1, keepdim=False)[0]#gets prbability of the class with max probability
        target = torch.cat(target, dim=0)
        target = target.data.cpu()

        Error.append(class_err(y=target, model_out=pred_mu_source))
        print(Error[-1])
        entropy_count = []
        confidence_count = []
        for inc in np.arange(0,2.5,0.01):
            entropy_count.append(sum(torch.logical_and(torch.Tensor(source_entropy)>=inc,torch.Tensor(source_entropy)<0.01+inc)))
            confidence_count.append(sum(pred_prob_source>=inc))

        Entropy_count.append(entropy_count)
        Confidence_count.append(confidence_count)
    
    np.save(savedir+'/_'+attack+'_entropy_count_ATUNODE.npy',Entropy_count)
    np.save(savedir+'/_'+attack+'_confidence_count_ATUNODE.npy',Confidence_count)
    np.save(savedir+'/_'+attack+'_Error_ATUNODE.npy',Error)
    np.save(savedir+'/_'+attack+'_Prob_vec_ATUNODE.npy',Prob_vec)
    
def OOD_rej(net, target_dataset,dname, data_dir, epochs=90, workers=4, resume='', savedir='./',save_all_epochs=False, q_nograd_its=0, batch_size=256,num_classes=10,n_samples=10):
    print(target_dataset,dname)
    net.model.n_samples= n_samples

    source_loader, target_loader = cross_load_dataset(dname, target_dataset, data_dir=data_dir,
                                                      batch_size=batch_size, cuda=True, workers=workers)
    
    i = 0
    nb_samples = 0
    for x, y in target_loader:
        nb_samples += x.shape[0]
    Probs = np.zeros((n_samples,nb_samples,num_classes))
    for x, y in target_loader:
        net.model.eval()
        with torch.no_grad():
            x, y = to_variable(var=(x, y), cuda=True)
   
            act_vec = net.model.forward_test(x)
            probs = net.prob_model.get_activations(act_vec,softmax=True).detach().cpu().numpy()
            Probs[:,i:i+x.shape[0],:] = probs
            i = i + x.shape[0]
    print("target ",Probs[:,0,:])
    pred_mu_target = torch.Tensor((Probs/n_samples).sum(0))
    pred_target = pred_mu_target.max(dim=1, keepdim=False)[1]
    pred_prob_target = pred_mu_target.max(dim=1)[0]
    target_entropy = -(pred_mu_target * pred_mu_target.clamp(min=1e-35).log()).sum(dim=1).cpu().numpy()
    np.save(savedir+'/confidence_Base_Out.npy',pred_prob_target.cpu().numpy())#For ROC

    i = 0
    nb_samples = 0
    target = []
    for x, y in source_loader:
        nb_samples += x.shape[0]
    Probs = np.zeros((n_samples,nb_samples,num_classes))
    for x, y in source_loader:
        net.model.eval()
        with torch.no_grad():
            x, y = to_variable(var=(x, y), cuda=True)
            target.append(y)
            act_vec = net.model.forward_test(x)
            probs = net.prob_model.get_activations(act_vec,softmax=True).detach().cpu().numpy()
            Probs[:,i:i+x.shape[0],:] = probs
            i = i + x.shape[0]
    print("source ",Probs[:,0,:])
    pred_mu_source = torch.Tensor((Probs/n_samples).sum(0))
    pred_source = pred_mu_source.max(dim=1, keepdim=False)[1]#[1] for indeces [0] for values
    pred_prob_source = pred_mu_source.max(dim=1, keepdim=False)[0]
    source_entropy = -(pred_mu_source * pred_mu_source.clamp(min=1e-35).log()).sum(dim=1).cpu().numpy()

    target = torch.cat(target, dim=0)
    target = target.data.cpu()

    target_entropy_count = []
    target_confidence_count = []
    for inc in np.arange(0,2.5,0.01):
        target_entropy_count.append(sum(torch.logical_and(torch.Tensor(target_entropy)>=inc,torch.Tensor(target_entropy)<0.01+inc)))
        target_confidence_count.append(sum(pred_prob_target>=inc))
    np.save(savedir+'/_ood_entropy_count_ATUNODE.npy',target_entropy_count)
    np.save(savedir+'/_ood_confidence_count_ATUNODE.npy',target_confidence_count)

    np.save(savedir+'/confidence_Base_In.npy',pred_prob_source.cpu().numpy())#For ROC

    err_vec_in = pred_source.ne(target.data).cpu().numpy()
    err_vec_out = np.ones(target_entropy.shape[0])
    full_err_vec = np.concatenate([err_vec_in, err_vec_out], axis=0)
    full_entropy_vec = np.concatenate([source_entropy, target_entropy], axis=0)
    sort_entropy_idxs = np.argsort(full_entropy_vec, axis=0)
    Npoints = sort_entropy_idxs.shape[0]

    np.save(savedir+'/confidence_Base_Succ.npy',pred_prob_source[err_vec_in==0].cpu().numpy())#For ROC
    np.save(savedir+'/confidence_Base_Err.npy',pred_prob_source[err_vec_in==1].cpu().numpy())#For ROC

    print('calculate metrics for OOD')
    callog.metric(savedir, 'OOD')
    print('calculate metrics for mis')
    callog.metric(savedir, 'mis')

    err_props = []
    rejection_step=0.005
    for rej_prop in np.arange(0, 1, rejection_step):
        N_reject = np.round(Npoints * rej_prop).astype(int)
        if N_reject > 0:
            accepted_idx = sort_entropy_idxs[:-N_reject]
        else:
            accepted_idx = sort_entropy_idxs

        err_props.append(full_err_vec[accepted_idx].sum() / accepted_idx.shape[0])

        assert err_props[-1].max() <= 1 and err_props[-1].min() >= 0

    print("source_entropy.target_entropy",np.sum(source_entropy)/source_entropy.shape[0],np.sum(target_entropy)/target_entropy.shape[0])
    np.save(savedir+'/_entropy_source_target.npy',[np.sum(source_entropy)/source_entropy.shape[0],np.sum(target_entropy)/target_entropy.shape[0]])
    np.save(savedir+'/_err_props_ATUNODE.npy',err_props)


def test_loop(net, dname, data_dir, epochs=90, workers=4, resume='', savedir='./',
               save_all_epochs=False, q_nograd_its=0, batch_size=256,num_classes=10,n_samples=10):
    mkdir(savedir)
    global best_err1

    # Load data here:
    _, train_loader, val_loader, _, _, Ntrain = \
        get_image_loader(dname, batch_size, cuda=True, workers=workers, distributed=False, data_dir=data_dir)

    net.N_train = Ntrain

    start_epoch = 0

    
    print("Initial alpha and Beta values ",net.prob_model.q_a,net.prob_model.q_b)
    # optionally resume from a checkpoint
    net.model.n_samples= n_samples
        

    tic = time.time()
    nb_samples = 0
    dev_loss = 0
    err_dev = 0

    target_vec = []
    for x, y in val_loader:
        nb_samples += x.shape[0]
        target_vec.append(y)
    target_vec = torch.cat(target_vec, dim=0)
    target_vect = target_vec.data.cpu()
    Probs = np.zeros((n_samples,nb_samples,num_classes))
    
    '''
    for s in range(n_samples):
        print("samples s:",s)
        i = 0
        for x, y in val_loader:
            net.model.eval()
            with torch.no_grad():
                x, y = to_variable(var=(x, y), cuda=True)
       
                act_vec = net.model.forward(x)
                probs = net.prob_model.get_activations(act_vec,softmax=True).detach().cpu().numpy()[0]
                Probs[s,i:i+x.shape[0]] = probs
                i = i + x.shape[0]

    pred_mu = (Probs/n_samples).sum(0)
    model_var = (((Probs**2)).sum(0)/num_samples - pred_mu**2)
    model_var = np.power(model_var,0.5)
    entropy = np.sum(-(pred_mu * np.log(pred_mu)))
    print("Entropy ",entropy/nb_samples)
    '''
    results = []
    for rot in range(0, 181, 15):
        Probs = np.zeros((n_samples,nb_samples,num_classes))
        val_loader = rotate_load_dataset(dname,rot)
        conf_count = []#number of samples classified as a class c with  prob > P, c may be correct or wron gclass
        acc_count = []#number of samples correctly classified as a class c with  prob > P, or accuracy 
        i = 0
        print('rot',rot)
        for x, y in val_loader:
            net.model.eval()
            with torch.no_grad():
                x, y = to_variable(var=(x, y), cuda=True)
       
                act_vec = net.model.forward_test(x)
                probs = net.prob_model.get_activations(act_vec,softmax=True).detach().cpu().numpy()
                Probs[:,i:i+x.shape[0],:] = probs
                i = i + x.shape[0]

        pred_mu = torch.Tensor((Probs/n_samples).sum(0))
        pred = pred_mu.max(dim=1, keepdim=False)[1]  # get the index of the max probability
        pred_prob = pred_mu.max(dim=1)[0] #class with max probability
        for inc in np.arange(0,1+0.01,0.01):
            
            conf_count.append(sum(pred_prob>=inc))#computes number of samples with >=inc in max proability

            Err = pred[pred_prob>=inc].ne(target_vec[pred_prob>=inc].data).sum().item() / sum(pred_prob>=inc)#target_vec.shape[0]
            #get the probabolotyes with >=inc values, egt the true class values of those samples with prob>=inc
            acc_count.append((1 - Err)*100.)

        np.save(savedir+'/_conf_count_ATUNODE%d.npy'%(rot),conf_count)
        np.save(savedir+'/_acc_count_ATUNODE%d.npy'%(rot),acc_count)
        brier = class_brier(y=target_vec, probs=pred_mu, log_probs=None)
        err = class_err(y=target_vec, model_out=pred_mu)
        ll = class_ll(y=target_vec, probs=pred_mu, log_probs=None, eps=1e-40)
        ece = class_ECE(y=target_vec, probs=pred_mu, log_probs=None, nbins=10)
        results.append([rot,err,ll,brier,ece])
        cprint('g', 'rot = %f, err = %f, ll = %f, brier = %f, ece =%f\n' % (rot,err,ll,brier,ece), end="")
    np.save(savedir+'/_results_ATUNODE.npy',results)
        

       


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
