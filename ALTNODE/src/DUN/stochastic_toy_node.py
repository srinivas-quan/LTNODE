import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.init as init
import numpy as np

__all__ = ['toy']


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class NODE(nn.Module):

    def __init__(self, dim):
        super(NODE, self).__init__()
        #self.norm1 = norm(dim)
        #self.tanh = nn.Tanh()#
        self.relu =nn.ReLU(inplace=True)
        self.lin1 = nn.Linear(dim+1, dim*2)
        #self.dp = nn.Dropout(0.3)
        self.norm2 = norm(dim*2)

        
        self.lin2 = nn.Linear(dim*2, dim*3)
        self.norm3 = norm(dim*3)
        
        
        self.lin3 = nn.Linear(dim*3, dim*2)
        self.norm4 = norm(dim*2)
        
        self.lin4 = nn.Linear(dim*2, dim)
        self.norm5 = norm(dim)
        
        

    def forward(self,t,x):
        #out = self.norm1(x)
        #out = self.relu(out)

        t_vec = torch.ones(x.shape[0], 1).type(x.type()) * t
        # Shape (batch_size, data_dim + 1)
        #print(t_vec.shape,x.shape)
        t_and_x = torch.cat([t_vec, x], 1)
        # Shape (batch_size, hidden_dim)
        #out = self.fc1(t_and_x)
        out = self.lin1(t_and_x)
        out = self.norm2(out)
        out = self.relu(out)
        
        #out = self.dp(out)
        
        out = self.lin2(out)
        out = self.norm3(out)
        out = self.relu(out)
        
        #out = self.dp(out)

        
        
        out = self.lin3(out)
        out = self.norm4(out)
        out = self.relu(out)
        #out = self.dp(out)
        
        out = self.lin4(out)
        out = self.norm5(out)
        out = self.relu(out)
        #out = self.dp(out)
               

        '''
        
        out = self.lin2(out)
        out = self.norm3(out)
        out = self.relu(out)

        
        out = self.norm3(out)
        out = self.relu(out)
        out = self.lin3(out)
        out = self.norm4(out)
        out = self.relu(out)
        out = self.lin4(out)
        '''

        #out = self.norm3(out)
        return out

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

def norm(dim):
    return nn.BatchNorm1d(dim)
    #return nn.GroupNorm(min(10, dim), dim)
def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0, std=0.1)
            init.constant_(m.bias, val=0)
Size = 50
class layers(nn.Module):
    def __init__(self, input_dim):
        super(layers, self).__init__()
        #self.norm1 = norm(dim_in)
        self.linear1 = nn.Linear(input_dim, Size)
        self.norm1 = norm(Size)
        #self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        
        
        self.linear2 = nn.Linear(Size, Size*2)
        self.norm2 = norm(Size*2)
        
        self.linear3 = nn.Linear(Size*2, Size*3)
        self.norm3 = norm(Size*3)
        
        self.linear4 = nn.Linear(Size*3, Size)
        self.norm4 = norm(Size)
        
        
        #self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.linear2(out)
        out = self.relu(out)
        out = self.norm2(out)
        
        out = self.linear3(out)
        out = self.norm3(out)
        out = self.relu(out)
        
        out = self.linear4(out)
        out = self.norm4(out)
        out = self.relu(out)
        
        
        return out
from torch_ACA import odesolve
class Net(nn.Module):
    def __init__(self, prob_model,n_samples,input_dim):
        super(Net, self).__init__()

        
        
        self.fcnn = layers(input_dim)
        self.Node = NODE(Size)
        self.Encoder = nn.Sequential(nn.Linear(input_dim,Size),nn.ReLU(inplace=True),norm(Size),nn.Linear(Size,Size),nn.ReLU(inplace=True),norm(Size),nn.Linear(Size,2))
            
        self.n_samples = n_samples
        self.prob_model = prob_model
        self.fc_layers = nn.Linear(Size, 1)#nn.Sequential(nn.Linear(Size,Size), nn.ReLU(inplace=True),norm(Size), nn.Linear(Size, 1))#
        self.deltat = 0.1
        self.output_dim = 1
        self.apply(init_params)

    def forward(self, x,samples = None):
        out = self.fcnn(x)
        alpha_beta = self.Encoder(x)


        alpha = torch.exp(alpha_beta[:,0])
        beta = torch.exp(alpha_beta[:,1])
        #print("Eoncded:",alpha,beta)
        self.prob_model.q_a = alpha
        self.prob_model.q_b = beta
        act_vec = x.new_zeros(1, x.shape[0], self.output_dim)
        if samples == None:
            samples = self.prob_model.get_samples(alpha,beta,alpha.shape[0])
        print("samples",samples.shape)
        samples = samples[:,0]
        samples[samples>=50]= 50

        max_T = samples.max()
        #temp = x.new_zeros(int(max_T/self.deltat)+1,x.shape[0], self.output_dim)#time_stampsxsamplesxclasses
        temp = x.new_zeros(out.shape)
        self.prob_model.current_posterior = samples

        self.prob_model.current_posterior_pdf = self.prob_model.get_q_probs()
        t = 0
        i = 0
        '''
        while t + self.deltat <= max_T:
            t = t + self.deltat
            out = out + self.Node(t,out)*self.deltat
            #print(t,max_T)
            temp[i,:,:] = self.fc_layers(out)    #stores every sample representaton at every time, here i is index of dt,2dt,3dt...
            i += 1

        indeces = samples/self.deltat #samples contains T values and are converted to indeces 
        indeces = indeces.type(torch.LongTensor)
        #act_vec = temp[indeces]
         
        for i in range(x.shape[0]):
            #print(i,samples[i],indeces[i],act_vec[i,:].shape,temp[indeces[i],i,:].shape)
            act_vec[0,i,:] = temp[indeces[i],i,:]# for ith sample, represnetation stored at index n*dt+eps, indexed by n->indeces[i]
        '''
        samples,indeces  = samples.sort()
        #for sample in samples:
        options = {}
        
        options.update({'method':'Dopri5'})
        options.update({'h': None})
        options.update({'t0': t})
        #options.update({'t1': sample.detach().cpu().numpy()[0]})
        options.update({'t_eval':list(samples.cpu().numpy())})
        options.update({'t1':max_T})#max_T}),sample
        options.update({'rtol': 1e-5})
        options.update({'atol': 1e-5})
        options.update({'print_neval': True})
        options.update({'neval_max': 5000})
        options.update({'regenerate_graph': True})
        out = odesolve(self.Node, out, options) #out is mxmxc, 
        #print(out.shape,temp.shape)
        for i in range(x.shape[0]):
            #temp[indeces[i],:] = out[i,i,:]
            #print(i,indeces[i])
            #temp[i,:] = out[indeces[i],i,:]#indeces[i],i,:
            temp[indeces[i],:] = out[i,indeces[i],:]
            #temp[i,:] = out[i,i,:]
        
        act_vec[0,:] = self.fc_layers(temp)
        #for param in self.fc_layers.parameters():

        
  
        del temp
        return act_vec
    def forward_test(self, x,samples=None):
        out = self.fcnn(x)
        alpha_beta = self.Encoder(x)


        alpha = torch.exp(alpha_beta[:,0])
        beta = torch.exp(alpha_beta[:,1])
        '''
        print("Eoncded: Left " ,alpha[0:200],beta[0:200])
        print("Eoncded: train1 " ,alpha[200:260],beta[200:260])
        print("Eoncded: middle " ,alpha[260:500],beta[260:500])
        print("Eoncded: train2 " ,alpha[500:580],beta[500:580])
        print("Eoncded: right " ,alpha[580:],beta[580:])
        '''  
        #np.save("./all_synthetic_alpha.npy",alpha.cpu())
        #np.save("./all_synthetic_beta.npy",beta.cpu())
        self.prob_model.q_a = alpha
        self.prob_model.q_b = beta
        print(" variational parameters", self.prob_model.q_a,self.prob_model.q_b)
        #act_vec = x.new_zeros(self.n_samples, x.shape[0], self.num_classes)
        act_vec = x.new_zeros(self.n_samples,x.shape[0], self.output_dim)#it stores only final for every datapoint.
        if samples == None:
            samples = self.prob_model.get_samples(alpha,beta,n=self.n_samples) #return self.n_samplesxN_b(batch_size) T samples for each a and b
        print("samples.shape",samples.shape)
        #samples = samples[0,:]        
        #samples[samples>=50]= 50

        max_T = samples.max()
        #print("max_T",max_T)
        #temp = x.new_zeros(int(max_T/self.deltat)+1,x.shape[0], self.output_dim)#time_stampsxsamplesxclasses
        
        #temp[0,:,:] = self.fc_layers(x)#if for any sample 'x',T is less than dt, x will be returned, 
        temp = x.new_zeros(out.shape)
        #self.prob_model.current_posterior = samples
        #self.prob_model.current_posterior_pdf = self.prob_model.get_q_probs()
        t = 0
        i = 1
        #print("max_T,max_T/self.deltat,int(max_T/self.deltat)",max_T,max_T/self.deltat,int(max_T/self.deltat))
        '''
        while t + self.deltat <= max_T:
            t = t + self.deltat
            out = out + self.Node(t,out)*self.deltat
            #print(t,max_T)
            temp[i,:,:] = self.fc_layers(out)    #stores every sample representaton at every time, here i is index of dt,2dt,3dt...
            i += 1

        for j in range(self.n_samples):
            indeces = samples[j]/self.deltat #samples contains T values and are converted to indeces 
            indeces = indeces.type(torch.LongTensor)
            #act_vec = temp[indeces]
             
            for i in range(x.shape[0]):
                #print(i,samples[i],indeces[i],act_vec[i,:].shape,temp[indeces[i],i,:].shape)
                act_vec[j,i,:] = temp[indeces[i],i,:]# for ith sample, represnetation stored at index n*dt+eps, indexed by n->indeces[i]
        '''
        for i in range(self.n_samples):
            samples1,indeces  = samples[i].sort()
            #samples1[samples1>=10]= 10
            max_T = samples1.max()
            #for sample in samples:
            print("in loop",samples1.shape)
            options = {}
        
            options.update({'method':'Dopri5'})
            options.update({'h': None})
            options.update({'t0': t})
            #options.update({'t1': sample.detach().cpu().numpy()[0]})
            options.update({'t_eval':list(samples1.cpu().numpy())})
            options.update({'t1':max_T})#max_T}),sample
            options.update({'rtol': 1e-5})
            options.update({'atol': 1e-5})
            options.update({'print_neval': False})
            options.update({'neval_max': 5000})
            options.update({'regenerate_graph': False})
            out1 = odesolve(self.Node, out, options) #out is mxmxc, 
        
            for j in range(x.shape[0]):
                #temp[indeces[i],:] = out[i,i,:]
                temp[indeces[j],:] = out1[j,indeces[j],:]
                #temp[i,:] = out[i,i,:]
        
            act_vec[i,:] = self.fc_layers(temp)
        #for param in self.fc_layers.parameters():

        del temp
        return act_vec
    

    
def toy(prob_model,n_samples,input_dim):
    return Net(prob_model,n_samples,input_dim)

    



