import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.init as init


__all__ = ['concentric']




class NODE(nn.Module):

    def __init__(self, dim):
        super(NODE, self).__init__()
        #self.norm1 = norm(dim)
        #self.tanh = nn.Tanh()#
        hidden_dim = dim
        self.relu =nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        

    def forward(self,t,x):

        t_vec = torch.ones(x.shape[0], 1).type(x.type()) * t
        t_and_x = torch.cat([t_vec, x], 1)
        #print("in forward t_and_x",t_and_x.shape)
        out = self.fc1(t_and_x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

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
Size = 32
class layers(nn.Module):
    def __init__(self, input_dim):
        super(layers, self).__init__()
    def forward(self, x):
        out = x       
        return out
from torch_ACA import odesolve
class Net(nn.Module):
    def __init__(self, prob_model,num_classes,n_samples,input_dim):
        super(Net, self).__init__()

        
        
        #self.fcnn = layers(input_dim)
        self.num_classes = num_classes
        self.Node = NODE(Size)
        #self.Encoder = nn.Sequential(nn.Linear(input_dim,Size),nn.ReLU(inplace=True),norm(Size),nn.Linear(Size,Size),nn.ReLU(inplace=True),norm(Size),nn.Linear(Size,2))
        self.Encoder = nn.Sequential(nn.Linear(input_dim,Size),nn.ReLU(inplace=True),nn.Linear(Size,Size),nn.ReLU(inplace=True),nn.Linear(Size,2))    
        self.n_samples = n_samples
        self.prob_model = prob_model
        self.output_dim = 1
        self.fc_layers = nn.Linear(input_dim, num_classes)#nn.Sequential(nn.Linear(Size,Size), nn.ReLU(inplace=True),norm(Size), nn.Linear(Size, 1))#
        self.deltat = 0.1
        
        self.apply(init_params)

    def forward(self, x,samples = None):
        #out = self.fcnn(x)
        alpha_beta = self.Encoder(x)

        #print("x.shape",x.shape)
        alpha = torch.exp(alpha_beta[:,0])
        beta = torch.exp(alpha_beta[:,1])
        #print("Eoncded:",alpha,beta)
        self.prob_model.q_a = alpha
        self.prob_model.q_b = beta
        act_vec = x.new_zeros(1, x.shape[0], self.num_classes)
        if samples == None:
            samples = self.prob_model.get_samples(alpha,beta,alpha.shape[0])
        #print("samples",samples.shape,samples[:,0].shape)
        samples = samples[:,0]
        samples[samples>=50]= 50

        max_T = samples.max()
        #temp = x.new_zeros(int(max_T/self.deltat)+1,x.shape[0], self.output_dim)#time_stampsxsamplesxclasses
        temp = x.new_zeros(x.shape)
        self.prob_model.current_posterior = samples

        self.prob_model.current_posterior_pdf = self.prob_model.get_q_probs()
        t = 0
        i = 0
        samples,indeces  = samples.sort()
        #for sample in samples:
        options = {}
        
        options.update({'method':'Dopri5'})
        options.update({'h': None})
        options.update({'t0': t})
        #options.update({'t1': sample.detach().cpu().numpy()[0]})
        options.update({'t_eval':list(samples.cpu().numpy())})
        options.update({'t1':max_T})#max_T}),sample
        options.update({'rtol': 1e-2})
        options.update({'atol': 1e-2})
        options.update({'print_neval': False})
        options.update({'neval_max': 5000})
        options.update({'regenerate_graph': True})
        out = odesolve(self.Node, x, options) #out is mxmxc, 
        #print(out.shape,temp.shape)
        for i in range(x.shape[0]):
            #temp[indeces[i],:] = out[i,i,:]
            #print(i,indeces[i])
            #temp[i,:] = out[indeces[i],i,:]#indeces[i],i,:
            temp[indeces[i],:] = out[i,indeces[i],:]
            #temp[i,:] = out[i,i,:]
        
        act_vec[0,:] = self.fc_layers(temp)
        #for param in self.fc_layers.parameters():

        
        #print("act_vec.shape",act_vec.shape)
        del temp
        return act_vec
    def forward_test(self, x,samples=None):
        #out = self.fcnn(x)
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
        act_vec = x.new_zeros(self.n_samples,x.shape[0], self.num_classes)#it stores only final for every datapoint.
        if samples == None:
            samples = self.prob_model.get_samples(alpha,beta,n=self.n_samples) #return self.n_samplesxN_b(batch_size) T samples for each a and b
        print("samples.shape",samples.shape)
        #samples = samples[0,:]        
        #samples[samples>=50]= 50

        max_T = samples.max()
        #temp = x.new_zeros(int(max_T/self.deltat)+1,x.shape[0], self.output_dim)#time_stampsxsamplesxclasses
        
        #temp[0,:,:] = self.fc_layers(x)#if for any sample 'x',T is less than dt, x will be returned, 
        temp = x.new_zeros(x.shape)
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
            max_T = samples1.max()
            #for sample in samples:
            #print("in loop",samples1.shape)
            options = {}
        
            options.update({'method':'Dopri5'})
            options.update({'h': None})
            options.update({'t0': t})
            #options.update({'t1': sample.detach().cpu().numpy()[0]})
            options.update({'t_eval':list(samples1.cpu().numpy())})
            options.update({'t1':max_T})#max_T}),sample
            options.update({'rtol': 1e-2})
            options.update({'atol': 1e-2})
            options.update({'print_neval': False})
            options.update({'neval_max': 5000})
            options.update({'regenerate_graph': False})
            out1 = odesolve(self.Node, x, options) #out is mxmxc, 
            
            temp = x.new_zeros(x.shape)
            for j in range(x.shape[0]):
                #temp[indeces[i],:] = out[i,i,:]
                temp[indeces[j],:] = out1[j,indeces[j],:]
                #temp[i,:] = out[i,i,:]
        
            act_vec[i,:] = self.fc_layers(temp)
        #for param in self.fc_layers.parameters():

        del temp
        return act_vec
    

    
def concentric(prob_model,num_classes,n_samples,input_dim):
    return Net(prob_model,num_classes,n_samples,input_dim)