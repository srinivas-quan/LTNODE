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
        hidden_dim = 32
        self.relu =nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        

    def forward(self,t,x):

        t_vec = torch.ones(x.shape[0], 1).type(x.type()) * t
        t_and_x = torch.cat([t_vec, x], 1)
        out = self.fc1(x)
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
Size = 50
class layers(nn.Module):
    def __init__(self, input_dim):
        super(layers, self).__init__()
    def forward(self, x):
        out = x       
        return out
from torch_ACA import odesolve as odesolve
class Net(nn.Module):
    def __init__(self, prob_model,n_samples,input_dim):
        super(Net, self).__init__()

        
        
        self.fcnn = layers(input_dim)
        self.Node = NODE(Size)
        self.n_samples = n_samples
        self.prob_model = prob_model
        self.fc_layers = nn.Linear(Size, 1)
        self.deltat = 1
        self.output_dim = 1
        self.apply(init_params)

    def forward(self, x,samples = None):
        out = self.fcnn(x)
        act_vec = x.new_zeros(self.n_samples, x.shape[0], 1)
        if samples == None:
            samples = self.prob_model.get_samples(self.n_samples)
        self.prob_model.current_posterior = samples
        self.prob_model.current_posterior_pdf = self.prob_model.get_q_probs()
        t = 0
        i = 0
        '''
        for sample in samples:
            while t + self.deltat <= sample:
                t = t + self.deltat
                out = out + self.Node(out)*self.deltat
            final_out = self.fc_layers(out)
            act_vec[i,:,:] = final_out
            i += 1
        '''
        samples=samples[:,0]
        #samples,indeces  = samples.sort()

        #for sample in samples:
        options = {}
        max_T = samples.max()
        #print(samples,max_T)
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
        #print("out.shape",out.shape)
        out = odesolve(self.Node, out, options) #out is Txmxc, 
        #temp = x.new_zeros(out.shape)
        #for i in range(self.n_samples):
        #    temp[indeces[i],:] = out[i,:]
            #print("indeces[i].......",indeces[i])

        act_vec = self.fc_layers(out)
        #del temp
        return act_vec
    

    
def toy(prob_model,n_samples,input_dim):
    return Net(prob_model,n_samples,input_dim)

    



