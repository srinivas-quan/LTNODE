import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.init as init


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
        self.norm2 = norm_batch(dim*2)

        
        self.lin2 = nn.Linear(dim*2, dim*3)
        self.norm3 = norm_batch(dim*3)
        
        
        self.lin3 = nn.Linear(dim*3, dim*2)
        self.norm4 = norm_batch(dim*2)
        
        self.lin4 = nn.Linear(dim*2, dim)
        self.norm5 = norm_batch(dim)
            
        

    def forward(self,t,x):
        #out = self.norm1(x)
        #out = self.relu(out)
        t_vec = torch.ones(x.shape[0], 1).type(x.type()) * t
        # Shape (batch_size, data_dim + 1)
        #print(t_vec.shape,x.shape)
        t_and_x = torch.cat([t_vec, x], 1)
        out = self.lin1(t_and_x)
        #print(out.shape)
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
    #return nn.BatchNorm1d(dim,affine=True,momentum=0)
    #return nn.GroupNorm(min(10, dim), dim)
    return nn.LayerNorm(dim)
def norm_batch(dim):
    return nn.BatchNorm1d(dim)#,affine=True,momentum=0)
    #return nn.GroupNorm(min(10, dim), dim)
    #return nn.LayerNorm(dim)
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
        self.norm1 = norm_batch(Size)
        #self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        
        
        self.linear2 = nn.Linear(Size, Size*2)
        self.norm2 = norm_batch(Size*2)
        
        self.linear3 = nn.Linear(Size*2, Size*3)
        self.norm3 = norm_batch(Size*3)
        
        self.linear4 = nn.Linear(Size*3, Size)
        self.norm4 = norm_batch(Size)
        
        
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
        print("samples",max_T)
        options.update({'method':'Dopri5'})
        options.update({'h': None})
        options.update({'t0': t})
        #options.update({'t1': sample.detach().cpu().numpy()[0]})
        options.update({'t_eval':list(samples.cpu().numpy())})
        options.update({'t1':max_T})#max_T}),sample
        options.update({'rtol': 1e-2})
        options.update({'atol': 1e-2})
        options.update({'print_neval': True})
        options.update({'neval_max': 5000})
        options.update({'regenerate_graph': True})
        #options.update('{')
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

    



