import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.init as init


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','simple','simple1']


class MC_Dropout2d(_DropoutNd):
    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,  bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_drop=0):
        super(BasicBlock, self).__init__()

        norm_layer = nn.BatchNorm2d
        self.p_drop = p_drop
        if self.p_drop > 0:
            self.drop_layer = MC_Dropout2d(p=self.p_drop, inplace=False)
        else:
            self.drop_layer = nn.Identity()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop_layer(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)  # TODO: maybe put this relu in the layer

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_drop=0):
        super(Bottleneck, self).__init__()

        norm_layer = nn.BatchNorm2d

        self.p_drop = p_drop
        if self.p_drop > 0:
            self.drop_layer = MC_Dropout2d(p=self.p_drop, inplace=False)
        else:
            self.drop_layer = nn.Identity()

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.drop_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True, initial_conv='1x7',
                 concat_pool=False, input_chanels=3, p_drop=0):
        super(ResNet, self).__init__()

        #print("In resnet")
        self.zero_init_residual = zero_init_residual

        self.num_classes = num_classes
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if initial_conv == '1x7':
            self.conv1 = [nn.Conv2d(input_chanels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                          self._norm_layer(self.inplanes), self.relu, self.maxpool]
        elif initial_conv == '1x3':
            self.conv1 = [conv3x3(input_chanels, self.inplanes), self._norm_layer(self.inplanes), self.relu]

        elif initial_conv == '3x3':
            self.conv1 = [
                conv3x3(input_chanels, self.inplanes), self._norm_layer(self.inplanes), self.relu,
                conv3x3(self.inplanes, self.inplanes), self._norm_layer(self.inplanes), self.relu,
                conv3x3(self.inplanes, self.inplanes), self._norm_layer(self.inplanes), self.relu, self.maxpool]
        self.conv1 = nn.Sequential(*self.conv1)

        self.layer_list = nn.ModuleList()
        self.layer_list += self._make_layer(block, 64, layers[0], p_drop=p_drop)
        self.layer_list += self._make_layer(block, 128, layers[1], stride=2, p_drop=p_drop)
        self.layer_list += self._make_layer(block, 256, layers[2], stride=2, p_drop=p_drop)
        self.layer_list += self._make_layer(block, 512, layers[3], stride=2, p_drop=p_drop)

        self.n_layers = len(self.layer_list)

        if concat_pool:
            self.pool = AdaptiveConcatPool2d((1, 1))
            self.output_block = nn.Linear(2 * 512 * block.expansion, self.num_classes)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_block = nn.Linear(512 * block.expansion, self.num_classes)
            # TODO: Make ^ more flexible so it can deal with multiple types of inputs

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, p_drop=0):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, p_drop=p_drop))

        return layers

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)

        for layer in self.layer_list:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.output_block(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ArchUncertResNet(ResNet):

    def __init__(self, block, layers, start_depth, end_depth,
                 num_classes=1000, zero_init_residual=True, initial_conv='1x7',
                 concat_pool=False, input_chanels=3, p_drop=0):
        super(ArchUncertResNet, self).__init__(block, layers, num_classes=num_classes,
                                               zero_init_residual=zero_init_residual,
                                               initial_conv=initial_conv, concat_pool=concat_pool,
                                               input_chanels=input_chanels, p_drop=p_drop)
        #print(" block, layers, start_depth, end_depth,num_classes=1000, zero_init_residual=True, initial_conv='1x7',concat_pool=False, input_chanels=3, p_drop=0",block, layers, start_depth, end_depth,num_classes, zero_init_residual, initial_conv,concat_pool, input_chanels, p_drop)
        self.start_depth = start_depth
        self.end_depth = end_depth
        self.n_layers = self.end_depth - self.start_depth

        self.channel_list = [0] * layers[0] + [1] * layers[1] + [2] * layers[2] + [3] * layers[3]

        self.adapt0 = nn.Sequential(conv1x1(64 * block.expansion, 128 * block.expansion, stride=2),
                                    self._norm_layer(128 * block.expansion), self.relu)
        self.adapt1 = nn.Sequential(conv1x1(128 * block.expansion, 256 * block.expansion, stride=2),
                                    self._norm_layer(256 * block.expansion), self.relu)
        self.adapt2 = nn.Sequential(conv1x1(256 * block.expansion, 512 * block.expansion, stride=2),
                                    self._norm_layer(512 * block.expansion), self.relu)
        self.adapt3 = nn.Identity()

        if self.end_depth <= layers[0]:
            self.adapt_layers = nn.ModuleList([self.adapt3])
        elif layers[0] < self.end_depth <= (layers[0] + layers[1]):
            self.adapt_layers = nn.ModuleList([self.adapt0,
                                               self.adapt3])
        elif (layers[0] + layers[1]) < self.end_depth <= (layers[0] + layers[1] + layers[2]):
            self.adapt_layers = nn.ModuleList([nn.Sequential(self.adapt0, self.adapt1),
                                               self.adapt1,
                                               self.adapt3])
        elif (layers[0] + layers[1] + layers[2]) < self.end_depth:
            self.adapt_layers = nn.ModuleList([nn.Sequential(self.adapt0, self.adapt1, self.adapt2),
                                               nn.Sequential(self.adapt1, self.adapt2),
                                               self.adapt2,
                                               self.adapt3])
        #print(layers[0] + layers[1] + layers[2],self.end_depth)
        #print(self.adapt_layers)
        self._init_layers()


    def fwd_input_block(self, x):
        #print("fwd_input_block before ",x.shape)
        x = self.conv1(x)
        #print("fwd_input_block after ",x.shape)
        for layer in self.layer_list[:self.start_depth]:
            x = layer(x)
            #print("fwd_input_block after ",x.shape)
        return x

    def fwd_output_block(self, x):
        for layer in self.layer_list[self.end_depth:]:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.output_block(x)
        return x

    def fast_forward_impl(self, x, layer_probs, min_prob=1e-2):
        # See note [TorchScript super()]
        x = self.fwd_input_block(x)
        act_vec = x.new_zeros(self.n_layers, x.shape[0], self.num_classes)
        for idx, layer_idx in enumerate(range(self.start_depth, self.end_depth)):
            layer = self.layer_list[layer_idx]
            x = layer(x)
            if layer_probs[idx] > min_prob:
                y = self.adapt_layers[self.channel_list[layer_idx]](x)
                y = self.fwd_output_block(y)
                act_vec[layer_idx-self.start_depth, :, :] = y
            else:
                print('skipping layer %d' % layer_idx)
        return act_vec

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.fwd_input_block(x)

        act_vec = x.new_zeros(self.n_layers, x.shape[0], self.num_classes)
        for layer_idx in range(self.start_depth, self.end_depth):
            layer = self.layer_list[layer_idx]
            #print("before ",x.shape)
            x = layer(x)
            #print("after ",x.shape , self.adapt_layers[self.channel_list[layer_idx]])
            y = self.adapt_layers[self.channel_list[layer_idx]](x)
            y = self.fwd_output_block(y)
            act_vec[layer_idx-self.start_depth, :, :] = y
        return act_vec

    def forward(self, x, depth=None):
        return self._forward_impl(x)

    def get_w_prior_loglike(self, k=0):
        # TODO: this function to add priors
        return self.adapt0[0].weight.data.new_zeros(self.n_layers)


def _resnet(block, layers, arch_uncert=True, **kwargs):
    if arch_uncert:
        model = ArchUncertResNet(block, layers, **kwargs)
    else:
        model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    r"""Modified ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    # TODO: proper doc string for when we publish code
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    r"""Modified ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    r"""Modified ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    r"""Modified ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

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
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)
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
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)
#from torch_ACA import odesolve_adjoint as odesolve
from torch_ACA import odesolve
class Net(nn.Module):
    def __init__(self, prob_model,num_classes,n_samples,input_chanels):
        super(Net, self).__init__()
        dim = 64
        #self.layer_depth = layer_depth
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(input_chanels, dim, 3, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
        )
        self.Encoder = nn.Sequential(
            nn.Conv2d(input_chanels, dim, 3, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            norm(dim), 
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), 
            nn.Linear(dim, 2),
            )
        self.Node = NODE(dim)
        self.n_samples = n_samples
        self.prob_model = prob_model
        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(norm(dim), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim, num_classes))
        self.deltat = 0.1
        self.apply(init_params)
        self.sigma = 500
    
    def forward(self, x,samples=None):
        out = self.downsampling_layers(x)
        alpha_beta = self.Encoder(x)


        alpha = torch.exp(alpha_beta[:,0])
        beta = torch.exp(alpha_beta[:,1])
        #print("Eoncded:",alpha.shape,beta.shape)
        self.prob_model.q_a = alpha
        self.prob_model.q_b = beta

        #act_vec = x.new_zeros(self.n_samples, x.shape[0], self.num_classes)
        act_vec = x.new_zeros(1,x.shape[0], self.num_classes)#it stores only final for every datapoint.
        if samples == None:
            samples = self.prob_model.get_samples(alpha,beta,alpha.shape[0]) #need to get N_b T samples for each a and b
        #print("samples.shape",samples.shape)
        samples = samples[:,0]#samples[0,:]
        samples[samples>=50]= 50

        max_T = samples.max()
        temp = x.new_zeros(int(max_T/self.deltat)+1,x.shape[0], self.num_classes)#time_stampsxsamplesxclasses
        #temp[0,:,:] = self.fc_layers(x)#if for any sample 'x',T is less than dt, x will be returned, 
        
        self.prob_model.current_posterior = samples
        self.prob_model.current_posterior_pdf = self.prob_model.get_q_probs()
        t = 0
        i = 1
        #print("max_T,max_T/self.deltat,int(max_T/self.deltat)",max_T,max_T/self.deltat,int(max_T/self.deltat))
        '''
        #Begin Euler
        while t + self.deltat <= max_T:
            t = t + self.deltat
            out = out + self.Node(t, out)*self.deltat
            #print(t,max_T)
            temp[i,:,:] = self.fc_layers(out)    #stores every sample representaton at every time, here i is index of dt,2dt,3dt...
            i += 1
        #End Euler
        '''
        #Adaptive method
        print(samples.sort()[0])
        options = {}
        options.update({'method':'Dopri5'})
        options.update({'h': None})
        options.update({'t0': t})
        #options.update({'t1': sample.detach().cpu().numpy()[0]})
        options.update({'t_eval':list(samples.sort()[0].numpy())})
        options.update({'t1':max_T})
        options.update({'rtol': 1e-2})
        options.update({'atol': 1e-2})
        options.update({'print_neval': False})
        options.update({'neval_max': 5000})
        out = odesolve(self.Node, out, options)
        print(out.shape)
        out = self.fc_layers(out)
        #t = sample.detach().cpu().numpy()[0]
        
        #End Adaptive method
        
        #indeces = samples/self.deltat #samples contains T values and are converted to indeces 
        #indeces = indeces.type(torch.LongTensor)
        indeces = samples.sort()[1]
         
        for i in range(x.shape[0]):
            #print(i,samples[i],indeces[i],act_vec[i,:].shape,temp[indeces[i],i,:].shape)
            #act_vec[0,i,:] = temp[indeces[i],i,:]# for ith sample, represnetation stored at index n*dt+eps, indexed by n->indeces[i]
            act_vec[0,i,:] = out[indeces[i],i,:] #TxNXC, The ith sample, represnetation stored at index indeces[i], indeces stores the index of actual value before sorted.
        del temp
        return act_vec
    def forward_test(self, x,samples=None):
        out = self.downsampling_layers(x)
        alpha_beta = self.Encoder(x)


        alpha = torch.exp(alpha_beta[:,0])
        beta = torch.exp(alpha_beta[:,1])
        #print("Eoncded:",alpha,beta)
        self.prob_model.q_a = alpha
        self.prob_model.q_b = beta

        #act_vec = x.new_zeros(self.n_samples, x.shape[0], self.num_classes)
        act_vec = x.new_zeros(self.n_samples,x.shape[0], self.num_classes)#it stores only final for every datapoint.
        if samples == None:
            samples = self.prob_model.get_samples(alpha,beta,n=self.n_samples) #return self.n_samplesxN_b(batch_size) T samples for each a and b
        #print("samples.shape",samples.shape)
        #samples = samples[0,:]
        #samples[samples>=50]= 50

        max_T = samples.max()
        temp = x.new_zeros(int(max_T/self.deltat)+1,x.shape[0], self.num_classes)#time_stampsxsamplesxclasses
        
        #temp[0,:,:] = self.fc_layers(x)#if for any sample 'x',T is less than dt, x will be returned, 
        
        self.prob_model.current_posterior = samples
        self.prob_model.current_posterior_pdf = self.prob_model.get_q_probs()
        t = 0
        i = 1
        #print("max_T,max_T/self.deltat,int(max_T/self.deltat)",max_T,max_T/self.deltat,int(max_T/self.deltat))
        while t + self.deltat <= max_T:
            t = t + self.deltat
            out = out + self.Node(t, out)*self.deltat
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
                
        del temp
        return act_vec
    
    

class Net1(nn.Module):
    def __init__(self, num_classes,n_samples,input_chanels):
        super(Net1, self).__init__()
        dim = 64
        #self.layer_depth = layer_depth
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(input_chanels, dim, 3, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
        )
        self.Node = NODE(dim)
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(norm(dim), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim, num_classes))
        self.deltat = 0.1
        self.apply(init_params)
        self.sigma = 500
    def forward(self, x):
        out = self.downsampling_layers(x)
        t = 0
        while t + self.deltat <= self.n_samples:
            t = t + self.deltat
            out = out + self.Node(t, out)*self.deltat
        out = self.fc_layers(out)
        return out
      
def simple(prob_model,num_classes,n_samples,input_chanels):
    return Net(prob_model,num_classes,n_samples,input_chanels)
def simple1(num_classes,n_samples,input_chanels):
    return Net1(num_classes,n_samples,input_chanels)

    



