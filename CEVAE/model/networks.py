import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import pyro
import pyro.distributions as dist


class FCNet(nn.Module):
    def __init__(self, input_size, layers, out_layers, activation):
        super(FCNet, self).__init__()

        self.activation = activation

        if layers:
            self.input = nn.Linear(input_size, layers[0])
            self.hidden_layers = nn.ModuleList()
            for i in range(1, len(layers)):
                self.hidden_layers.append(nn.Linear(layers[i], layers[i]))
            self.output_layers = nn.ModuleList()
            self.output_activations = []
            for i, (outdim, activation) in enumerate(out_layers):
                self.output_layers.append(nn.Linear(layers[-1], outdim))
                self.output_activations.append(activation)
        else:
            self.output_layers = nn.ModuleList()
            self.output_activations = []
            for i, (outdim, activation) in enumerate(out_layers):
                self.output_layers.append(nn.Linear(input_size, outdim))
                self.output_activations.append(activation)

    def forward(self, x):
        _input = self.input(x)
        x = self.activation(_input)
        try:
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
        except AttributeError:
            pass
        if self.output_layers:
            outputs = []
            for output_layer, output_activation in zip(self.output_layers, self.output_activations):
                if output_activation:
                    outputs.append(output_activation(output_layer(x)))
                else:
                    outputs.append(output_layer(x))
            return outputs if len(outputs) > 1 else outputs[0]
        else:
            return x


class Decoder(nn.Module):
    def __init__(self, binfeats, contfeats, n_z, h, nh, activation, cuda):
        super(Decoder, self).__init__()
        self.cuda = cuda

        # p(x|z)
        self.hx = FCNet(n_z, (nh - 1) * [h], [], activation)
        self.logits_1 = FCNet(h, [h], [[binfeats, torch.sigmoid]], activation)

        self.mu_sigma = FCNet(
            h, [h], [[contfeats, None], [contfeats, F.softplus]], activation)

        # p(t|z)
        self.logits_2 = FCNet(n_z, [h], [[1, torch.sigmoid]], activation)

        # p(y|t,z)
        self.mu2_t0 = FCNet(n_z, nh * [h], [[1, None]], activation)
        self.mu2_t1 = FCNet(n_z, nh * [h], [[1, None]], activation)

    def forward(self, z):
        # p(x|z)
        hx = self.hx.forward(z)
        x_logits = self.logits_1.forward(hx)
        x_loc, x_scale = self.mu_sigma.forward(hx)

        # p(t|z)
        t_logits = self.logits_2(z)

        # p(y|t,z)
        y_loc_t0 = self.mu2_t0(z)
        y_loc_t1 = self.mu2_t1(z)
        y_scale = Variable(torch.ones(
            y_loc_t0.size()).type(torch.DoubleTensor))
        if self.cuda:
            y_scale = y_scale.cuda()

        return (x_logits, x_loc, x_scale), (t_logits), (y_loc_t0, y_loc_t1, y_scale)

    def forward_y(self, z, treated):
        # p(y|t=1,z) or p(y|t=0,z)
        y_loc_t0 = self.mu2_t0(z)
        y_loc_t1 = self.mu2_t1(z)
        y_scale = Variable(torch.ones(
            y_loc_t0.size()).type(torch.DoubleTensor))
        if self.cuda:
            y_scale = y_scale.cuda()

        if treated:
            return y_loc_t1, y_scale
        else:
            return y_loc_t0, y_scale


class Encoder(nn.Module):
    def __init__(self, binfeats, contfeats, d, h, nh, activation, cuda):
        super(Encoder, self).__init__()
        self.cuda = cuda
        in_size = binfeats + contfeats
        in2_size = in_size + 1

        # q(t|x)
        self.logits_t = FCNet(in_size, [d], [[1, torch.sigmoid]], activation)

        # q(y|x,t)
        self.hqy = FCNet(in_size, (nh - 1) * [h], [], activation)
        self.mu_qy_t0 = FCNet(h, [h], [[1, None]], activation)
        self.mu_qy_t1 = FCNet(h, [h], [[1, None]], activation)

        # q(z|x,t,y)
        self.hqz = FCNet(in2_size, (nh - 1) * [h], [], activation)
        self.muq_t0_sigmaq_t0 = FCNet(
            h, [h], [[d, None], [d, F.softplus]], activation)
        self.muq_t1_sigmaq_t1 = FCNet(
            h, [h], [[d, None], [d, F.softplus]], activation)

    def forward(self, x):
        # q(t|x)
        t_logits = self.logits_t.forward(x)
        t = pyro.sample('t', dist.Bernoulli(t_logits).to_event(1))

        # q(y|x,t)
        hqy = self.hqy.forward(x)
        y_loc_t0 = self.mu_qy_t0.forward(hqy)
        y_loc_t1 = self.mu_qy_t1.forward(hqy)
        y_loc = t * y_loc_t1 + (1. - t) * y_loc_t0
        y_scale = Variable(torch.ones(
            y_loc_t0.size()).type(torch.DoubleTensor))
        if self.cuda:
            y_scale = y_scale.cuda()
        y = pyro.sample('y', dist.Normal(y_loc, y_scale).to_event(1))

        # q(z|x,t,y)
        hqz = self.hqz.forward(torch.cat((x, y), 1))
        z_loc_t0, z_scale_t0 = self.muq_t0_sigmaq_t0.forward(hqz)
        z_loc_t1, z_scale_t1 = self.muq_t1_sigmaq_t1.forward(hqz)
        z_loc = t * z_loc_t1 + (1. - t) * z_loc_t0
        z_scale = t * z_scale_t1 + (1. - t) * z_scale_t0

        return z_loc, z_scale

    def forward_z(self, x):
        # q(t|x)
        t_logits = self.logits_t.forward(x)
        t = pyro.sample('t', dist.Bernoulli(t_logits).to_event(1))

        # q(y|x,t)
        hqy = self.hqy.forward(x)
        y_loc_t0 = self.mu_qy_t0.forward(hqy)
        y_loc_t1 = self.mu_qy_t1.forward(hqy)
        y_loc = t * y_loc_t1 + (1. - t) * y_loc_t0
        y_scale = Variable(torch.ones(
            y_loc_t0.size()).type(torch.DoubleTensor))
        if self.cuda:
            y_scale = y_scale.cuda()
        y = pyro.sample('y', dist.Normal(y_loc, y_scale).to_event(1))

        # q(z|x,t,y)
        hqz = self.hqz.forward(torch.cat((x, y), 1))
        z_loc_t0, z_scale_t0 = self.muq_t0_sigmaq_t0.forward(hqz)
        z_loc_t1, z_scale_t1 = self.muq_t1_sigmaq_t1.forward(hqz)

        return z_loc_t0, z_loc_t1
