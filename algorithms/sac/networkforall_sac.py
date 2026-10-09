import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch.distributions.normal import Normal
from utilities.observation_schema import (
    BASE_OBS_DIM,
    DYNAMIC_OBS_FEATURE_DIM,
    STATIC_OBS_FEATURE_DIM,
)

'''
Code partially extracted from:
https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/sac.py
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py

'''
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_size,input_rnn_size, hidden_in_dim, hidden_out_dim, output_dim, rnn_num_layers, rnn_hidden_size, device, actor=False, rnn=True):
        super(Network, self).__init__()
        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        self.device = device
        self.rnn_num_layers = rnn_num_layers
        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_active = rnn
        self.aux_mul = 1
        self.reparam_noise = 1e-6

        self.rnn_clip = input_size - input_rnn_size
        # Linear NN layers
        if actor == True:
            # Recurrent NN layers (LSTM)
            if self.rnn_active:
                # self.rnn = nn.RNN(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                # self.rnn = nn.GRU(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                self.rnn = nn.LSTM(input_rnn_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                self.aux_mul = 2
            self.fc0 = nn.Linear(input_size - 1 ,rnn_hidden_size)
            self.fc1 = nn.Linear(rnn_hidden_size*self.aux_mul,hidden_in_dim)
            self.fc_mu = nn.Linear(hidden_in_dim, output_dim)
            self.fc_sigma = nn.Linear(hidden_in_dim, output_dim)
        else:
            # Recurrent NN layers (LSTM)
            # self.rnn = nn.RNN(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            # self.rnn = nn.GRU(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            #Q1
            if self.rnn_active:
                self.rnn_q1 = nn.LSTM(input_rnn_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                self.aux_mul = 2
            self.fc0_q1 = nn.Linear(input_size,rnn_hidden_size)
            self.fc1_q1 = nn.Linear(rnn_hidden_size*self.aux_mul,hidden_in_dim)
            self.fc2_q1 = nn.Linear(hidden_in_dim,output_dim)
            #Q2
            if self.rnn_active:
                self.rnn_q2 = nn.LSTM(input_rnn_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                self.aux_mul = 2
            self.fc0_q2 = nn.Linear(input_size,rnn_hidden_size)
            self.fc1_q2 = nn.Linear(rnn_hidden_size*self.aux_mul,hidden_in_dim)
            self.fc2_q2 = nn.Linear(hidden_in_dim,output_dim)     
        self.nonlin = f.relu #leaky_relu
        self.nonlin_tanh = torch.tanh #tanh
        self.actor = actor
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.actor == True:
            # self.rnn.weight.data.uniform_(*hidden_init(self.rnn))
            self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc_mu.weight.data.uniform_(*hidden_init(self.fc_mu))
            self.fc_sigma.weight.data.uniform_(*hidden_init(self.fc_sigma))
        else:
            #Q1
            # self.rnn_q1.weight.data.uniform_(*hidden_init(self.rnn_q1))
            self.fc0_q1.weight.data.uniform_(*hidden_init(self.fc0_q1))
            self.fc1_q1.weight.data.uniform_(*hidden_init(self.fc1_q1))
            self.fc2_q1.weight.data.uniform_(*hidden_init(self.fc2_q1))
            #Q2
            # self.rnn_q2.weight.data.uniform_(*hidden_init(self.rnn_q2))
            self.fc0_q2.weight.data.uniform_(*hidden_init(self.fc0_q2))
            self.fc1_q2.weight.data.uniform_(*hidden_init(self.fc1_q2))
            self.fc2_q2.weight.data.uniform_(*hidden_init(self.fc2_q2))

    def forward(self, x1, x2):
        if self.actor:
            # return a vector of the force
            # RNN
            if self.rnn_active:
                h0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                c0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                # out, _ = self.rnn(x1,h0)
                # x1_input = []
                # for i in range(len(x1)):
                #     x1_input.append(x1[i][:, :-2])
                # x1_input = torch.cat(x1_input, dim=1)

                x1_input = x1[:, :, :-self.rnn_clip]
                out, _ = self.rnn(x1_input,(h0,c0))
                # out: batch_size, seq_legnth, hidden_size
                out = out[:,-1,:]
                # out: batch_size, hidden_size
                h00 = self.nonlin(self.fc0(x2))
                prob = torch.cat((out,h00), dim=1)
            else:
                prob = self.nonlin(self.fc0(x2))
            # Linear
            prob = self.nonlin(self.fc1(prob))           
            mean = self.fc_mu(prob)
            log_std = self.fc_sigma(prob)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) #from https://github.com/pranz24/pytorch-soft-actor-critic/blob/398595e0d9dca98b7db78c7f2f939c969431871a/model.py#L94
            return mean, log_std
        else:
            # critic network simply outputs a number       
            #Q1
            if self.rnn_active:#目前有点问题，后续需要用RNN再改，关于这个RNN剪切掉预测的输入问题
                # RNN
                h0_q1 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                c0_q1 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                # out, _ = self.rnn(x1,h0)
                x1_input = x1[:, :, :-self.rnn_clip]
                out_q1, _ = self.rnn_q1(x1_input,(h0_q1,c0_q1))
                # out: batch_size, seq_legnth, hidden_size
                out_q1 = out_q1[:,-1,:]
                # out: batch_size, hidden_size
                h00_q1 = self.nonlin(self.fc0_q1(x2))
                x_q1 = torch.cat((out_q1,h00_q1), dim=1)
            else:
                x_q1 = self.nonlin(self.fc0_q1(x2))
            # Linear
            h1_q1 = self.nonlin(self.fc1_q1(x_q1))       
            h2_q1 = (self.fc2_q1(h1_q1))
            
            #Q2
            if self.rnn_active:
                # RNN
                h0_q2 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                c0_q2 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                # out, _ = self.rnn(x1,h0)
                x1_input = x1[:, :, :-self.rnn_clip]
                out_q2, _ = self.rnn_q2(x1_input,(h0_q2,c0_q2))
                # out: batch_size, seq_legnth, hidden_size
                out_q2 = out_q2[:,-1,:]
                # out: batch_size, hidden_size
                h00_q2 = self.nonlin(self.fc0_q2(x2))
                x_q2 = torch.cat((out_q2,h00_q2), dim=1)
            else:
                x_q2 = self.nonlin(self.fc0_q2(x2))
            # Linear
            h1_q2 = self.nonlin(self.fc1_q2(x_q2))       
            h2_q2 = (self.fc2_q2(h1_q2))
            
            return h2_q1, h2_q2
        
        
    def sample_normal(self, x1, x2):
        mean, log_std = self.forward(x1,x2)
        std = log_std.exp()
        
        normal = Normal(mean,std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t) # for squashed Gaussian distribution (which means that is bounded between -1 and 1)
                
        log_prob = normal.log_prob(x_t)
        #Enforcing Action Bound
        log_prob -= torch.log(1-action.pow(2)+self.reparam_noise) #done as https://arxiv.org/pdf/2007.14430.pdf
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class EntityEncoderNetwork(nn.Module):
    """Actor/critic network with typed obstacle encoders and masked max pooling.

    The raw observation layout is the compact 3-DOF body-frame base followed
    by dynamic and static entity slots.
    This module only changes how the flat observation is interpreted inside the
    network, so replay buffers and environments do not need a different format.
    """

    def __init__(
        self,
        input_size,
        input_rnn_size,
        hidden_in_dim,
        hidden_out_dim,
        output_dim,
        rnn_num_layers,
        rnn_hidden_size,
        device,
        actor=False,
        rnn=True,
        obs_dim=None,
        action_dim=None,
        num_dynamic_slots=3,
        num_static_slots=1,
        entity_latent_dim=32,
    ):
        super(EntityEncoderNetwork, self).__init__()
        if rnn:
            raise ValueError("EntityEncoderNetwork currently supports RNN=False only.")
        self.device = device
        self.actor = actor
        self.reparam_noise = 1e-6
        self.base_dim = BASE_OBS_DIM
        self.dynamic_feature_dim = DYNAMIC_OBS_FEATURE_DIM
        self.static_feature_dim = STATIC_OBS_FEATURE_DIM
        self.num_dynamic_slots = int(num_dynamic_slots)
        self.num_static_slots = int(num_static_slots)
        self.obs_dim = int(
            obs_dim
            if obs_dim is not None
            else self.base_dim
            + self.dynamic_feature_dim * self.num_dynamic_slots
            + self.static_feature_dim * self.num_static_slots
        )
        inferred_action_dim = max(int(input_size) - self.obs_dim, 0)
        self.action_dim = int(action_dim if action_dim is not None else inferred_action_dim)
        latent_dim = int(entity_latent_dim)

        self.ego_goal_encoder = nn.Sequential(
            nn.Linear(self.base_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(self.dynamic_feature_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(self.static_feature_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        fused_dim = latent_dim * 3
        self.nonlin = f.relu
        if self.actor:
            self.fc1 = nn.Linear(fused_dim, hidden_in_dim)
            self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
            self.fc_mu = nn.Linear(hidden_out_dim, output_dim)
            self.fc_sigma = nn.Linear(hidden_out_dim, output_dim)
        else:
            critic_in = fused_dim + self.action_dim
            self.fc1_q1 = nn.Linear(critic_in, hidden_in_dim)
            self.fc2_q1 = nn.Linear(hidden_in_dim, hidden_out_dim)
            self.fc3_q1 = nn.Linear(hidden_out_dim, output_dim)
            self.fc1_q2 = nn.Linear(critic_in, hidden_in_dim)
            self.fc2_q2 = nn.Linear(hidden_in_dim, hidden_out_dim)
            self.fc3_q2 = nn.Linear(hidden_out_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        modules = []
        if self.actor:
            modules = [self.fc1, self.fc2, self.fc_mu, self.fc_sigma]
        else:
            modules = [
                self.fc1_q1,
                self.fc2_q1,
                self.fc3_q1,
                self.fc1_q2,
                self.fc2_q2,
                self.fc3_q2,
            ]
        for module in modules:
            module.weight.data.uniform_(*hidden_init(module))

    def _masked_max_pool(self, encoded, mask):
        if encoded.size(1) == 0:
            return torch.zeros(encoded.size(0), encoded.size(2), device=encoded.device, dtype=encoded.dtype)
        valid_mask = mask > 0.5
        pooled = encoded.masked_fill(~valid_mask, -1.0e9).max(dim=1).values
        has_valid = valid_mask.any(dim=1)
        return torch.where(has_valid, pooled, torch.zeros_like(pooled))

    def _encode_obs(self, x):
        obs = x[:, : self.obs_dim]
        base = obs[:, : self.base_dim]
        offset = self.base_dim

        if self.num_dynamic_slots > 0:
            dyn = obs[:, offset : offset + self.num_dynamic_slots * self.dynamic_feature_dim]
            dyn = dyn.reshape(-1, self.num_dynamic_slots, self.dynamic_feature_dim)
            dyn_encoded = self.dynamic_encoder(dyn)
            dyn_mask = (dyn[:, :, 6:7].abs() > 1.0e-6).to(dtype=torch.bool)
            dyn_pooled = self._masked_max_pool(dyn_encoded, dyn_mask)
        else:
            latent_dim = self.ego_goal_encoder[-2].out_features
            dyn_pooled = torch.zeros(obs.size(0), latent_dim, device=obs.device, dtype=obs.dtype)
        offset += self.num_dynamic_slots * self.dynamic_feature_dim

        if self.num_static_slots > 0:
            stat = obs[:, offset : offset + self.num_static_slots * self.static_feature_dim]
            stat = stat.reshape(-1, self.num_static_slots, self.static_feature_dim)
            stat_encoded = self.static_encoder(stat)
            stat_mask = (stat[:, :, 4:5].abs() > 1.0e-6).to(dtype=torch.bool)
            stat_pooled = self._masked_max_pool(stat_encoded, stat_mask)
        else:
            latent_dim = self.ego_goal_encoder[-2].out_features
            stat_pooled = torch.zeros(obs.size(0), latent_dim, device=obs.device, dtype=obs.dtype)

        base_encoded = self.ego_goal_encoder(base)
        return torch.cat((base_encoded, dyn_pooled, stat_pooled), dim=1)

    def forward(self, x1, x2):
        encoded = self._encode_obs(x2)
        if self.actor:
            h = self.nonlin(self.fc1(encoded))
            h = self.nonlin(self.fc2(h))
            mean = self.fc_mu(h)
            log_std = self.fc_sigma(h)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            return mean, log_std

        action = x2[:, self.obs_dim : self.obs_dim + self.action_dim]
        critic_input = torch.cat((encoded, action), dim=1)
        h_q1 = self.nonlin(self.fc1_q1(critic_input))
        h_q1 = self.nonlin(self.fc2_q1(h_q1))
        q1 = self.fc3_q1(h_q1)
        h_q2 = self.nonlin(self.fc1_q2(critic_input))
        h_q2 = self.nonlin(self.fc2_q2(h_q2))
        q2 = self.fc3_q2(h_q2)
        return q1, q2

    def sample_normal(self, x1, x2):
        mean, log_std = self.forward(x1, x2)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
