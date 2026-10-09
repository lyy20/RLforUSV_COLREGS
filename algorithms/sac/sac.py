# individual network settings for each actor + critic pair
# see networkforall for details
'''
An addaption from:

Code partially extracted from:
https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/sac.py
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py


'''
from SCT import StateCoverageTracker
from algorithms.sac.networkforall_sac import EntityEncoderNetwork, Network
from utilities.utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np

from algorithms.sac.ICM import ICM
# add OU noise for exploration
from utilities.OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class SACAgent():
    def __init__(self, in_actor,in_rnn , hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic , lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu', rnn = True, alpha = 0.2, automatic_entropy_tuning = True,CTIP_active = False, num_ob=3, lr_alpha=None, num_static_ob_slots=1, entity_encoder_enabled=False, entity_encoder_latent_dim=32):
        super(SACAgent, self).__init__()

        self.entity_encoder_enabled = bool(entity_encoder_enabled)
        self.num_static_ob_slots = int(num_static_ob_slots)
        obs_dim = int(in_actor) - 1
        if self.entity_encoder_enabled:
            network_cls = EntityEncoderNetwork
            common_kwargs = {
                "obs_dim": obs_dim,
                "action_dim": out_actor,
                "num_dynamic_slots": num_ob,
                "num_static_slots": self.num_static_ob_slots,
                "entity_latent_dim": entity_encoder_latent_dim,
            }
        else:
            network_cls = Network
            common_kwargs = {}

        self.actor = network_cls(in_actor,in_rnn, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device,actor=True, rnn = rnn, **common_kwargs).to(device)
        self.critic = network_cls(in_critic,in_rnn, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn = rnn, **common_kwargs).to(device)
        # self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device, actor=True, rnn = rnn).to(device)
        self.target_critic = network_cls(in_critic,in_rnn, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn = rnn, **common_kwargs).to(device)

        self.noise = OUNoise(out_actor, scale=1.0 )
        self.device = device
        
        # from torchsummary import summary
        
        # import pdb; pdb.set_trace()
        # summary(self.actor, (3, 224, 224))

        # initialize targets same as original networks
        # hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
        # self.actor_optimizer = AdamW(self.actor.parameters(), lr=lr_actor, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # self.critic_optimizer = AdamW(self.critic.parameters(), lr=lr_critic, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        
        # Alpha 
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.lr_alpha = float(lr_actor if lr_alpha is None else lr_alpha)
        self.alpha = alpha
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -float(out_actor)
            self.log_alpha = (torch.zeros(1, requires_grad=True, device=self.device)+np.log(self.alpha)).detach().requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_alpha)
        self.CTIP_active = CTIP_active
        if self.CTIP_active:
            # 初始化 CTIP 模块
            state_dim = in_actor -1  # 状态维度30, actor 输入31
            action_dim = out_actor
            hidden_dim = 256
            self.icm = ICM(state_dim, action_dim, hidden_dim)
            self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=lr_actor)

        # 定义分箱参数
        self.num_ob = int(num_ob)
        bin_params = {
            'q_r': (0.0, 2.0, 100),        # 目标距离：0-100m，分为10箱
            'q_theta': (0.0, 2 * np.pi, 24), # 目标方位角：0-2π，分为8箱
        }
        for i in range(self.num_ob):
            bin_params[f'obstacle_dist_{i}'] = (0.0, 0.5, 5)  # 障碍物距离：0-50m，分为5箱
            bin_params[f'obstacle_angle_{i}'] = (0.0, 2 * np.pi, 8)  # 障碍物方位角：0-2π，分为4箱
            bin_params[f'obstacle_size_{i}'] = (0.0, 0.15, 3)  # 障碍物大小：0.05-0.15，分为4箱

        # 定义权重
        for i in range(self.num_ob):
            bin_params[f'obstacle_dist_{i}'] = (0.0, 1.0, 5)
            bin_params[f'obstacle_angle_{i}'] = (0.0, 2 * np.pi, 8)
            bin_params[f'obstacle_size_{i}'] = (0.0, 1.0, 3)
        weight_dict = {
            'q_r': 0.3,
            'q_theta': 0.3,
        }
        for i in range(self.num_ob):
            weight_dict[f'obstacle_dist_{i}'] = 0.1
            weight_dict[f'obstacle_angle_{i}'] = 0.1
            weight_dict[f'obstacle_size_{i}'] = 0.1
        # 定义有效取值范围（关键改进）
        valid_ranges = {
            'q_r': (0.0, 2.0),           # 目标最大可见距离为80m
        }
        for i in range(self.num_ob):
            valid_ranges[f'obstacle_dist_{i}'] = (0.0, 0.5)  # 障碍物检测范围为40m
    
        for i in range(self.num_ob):
            valid_ranges[f'obstacle_dist_{i}'] = (0.0, 1.0)

        self.tracker = StateCoverageTracker(bin_params, lsh_epsilon=0.1, weight_dict=weight_dict,
                                   valid_ranges=valid_ranges, prune_invalid=True, num_obstacles=self.num_ob)

    def act(self, his, obs, noise=0.0):
        his = his.to(self.device)
        obs = obs.to(self.device)
        # Environment interaction never backpropagates through the actor.
        # Keeping this path under no_grad avoids building a temporary graph
        # for every environment step without changing sampled actions.
        with torch.no_grad():
            if noise > 0.0:
                action, _ = self.actor.sample_normal(his, obs)
            else:
                mean, _ = self.actor.forward(his, obs)
                action = torch.tanh(mean)
        return action.cpu()
    

    def act_prob(self, his, obs, noise=0.0):
        his = his.to(self.device)
        obs = obs.to(self.device)
        #before 5/12/2022
        # actions, log_probs = self.actor.sample_normal(his,obs) 
        #After 5/12/2022
        #from https://github.com/kengz/SLM-Lab/blob/dda02d00031553aeda4c49c5baa7d0706c53996b/slm_lab/agent/algorithm/sac.py
        #and https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
        action, log_probs = self.actor.sample_normal(his,obs)
        return action.cpu(), log_probs
