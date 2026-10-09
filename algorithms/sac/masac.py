# -*- coding: latin-1 -*-
# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from algorithms.sac.sac import SACAgent
import torch
from utilities.utilities import soft_update, transpose_to_tensor, transpose_list, gumbel_softmax
from utilities.observation_schema import (
    BASE_INDEX,
    BASE_OBS_DIM,
    DYNAMIC_OBS_FEATURE_DIM,
    STATIC_OBS_FEATURE_DIM,
    observation_dim,
)
import numpy as np
import SCT
import math
import torch.nn.functional as f

DEFAULT_ACTION_DIM = 1
'''
An addaption from:

Code partially extracted from:
https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/sac.py
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py

'''
class MASAC:
    def __init__(self, num_agents = 3, num_landmarks = 1 ,num_ob = 3, landmark_depth=15., discount_factor=0.95, tau=0.02, lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu', rnn = True, alpha = 0.2, automatic_entropy_tuning = True, dim_1=64, dim_2=32 ,CTIP_active = False, action_dim=DEFAULT_ACTION_DIM, num_static_ob_slots=1, lr_alpha=None, entity_encoder_enabled=False, entity_encoder_latent_dim=32, rule_buffer_enabled=False, rule_imitation_weight=0.15):
        super(MASAC, self).__init__()
        
        # ([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [entity_range] + [entity_depth] + [agent.state.p_pos_origin]) + action(for critic not actor)
        #in_actor = 1*2*2 + num_landmarks*2 + (num_agents-1)*2 + num_landmarks + 1*num_landmarks + 2 +1#test with target depth and agent's origin for science

        # obs = compact 3-DOF body-frame base state + typed obstacle slots.

        self.dynamic_ob_feature_dim = DYNAMIC_OBS_FEATURE_DIM
        self.static_ob_feature_dim = STATIC_OBS_FEATURE_DIM
        self.num_static_ob_slots = int(num_static_ob_slots)
        self.entity_encoder_enabled = bool(entity_encoder_enabled)
        self.entity_encoder_latent_dim = int(entity_encoder_latent_dim)
        if self.entity_encoder_enabled and rnn:
            raise ValueError('ENTITY_ENCODER_ENABLED currently requires RNN=False.')
        obs_dim = observation_dim(num_ob, self.num_static_ob_slots)
        in_actor = obs_dim + 1
        in_rnn = obs_dim - 1
        hidden_in_actor = dim_2
        hidden_out_actor = int(hidden_in_actor/2)
        out_actor = action_dim
        # in_critic = in_actor * num_agents # the critic input is all agents concatenated
        in_critic = obs_dim + action_dim
        hidden_in_critic = dim_2
        hidden_out_critic = int(hidden_in_critic/2)
        #RNN
        rnn_num_layers = 2 #two stacked RNN to improve the performance (default = 1)
        rnn_hidden_size_actor = dim_1
        rnn_hidden_size_critic = dim_1
        
        # print('Actor NN configuration:')
        # print('Input nodes number:            ',in_actor)
        # print('Hidden 1st layer nodes number: ',hidden_in_actor)
        # print('Hidden 2nd layer nodes number: ',hidden_out_actor)
        # print('Output nodes number:           ',out_actor)
        # print('RNN hidden size actor :        ',rnn_hidden_size_actor)
        # print('Critic NN configuration:')
        # print('Input nodes number:            ',in_critic)
        # print('Hidden 1st layer nodes number: ',hidden_in_critic)
        # print('Hidden 2nd layer nodes number: ',hidden_out_critic)
        # print('Output nodes number:           ',out_actor)
        # print('RNN hidden size critic:        ',rnn_hidden_size_critic)
        
        self.masac_agent = [SACAgent(in_actor, in_rnn , hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic, lr_actor=lr_actor, lr_critic=lr_critic, lr_alpha=lr_alpha, weight_decay=weight_decay, device=device, rnn = rnn, alpha = alpha, automatic_entropy_tuning=automatic_entropy_tuning,CTIP_active = CTIP_active, num_ob=num_ob, num_static_ob_slots=self.num_static_ob_slots, entity_encoder_enabled=self.entity_encoder_enabled, entity_encoder_latent_dim=self.entity_encoder_latent_dim) for _ in range(num_agents)]
        # self.masac_agent = [DDPGAgent(14, 128, 128, 2, 48, 128, 128, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device) for _ in range(num_agents)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.iter_delay = 0
        
        self.policy_freq = 1
        self.num_agents = num_agents
        self.action_dim = action_dim
        # Optional second replay batch.  It is deliberately kept separate
        # from the main SAC/PER update; see update(..., rule_samples=...).
        self.rule_buffer_enabled = bool(rule_buffer_enabled)
        self.rule_imitation_weight = float(rule_imitation_weight)
        if self.rule_imitation_weight < 0.0:
            raise ValueError('rule_imitation_weight must be non-negative.')
        self.last_rule_update_stats = {}
        # 保存最近一次 SAC 更新的诊断快照，便于短 smoke 或断点排查；
        # 该字典不参与梯度计算，也不改变任何动作或损失的数值。
        self.last_sac_update_stats = {}
        
        #initial priority for the experienced replay buffer
        self.priority = 1.
        
        #device 'cuda' or 'cpu'
        self.device = device
        
        #To update alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [sac_agent.actor for sac_agent in self.masac_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [sac_agent.target_actor for sac_agent in self.masac_agent]
        return target_actors

    def act(self, his_all_agents, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions_next = [agent.act(his, obs, noise) for agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents)]
        return actions_next

    def act_prob(self, his_all_agents, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        actions_next = []
        log_probs = []
        for sac_agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents):
            action, log_prob = sac_agent.act_prob(his, obs)
            log_prob = log_prob.view(-1)
            actions_next.append(action)
            log_probs.append(log_prob)
        # target_actions_next = [sac_agent.target_actor.sample_normal(his, obs, noise) for sac_agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents)]
        # for i,aux in enumerate(log_probs):
        #     log_probs[i]=aux.view(-1,1)
        return actions_next, log_probs

    def _assert_finite(self, value, field_name, agent_number):
        """严格检查 SAC 更新链中的数值，发现 NaN/Inf 时立即暴露源头。

        这里故意不使用 ``torch.nan_to_num``。把非有限值替换成正常数会让
        replay、梯度或优先级继续运行，但会掩盖真正的动力学/网络/数据错误。
        """
        if value is None:
            return
        if torch.is_tensor(value):
            tensor = value.detach()
            finite = torch.isfinite(tensor)
            if bool(torch.all(finite).item()):
                return
            bad_indices = torch.nonzero(~finite, as_tuple=False)[:8]
            bad_indices = bad_indices.detach().cpu().tolist()
            raise FloatingPointError(
                'Non-finite SAC value: agent=%d iter=%d field=%s shape=%s indices=%s'
                % (agent_number, self.iter, field_name, tuple(tensor.shape), bad_indices)
            )
        try:
            array = np.asarray(value)
            finite = np.isfinite(array)
        except (TypeError, ValueError):
            raise ValueError(
                'SAC finite check failed: agent=%d iter=%d field=%s shape=%s'
                % (agent_number, self.iter, field_name, getattr(value, 'shape', None))
            )
        if bool(np.all(finite)):
            return
        bad_indices = np.argwhere(~finite)[:8].tolist()
        raise FloatingPointError(
            'Non-finite SAC value: agent=%d iter=%d field=%s shape=%s indices=%s'
            % (agent_number, self.iter, field_name, array.shape, bad_indices)
        )

    def _assert_finite_gradients(self, module, module_name, agent_number):
        """检查模块反向传播后的梯度，避免非有限梯度静默污染参数。"""
        for parameter_name, parameter in module.named_parameters():
            if parameter.grad is not None:
                self._assert_finite(
                    parameter.grad,
                    '%s.%s.grad' % (module_name, parameter_name),
                    agent_number,
                )

    def _assert_finite_parameters(self, module, module_name, agent_number):
        """检查优化器更新后的参数，及时拦截参数爆炸或非有限值。"""
        for parameter_name, parameter in module.named_parameters():
            self._assert_finite(
                parameter,
                '%s.%s.parameter' % (module_name, parameter_name),
                agent_number,
            )

    def _finite_mean(self, value, field_name, agent_number):
        """返回可写入 TensorBoard 的有限均值，并复用严格数值检查。"""
        self._assert_finite(value, field_name, agent_number)
        if torch.is_tensor(value):
            tensor = value.detach()
            return float(tensor.mean().cpu().item()) if tensor.numel() else 0.0
        array = np.asarray(value)
        return float(array.mean()) if array.size else 0.0

    def update(self, samples, agent_number, logger, rule_samples=None):
        """update the critics and actors of all the agents 
            Update parameters of agent model based on sample from replay buffer
            Inputs:
                samples: tuple of (observations, full observations, actions, rewards, next
                        observations, full next observations, and episode end masks) sampled randomly from
                        the replay buffer. Each is a list with entries
                        corresponding to each agent
                agent_number (int): index of agent to update
            logger (SummaryWriter from Tensorboard-Pytorch):
                     If passed in, important quantities will be logged
                rule_samples: optional dict sampled from RuleReplayBuffer. It
                    is used only for the actor imitation term and never for
                    critic, alpha, CTIP, or PER calculations.
        """
        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        # obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        # 新 transition 的 12 个字段为：基础 7 项 +
        # raw_action/constrained_action/rule_action/smoothed_action/filter_active。
        # 仍保留 7/9 字段回退，便于读取旧主池；旧格式无法提供完整动作链，
        # 只按它自身的 legacy 语义计算，不把旧数据伪装成新格式。
        base_samples = samples[:7]
        extra_samples = samples[7:]
        his_obs, his_act, obs, transition_action, reward, next_obs, done = map(
            transpose_to_tensor,
            base_samples,
        )
        # 先检查 replay 中的基础字段，再进行拼接和网络前向计算。
        # 这样错误会定位到具体的 agent/字段，而不是晚些时候才表现成 loss NaN。
        base_field_groups = (
            ('history_observation', his_obs),
            ('history_action', his_act),
            ('observation', obs),
            ('transition_action', transition_action),
            ('reward', reward),
            ('next_observation', next_obs),
            ('done', done),
        )
        for field_name, field_values in base_field_groups:
            for field_agent, field_value in enumerate(field_values):
                self._assert_finite(field_value, field_name, agent_number)
        raw_action_by_agent = None
        constrained_action_by_agent = None
        rule_action_by_agent = None
        smoothed_action_by_agent = None
        filter_active_by_agent = None
        if len(extra_samples) >= 5:
            # 当前格式：第 4 项已经是 raw_action；附加字段只负责审计和
            # 执行链复原，避免把同一动作误解释为多个层级。
            raw_action_by_agent = transpose_to_tensor(extra_samples[0])
            constrained_action_by_agent = transpose_to_tensor(extra_samples[1])
            rule_action_by_agent = transpose_to_tensor(extra_samples[2])
            smoothed_action_by_agent = transpose_to_tensor(extra_samples[3])
            filter_active_by_agent = transpose_to_tensor(extra_samples[4])
            extra_field_groups = (
                ('raw_action', raw_action_by_agent),
                ('constrained_action', constrained_action_by_agent),
                ('rule_action', rule_action_by_agent),
                ('smoothed_action', smoothed_action_by_agent),
                ('filter_active', filter_active_by_agent),
            )
            for field_name, field_values in extra_field_groups:
                for field_agent, field_value in enumerate(field_values):
                    self._assert_finite(field_value, field_name, agent_number)
            critic_action_by_agent = transition_action
            execution_action_by_agent = smoothed_action_by_agent
        elif len(extra_samples) >= 2:
            # 旧 9 字段格式：基础第 4 项是 applied_action，附加第 1
            # 项才是 raw_action；这里保留旧数据的可恢复性。
            raw_action_by_agent = transpose_to_tensor(extra_samples[0])
            filter_active_by_agent = transpose_to_tensor(extra_samples[1])
            self._assert_finite(raw_action_by_agent[agent_number], 'raw_action', agent_number)
            self._assert_finite(filter_active_by_agent[agent_number], 'filter_active', agent_number)
            critic_action_by_agent = raw_action_by_agent
            execution_action_by_agent = transition_action
        else:
            # 最老的 7 字段样本没有动作层级信息，只能把唯一动作同时
            # 作为 Critic 动作和执行历史动作。
            critic_action_by_agent = transition_action
            execution_action_by_agent = transition_action
        
        # obs_full = torch.stack(obs_full)
        # next_obs_full = torch.stack(next_obs_full)
        
        obs_full = torch.cat(obs, dim=1)
        next_obs_full = torch.cat(next_obs, dim=1)
        raw_action_full = torch.cat(critic_action_by_agent, dim=1)
        execution_action_full = torch.cat(execution_action_by_agent, dim=1)
        self._assert_finite(obs_full, 'obs_full', agent_number)
        self._assert_finite(next_obs_full, 'next_obs_full', agent_number)
        self._assert_finite(raw_action_full, 'critic_action_raw_full', agent_number)
        self._assert_finite(execution_action_full, 'execution_action_smoothed_full', agent_number)
        # 当前 Critic 的动作语义是 raw_action；这个选择必须和主 Replay
        # 的第 4 项一致。平滑执行动作仅用于下一历史状态和辅助动力学模型。
        action = raw_action_full
        obs_act_full = torch.cat((obs_full, raw_action_full), dim=1)
        his = []
        for i in range(len(his_obs)):
            his.append(torch.cat((his_obs[i],his_act[i]), dim=2))
        his_full = torch.cat(his,dim=2)
        self._assert_finite(his_full, 'history_full', agent_number)
        
        # next_his = []
        # for i in range(len(his_obs)):
        #     aux = torch.cat((his_obs[i],obs[i].reshape(his_obs[i].shape[0],1,his_obs[i].shape[2])),dim=1)
        #     aux = np.delete(aux,0,1)
        #     aux_a = torch.cat((his_act[i],action.reshape(his_act[i].shape[0],1,his_act[i].shape[2])),dim=1)
        #     aux_a = np.delete(aux_a,0,1)
        #     next_his.append(torch.cat((aux,aux_a), dim=2))

        
        agent = self.masac_agent[agent_number]

        states = []
        for i in range(obs_full.shape[0]):
            q_t = (
                obs_full[i, BASE_INDEX['target_x_body']].item(),
                obs_full[i, BASE_INDEX['target_y_body']].item(),
            )

            obstacles = []
            dynamic_start = BASE_OBS_DIM
            dynamic_feature_dim = DYNAMIC_OBS_FEATURE_DIM
            for count in range(agent.num_ob):
                j = dynamic_start + count * dynamic_feature_dim
                type_mask = obs_full[i, j + 6].item()
                if abs(type_mask) <= 1.0e-9:
                    continue
                entity_pos_body = (obs_full[i, j].item(), obs_full[i, j + 1].item())
                o_polar = agent.tracker._polar_transform(entity_pos_body)
                size = obs_full[i, j + 3].item()
                obstacles.append({'distance': o_polar[0], 'angle': o_polar[1], 'size': size})

            states.append({'q_t': q_t, 'obstacles': obstacles})

        for i, state in enumerate(states):
            agent.tracker.update_coverage(state)
        agent.critic_optimizer.zero_grad()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        
        # target_actions_next = self.target_act(his,next_obs) 
        # target_actions_next, log_probs = self.target_act(his, next_obs)
        # Target actions and target log-probabilities are not part of the
        # actor update graph. Avoid retaining a graph for every critic
        # target calculation.
        with torch.no_grad():
            actions_next, log_probs = self.act_prob(
                his, next_obs
            ) # SAC uses the current policy for next-state target actions.
            # Target Critic 使用的下一状态动作必须仍然是策略 raw action；
            # 这里的检查同时覆盖每个并行样本和每个 agent 的 log-prob。
            for next_agent, next_action in enumerate(actions_next):
                self._assert_finite(next_action, 'target_next_raw_action_%d' % next_agent, agent_number)
            for next_agent, next_log_prob in enumerate(log_probs):
                self._assert_finite(next_log_prob, 'target_next_log_prob_%d' % next_agent, agent_number)
            actions_next = torch.cat(actions_next, dim=1)
            self._assert_finite(actions_next, 'target_next_raw_action_full', agent_number)
        # log_probs = torch.cat(log_probs, dim=0)
        next_his_append = torch.cat((obs_full, execution_action_full), dim=1)
        next_his_full = torch.cat(
            (
                his_full[:, 1:, :],
                next_his_append.reshape(
                    next_his_append.shape[0],
                    1,
                    next_his_append.shape[1],
                ),
            ),
            dim=1,
        )
        next_obs_act_full = torch.cat((next_obs_full,actions_next), dim=1)
        self._assert_finite(next_his_full, 'target_next_history_full', agent_number)
        self._assert_finite(next_obs_act_full, 'target_next_observation_action', agent_number)
        
        # Compute Q targets (y) for current states (y_i)
        with torch.no_grad():
            target_Q1, target_Q2 = agent.target_critic(next_his_full.to(self.device), next_obs_act_full.to(self.device))
            self._assert_finite(target_Q1, 'target_Q1', agent_number)
            self._assert_finite(target_Q2, 'target_Q2', agent_number)
            target_V = torch.min(target_Q1, target_Q2) - agent.alpha*log_probs[agent_number].view(-1,1)
            self._assert_finite(target_V, 'target_V', agent_number)
            target_Q = reward[agent_number].view(-1, 1).to(self.device) + self.discount_factor * target_V * (1 - done[agent_number].view(-1, 1)).to(self.device)
            self._assert_finite(target_Q, 'target_Q', agent_number)

        # Compute Q expected (q) 
        current_Q1, current_Q2 = agent.critic(his_full.to(self.device), obs_act_full.to(self.device))
        self._assert_finite(current_Q1, 'current_Q1', agent_number)
        self._assert_finite(current_Q2, 'current_Q2', agent_number)
        
        # Compute critic loss
        loss_mse = torch.nn.MSELoss()
        critic_loss = loss_mse(current_Q1, target_Q.detach()) + loss_mse(current_Q2, target_Q.detach())
        self._assert_finite(critic_loss, 'critic_loss', agent_number)
        # Minimize the loss
        critic_loss.backward()
        self._assert_finite_gradients(agent.critic, 'critic', agent_number)
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        self._assert_finite_gradients(agent.critic, 'critic_after_clip', agent_number)
        agent.critic_optimizer.step()
        self._assert_finite_parameters(agent.critic, 'critic_after_update', agent_number)
        
        # 以下变量用于 SAC 更新链的 TensorBoard 诊断。默认值表示本次只做了
        # Critic 更新；通过 actor_update_applied 可以区分“未更新”与“数值为零”。
        actor_update_applied = 0.0
        policy_action_mean = 0.0
        policy_log_prob_mean = 0.0
        actor_mean_mean = 0.0
        actor_log_std_mean = 0.0
        actor_loss_value = 0.0
        sac_actor_loss_value = 0.0
        alpha_loss_value = 0.0

        # Delayed policy updates
        if self.iter_delay % self.policy_freq == 0:
            # ---------------------------- update actor ---------------------------- #
            #update actor network using policy gradient
            # Compute actor loss
            agent.actor_optimizer.zero_grad()
            # 记录 actor 的未 squashing 均值和 log_std；放在 no_grad 中只做
            # 诊断，不会额外建立反向图，也不会改变策略采样的随机数序列。
            with torch.no_grad():
                actor_mean_diag, actor_log_std_diag = agent.actor.forward(
                    his[agent_number].to(self.device),
                    obs[agent_number].to(self.device),
                )
            self._assert_finite(actor_mean_diag, 'actor_mean', agent_number)
            self._assert_finite(actor_log_std_diag, 'actor_log_std', agent_number)
            actor_mean_mean = self._finite_mean(actor_mean_diag, 'actor_mean', agent_number)
            actor_log_std_mean = self._finite_mean(actor_log_std_diag, 'actor_log_std', agent_number)
            # make input to agent
            # curr_q_input = self.masac_agent[agent_number].actor(his[agent_number].to(self.device), obs[agent_number].to(self.device))
            actions, log_probs = self.masac_agent[agent_number].actor.sample_normal(his[agent_number].to(self.device), obs[agent_number].to(self.device))
            log_probs = log_probs.view(-1)
            self._assert_finite(actions, 'actor_sampled_raw_action', agent_number)
            self._assert_finite(log_probs, 'actor_log_prob', agent_number)
            # use Gumbel-Softmax sample
            # curr_q_input = gumbel_softmax(curr_q_input, hard = True) # this should be used only if the action is discrete (for example in comunications, but in general the action is not discrete)
            # detach the other agents to save computation
            # saves some time for computing derivative
            # q_input = [ self.masac_agent[i].actor(ob.to(self.device)) if i == agent_number \
            #            else self.masac_agent[i].actor(ob.to(self.device)).detach()
            #            for i, ob in enumerate(obs) ]
            # q_input = [ curr_q_input if i == agent_number \
            #            else self.masac_agent[i].actor.sample_normal(his[i].to(self.device),ob.to(self.device)).detach()
            #            for i, ob in enumerate(obs) ]
                
            q_actions = []
            q_log_probs = []
            for i, ob in enumerate(obs):
                if i == agent_number:
                    q_actions.append(actions)
                    q_log_probs.append(log_probs)
                else:
                    actions_aux, log_probs_aux = self.masac_agent[i].actor.sample_normal(his[i].to(self.device),ob.to(self.device))
                    log_probs_aux = log_probs_aux.view(-1)
                    q_actions.append(actions_aux.detach())
                    q_log_probs.append(log_probs_aux.detach())   
                self._assert_finite(q_actions[-1], 'actor_q_action_%d' % i, agent_number)
                self._assert_finite(q_log_probs[-1], 'actor_q_log_prob_%d' % i, agent_number)
              
            q_actions = torch.cat(q_actions, dim=1)
            self._assert_finite(q_actions, 'actor_q_actions_full', agent_number)
            # q_log_probs = torch.cat(q_log_probs, dim=0)
            # combine all the actions and observations for input to critic
            # many of the obs are redundant, and obs[1] contains all useful information already
            obs_q_full = torch.cat((obs_full.to(self.device),q_actions), dim=1)
            actor_Q1, actor_Q2 = agent.critic(his_full.to(self.device),obs_q_full)
            self._assert_finite(actor_Q1, 'actor_Q1', agent_number)
            self._assert_finite(actor_Q2, 'actor_Q2', agent_number)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            sac_actor_loss = (agent.alpha*q_log_probs[agent_number].view(-1,1)-actor_Q).mean() # get the policy gradient
            self._assert_finite(actor_Q, 'actor_Q', agent_number)
            self._assert_finite(sac_actor_loss, 'sac_actor_loss', agent_number)
            policy_action_mean = self._finite_mean(actions, 'actor_sampled_raw_action', agent_number)
            policy_log_prob_mean = self._finite_mean(log_probs, 'actor_log_prob', agent_number)
            rule_imitation_loss = torch.zeros((), device=self.device)
            rule_active_ratio = 0.0
            rule_batch_size = 0
            rule_unique_sample_ratio = 0.0
            rule_reuse_ratio = 0.0
            rule_sample_time_ms = 0.0
            rule_raw_teacher_distance = 0.0
            rule_actor_teacher_distance = 0.0
            rule_warmup_skipped = 0.0

            # This diagnostic describes how many records in the main SAC
            # batch came from an active filter.  It is measured even when the
            # independent rule batch is enabled, but it is never used as the
            # rule-loss mask in that mode.
            if filter_active_by_agent is not None:
                main_active_mask = filter_active_by_agent[agent_number].to(self.device).view(-1, 1)
                rule_active_ratio = float(
                    main_active_mask.sum().detach().cpu().item()
                    / max(float(main_active_mask.numel()), 1.0)
                )

            # With the optional rule buffer enabled, the main batch no longer
            # applies the old active-mask loss.  This prevents duplicate
            # supervision while the independent rule batch remains sparse.
            use_legacy_main_rule_loss = (
                not self.rule_buffer_enabled and
                raw_action_by_agent is not None and
                filter_active_by_agent is not None
            )
            if use_legacy_main_rule_loss:
                active_mask = filter_active_by_agent[agent_number].to(self.device).view(-1, 1)
                active_count = float(active_mask.sum().detach().cpu().item())
                rule_active_ratio = active_count / max(float(active_mask.numel()), 1.0)
                if active_count > 0.0:
                    if rule_action_by_agent is not None:
                        # 新格式的教师动作必须是纯规则输出，不能使用
                        # smoothed_action/applied_action。
                        teacher_action = rule_action_by_agent[agent_number].to(self.device).detach()
                        constrained_action_agent = constrained_action_by_agent[agent_number].to(self.device).detach()
                    else:
                        # 旧主池没有 rule_action，只保留旧格式的兼容回退。
                        teacher_action = execution_action_by_agent[agent_number].to(self.device).detach()
                        constrained_action_agent = None
                    raw_action_agent = raw_action_by_agent[agent_number].to(self.device).detach()
                    self._assert_finite(teacher_action, 'legacy_rule_teacher_action', agent_number)
                    self._assert_finite(raw_action_agent, 'legacy_raw_action', agent_number)
                    if constrained_action_agent is not None:
                        self._assert_finite(constrained_action_agent, 'legacy_constrained_action', agent_number)
                    mean_action, _ = self.masac_agent[agent_number].actor.forward(his[agent_number].to(self.device), obs[agent_number].to(self.device))
                    self._assert_finite(mean_action, 'legacy_actor_mean', agent_number)
                    mean_action = torch.tanh(mean_action)
                    self._assert_finite(mean_action, 'legacy_actor_mean_tanh', agent_number)
                    if constrained_action_agent is None:
                        correction_norm = torch.norm(teacher_action - raw_action_agent, dim=1, keepdim=True)
                    else:
                        correction_norm = torch.norm(teacher_action - constrained_action_agent, dim=1, keepdim=True)
                    self._assert_finite(correction_norm, 'legacy_rule_correction_norm', agent_number)
                    correction_weight = torch.clamp(correction_norm / math.sqrt(max(self.action_dim, 1)), min=0.25, max=1.0)
                    sample_weight = active_mask * correction_weight
                    imitation_error = torch.sum((mean_action - teacher_action) ** 2, dim=1, keepdim=True)
                    self._assert_finite(imitation_error, 'legacy_rule_imitation_error', agent_number)
                    rule_imitation_loss = torch.sum(imitation_error * sample_weight) / torch.clamp(sample_weight.sum(), min=1.0)
                    self._assert_finite(rule_imitation_loss, 'legacy_rule_imitation_loss', agent_number)
            elif self.rule_buffer_enabled and rule_samples is not None:
                # The rule batch is already restricted to filter_active=True.
                # ``rule_action`` is the pure COLREGs teacher; ``applied_action``
                # is only the final smoothed action used for execution auditing.
                try:
                    rule_his_obs = torch.as_tensor(
                        rule_samples['history_obs'],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    rule_his_action = torch.as_tensor(
                        rule_samples['history_action'],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    rule_obs = torch.as_tensor(
                        rule_samples['observation'],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    teacher_action = torch.as_tensor(
                        rule_samples['rule_action'],
                        dtype=torch.float32,
                        device=self.device,
                    ).detach()
                    constrained_teacher_action = torch.as_tensor(
                        rule_samples['constrained_action'],
                        dtype=torch.float32,
                        device=self.device,
                    ).detach()
                    raw_teacher_action = torch.as_tensor(
                        rule_samples['raw_action'],
                        dtype=torch.float32,
                        device=self.device,
                    ).detach()
                    self._assert_finite(rule_his_obs, 'rule_history_observation', agent_number)
                    self._assert_finite(rule_his_action, 'rule_history_action', agent_number)
                    self._assert_finite(rule_obs, 'rule_observation', agent_number)
                    self._assert_finite(teacher_action, 'rule_teacher_action', agent_number)
                    self._assert_finite(constrained_teacher_action, 'rule_constrained_action', agent_number)
                    self._assert_finite(raw_teacher_action, 'rule_raw_action', agent_number)
                    if rule_his_obs.ndim != 3 or rule_his_action.ndim != 3 or rule_obs.ndim != 2:
                        raise ValueError('rule batch tensors have invalid ranks')
                    if rule_his_obs.shape[0] != rule_obs.shape[0]:
                        raise ValueError('rule batch history/observation batch sizes differ')
                    rule_his = torch.cat((rule_his_obs, rule_his_action), dim=2)
                    self._assert_finite(rule_his, 'rule_history_full', agent_number)
                    mean_rule_action, _ = agent.actor.forward(rule_his, rule_obs)
                    self._assert_finite(mean_rule_action, 'rule_actor_mean', agent_number)
                    mean_rule_action = torch.tanh(mean_rule_action)
                    self._assert_finite(mean_rule_action, 'rule_actor_mean_tanh', agent_number)

                    correction_norm = rule_samples.get('correction_norm')
                    if correction_norm is None:
                        correction_norm = torch.norm(
                            teacher_action - constrained_teacher_action,
                            dim=1,
                            keepdim=True,
                        )
                    else:
                        correction_norm = torch.as_tensor(
                            correction_norm,
                            dtype=torch.float32,
                            device=self.device,
                        ).view(-1, 1)
                    self._assert_finite(correction_norm, 'rule_correction_norm', agent_number)
                    correction_weight = torch.clamp(
                        correction_norm / math.sqrt(max(self.action_dim, 1)),
                        min=0.25,
                        max=1.0,
                    )
                    # 可选教师元数据仅屏蔽/加权模仿项，不参与 Critic、Alpha 或 PER。
                    if 'teacher_valid' in rule_samples:
                        valid = torch.as_tensor(rule_samples['teacher_valid'], dtype=torch.float32, device=self.device).view(-1, 1)
                        confidence = torch.as_tensor(rule_samples.get('teacher_confidence', np.ones(rule_obs.shape[0])), dtype=torch.float32, device=self.device).view(-1, 1)
                        correction_weight = correction_weight * valid * confidence.clamp(0.0, 1.0)
                    imitation_error = torch.sum(
                        (mean_rule_action - teacher_action) ** 2,
                        dim=1,
                        keepdim=True,
                    )
                    self._assert_finite(imitation_error, 'rule_imitation_error', agent_number)
                    rule_imitation_loss = torch.sum(
                        imitation_error * correction_weight
                    ) / torch.clamp(correction_weight.sum(), min=1.0e-6)
                    self._assert_finite(rule_imitation_loss, 'rule_imitation_loss', agent_number)
                    rule_batch_size = int(rule_obs.shape[0])
                    rule_raw_teacher_distance = float(
                        torch.norm(teacher_action - raw_teacher_action, dim=1)
                        .mean().detach().cpu().item()
                    )
                    rule_actor_teacher_distance = float(
                        torch.norm(mean_rule_action - teacher_action, dim=1)
                        .mean().detach().cpu().item()
                    )
                    sample_stats = rule_samples.get('_sample_stats', {})
                    rule_unique_sample_ratio = float(
                        sample_stats.get('unique_sample_ratio', 0.0)
                    )
                    rule_reuse_ratio = float(sample_stats.get('reuse_ratio', 0.0))
                    rule_sample_time_ms = float(sample_stats.get('sample_time_ms', 0.0))
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    raise ValueError('Invalid rule replay batch for actor update.') from exc
            elif self.rule_buffer_enabled:
                # The pool has not reached the configured warm-up threshold
                # (or this agent has no active records yet). Keep the SAC
                # update valid and expose the skipped rule update explicitly.
                rule_warmup_skipped = 1.0
            actor_loss = sac_actor_loss + self.rule_imitation_weight * rule_imitation_loss            
            self._assert_finite(actor_loss, 'actor_loss', agent_number)
            # Minimize the loss
            actor_loss.backward()
            self._assert_finite_gradients(agent.actor, 'actor', agent_number)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
            self._assert_finite_gradients(agent.actor, 'actor_after_clip', agent_number)
            agent.actor_optimizer.step()
            self._assert_finite_parameters(agent.actor, 'actor_after_update', agent_number)
            actor_update_applied = 1.0
            actor_loss_value = self._finite_mean(actor_loss, 'actor_loss', agent_number)
            sac_actor_loss_value = self._finite_mean(sac_actor_loss, 'sac_actor_loss', agent_number)
            
            #Update alpha
            if self.automatic_entropy_tuning:
                # import pdb; pdb.set_trace()
                self._assert_finite(agent.log_alpha, 'log_alpha_before_update', agent_number)
                alpha_loss = -(agent.log_alpha * (q_log_probs[agent_number].view(-1,1) + agent.target_entropy).detach()).mean()    
                self._assert_finite(alpha_loss, 'alpha_loss', agent_number)
                agent.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self._assert_finite(agent.log_alpha.grad, 'log_alpha.grad', agent_number)
                agent.alpha_optimizer.step()    
                agent.alpha = agent.log_alpha.exp()
                self._assert_finite(agent.log_alpha, 'log_alpha_after_update', agent_number)
                self._assert_finite(agent.alpha, 'alpha_after_update', agent_number)
                alpha_loss_value = self._finite_mean(alpha_loss, 'alpha_loss', agent_number)
            else:
                # 固定 alpha 的配置没有 log_alpha/alpha_optimizer；仅记录当前熵权重。
                self._assert_finite(agent.alpha, 'alpha_fixed', agent_number)
    
            al = actor_loss.cpu().detach().item()
            sac_al = sac_actor_loss.cpu().detach().item()
            ri = rule_imitation_loss.cpu().detach().item()
            cl = critic_loss.cpu().detach().item()
            self.last_rule_update_stats[agent_number] = {
                'actor_loss': float(al),
                'critic_loss': float(cl),
                'alpha_loss': float(alpha_loss_value),
                'rule_batch_size': int(rule_batch_size),
                'rule_imitation_loss': float(ri),
                'weighted_rule_loss': float(self.rule_imitation_weight * ri),
                'rule_active_ratio': float(rule_active_ratio),
                'rule_unique_sample_ratio': float(rule_unique_sample_ratio),
                'rule_reuse_ratio': float(rule_reuse_ratio),
                'rule_sample_time_ms': float(rule_sample_time_ms),
                'rule_raw_teacher_distance': float(rule_raw_teacher_distance),
                'rule_actor_teacher_distance': float(rule_actor_teacher_distance),
                'rule_warmup_skipped': float(rule_warmup_skipped),
            }
            logger.add_scalars('agent%i/losses' % agent_number,
                               {'critic loss': cl,
                                'actor_loss': al,
                                'sac_actor_loss': sac_al,
                                'rule_imitation_loss': ri,
                                'weighted_rule_loss': self.rule_imitation_weight * ri,
                                'rule_filter_active_ratio': rule_active_ratio,
                                'rule_batch_size': rule_batch_size,
                                'rule_unique_sample_ratio': rule_unique_sample_ratio,
                                'rule_reuse_ratio': rule_reuse_ratio,
                                'rule_sample_time_ms': rule_sample_time_ms,
                                'rule_raw_teacher_distance': rule_raw_teacher_distance,
                                'rule_actor_teacher_distance': rule_actor_teacher_distance},
                               self.iter)
            logger.add_scalars(
                'agent%i/rule_update' % agent_number,
                {
                    'actual_batch_size': float(rule_batch_size),
                    'unique_sample_ratio': float(rule_unique_sample_ratio),
                    'reuse_ratio': float(rule_reuse_ratio),
                    'sample_time_ms': float(rule_sample_time_ms),
                    'rule_imitation_loss': float(ri),
                    'weighted_rule_loss': float(self.rule_imitation_weight * ri),
                    'raw_teacher_distance': float(rule_raw_teacher_distance),
                    'actor_teacher_distance': float(rule_actor_teacher_distance),
                    'warmup_skipped': float(rule_warmup_skipped),
                },
                self.iter,
            )
        # 计算TD误差
        loss_mse = torch.nn.MSELoss(reduction='none')
        td_error_1 = loss_mse(current_Q1, target_Q.detach())
        td_error_2 = loss_mse(current_Q2, target_Q.detach())
        td_error = torch.max(td_error_1, td_error_2).squeeze()
        self._assert_finite(td_error_1, 'td_error_q1_squared', agent_number)
        self._assert_finite(td_error_2, 'td_error_q2_squared', agent_number)
        self._assert_finite(td_error, 'td_error_squared', agent_number)
        td_error_abs = torch.max(
            torch.abs(current_Q1 - target_Q.detach()),
            torch.abs(current_Q2 - target_Q.detach()),
        )
        self._assert_finite(td_error_abs, 'td_error_absolute', agent_number)
        if agent.CTIP_active:
            # 计算 ICM 内在奖励
            state = obs_full
            next_state = next_obs_full
            # ICM 学的是环境真实转移，因此输入最终平滑执行动作，而不是
            # 仅存在于 Critic 归因链中的 raw_action。
            predicted_next_state_encoded, predicted_action = agent.icm(
                state,
                next_state,
                execution_action_full,
            )
            next_state_encoded = agent.icm.encoder(next_state)
            self._assert_finite(predicted_next_state_encoded, 'icm_predicted_next_state', agent_number)
            self._assert_finite(predicted_action, 'icm_predicted_action', agent_number)
            self._assert_finite(next_state_encoded, 'icm_next_state_encoded', agent_number)
            
            forward_loss = f.mse_loss(predicted_next_state_encoded, next_state_encoded, reduction='none')
            # 对每个样本的损失在合适的维度上求和（假设是最后一个维度）
            forward_loss = forward_loss.sum(dim=-1) 
            self._assert_finite(forward_loss, 'icm_forward_loss_per_sample', agent_number)

            # 反向模型损失
            inverse_loss = f.mse_loss(predicted_action, execution_action_full)
            icm_loss = forward_loss.mean() + inverse_loss  # 对前向损失求平均后再与反向损失相�?            
            self._assert_finite(inverse_loss, 'icm_inverse_loss', agent_number)
            self._assert_finite(icm_loss, 'icm_loss', agent_number)
            agent.icm_optimizer.zero_grad()
            icm_loss.backward()
            self._assert_finite_gradients(agent.icm, 'icm', agent_number)
            agent.icm_optimizer.step()
            self._assert_finite_parameters(agent.icm, 'icm_after_update', agent_number)
            # 计算内在奖励
            # intrinsic_reward = forward_loss.detach()


            td_error_values = td_error.cpu().detach().numpy()
            eicm_values = forward_loss.detach().cpu().numpy()  # EICM即前向损�?            # 定义经验分类参数
            exploration_threshold = 0.1 * max(int(len(eicm_values) / 256) * 10,1) # 探索类经验的阈值（可调整）
            exploration_amplify_coef = 2.0  # 探索类经验的TD-error放大系数（可调整�?            
            
            # 初始化优先级数组
            new_priorities = np.zeros_like(td_error_values)
            #计算n分位的阈�?            # exploration_threshold = np.percentile(eicm_values, 80)
            #计算第N大的阈�?            # k = int(min(len(eicm_values) / 2, 100))
            # partitioned = np.partition(eicm_values, k)
            # exploration_threshold = partitioned[k]
            # 经验分类与优先级计算（改良部分）
            for i in range(len(eicm_values)):
                if eicm_values[i] >= exploration_threshold:
                    td_error_values[i] = np.clip((td_error_values[i] + 10. * eicm_values[i]), 0.0, 10.0)
                    # print("探索类经验：放大TD-error作为优先�?,td_error_values[i] * exploration_amplify_coef)
                    new_priorities[i] = td_error_values[i] * exploration_amplify_coef
                else:
                    td_error_values[i] = np.clip(td_error_values[i], 0.0, 10.0)
                    # print("利用类经验：传统PER的TD-error优先�?,td_error_values[i])
                    new_priorities[i] = td_error_values[i]
        else:
            td_error_values = td_error.cpu().detach().numpy()
            # 初始化优先级数组
            new_priorities = np.zeros_like(td_error_values)
            for i in range(len(td_error_values)):
                td_error_values[i] = np.clip(td_error_values[i], 0.0, 10.0)
                new_priorities[i] = td_error_values[i]
        self._assert_finite(new_priorities, 'new_priorities', agent_number)

        # SAC 更新链的统一诊断指标。所有指标都在写入前经过有限值检查；
        # ``finite_guard_pass=1`` 只表示本次 update 的张量、梯度和优先级检查通过。
        sac_update_metrics = {
            'current_q1_mean': self._finite_mean(current_Q1, 'current_Q1', agent_number),
            'current_q2_mean': self._finite_mean(current_Q2, 'current_Q2', agent_number),
            'target_q_mean': self._finite_mean(target_Q, 'target_Q', agent_number),
            'td_error_abs_mean': self._finite_mean(td_error_abs, 'td_error_absolute', agent_number),
            'td_error_squared_mean': self._finite_mean(td_error, 'td_error_squared', agent_number),
            'policy_action_mean': float(policy_action_mean),
            'policy_log_prob_mean': float(policy_log_prob_mean),
            'actor_mean_mean': float(actor_mean_mean),
            'actor_log_std_mean': float(actor_log_std_mean),
            'critic_loss': self._finite_mean(critic_loss, 'critic_loss', agent_number),
            'actor_loss': float(actor_loss_value),
            'sac_actor_loss': float(sac_actor_loss_value),
            'alpha': self._finite_mean(agent.alpha, 'alpha', agent_number),
            'target_entropy': float(agent.target_entropy) if self.automatic_entropy_tuning else 0.0,
            'alpha_loss': float(alpha_loss_value),
            'actor_update_applied': float(actor_update_applied),
            'new_priority_mean': self._finite_mean(new_priorities, 'new_priorities', agent_number),
            'finite_guard_pass': 1.0,
        }
        for metric_name, metric_value in sac_update_metrics.items():
            if not np.isfinite(float(metric_value)):
                raise FloatingPointError(
                    'Non-finite SAC metric: agent=%d iter=%d field=%s value=%r'
                    % (agent_number, self.iter, metric_name, metric_value)
                )
        self.last_sac_update_stats[agent_number] = dict(sac_update_metrics)
        logger.add_scalars(
            'agent%i/sac_update' % agent_number,
            sac_update_metrics,
            self.iter,
        )
        return new_priorities
        # if agent_number+1 == self.num_agents: #this works test 78
        #     self.iter_delay += 1

    def update_targets(self):
        """soft update targets"""
        self.iter += 1 #this doesnt work as well as the other test 80
        self.iter_delay += 1
        # ----------------------- update target networks ----------------------- #
        for sac_agent in self.masac_agent:
            # soft_update(sac_agent.target_actor, sac_agent.actor, self.tau)
            soft_update(sac_agent.target_critic, sac_agent.critic, self.tau)
            
            
            
