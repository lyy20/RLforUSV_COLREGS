# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:37:13 2021

@author: Usuari

5. Watch a Smart Agent!
In the next code cell, you will load the trained weights from file to watch a smart agent!
"""
from algorithms.ppo.mappo import MAPPO
from utilities import envs
from utilities.buffer import ReplayBuffer, ReplayBuffer_SummTree
from algorithms.ddpg.maddpg import MADDPG
from algorithms.sac.masac import MASAC
from algorithms.td3.matd3_bc import MATD3_BC
from algorithms.ACLSAC.maaclsac import MAACLSAC
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities.utilities import transpose_list, transpose_to_tensor, circle_path, random_levy
from utilities.paths import training_model_dir
import time
import copy
import matplotlib.pyplot as plt
import pickle
import sys
from configparser import ConfigParser
from matplotlib.patches import Polygon
import math
from matplotlib import transforms  # 新增导入
import random
# for saving gif
import imageio
def draw_ship(ax, x, y, angle, scale=1.0, ship_length=8, ship_width=3):
    # 应用缩放因子
    ship_length *= scale / 256
    ship_width *= scale / 256
    
    # 基础变换：平移+旋转
    base_transform = (
        transforms.Affine2D()
        .translate(x, y)
        .rotate(angle)
        + ax.transData
    )
    
    # 1. 船体（主体）
    hull_points = np.array([
        [-ship_length/2, -ship_width/2.5],  # 船尾左侧
        [ship_length/2 * 0.5, -ship_width/2],  # 船头左侧
        [ship_length/2, 0],  # 船头尖端
        [ship_length/2 * 0.5, ship_width/2],  # 船头右侧
        [-ship_length/2, ship_width/2.5],  # 船尾右侧
        [-ship_length/2, -ship_width/2.5]  # 闭合
    ])
    hull = Polygon(hull_points, closed=True, facecolor='#8B4513', edgecolor='black', linewidth=1.5, transform=base_transform)
    ax.add_patch(hull)
    
    # 2. 甲板
    deck_points = np.array([
        [-ship_length/2 * 0.8, -ship_width/4],
        [ship_length/2 * 0.4, -ship_width/3],
        [ship_length/2 * 0.4, ship_width/3],
        [-ship_length/2 * 0.8, ship_width/4],
        [-ship_length/2 * 0.8, -ship_width/4]
    ])
    deck = Polygon(deck_points, closed=True, facecolor='#A0522D', edgecolor='black', linewidth=1, transform=base_transform)
    ax.add_patch(deck)
    
    # 3. 驾驶舱
    bridge_points = np.array([
        [-ship_length/4, -ship_width/8],
        [ship_length/8, -ship_width/8],
        [ship_length/8, ship_width/8],
        [-ship_length/4, ship_width/8],
        [-ship_length/4, -ship_width/8]
    ])
    bridge = Polygon(bridge_points, closed=True, facecolor='#4682B4', edgecolor='black', transform=base_transform)
    ax.add_patch(bridge)
    
    # 6. 船头标识（可选）
    bow_marker = plt.Circle(
        (ship_length/2 * 0.7, 0), ship_width/12, 
        facecolor='yellow', edgecolor='black', transform=base_transform
    )
    ax.add_patch(bow_marker)
def transform_coordinates(A, B, C):
    """将三组坐标转换到以A为X轴的新坐标系"""
    # 获取A的起始点和终点
    p0 = A[0]
    p1 = A[-1]
    
    # 计算A的原始长度
    original_length = np.linalg.norm(p1 - p0)
    
    # 计算A的方向向量（归一化）
    v = (p1 - p0) / original_length
    
    # 计算垂直于v的法向量
    if v[0] == 0 and v[1] == 0:
        # 处理特殊情况：如果v是零向量（不应该发生在匀速直线运动中）
        u = np.array([1, 0])
    else:
        # 旋转90度得到法向量
        u = np.array([-v[1], v[0]])
    
    # 构建旋转矩阵：第一行为v，第二行为u
    rotation_matrix = np.array([v, u])
    
    # 将A的起始点设为原点(0,0)
    A_centered = A - p0
    B_centered = B - p0
    C_centered = C - p0
    
    # 应用旋转变换，将A旋转到X轴
    A_transformed = np.dot(A_centered, rotation_matrix.T)
    B_transformed = np.dot(B_centered, rotation_matrix.T)
    C_transformed = np.dot(C_centered, rotation_matrix.T)
    
    # 缩放A的总长度至2
    scale_factor = 2.0 / original_length
    A_scaled = A_transformed * scale_factor
    B_scaled = B_transformed * scale_factor
    C_scaled = C_transformed * scale_factor
    
    return A_scaled, B_scaled, C_scaled, original_length, scale_factor
# Read config file argument if its necessary
if( len( sys.argv ) > 1 ):
    configFile = sys.argv[1]
else:
    configFile = 'trained_saca'
#print ('Configuration File   =  ',configFile +'.txt')

config = ConfigParser()
config.read(configFile+'.txt')

BUFFER_SIZE    = config.getint('hyperparam','BUFFER_SIZE')
BATCH_SIZE     = config.getint('hyperparam','BATCH_SIZE')
GAMMA          = config.getfloat('hyperparam','GAMMA')
TAU            = config.getfloat('hyperparam','TAU')
LR_ACTOR       = config.getfloat('hyperparam','LR_ACTOR')
LR_CRITIC      = config.getfloat('hyperparam','LR_CRITIC')
WEIGHT_DECAY   = config.getfloat('hyperparam','WEIGHT_DECAY')
CLIP_EPISILON = config.getfloat('hyperparam', 'CLIP_EPISILON')
UPDATE_EVERY   = config.getint('hyperparam','UPDATE_EVERY')
UPDATE_TIMES   = config.getint('hyperparam','UPDATE_TIMES')
SEED           = config.getint('hyperparam','SEED')
BENCHMARK      = config.getboolean('hyperparam','BENCHMARK')
EXP_REP_BUF    = config.getboolean('hyperparam','EXP_REP_BUF')
PRE_TRAINED    = config.getboolean('hyperparam','PRE_TRAINED')
#Scenario used to train the networks
SCENARIO       = config.get('hyperparam','SCENARIO')
RENDER         = config.getboolean('hyperparam','RENDER')
PROGRESS_BAR   = config.getboolean('hyperparam','PROGRESS_BAR')
RNN            = config.getboolean('hyperparam','RNN')
ICM = config.getboolean('hyperparam', 'ICM')
ICM_GAMMA = config.getfloat('hyperparam', 'ICM_GAMMA')
PPO_method = config.get('hyperparam', 'PPO_method')
HISTORY_LENGTH = config.getint('hyperparam','HISTORY_LENGTH')
DNN            = config.get('hyperparam','DNN')
START_STEPS    = config.getint('hyperparam','START_STEPS')
REWARD_WINDOWS = config.getint('hyperparam','REWARD_WINDOWS')
LANDMARK_ERROR_WINDOWS     = config.getint('hyperparam','LANDMARK_ERROR_WINDOWS')
COLLISION_OUTWORLD_WINDOWS = config.getint('hyperparam','COLLISION_OUTWORLD_WINDOWS')
ALPHA          = config.getfloat('hyperparam','ALPHA')
AUTOMATIC_ENTROPY = config.getboolean('hyperparam','AUTOMATIC_ENTROPY')
DIM_1          = config.getint('hyperparam','DIM_1')
DIM_2          = config.getint('hyperparam','DIM_2')
# number of parallel agents
parallel_envs  = config.getint('hyperparam','parallel_envs')
# number of agents per environment
num_agents     = config.getint('hyperparam','num_agents')
# number of landmarks (or targets) per environment
num_landmarks  = config.getint('hyperparam','num_landmarks')
# 障碍物的数量
num_obstacles = config.getint('hyperparam', 'num_obstacles')
# range of obsevation
ob_range = config.getfloat('hyperparam', 'ob_range')
# num of obsevation (obstacles)
num_ob = config.getint('hyperparam', 'num_ob')
landmark_depth = config.getfloat('hyperparam','landmark_depth')
landmark_movable = config.getboolean('hyperparam', 'landmark_movable')
obstacle_movable = config.getboolean('hyperparam', 'obstacle_movable')
landmark_vel = config.getfloat('hyperparam', 'landmark_vel')
movement = config.get('hyperparam', 'movement')
obstacle_movement = config.get('hyperparam', 'obstacle_movement')
pre_method        = config.get('hyperparam','pre_method')
rew_err_th       = config.getfloat('hyperparam','rew_err_th')
rew_dis_th       = config.getfloat('hyperparam','rew_dis_th')
max_range = config.getfloat('hyperparam', 'max_range')
max_current_vel = config.getfloat('hyperparam', 'max_current_vel')
range_dropping = config.getfloat('hyperparam', 'range_dropping')
# number of training episodes.
# change this to higher number to experiment. say 30000.
number_of_episodes = config.getint('hyperparam','number_of_episodes')
episode_length = config.getint('hyperparam','episode_length')
# how many episodes to save policy and gif
save_interval  = config.getint('hyperparam','save_interval')
# amplitude of OU noise
# this slowly decreases to 0
noise          = config.getfloat('hyperparam','noise')
noise_reduction= config.getfloat('hyperparam','noise_reduction')    
fol_in = int(np.random.rand()*1000)
try:
    max_vel = config.getfloat('hyperparam', 'max_vel')
    random_vel = config.getboolean('hyperparam', 'random_vel')
except:
    print('no max_vel or random_vel found in config file')
    max_vel = 0.
    random_vel = False
#Chose device
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run the pytorch tensors on cuda GPU
DEVICE = 'cpu'

CIRCLE = False
CIRCLE_RADI = 110.

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # as reproducibility docs
    torch.backends.cudnn.deterministic = True  # as reproducibility docs

def main():
    Count = 1
    RMSE = 0.
    STD = 0.
    MEAN_SCORE = 0.
    done_count = 0 
    for temp in range(Count):
        SCENARIO = 'tracking6.27_ship'
        # global RNN
        global SEED
        SEED = np.random.randint(0,10000)
        # SEED = 2658
        print("本次测试种子为:",SEED)
        seeding(seed = SEED)
        # number of parallel agents
        parallel_envs = 1
        episode_length = 200
        landmark_movable = True
        movement = 'linear'
        obstacle_movable = True
        obstacle_movement = 'linear'
        random_vel = True
        #velocities in km/s
        landmark_vel = 0.0003
        max_vel = 0.0003

        pre_method = 'KM'
        # number of agents per environment
        # num_agents = 3
        # number of landmarks (or targets) per environment
        # num_landmarks = 3
        # number of obstacles per environment
        num_obstacles = 8
        #landmark depth
        # landmark_depth = landmark_depth
        PRE_TRAINED = True
        # initialize environment
        torch.set_num_threads(parallel_envs)
        # env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED, num_agents=num_agents, num_landmarks=num_landmarks, landmark_depth=landmark_depth, benchmark = BENCHMARK)
        env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED , num_agents=num_agents, num_landmarks=num_landmarks,num_obstacles = num_obstacles,ob_range = ob_range,num_ob = num_ob, landmark_depth=landmark_depth,landmark_movable = landmark_movable,obstacle_movable = obstacle_movable, landmark_vel=landmark_vel, max_vel=max_vel,
                                    random_vel=random_vel, movement=movement, obstacle_movement=obstacle_movement, pre_method=pre_method, rew_err_th=rew_err_th, rew_dis_th=rew_dis_th,max_range=max_range,
                                    max_current_vel=max_current_vel, range_dropping=range_dropping,  benchmark = BENCHMARK)
        # initialize policy and critic
        if DNN == 'MADDPG':
                maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN, dim_1=DIM_1, dim_2=DIM_2)
        elif DNN == 'MATD3_BC':
                maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN, dim_1=DIM_1, dim_2=DIM_2)
        elif DNN == 'MASAC':
                maddpg =    MASAC(num_agents = num_agents, num_landmarks = num_landmarks,num_ob = num_ob, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2)
        elif DNN == 'MAACLSAC':
                maddpg =    MAACLSAC(num_agents = num_agents, num_landmarks = num_landmarks,num_obstacles = num_obstacles, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2)
        elif DNN == 'MAPPO':
            maddpg = MAPPO(num_agents=num_agents, num_landmarks=num_landmarks, landmark_depth=landmark_depth,
                        num_obstacles=num_obstacles, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR,
                        lr_critic=LR_CRITIC, gamma=GAMMA, clip_epsilon=CLIP_EPISILON, device=DEVICE, rnn=RNN,icm_active= ICM,icm_gamma= ICM_GAMMA,PPO_method= PPO_method,
                        dim_1=DIM_1, dim_2=DIM_2)
        else:
            print('ERROR UNKNOWN DNN ARCHITECTURE')
        agents_reward = []
        for n in range(num_agents):
            agents_reward.append([])
        
        if PRE_TRAINED == True:
            
            #New test using LSTM
            #trained_checkpoint = os.getcwd()+'/logs/' + configFile+ '/model_dir/episode' #Test SAC auto
            trained_checkpoint = str(training_model_dir(configFile) / 'episode')  # Test SAC auto
            # RNN = True
            
            aux = torch.load(trained_checkpoint+'_last.pt')
            if DNN == 'MASAC'or DNN == 'MAHRSAC' or DNN == 'MAATLSAC':
                with open(trained_checkpoint +  '_target_entropy_last.file', "rb") as f:
                    target_entropy_aux = pickle.load(f)
                with open(trained_checkpoint +  '_log_alpha_last.file', "rb") as f:
                    log_alpha_aux = pickle.load(f)
                with open(trained_checkpoint + '_alpha_last.file', "rb") as f:
                    alpha_aux = pickle.load(f)
            for i in range(num_agents):  
                if DNN == 'MADDPG':
                    maddpg.maddpg_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.maddpg_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                elif DNN == 'MATD3_BC':
                    maddpg.matd3_bc_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.matd3_bc_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                elif DNN == 'MASAC'or DNN == 'MAHRSAC' or DNN == 'MAATLSAC':
                    if AUTOMATIC_ENTROPY:
                        maddpg.masac_agent[i].actor.load_state_dict(aux[0]['actor_params'])
                        maddpg.masac_agent[i].critic.load_state_dict(aux[0]['critic_params'])
                        # maddpg.masac_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                        # maddpg.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                        # maddpg.masac_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
                        # maddpg.masac_agent[i].alpha_optimizer.load_state_dict(aux[i]['alpha_optim_params'])
                        #load agents alpha parameters
                        maddpg.masac_agent[i].target_entropy = target_entropy_aux[0]
                        maddpg.masac_agent[i].log_alpha = log_alpha_aux[0]
                        maddpg.masac_agent[i].alpha = alpha_aux[0]
                    else:
                        maddpg.masac_agent[i].actor.load_state_dict(aux[0]['actor_params'])
                        maddpg.masac_agent[i].critic.load_state_dict(aux[0]['critic_params'])
                        # maddpg.masac_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                        # maddpg.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                        # maddpg.masac_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
                elif DNN == 'MAPPO':
                    # 后续需要加PPO的算法
                    maddpg.mappo_agents[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.mappo_agents[i].critic.load_state_dict(aux[i]['critic_params'])
                    # maddpg.mappo_agents[i].target_actor.load_state_dict(aux[i]['target_actor_params'])
                    # maddpg.mappo_agents[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                    # maddpg.mappo_agents[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                    # maddpg.mappo_agents[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
                else:    
                    break
        
        #Reset the environment
        all_obs = env.reset() 
        # flip the first two indices
        obs_roll = np.rollaxis(all_obs,1)
        obs = transpose_list(obs_roll)
        
        #Reset landmark error benchmark
        landmark_error = []
        for i in range(num_landmarks):
            landmark_error.append([])
        landmark_error_episode = []
        for i in range(num_landmarks):
            landmark_error_episode.append([])
        
        #Initialize history buffer with 0.
        obs_size = obs[0][0].size
        history = copy.deepcopy(obs)
        for n in range(parallel_envs):
            for m in range(num_agents):
                for i in range(HISTORY_LENGTH-1):
                    if i == 0:
                        history[n][m] = history[n][m].reshape(1,obs_size)*0.
                    aux = obs[n][m].reshape(1,obs_size)*0.
                    history[n][m] = np.concatenate((history[n][m],aux),axis=0)
        #Initialize action history buffer with 0.
        history_a = np.zeros([parallel_envs,num_agents,HISTORY_LENGTH,1]) #the last entry is the number of actions, here is 2 (x,y)
        
        scores = 0            
        scores_list = []    
        t = 0
        
        #save gif
        frames = []
        gif_folder = ''
        main_folder = trained_checkpoint.split('\\')
        for i in range(len(main_folder)-2):
            gif_folder += main_folder[i]
            gif_folder += '\\'
        total_rewards = []
        steps = []
        agent_x = []
        agent_y = []
        agent_size = []
        agent_angle = []
        range_total = []
        for i in range(num_agents):
            agent_x.append([])
            agent_y.append([])
            agent_size.append([])
            agent_angle.append([])
            range_total.append([])
        landmark_x = []
        landmark_y = []
        landmark_p_x = []
        landmark_p_y = []
        landmark_size = []
        for i in range(num_landmarks):
            landmark_x.append([])
            landmark_y.append([])
            landmark_p_x.append([])
            landmark_p_y.append([])
            landmark_size.append([])
        obstacle_x = []
        obstacle_y = []
        obstacle_size = []
        for i in range(num_obstacles):
            obstacle_x.append([])
            obstacle_y.append([])
            obstacle_size.append([])
        episodes = 0
        episodes_total = []
        while t<episode_length:
            frames.append(env.render('rgb_array'))
            t +=1
            # select an action
            his = []
            for i in range(num_agents):
                his.append(torch.cat((transpose_to_tensor(history)[i],transpose_to_tensor(history_a)[i]), dim=2))
            # actions = maddpg.act(transpose_to_tensor(obs), noise=0.)       
            # actions = maddpg.act(transpose_to_tensor(history), noise=0.) 
            actions = maddpg.act(his,transpose_to_tensor(obs) , noise=0.0) 
            
            # print('actions=',actions)
            
            actions_array = torch.stack(actions).detach().numpy()
            actions_for_env = np.rollaxis(actions_array,1)
            
            #cirlce path using my previous functions
            if CIRCLE == True:
                actions_for_env = circle_path(obs,CIRCLE_RADI,t) #radius of the desired agent circunference, between 50m and 1000m
            # print('actions=',actions_for_env)
            
            
            # actions_for_env = np.array([[[np.pi*2./10./0.3]]])
            # if t  > 10:
            #     actions_for_env = np.array([[[0.,0.1]]])
            # if t  > 20:
            #     actions_for_env = np.array([[[0.,0.1]]])
            # if t  > 30:
            #     actions_for_env = np.array([[[0.,0.1]]])
            # if t  > 40:
            #     actions_for_env = np.array([[[1.,0.1]]])
            
            #see a random agent
            # actions_for_env = np.array([[np.random.rand(1)*2-1]])
            beta = 1.99 #must be between 1 and 2
            # actions_for_env = random_levy(beta)
            
            # import pdb; pdb.set_trace()
            
            # send all actions to the environment
            next_obs, rewards, dones, info = env.step(actions_for_env)

            # Update history buffers
            # Add obs to the history buffer
            for n in range(parallel_envs):
                for m in range(num_agents):
                    aux = obs[n][m].reshape(1,obs_size)
                    history[n][m] = np.concatenate((history[n][m],aux),axis=0)
                    history[n][m] = np.delete(history[n][m],0,0)
            # Add actions to the history buffer
            history_a = np.concatenate((history_a,actions_for_env.reshape(parallel_envs,num_agents,1,1)),axis=2)
            history_a = np.delete(history_a,0,2)
                        
            # update the score (for each agent)
            scores += np.sum(rewards)  
            scores_list.append(scores)
            # Save values to plot later on
            total_rewards.append(np.sum(rewards))
            steps.append(t)          
            for n in range(parallel_envs):
                for m in range(num_agents):
                    agent_x[m].append(obs[n][m][2])
                    agent_y[m].append(obs[n][m][3])
                    range_total[m].append(obs[n][m][6])
                    agent_size[m].append(info[n]['n'][m][8][m])
                    agent_angle[m].append(info[n]['n'][m][9][m])

                    landmark_x[m].append(info[n]['n'][m][1][m][0])
                    landmark_y[m].append(info[n]['n'][m][1][m][1])
                    landmark_p_x[m].append(info[n]['n'][m][6][m][0])
                    landmark_p_y[m].append(info[n]['n'][m][6][m][1])
                    landmark_size[m].append(info[n]['n'][m][8][m + num_agents])

                for i in range(num_obstacles):
                    obstacle_x[i].append(info[n]['n'][0][7][i][0])
                    obstacle_y[i].append(info[n]['n'][0][7][i][1])
                    obstacle_size[i].append(info[n]['n'][0][8][i + num_agents + num_landmarks])
                    

            # for e, inf in enumerate(info):
            #     for a in range(num_agents):
            #         agent_info[a] = np.add(agent_info[a],(inf['n'][a]))
                    
            # print ('\r\n Rewards at step %i = %.3f'%(t,scores))
            # roll over states to next time step  
            obs = next_obs     

            # print("Score: {}".format(scores))
            episodes += 1
            episodes_total.append(episodes)
            if np.any(dones):
                # print('done')
                done_count += 1
                break
                episodes = 0
                #env_wrapper line 18: env.reset(). Therefore, if you don't want an env.reset, comment this line.
                # break
        agent_color = ["#00008B","#006400","#00CCFF","#6df744","#f83ddf","#fbff02","#ff07b4"]
        landmark_color = ["#FFA500","#FF0000","#8A2BE2","#0e0d75","#3deb11","#fbff02","#ff07b4"]
        landmark_P_color = ["#FFD700","#FFC0CB","#D8BFD8","#0e0d75","#3deb11","#fbff02","#ff07b4"]
        #save .gif
        # imageio.mimsave(os.path.join(gif_folder, '{}seed-{}.gif'.format(configFile,SEED)), 
        #                             frames, duration=.04)
        # # 每回合奖励得分，用于测定各回合效果
        # plt.figure(figsize=(5,5),num= "每回合奖励图")
        # plt.plot(steps,total_rewards,'bo-')
        # plt.ylabel('Rewards')
        # plt.xlabel('Steps')
        # plt.title('Trained agent (RL)')
        # # plt.title('Predefined cricumference')
        # plt.show()
        # # 总得分图
        # plt.figure(figsize=(5,5),num= "得分图")
        # plt.plot(steps,scores_list,'bo-')
        # plt.ylabel('Scores')
        # plt.xlabel('Steps')
        # plt.title('Trained agent (RL)')
        # # plt.title('Predefined cricumference')
        # plt.show()
        # # 每回合目标和智能体的速度
        # colors = ['royalblue','orangered','gold','seagreen']
        # plt.figure(figsize=(5,5),num= "速度图")
        # for i in range(num_agents):
        #     agent_xv = np.array(agent_x[i])[:-1]-np.array(agent_x[i])[1:]
        #     agent_yv = np.array(agent_y[i])[:-1]-np.array(agent_y[i])[1:]
        #     agent_v = np.sqrt(agent_xv**2 + agent_yv**2)
        #     plt.plot(steps[1:],agent_v,'bo--',  alpha=0.5,label='Agent')
        # landmark_xv = np.array(landmark_x)[:-1]-np.array(landmark_x)[1:]
        # landmark_yv = np.array(landmark_y)[:-1]-np.array(landmark_y)[1:]
        # landmark_v = np.sqrt(landmark_xv**2 + landmark_yv**2)
        # plt.plot(steps[1:],landmark_v,'k^--',alpha=0.5,label='Landmark')
        # plt.ylabel('relative velocity')
        # plt.xlabel('Steps')
        # plt.title('Trained agent (RL)')
        # # plt.title('Predefined cricumference')
        # plt.legend()
        # plt.show()
        # 运行轨迹图，以及预测图
        # fig_width = 8
        # fig_length = 8
        # fig, ax = plt.subplots(figsize=(fig_width, fig_length))
        # # 设置坐标轴背景色
        # ax.set_facecolor('#B0E0E6')
        # plt.xlim(-2., 2.)
        # plt.ylim(-2., 2.)
        # d = 72 / ( fig_width / 5 ) * 2 * 2
        # # 绘制轨迹并保存句柄
        # handles = []
        # for i in range(num_agents):
        #     if len(landmark_x[i]) > 1:
        #         plt.plot(agent_x[i][:-1],agent_y[i][:-1],'o--',color = agent_color[i], alpha=0.5)
        #         line, = plt.plot(agent_x[i][-1],agent_y[i][-1],'o',color = agent_color[i], markersize=agent_size[i][0] * d,label='Agent {}'.format(i+1))
        #         handles.append(line)
        # for i in range(num_landmarks):
        #     if len(landmark_x[i]) > 1:
        #         line, = plt.plot(landmark_p_x[i],landmark_p_y[i],'D--',color = landmark_P_color[i],markersize=6 , alpha=0.5,label='Landmark Predicted {}'.format(i+1))
        #         handles.append(line)
        #         plt.plot(landmark_x[i][:-1],landmark_y[i][:-1],color = landmark_color[i],linewidth=4, alpha=0.5)
        #         line, = plt.plot(landmark_x[i][-1],landmark_y[i][-1],'o',color = landmark_color[i], markersize=landmark_size[i][0] * d,label='Landmark Real {}'.format(i+1))
        #         handles.append(line)
                
        # for i in range(num_obstacles):
        #     if i == 0:
        #         line, = plt.plot(obstacle_x[i],obstacle_y[i],color = "#696969",marker = 'o',markersize=obstacle_size[i][0] * d, label='Obstacle')
        #     else:
        #         line, = plt.plot(obstacle_x[i],obstacle_y[i],color = "#696969",marker = 'o',markersize=obstacle_size[i][0] * d, )
        #     handles.append(line)  # 保存句柄
        # plt.xlabel('X position')
        # plt.ylabel('Y position')
        # # plt.title('Test 2b')
        # plt.axis('equal')
        # # plt.xlim(0.26,0.3)
        # # plt.ylim(-0.14,-0.08)
        
        # # plt.xlim(0.4,1.1)
        # # plt.ylim(-0.9,-0.3)
        # # 创建图例
        # legend = ax.legend(handles=handles, loc='lower left',framealpha=0.2)

        # # 统一设置图例中所有项的标记大小为 8
        # for handle in legend.legendHandles:
        #     handle.set_markersize(5)  # 设置标记大小（单位：点，points）
        # plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 0.75, length=2.5)
        # plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.1, length=1.5)
        # plt.grid(which='major', linestyle='-', linewidth='.8', alpha=0.4)
        # plt.grid(which='minor', linestyle='-', linewidth='.4', alpha=0.4)
        
        # # plt.title('Predefined cricumference')
        # # plt.savefig('depth_test.png',format='png', dpi=800 ,bbox_inches='tight',pad_inches = 0.02)
        # plt.show()
        # # 每回合预测误差图
        # target_error = np.sqrt((np.array(landmark_p_x)-np.array(landmark_x))**2+(np.array(landmark_p_y)-np.array(landmark_y))**2)
        # plt.figure(figsize=(5,5),num= "预测误差图")
        # plt.plot(steps,target_error,'bo-')
        # plt.hlines(rew_err_th,0,episode_length, colors='k', linestyles='--')
        # plt.ylabel('Target prediction error (RMSE)')
        # plt.xlabel('Steps')
        # plt.title('Trained agent (RL)')
        # plt.ylim(0,0.1)
        # # plt.title('Predefined cricumference')
        # plt.show()
        # if len(target_error) >= 100:
        #     print('RMSE= %.3f m; STD = %.3f m' % (np.mean(target_error[-100:])*1000., np.std(target_error[-100:])*1000.))
        #     RMSE += np.mean(target_error[-100:])*1000.
        #     STD += np.std(target_error[-100:])*1000.
        # else :
        #     print('RMSE= %.3f m; STD = %.3f m' % (np.mean(target_error[-len(target_error):])*1000., np.std(target_error[-len(target_error):])*1000.))
        #     RMSE += np.mean(target_error[-len(target_error):])*1000.
        #     STD += np.std(target_error[-len(target_error):])*1000.
        
        # # 每回合智能体与目标的距离
        # plt.figure(figsize=(5,5),num= "距离图")
        # for i in range(num_agents):
        #     plt.plot(steps,range_total[i],'bo-')
        # plt.hlines(rew_dis_th,0,episode_length, colors='k', linestyles='--')
        # plt.ylabel('Range')
        # plt.xlabel('Steps')
        # plt.title('Trained agent (RL)')
        # plt.grid()
        # plt.ylim(0,0.5)
        # # plt.title('Predefined cricumference')
        # plt.show()
        # print('avg range = %.1f m; STD = %.3f m'% (np.mean(range_total[0][-100:])*1000., np.std(range_total[0][-100:])*1000.))
        # # 存活时长
        # plt.figure(figsize=(5,5),num= "智能体存活时长图")
        # plt.plot(steps,episodes_total,'bo-')
        # plt.ylabel('Number of episodes')
        # plt.xlabel('Steps')
        # plt.title('Trained agent (RL)')
        # # plt.title('Predefined cricumference')
        # plt.show()
        
        # print('MEAN SCORE = ',scores)
        # MEAN_SCORE += scores
        # print('TOTAL LAST SCORE = ',np.mean(total_rewards[::-1][:10]))
        
        # while True:
        #     a = 0
        #     break
        # 创建一个 2×4 的子图布局
        if True:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            Step_len = int(t / (8))
            for i_x in range(2):
                for i_y in range(4):
                    # 设置坐标轴背景色
                    axes[i_x, i_y].set_facecolor('#B0E0E6')
                    axes[i_x, i_y].set_xlim(-2., 2.)
                    axes[i_x, i_y].set_ylim(-2., 2.)
                    d = 72 / ( 4 / 5 ) 
                    # 绘制轨迹并保存句柄
                    handles = []
                    for i in range(num_obstacles):
                        if i == 0:
                            axes[i_x, i_y].plot(obstacle_x[i][:(Step_len * (4 * i_x + i_y + 1) - 1):2],obstacle_y[i][:(Step_len * (4 * i_x + i_y + 1)) - 1:2],
                                                'o--',color = "#008d6e",
                                                markersize=2,  alpha=0.3)
                            line4, = axes[i_x, i_y].plot(obstacle_x[i][(Step_len * (4 * i_x + i_y + 1)) - 1],obstacle_y[i][(Step_len * (4 * i_x + i_y + 1)) - 1],
                                                        'o',
                                                        markersize=obstacle_size[i][0] * d,
                                                        markerfacecolor='none',  # 设置空心
                                                        markeredgecolor='#008d6e',
                                                        label='Moving Obstacle')
                            handles.append(line4)  # 保存句柄
                        else:
                            axes[i_x, i_y].plot(obstacle_x[i][:(Step_len * (4 * i_x + i_y + 1) - 1):2],obstacle_y[i][:(Step_len * (4 * i_x + i_y + 1)) - 1:2],
                                                'o--',color = "#008d6e",
                                                markersize=2,  alpha=0.3)
                            line4, = axes[i_x, i_y].plot(obstacle_x[i][(Step_len * (4 * i_x + i_y + 1)) - 1],obstacle_y[i][(Step_len * (4 * i_x + i_y + 1)) - 1],
                                                        'o',
                                                        markersize=obstacle_size[i][0] * d,
                                                        markerfacecolor='none',  # 设置空心
                                                        markeredgecolor='#008d6e')
                    for i in range(num_landmarks):
                        if len(landmark_x[i]) > 1:
                            line2, = axes[i_x, i_y].plot(landmark_p_x[i][:(Step_len * (4 * i_x + i_y + 1) - 1):2],landmark_p_y[i][:(Step_len * (4 * i_x + i_y + 1) - 1):2],'D--',color = landmark_P_color[i],markersize=6 , alpha=0.7,label='Target Predicted')
                            handles.append(line2)
                            axes[i_x, i_y].plot(landmark_x[i][:(Step_len * (4 * i_x + i_y + 1) - 1):2],landmark_y[i][:(Step_len * (4 * i_x + i_y + 1) - 1):2],color = landmark_color[i],linewidth=4, alpha=0.7)
                            line3, = axes[i_x, i_y].plot(landmark_x[i][(Step_len * (4 * i_x + i_y + 1) - 1)],landmark_y[i][(Step_len * (4 * i_x + i_y + 1) - 1)],'o',color = landmark_color[i], markersize=landmark_size[i][0] * d,label='Target')
                            handles.append(line3)
                    for i in range(num_agents):
                        if len(landmark_x[i]) > 1:
                            axes[i_x, i_y].plot(agent_x[i][:(Step_len * (4 * i_x + i_y + 1) - 1):2],agent_y[i][:(Step_len * (4 * i_x + i_y + 1)) - 1:2],'o--',color = agent_color[i],markersize=3,  alpha=0.3)
                            line1, = axes[i_x, i_y].plot(agent_x[i][(Step_len * (4 * i_x + i_y + 1)) - 1],agent_y[i][(Step_len * (4 * i_x + i_y + 1)) - 1],'o',color = agent_color[i], markersize=agent_size[i][0] * d,alpha=0.7,label='Agent')
                            # draw_ship(axes[i_x, i_y],agent_x[i][(Step_len * (4 * i_x + i_y + 1)) - 1],agent_y[i][(Step_len * (4 * i_x + i_y + 1)) - 1],agent_angle[i][(Step_len * (4 * i_x + i_y + 1)) - 1],agent_size[i][0] * d)
                            # 基础变换：平移+旋转
                            base_transform = (
                                transforms.Affine2D()
                                .translate(0, 0)
                                .rotate(agent_angle[i][(Step_len * (4 * i_x + i_y + 1)) - 1])
                                + axes[i_x, i_y].transData
                            )
                            x = agent_x[i][(Step_len * (4 * i_x + i_y + 1)) - 1]
                            y = agent_y[i][(Step_len * (4 * i_x + i_y + 1)) - 1]
                            ship_length=8 / 256
                            ship_width=3 / 256
                            # 1. 船体（主体）
                            hull_points = np.array([
                                [-ship_length/2, -ship_width/2.5],  # 船尾左侧
                                [ship_length/2 * 0.5, -ship_width/2],  # 船头左侧
                                [ship_length/2, 0],  # 船头尖端
                                [ship_length/2 * 0.5, ship_width/2],  # 船头右侧
                                [-ship_length/2, ship_width/2.5],  # 船尾右侧
                                [-ship_length/2, -ship_width/2.5]  # 闭合
                            ]) + np.array([x, y])
                            hull = Polygon(hull_points, closed=True, facecolor='#8B4513', edgecolor='black', linewidth=1.5, transform=base_transform)
                            axes[i_x, i_y].add_patch(hull)
                            
                            # 2. 甲板
                            deck_points = np.array([
                                [-ship_length/2 * 0.8, -ship_width/4],
                                [ship_length/2 * 0.4, -ship_width/3],
                                [ship_length/2 * 0.4, ship_width/3],
                                [-ship_length/2 * 0.8, ship_width/4],
                                [-ship_length/2 * 0.8, -ship_width/4]
                            ]) + np.array([x, y])
                            deck = Polygon(deck_points, closed=True, facecolor='#A0522D', edgecolor='black', linewidth=1, transform=base_transform)
                            axes[i_x, i_y].add_patch(deck)
                            
                            # 3. 驾驶舱
                            bridge_points = np.array([
                                [-ship_length/4, -ship_width/8],
                                [ship_length/8, -ship_width/8],
                                [ship_length/8, ship_width/8],
                                [-ship_length/4, ship_width/8],
                                [-ship_length/4, -ship_width/8]
                            ]) + np.array([x, y])
                            bridge = Polygon(bridge_points, closed=True, facecolor='#4682B4', edgecolor='black', transform=base_transform)
                            axes[i_x, i_y].add_patch(bridge)
                            
                            # 6. 船头标识（可选）
                            bow_marker = plt.Circle(
                                (x + ship_length/2 * 0.7, y + 0), ship_width/12, 
                                facecolor='yellow', edgecolor='black', transform=base_transform
                            )
                            axes[i_x, i_y].add_patch(bow_marker)
                            handles.append(line1) 
                    axes[i_x, i_y].axis('equal')
                    # 创建图例
                    legend = axes[i_x, i_y].legend(
                                                    handles=handles, 
                                                    loc='upper left',
                                                    frameon=True,
                                                    edgecolor='black',
                                                    facecolor='white',
                                                    framealpha=0.5,
                                                    fontsize=5)

                    # 统一设置图例中所有项的标记大小
                    for handle in legend.legendHandles:
                        handle.set_markersize(5)  # 设置标记大小（单位：点，points）
                    axes[i_x, i_y].tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 0.75, length=2.5)
                    axes[i_x, i_y].tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.1, length=1.5)
                    axes[i_x, i_y].grid(which='major', linestyle='-', linewidth='.8', alpha=0.4)
                    axes[i_x, i_y].grid(which='minor', linestyle='-', linewidth='.4', alpha=0.4)
            # 添加主标题
            plt.suptitle('Trajectory diagram', fontsize=16)
            # 显示图形
            plt.show()




        env.close()
    



    # print("本次测试总次数为:",Count)
    # print('RMSE= %.3f m; STD = %.3f m' % (RMSE/Count, STD/Count))
    # print('MEAN SCORE = ',MEAN_SCORE/Count)
    # print("存活率：%.3f %%" % (((Count - done_count) * 100.) / Count ))
if __name__=='__main__':
    main()
    
