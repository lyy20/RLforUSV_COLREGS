import numpy as np

from A_Star import calculate_direction_angle
from multiagent.core import World, Agent, Landmark ,Obstacle
from multiagent.scenario import BaseScenario
from tracking.target_pf import Target
from utilities.utilities import random_levy
old_dist = 10.
def transform_to_local_coordinates(agent_pos, agent_angle, entity_pos):
    """
    将实体位置转换为以Agent位置为原点，Agent运行方向为X轴正方向的局部坐标系
    :param agent_pos: Agent的位置
    :param agent_angle: Agent的运行方向（弧度）
    :param entity_pos: 实体的位置
    :return: 局部坐标系下的实体位置
    """
    dx = entity_pos[0] - agent_pos[0]
    dy = entity_pos[1] - agent_pos[1]
    new_x = dx * np.cos(agent_angle) + dy * np.sin(agent_angle)
    new_y = -dx * np.sin(agent_angle) + dy * np.cos(agent_angle)
    return np.array([new_x, new_y])
class Scenario(BaseScenario):
    
    def make_world(self, num_agents=3, num_landmarks=3,num_obstacles=5,ob_range = 1.,num_ob = 3, landmark_depth=15., landmark_movable = False,obstacle_movable = False, landmark_vel=0.05, max_vel=0.2, random_vel=False, movement='linear',obstacle_movement = 'linear', pre_method = 'LS', rew_err_th=0.0003, rew_dis_th=0.3, max_range = 2., max_current_vel=0.,range_dropping = 0.2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        world.num_obstacles = num_obstacles
        world.ob_range = ob_range
        world.num_ob = num_ob
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.04
            agent.max_a_speed = 3.1415
            # agent has_captured target
            agent.has_captured = False
            # Time ( agent has_captured target )
            agent.consecutive_hold = 0
        # add landmarks 预测地标需要大于本体
        world.landmarks = [Landmark() for i in range(num_landmarks*2)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_landmarks:
                landmark.name = 'landmark %d' % i
                landmark.collide = False
                landmark.movable = landmark_movable
            else:
                landmark.name = 'landmark_estimation %d' % (i-num_landmarks)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.002
        # 创建障碍物
        # 翻倍障碍物，为了估计观测,且障碍物预测体需要大于本体
        world.obstacles = [Obstacle() for i in range(num_obstacles*2)]
        for i, obstacle in enumerate(world.obstacles):
            if i < num_obstacles:
                obstacle.name = 'obstacle %d' % i
                obstacle.collide = False
                obstacle.movable = obstacle_movable
            else:
                obstacle.name = 'obstacle_estimation %d' % (i-num_obstacles)
                obstacle.collide = False
                obstacle.movable = False

        # make initial conditions
        world.cov = np.ones(num_landmarks)/30.
        world.error = np.ones(num_landmarks)
        
        #make initial world current 
        self.max_vel_ocean_current = max_current_vel
        world.vel_ocean_current = 0 #initial random strength
        world.angle_ocean_current = 0 #initial landmark direction
        # world.vel_ocean_current = 0.05
        # world.angle_ocean_current = np.pi/2.*3.
        
        self.landmark_vel = landmark_vel
        # print('test',landmark_vel)
        
        #benchmark variables  基准参数   增加与障碍物的碰撞
        self.pre_error = np.ones(num_landmarks)
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        self.obstacle_collision = 0
        self.pre_error = np.ones(num_obstacles)
        #Scenario initial conditions
        self.max_landmark_depth = landmark_depth
        #set random target depth
        self.landmark_depth = 0
        # if self.landmark_depth<15.:
        #     self.landmark_depth = 15.
        self.ra = 0 #initial landmark direction
        self.movement = movement
        self.obstacle_movement = obstacle_movement
        self.pre_method = pre_method
        self.rew_err_th = rew_err_th
        self.rew_dis_th = rew_dis_th
        self.set_max_range = max_range
        #random variables for target
        self.max_vel = max_vel
        self.random_vel = random_vel
        self.reset_world(world)
        #
        world.damping = landmark_vel/5.
        
        self.range_dropping = range_dropping
        
        return world

    def reset_world(self, world):
        world.vel_ocean_current = np.random.rand(1).item(0)*self.max_vel_ocean_current #initial random strength
        world.angle_ocean_current = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                landmark.color = np.array([0.25, 0.25, 0.25])
            else:
                landmark.color = np.array([0.55, 0.0, 0.0])
        # 属性给障碍物
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                obstacle.color = np.array([0.0, 1.0, 0.0]) #绿
            else:
                obstacle.color = np.array([0.0, 1.0, 1.0]) #青色

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.5, 0.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_vel_old = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.a_vel = 0.
            agent.state.p_pos_origin = agent.state.p_pos.copy()
            # agent has_captured target
            agent.has_captured = False
            # Time ( agent has_captured target )
            agent.consecutive_hold = 0
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                dis = np.random.uniform(1.0 , 1.2)
                # dis = np.random.uniform(0.1, 1.) #改成这个避免碰撞次数太高
                rad = np.random.uniform(0, np.pi*2)
                landmark.state.p_pos = agent.state.p_pos + np.array([np.cos(rad),np.sin(rad)])*dis
                landmark.state.p_vel = np.zeros(world.dim_p)
            # if i < world.num_landmarks:
            #     dis = np.random.uniform(0.9, 1.8)
            #     rad = np.random.uniform(0, np.pi*2)
            #     landmark.state.p_pos = world.agents[0].state.p_pos + np.array([np.cos(rad),np.sin(rad)])*dis
            #     while landmark.state.p_pos[0] < -0.95 or landmark.state.p_pos[0] > 0.95 or landmark.state.p_pos[1] < -0.95 or landmark.state.p_pos[1] > 0.95:
            #         dis = np.random.uniform(0.4, 1.8)
            #         rad = np.random.uniform(0, np.pi * 2)
            #         landmark.state.p_pos = world.agents[0].state.p_pos + np.array([np.cos(rad), np.sin(rad)]) * dis
            #     landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = world.landmarks[i-world.num_landmarks].state.p_pos
                landmark.state.p_vel = np.zeros(world.dim_p)
        #障碍物随机定位，假设只有一个目标点，让障碍物不接触目标点
        target_landmark = world.landmarks[0]
        agent = world.agents[0]
        # 智能体与目标点之间的向量
        agent_to_target = target_landmark.state.p_pos - agent.state.p_pos
        agent_to_target_dist = np.linalg.norm(agent_to_target)
        # 智能体的大小，假设为一个常量，可根据实际情况调整
        agent_size = agent.size
        min_gap = agent_size * 4  # 两相近障碍物之间的最小缝隙
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                obstacle.size = np.random.uniform(0.05, 0.15)
                world.obstacles[i + world.num_obstacles].size = obstacle.size
                valid_position = False
                while not valid_position:
                    # 以智能体与目标点连线为直径创建圆形区域
                    # 先随机生成一个角度
                    angle = np.random.uniform(0, 2 * np.pi)
                    # 随机生成一个半径
                    radius = np.random.uniform(0, (3 * agent_to_target_dist) / 2)
                    # 计算圆形区域内的随机点相对于智能体的偏移向量
                    offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                    # 计算障碍物的位置
                    O_pos = agent.state.p_pos + agent_to_target / 2 + offset
                    # 检查是否覆盖目标点、智能体以及与其他障碍物重叠，并且距离目标 0.6 以上
                    valid = True
                    # 增加距离目标 0.6 以上的判断条件
                    if np.linalg.norm(O_pos - target_landmark.state.p_pos) <= 0.6 or np.linalg.norm(O_pos - agent.state.p_pos) <= (obstacle.size + agent_size + 0.3):
                        valid = False
                    for other_obstacle in world.obstacles[:i]:
                        dist = np.linalg.norm(O_pos - other_obstacle.state.p_pos)
                        if dist <= (obstacle.size + other_obstacle.size + min_gap):
                            valid = False
                            break
                    if valid:
                        valid_position = True

                obstacle.state.p_pos = O_pos
                obstacle.state.p_vel = np.zeros(world.dim_p)
            else:
                obstacle.state.p_pos = world.obstacles[i - world.num_obstacles].state.p_pos
                obstacle.state.p_vel = np.zeros(world.dim_p)
        #benchmark variables
        self.pre_error = np.ones(world.num_landmarks)
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        self.obstacle_collision = 0
        self.ob_pre_error = np.ones(world.num_obstacles)
        #tacke a random velocity
        for landmark in world.landmarks:
            if landmark.movable:
                if self.random_vel == True:
                    landmark.landmark_vel = np.random.rand(1).item(0)*self.max_vel
                else:
                    landmark.landmark_vel = self.landmark_vel
            else:
                landmark.landmark_vel = 0.
            landmark.max_speed = landmark.landmark_vel
                
        #take a random direction
        for landmark in world.landmarks:
            landmark.ra = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
            #take a random target depth
            landmark.landmark_depth = 0.
            # 给障碍物设置一个随机的速度
        for obstacle in world.obstacles:
            if self.random_vel == True:
                obstacle.obstacle_vel = np.random.rand(1).item(0) * self.max_vel
            else:
                obstacle.obstacle_vel = self.landmark_vel  #后续可以考虑给障碍物加一个配置属性，最大深度，最大速度，等等
            obstacle.max_speed = obstacle.obstacle_vel
        # 给障碍物设置一个随机的方向
        for obstacle in world.obstacles:
            obstacle.ra = (np.random.rand(1) * np.pi * 2.).item(0)  # initial obstacle direction
            obstacle.obstacle_depth = 0.

        #Initailize the landmark estimated positions
        world.landmarks_estimated = [Target(L_pos=world.landmarks[i].state.p_pos,V=[world.landmarks[i].landmark_vel*np.cos(world.landmarks[i].ra),world.landmarks[i].landmark_vel*np.sin(world.landmarks[i].ra)]) for i in range(world.num_landmarks)]
        # 定义障碍物估计位置
        world.obstacles_estimated = [Target(L_pos=world.obstacles[i].state.p_pos,V=[world.obstacles[i].obstacle_vel*np.cos(world.obstacles[i].ra),world.obstacles[i].obstacle_vel*np.sin(world.obstacles[i].ra)]) for i in range(world.num_obstacles)]
        #initialize the ocean current at random
        world.vel_ocean_current = np.random.rand(1).item(0)*self.max_vel_ocean_current #initial random strength
        world.angle_ocean_current = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
    
       
    def benchmark_data(self, agent, world):
        landmarks_real_p = []
        landmarks_predict_p = []
        obstacles_p = []
        entity_size = []
        for i in range(world.num_agents):
            entity_size.append(world.agents[i].size)
        for i in range(world.num_landmarks):
            landmarks_real_p.append(world.landmarks[i].state.p_pos)
            entity_size.append(world.landmarks[i].size)
        for i in range(world.num_landmarks):
            landmarks_predict_p.append(world.landmarks[i + world.num_landmarks].state.p_pos)
        for i in range(world.num_obstacles):
            obstacles_p.append(world.obstacles[i].state.p_pos)
            entity_size.append(world.obstacles[i].size)
        # return (rew, collisions, min_dists, occupied_landmarks,landmarks_real_p)
        return(self.pre_error,landmarks_real_p, self.agent_outofworld, self.landmark_collision, self.agent_collision,self.obstacle_collision,landmarks_predict_p,obstacles_p,entity_size) #后续要加入一个障碍物与智能体碰撞次数


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    done_state = False
    def reward(self, agent, world):
        global done_state
        done_state = False
        # Agents are rewarded based on landmarks_estimated covariance_vals, penalized for collisions
        rew = 0.
        
        for i,o in enumerate(world.obstacles):
            if i < world.num_obstacles and self.is_collision(o, agent):
                self.obstacle_collision += 1
                rew -= 10.
                done_state = True
                return rew
            elif i < world.num_obstacles:
                dist = np.sqrt(np.sum(np.square(world.obstacles[i].state.p_pos - agent.state.p_pos)))
                if dist < self.rew_dis_th:
                    rew += np.log(abs(dist - world.obstacles[i].size))
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 10.
                    self.agent_collision += 1
                    done_state = True
                    return rew
        for i,l in enumerate(world.landmarks):
            if i < world.num_landmarks and self.is_collision(l, agent) and agent.name[-1] != l.name[-1]:
                self.landmark_collision += 1
                rew -= 10.
                done_state = True
                return rew
            elif i < world.num_landmarks and agent.name[-1] == l.name[-1]:
                dist = np.sqrt((world.landmarks_estimated[i].lsxs[-1][0]-agent.state.p_pos[0])**2+(world.landmarks_estimated[i].lsxs[-1][2]-agent.state.p_pos[1])**2)
                if dist > 2 * self.set_max_range:
                    rew -= 100
                    done_state = True
                    self.agent_outofworld += 1
                    return rew
                elif dist < agent.size + l.size:
                    self.landmark_collision += 1
                    rew -= 10.
                    done_state = True
                    return rew
                # ===== 捕获+保持型稀疏奖励 =====
                # 1. 捕获阶段：首次进入安全跟踪范围的奖励
                elif not agent.has_captured and (agent.size + l.size) <= dist <= self.rew_dis_th:
                    rew += 10.  # 一次性高奖励，激励捕获目标
                    agent.has_captured = True
                    agent.consecutive_hold = 0  # 重置保持计数
                
                # 2. 保持阶段：持续停留在安全范围内的奖励
                elif agent.has_captured and (agent.size + l.size) <= dist <= self.rew_dis_th:
                    # 连续保持在范围内的时间越长，奖励越高 (阶梯式稀疏奖励)
                    agent.consecutive_hold += 1
                    
                    # 每达到一定时间步长给予额外奖励
                    if agent.consecutive_hold % 10 == 0:  # 每10步
                        rew += 10.  # 保持奖励
                
                # 3. 脱离观测范围惩罚
                elif agent.has_captured and dist > world.ob_range:
                    # 脱离范围的惩罚与脱离时间相关
                    rew -= 10
                    agent.has_captured = False  # 重置捕获状态
        for i,l in enumerate(world.landmarks_estimated): #计算估计地标位置与真实地标位置关系，给予奖惩
            if agent.name[-1] == world.landmarks[i].name[-1]:
                if self.pre_method == 'PF':
                    self.pre_error[i] = np.sqrt((l.pfxs[0]-world.landmarks[i].state.p_pos[0])**2+(l.pfxs[2]-world.landmarks[i].state.p_pos[1])**2) #Error from PF
                elif self.pre_method == 'LS' or self.pre_method == 'KM':
                    self.pre_error[i] = np.sqrt((l.lsxs[-1][0]-world.landmarks[i].state.p_pos[0])**2+(l.lsxs[-1][2]-world.landmarks[i].state.p_pos[1])**2) #Error from LS
                if self.pre_error[i]<self.rew_err_th:
                    rew += 1
        return rew

    def observation(self, agent, world):
        # 计算Agent的运行方向
        agent_angle = np.arctan2(agent.state.p_vel[1], agent.state.p_vel[0])
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_range = []
        entity_depth = []
        entity_range_estimated = []
        entity_error = []
        for i, entity in enumerate(world.landmarks):
            if i < world.num_landmarks and agent.name[-1] == entity.name[-1]: 
                #Update the landmarks_estiamted position using Particle Fileter
                #1:Compute radius between the agent and each landmark
                slant_range = np.sqrt(((entity.state.p_pos - agent.state.p_pos)[0])**2+((entity.state.p_pos - agent.state.p_pos)[1])**2)
                target_depth = entity.landmark_depth/1000. #normalize the target depth ###修改一下，把地标深度改为0，因为最小二乘法里面并没有对深度进行模拟  #  尝试完毕，效果不明显
                slant_range = np.sqrt(slant_range**2+target_depth**2) #add target depth to the range measurement
                # Add some systematic error in the measured range
                slant_range *= 1.01 # where 0.99 = 1% of sound speed difference = 1495 m/s
                # Add some noise in the measured range
                slant_range += np.random.uniform(-0.001, +0.001)
                # Return to a planar range
                slant_range = np.sqrt(abs(slant_range**2-target_depth**2))
                # 计算差值
                dx = entity.state.p_pos[0] - agent.state.p_pos[0]
                dy = entity.state.p_pos[1] - agent.state.p_pos[1]

                # 计算角度（弧度制）
                slant_angle = np.arctan2(dy, dx)
                #set a maximum range between target and agent where the measurement can not be conducted.
                if slant_range > self.set_max_range * 2 or np.random.rand() < self.range_dropping:
                    slant_range = -1.
                    new_range = False
                else:
                    new_range = True
                if self.pre_method == 'KM':
                    world.landmarks_estimated[i].updateEKF(dt=0.01, new_range=new_range, z=slant_range,
                                                                    L_info=[agent.state.p_pos[0],agent.state.p_pos[1]])
                elif self.pre_method == 'PF':
                #2:Update the PF
                    add_pos_error = False
                    if add_pos_error == True:
                        world.landmarks_estimated[i].updatePF(dt=30., new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0]+np.random.randn(1).item(0)*3/1000.,0.,agent.state.p_pos[1]+np.random.randn(1).item(0)*3/1000.,0.], update=new_range)
                    else:
                        world.landmarks_estimated[i].updatePF(dt=30., new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.], update=new_range)
                elif self.pre_method == 'LS':
                    #2b: Update the LS
                    add_pos_error = False
                    if add_pos_error == True:
                        world.landmarks_estimated[i].updateLS(dt=0.04, new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0]+np.random.randn(1).item(0)*3/1000.,0.,agent.state.p_pos[1]+np.random.randn(1).item(0)*3/1000.,0.])
                    else:
                        world.landmarks_estimated[i].updateLS(dt=0.01, new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.],L_pos = world.landmarks[i].state.p_pos) ##尝试更改步长，有一定效果但不明显
                else:
                    pass
                #3:Publish the new estimated position
                try:
                    if self.pre_method == 'PF':
                        world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].pfxs[0],world.landmarks_estimated[i].pfxs[2]] #Using PF
                    elif self.pre_method == 'LS' or self.pre_method == 'KM':
                        world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].lsxs[-1][0],world.landmarks_estimated[i].lsxs[-1][2]] #Using LS or KM #将LS或者KM计算的估计地表坐标录入  预测地标实体内
                except:
                    #An error will be produced if its the initial time and no good range measurement has been conducted yet. In this case, we supose that the target 
                    #is at the same position of the agent.
                    world.landmarks[i+world.num_landmarks].state.p_pos = world.landmarks[i].state.p_pos.copy() #原本是Agent位置，解释是说在初期没有进行良好的距离判断，所以无法得知地标位置，但暂时现在不考虑这个问题，假设地标探测一直良好，所以把agent改为对应的地标landmark(效果变好了一点)
                #Append the position of the landmark to generate the observation state
                #Using the true landmark position
                # entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                #Using the estimated landmark position  从此再无实际位置，实际位置有误差，完全依靠预测
                # 转换为局部坐标系
                local_pos = transform_to_local_coordinates(agent.state.p_pos, agent_angle, entity.state.p_pos)
                entity_pos.append(local_pos)
                # entity_pos.append(world.landmarks[i].state.p_pos - agent.state.p_pos) #将实际相对位置输入
                #Using the estimated landmark position but without delating the agent position. so it has a global position.
                # entity_pos.append(world.landmarks[i+world.num_landmarks].state.p_pos)
                entity_range.append(slant_range) #将测量距离输入
                entity_depth.append(target_depth)
                slant_range_estimated = np.sqrt(((world.landmarks[i + world.num_landmarks].state.p_pos - agent.state.p_pos)[0])**2+((world.landmarks[i + world.num_landmarks].state.p_pos - agent.state.p_pos)[1])**2)
                entity_range_estimated.append(slant_range_estimated)
                error = np.sqrt(((world.landmarks[i + world.num_landmarks].state.p_pos - world.landmarks[i].state.p_pos)[0])**2+((world.landmarks[i + world.num_landmarks].state.p_pos - world.landmarks[i].state.p_pos)[1])**2)
                entity_error.append(error)
                # Move the landmark if movable
                if entity.movable:
                    if self.movement == 'linear':
                        #linear movement
                        u_force = entity.landmark_vel
                        entity.ra += np.random.uniform(-0.1, +0.1)
                        entity.action.u = np.array([np.cos(entity.ra)*u_force,np.sin(entity.ra)*u_force])
                    
                    elif self.movement == 'random':
                        # random movement
                        entity.action.u = np.random.randn(2)/2.
                    
                    elif self.movement == 'levy':
                        #random walk Levy movement
                        beta = 1.9 #must be between 1 and 2
                        entity.action.u = random_levy(beta)
                        if entity.state.p_pos[0] > 0.8:
                            entity.action.u[0] = -abs(entity.action.u[0])
                        if entity.state.p_pos[0] < -0.8:
                            entity.action.u[0] = abs(entity.action.u[0])
                        if entity.state.p_pos[1] > 0.8:
                            entity.action.u[1] = -abs(entity.action.u[1])
                        if entity.state.p_pos[1] < -0.8:
                            entity.action.u[1] = abs(entity.action.u[1])
                    elif self.movement == "escape":
                        #增加逃逸的运动方式,采用逃离距离最近智能体的策略
                        dist_min = 1000
                        for agent in world.agents:
                            if np.sqrt(((entity.state.p_pos - agent.state.p_pos)[0])**2+((entity.state.p_pos - agent.state.p_pos)[1])**2) < dist_min:
                                dist_min = np.sqrt(((entity.state.p_pos - agent.state.p_pos)[0])**2+((entity.state.p_pos - agent.state.p_pos)[1])**2)
                                entity.ra = calculate_direction_angle(agent.state.p_pos, entity.state.p_pos)
                        u_force = entity.landmark_vel
                        entity.action.u = np.array([np.cos(entity.ra) * u_force, np.sin(entity.ra) * u_force])
        # all other entity
        entity_inside = []
        for entity_a in world.agents:
            if entity_a is agent: continue
            else:
                dist = np.sqrt(np.sum(np.square(entity_a.state.p_pos - agent.state.p_pos)))
                if dist < world.ob_range:
                    flag = True
                    for i_in, entity_in in enumerate(entity_inside):
                        if dist < np.sqrt(np.sum(np.square(entity_in.state.p_pos - agent.state.p_pos))):
                            entity_inside.insert( i_in ,entity_a)
                            flag = False
                            break
                    if flag:
                        entity_inside.append(entity_a)
        for i,entity_l in enumerate(world.landmarks):
            if i < world.num_landmarks and agent.name[-1] == entity_l.name[-1]:continue
            else:
                dist = np.sqrt(np.sum(np.square(entity_l.state.p_pos - agent.state.p_pos)))
                if dist < world.ob_range:
                    flag = True
                    for i_in, entity_in in enumerate(entity_inside):
                        if dist < np.sqrt(np.sum(np.square(entity_in.state.p_pos - agent.state.p_pos))):
                            entity_inside.insert( i_in ,entity_l)
                            flag = False
                            break
                    if flag:
                        entity_inside.append(entity_l)
        for i,entity_o in enumerate(world.obstacles):
            if i < world.num_obstacles:
                dist = np.sqrt(np.sum(np.square(entity_o.state.p_pos - agent.state.p_pos)))
                if dist < world.ob_range:
                    flag = True
                    for i_in, entity_in in enumerate(entity_inside):
                        if dist < np.sqrt(np.sum(np.square(entity_in.state.p_pos - agent.state.p_pos))):
                            entity_inside.insert( i_in ,entity_o)
                            flag = False
                            break
                    if flag:
                        entity_inside.append(entity_o)
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                # Move the landmark if movable
                if obstacle.movable:
                    if self.obstacle_movement == 'linear':
                        # linear movement
                        u_force = obstacle.obstacle_vel
                        obstacle.ra += np.random.uniform(-0.1, +0.1)
                        obstacle.action.u = np.array([np.cos(obstacle.ra)*u_force,np.sin(obstacle.ra)*u_force])

                    elif self.obstacle_movement == 'random':
                        # random movement
                        obstacle.action.u = np.random.randn(2) / 2.
                    elif self.obstacle_movement == 'levy':
                        # random walk Levy movement
                        beta = 1.9  # must be between 1 and 2
                        obstacle.action.u = random_levy(beta)
                        if obstacle.state.p_pos[0] > 0.8:
                            obstacle.action.u[0] = -abs(obstacle.action.u[0])
                        if obstacle.state.p_pos[0] < -0.8:
                            obstacle.action.u[0] = abs(obstacle.action.u[0])
                        if obstacle.state.p_pos[1] > 0.8:
                            obstacle.action.u[1] = -abs(obstacle.action.u[1])
                        if obstacle.state.p_pos[1] < -0.8:
                            obstacle.action.u[1] = abs(obstacle.action.u[1])
        obs = np.concatenate( entity_pos + [entity_range])
        count_now = 0
        for en in entity_inside:
            if count_now < world.num_ob:
                local_pos = transform_to_local_coordinates(agent.state.p_pos,agent_angle, en.state.p_pos)
                dist = np.sqrt(np.sum(np.square(local_pos)))
                obs = np.concatenate([
                                        obs,
                                        local_pos,  # 位置信息
                                        np.array([dist]),  # 距离
                                        np.array([en.size])  # 大小
                                    ])
                count_now += 1
        # use_count = count_now
        while count_now < world.num_ob:
            obs = np.concatenate([
                                        obs,
                                        np.array([0, 0]),
                                        np.array([0]),
                                        np.array([0])
                                    ])
            count_now += 1
        obs = np.concatenate([obs  , np.array([slant_range_estimated]), np.array([error])])
        return obs
    def smooth_data(self, data):
        """
        简单的移动平均平滑处理
        """
        window_size = 3  # 窗口大小
        smoothed_data = []
        for i in range(len(data)):
            if i < window_size:
                smoothed_data.append(data[i])
            else:
                smoothed_data.append(np.mean(data[i - window_size:i], axis=0))
        return smoothed_data
    def done(self, agent, world):
        # episodes are done based on the agents minimum distance from a landmark.
        global done_state
        if done_state:
            done = True
        else:
            done = False
        return done
