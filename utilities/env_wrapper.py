"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe, set_start_method

try:
    from baselines.common.vec_env import VecEnv, CloudpickleWrapper
    from baselines.common.tile_images import tile_images
except ImportError:
    try:
        import cloudpickle
    except ImportError:
        cloudpickle = None
    import pickle

    class VecEnv(object):
        """Minimal local replacement for the OpenAI Baselines VecEnv API."""

        def __init__(self, num_envs, observation_space, action_space):
            self.num_envs = num_envs
            self.observation_space = observation_space
            self.action_space = action_space

        def step(self, actions):
            self.step_async(actions)
            return self.step_wait()

        def step_async(self, actions):
            raise NotImplementedError

        def step_wait(self):
            raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def reset_done(self, done_mask):
        """Reset only workers whose current episode has terminated."""
        raise NotImplementedError

        def close(self):
            pass

    class CloudpickleWrapper(object):
        """Pickle wrapper for multiprocessing environment factory closures."""

        def __init__(self, x):
            self.x = x

        def __getstate__(self):
            if cloudpickle is not None:
                return cloudpickle.dumps(self.x)
            return pickle.dumps(self.x)

        def __setstate__(self, ob):
            self.x = pickle.loads(ob)

    def tile_images(img_nhwc):
        """Tile N images into one image, matching Baselines' helper behavior."""
        img_nhwc = np.asarray(img_nhwc)
        if img_nhwc.ndim < 4:
            return img_nhwc
        n_images, height, width, channels = img_nhwc.shape
        grid_h = int(np.ceil(np.sqrt(n_images)))
        grid_w = int(np.ceil(float(n_images) / grid_h))
        padded = np.zeros((grid_h * grid_w, height, width, channels), dtype=img_nhwc.dtype)
        padded[:n_images] = img_nhwc
        tiled = padded.reshape(grid_h, grid_w, height, width, channels)
        tiled = tiled.transpose(0, 2, 1, 3, 4)
        return tiled.reshape(grid_h * height, grid_w * width, channels)


def _unpack_worker_result(result, expected_items=4):
    """兼容旧 worker 的四元组和带 timing 的结果。"""
    if len(result) == expected_items + 1:
        return result[:-1], result[-1]
    return result, {}


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # ob, ob_full, reward, done, info = env.step(data)
            ob, reward, done, info = env.step(data)
            if all(done):
                # ob = env.reset()
                pass
            # remote.send((ob, ob_full, reward, done, info))
            remote.send((ob, reward, done, info, getattr(env, 'last_step_timing', {})))
        elif cmd == 'reset':
            # ob, ob_full = env.reset()
            # remote.send((ob, ob_full))
            ob = env.reset()
            remote.send((ob, getattr(env, 'last_reset_timing', {})))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        # elif cmd == 'A_Star_Info':#返回的信息包括（  智能体数量，障碍物数量，目标数量，智能体基础信息，障碍物基础信息，目标基础信息（位置，大小），智能体朝向角集合 ） !!!需要拿观测值作为障碍物和目标的基础信息，
        #     agents_info = []
        #     for agent in env.world.agents:
        #         agents_info.append([agent.state.p_pos[0],agent.state.p_pos[1],agent.size])
        #     obstacles_info = []
        #     for i in env.world.num_obstacles:
        #         obstacles_info.append([env.world.obstacles[i+env.world.num_obstacles].state.p_pos[0],env.world.obstacles[i+env.world.num_obstacles].state.p_pos[1],env.world.obstacles[i+env.world.num_obstacles].size])
        #     landmarks_info = []
        #     for i in env.world.num_landmarks:
        #         landmarks_info.append([env.world.landmarks[i+env.world.num_landmarks].state.p_pos[0],env.world.landmarks[i+env.world.num_landmarks].state.p_pos[1],env.world.landmarks[i+env.world.num_landmarks].size])
        #     A_Star_Info = np.concatenate([env.world.num_agents], [env.world.num_obstacles], [env.world.num_landmarks],agents_info,obstacles_info,landmarks_info,env.world.angle)
        #     remote.send(A_Star_Info)
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self._waiting_command = None
        self.closed = False
        self.last_step_timings = []
        self.last_reset_timings = []
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        #print('env_wrapper: Process')
        #he possible start methods are 'fork', 'spawn' and 'forkserver'
        # set_start_method('spawn')
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            #print('env_wrapper: p.daemon')    
            p.daemon = True # if the main process crashes, we should not cause things to hang
            #print('env_wrapper: p.start')
            p.start()
        #print('env_wrapper: remote.close')
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        #print('env_wrapper: VecEnv.__init__')
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        self._waiting_command = 'step'

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        self._waiting_command = None
        unpacked = [_unpack_worker_result(result) for result in results]
        results = [item[0] for item in unpacked]
        self.last_step_timings = [item[1] for item in unpacked]
        # obs, obs_full, rews, dones, infos = zip(*results)
        # return np.stack(obs), np.stack(obs_full), np.stack(rews), np.stack(dones), infos
        obs, rews, dones, infos = zip(*results)
        if infos and all(isinstance(item, np.ndarray) for item in infos):
            try:
                infos = np.stack(infos)
            except ValueError:
                pass
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset_async(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        self.waiting = True
        self._waiting_command = 'reset'

    def reset_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        self._waiting_command = None
        unpacked = [_unpack_worker_result(result, expected_items=1) for result in results]
        observations = []
        self.last_reset_timings = []
        for payload, timing in unpacked:
            observations.append(payload[0] if isinstance(payload, tuple) else payload)
            self.last_reset_timings.append(timing)
        return np.stack(observations)

    def reset(self):
        self.reset_async()
        return self.reset_wait()

    def reset_done(self, done_mask):
        """Synchronously reset a selected subset of subprocess environments.

        The returned observations belong only to the selected workers.  Keeping
        unfinished workers alive is necessary for valid parallel RL rollouts:
        one early collision must not truncate the other trajectories.
        """
        mask = np.asarray(done_mask, dtype=bool).reshape(-1)
        if mask.size != self.num_envs:
            raise ValueError(
                'done_mask length %d does not match num_envs %d.'
                % (mask.size, self.num_envs)
            )
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return indices, np.empty((0,), dtype=float)

        for index in indices:
            self.remotes[int(index)].send(('reset', None))

        observations = []
        timings = []
        for index in indices:
            result = self.remotes[int(index)].recv()
            payload, timing = _unpack_worker_result(result, expected_items=1)
            observations.append(payload[0] if isinstance(payload, tuple) else payload)
            timings.append(timing)

        self.last_reset_timings = timings
        return indices, np.stack(observations)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
            self.waiting = False
            self._waiting_command = None
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        # code doesn't work all that well
        # TODO: need to clean up
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError
    #
    # def A_star_Info(self):
    #     for remote in self.remotes:
    #         remote.send(('A_star_Info', None))
    #     return np.stack([remote.recv() for remote in self.remotes])


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None
        self.last_step_timings = []
        self.last_reset_timings = []

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = zip(*results)
        self.last_step_timings = [
            getattr(env, 'last_step_timing', {}) for env in self.envs
        ]
        self.ts += 1
        self.actions = None
        if infos and all(isinstance(item, np.ndarray) for item in infos):
            try:
                infos = np.stack(infos)
            except ValueError:
                pass
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        self.reset_async()
        return self.reset_wait()

    def reset_done(self, done_mask):
        """Reset only completed local environments, matching SubprocVecEnv."""
        mask = np.asarray(done_mask, dtype=bool).reshape(-1)
        if mask.size != self.num_envs:
            raise ValueError(
                'done_mask length %d does not match num_envs %d.'
                % (mask.size, self.num_envs)
            )
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return indices, np.empty((0,), dtype=float)

        observations = []
        timings = []
        for index in indices:
            env = self.envs[int(index)]
            observations.append(env.reset())
            timings.append(getattr(env, 'last_reset_timing', {}))
        self.last_reset_timings = timings
        return indices, np.asarray(observations)

    def reset_async(self):
        # DummyVecEnv 没有 worker，接口保持一致并立即执行 reset。
        self._pending_reset = [env.reset() for env in self.envs]
        self.last_reset_timings = [
            getattr(env, 'last_reset_timing', {}) for env in self.envs
        ]

    def reset_wait(self):
        return np.array(self._pending_reset)

    def close(self):
        return
