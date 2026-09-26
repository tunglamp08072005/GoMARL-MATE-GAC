import sys
import numpy as np
import mate
from mate.agents import GreedyTargetAgent
from mate.wrappers import DiscreteCamera, MultiCamera
from envs.multiagentenv import MultiAgentEnv
import copy

def normalize_state(s, num_cameras, num_targets, num_obstacles):
    state = copy.deepcopy(s)
    for c in range(num_cameras):
        start_idx = 9 * c
        state[start_idx:start_idx + 3] = state[start_idx:start_idx + 3] / 1000.0
        state[start_idx + 3:start_idx + 5] = state[start_idx + 3:start_idx + 5] / state[start_idx + 6]
        state[start_idx + 5] = state[start_idx + 5] / 180.0
        state[start_idx + 6] = 1.0
        state[start_idx + 7:start_idx + 9] = state[start_idx + 7:start_idx + 9] / 180.0

    for t in range(num_targets):
        start_idx = 9 * num_cameras + 14 * t
        state[start_idx:start_idx + 3] = state[start_idx:start_idx + 3] / 1000.0
        state[start_idx + 4] = state[start_idx + 4] / 1000.0

    for o in range(num_obstacles):
        start_idx = 9 * num_cameras + 14 * num_targets + 3 * o
        state[start_idx:start_idx + 3] = state[start_idx:start_idx + 3] / 1000.0

    return state


def normalize_obs(obs):
    state = copy.deepcopy(obs)
    n_c = int(state[0][0])
    n_t = int(state[0][1])
    n_o = int(state[0][2])

    for camera_index in range(n_c):
        val = int(state[camera_index][3])
        mapping = {
            0: [0, 0, 0, 1],
            1: [0, 0, 1, 0],
            2: [0, 1, 0, 0],
            3: [1, 0, 0, 0]
        }

        original_radius = state[camera_index][19]
        encoded_bits = mapping.get(val, [0, 0, 0, 0])

        state[camera_index][0:4] = np.array(encoded_bits, dtype=float)
        state[camera_index][4:16] = state[camera_index][4:16] / 1000.0
        state[camera_index][16:18] = state[camera_index][16:18] / state[camera_index][19]
        state[camera_index][18] = state[camera_index][18] / 180.0
        state[camera_index][19] = 1.0
        state[camera_index][20:22] = state[camera_index][20:22] / 180.0

        for target_index in range(n_t):
            start = 22 + 5 * target_index
            state[camera_index][start:start + 3] = state[camera_index][start:start + 3] / 1000.0

        for obstacle_index in range(n_o):
            start = 22 + 5 * n_t + 4 * obstacle_index
            state[camera_index][start:start + 3] = state[camera_index][start:start + 3] / 1000.0

        for tm_index in range(n_c):
            start = 22 + 5 * n_t + 4 * n_o + 7 * tm_index
            state[camera_index][start:start + 3] = state[camera_index][start:start + 3] / 1000.0
            state[camera_index][start + 3:start + 5] = state[camera_index][start + 3:start + 5] / original_radius
            state[camera_index][start + 5] = state[camera_index][start + 5] / 180.0

    return state


class MATEEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        super().__init__()

        levels = kwargs.get("levels", 9)
        self.episode_limit = kwargs.get("episode_limit", 900)
        config_path = kwargs.get("config", None)

        self.levels = levels

        base_env = mate.make_environment(config=config_path)
        base_env = MultiCamera(
            base_env,
            target_agent=GreedyTargetAgent()
        )
        self.env = base_env

        self.n_agents = self.env.unwrapped.num_cameras
        self.n_targets = self.env.unwrapped.num_targets
        self.n_obstacles = 9

        # discrete actions = levels^2
        self.n_actions = self.levels * self.levels

        # dùng grid của wrapper DiscreteCamera để tự convert discrete -> continuous
        self.normalized_action_grid = DiscreteCamera.discrete_action_grid(levels=self.levels)

        # biên độ action continuous của camera: [rotation_step, zooming_step]
        self.action_high = np.asarray(
            [
                self.env.unwrapped.camera_rotation_step,
                self.env.unwrapped.camera_zooming_step,
            ],
            dtype=np.float32,
        )

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.last_obs = obs
        self.last_reward = 0.0
        self.t = 0

        # Theo dõi coverage trong 1 episode để cuối episode log mean coverage
        self.episode_coverage_sum = 0.0
        self.episode_coverage_count = 0

        flat_obs = [o.flatten() for o in obs]
        self.obs_shape = max(o.shape[0] for o in flat_obs)
        self.state_shape = self.get_state().shape[0]

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.last_obs = obs
        self.t = 0
        self.last_reward = 0.0

        self.episode_coverage_sum = 0.0
        self.episode_coverage_count = 0

        return normalize_obs(obs=self.last_obs)

    def step(self, actions):
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()

        # actions từ GoMARL: shape (n_agents,), mỗi phần tử là 1 discrete action
        actions = np.asarray(actions).astype(int).ravel()

        # map discrete -> continuous, ra shape (n_agents, 2)
        camera_joint_action_continuous = (
            self.action_high * self.normalized_action_grid[actions]
        ).astype(np.float32)

        result = self.env.step(camera_joint_action_continuous)

        # gymnasium-style: obs, reward, terminated, truncated, info
        if len(result) == 5:
            obs, reward, terminated, truncated, raw_info = result
            done_env = bool(terminated or truncated)
        else:
            obs, reward, terminated, raw_info = result
            done_env = bool(terminated)

        if isinstance(obs, tuple):
            obs = obs[0]

        step_coverage = float(raw_info[0]["coverage_rate"])

        self.episode_coverage_sum += step_coverage
        self.episode_coverage_count += 1

        self.last_obs = obs
        self.t += 1

        done = done_env
        if self.t >= self.episode_limit:
            done = True

        info = {
            "episode_limit": self.t >= self.episode_limit,
            "step_coverage_rate": step_coverage,
        }

        if done:
            info["episode_coverage_rate"] = (
                self.episode_coverage_sum / max(1, self.episode_coverage_count)
            )

        self.last_reward = step_coverage

        # giữ reward train như cũ
        return (step_coverage * 10), done, info

    def get_obs(self):
        obs = normalize_obs(self.last_obs)
        return np.array(obs, dtype=np.float32)

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.obs_shape

    def get_state(self):
        obs = normalize_obs(self.last_obs)
        state = np.array(obs, dtype=np.float32).flatten()
        return state

    def get_state_size(self):
        return self.state_shape

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions), dtype=np.int32)

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.n_actions, dtype=np.int32)

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        return {
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_stats(self):
        return {}

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    def render(self):
        if hasattr(self.env, "render"):
            self.env.render()


if __name__ == "__main__":
    print("===== TEST MATE ENV =====")

    env = MATEEnv(levels=9)

    env_info = env.get_env_info()
    print("\nENV INFO:", env_info)

    obs = env.reset()
    print("\nReset done")
    print("obs shape:", obs.shape)

    state = env.get_state()
    print("state shape:", state.shape)

    print("\nRunning random policy...\n")

    done = False
    step = 0

    while not done:
        actions = []
        for agent_id in range(env.n_agents):
            action = np.random.randint(env.n_actions)
            actions.append(action)

        reward, done, info = env.step(actions)

        obs = env.get_obs()
        state = env.get_state()

        print("------------------------------------------------")
        print(
            f"step {step:03d} | reward {reward:.4f} | "
            f"obs {obs.shape} | state {state.shape} | info {info}"
        )

        step += 1

    print("\nEpisode finished")
    env.close()