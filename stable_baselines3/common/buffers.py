import warnings
from typing import Generator, Optional, Union

import numpy as np
import torch as th
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape, get_obs_dtype
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class BaseBuffer(object):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[th.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.obs_dtype = get_obs_dtype(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: Union[np.ndarray, dict, tuple]) -> Union[np.ndarray, dict, tuple]:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr: (np.ndarray)
        :return: (np.ndarray)
        """

        if isinstance(arr, dict):
            return {k: BaseBuffer.swap_and_flatten(a) for k, a in arr.items()}
        if isinstance(arr, tuple):
            return tuple(BaseBuffer.swap_and_flatten(a) for a in arr)
        elif isinstance(arr, np.ndarray):
            shape = arr.shape
            if len(shape) < 3:
                shape = shape + (1,)
            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):
        """
        :param batch_inds: (th.Tensor)
        :param env: (Optional[VecNormalize])
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        raise NotImplementedError()

    def to_torch(self, array: Union[np.ndarray, dict, tuple], copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array: (np.ndarray)
        :param copy: (bool) Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return: (th.Tensor)
        """
        if isinstance(array, dict):
            return {k: self.to_torch(arr, copy=copy) for k, arr in array.items()}
        elif isinstance(array, tuple):
            return tuple(self.to_torch(arr, copy=copy) for arr in array)
        elif isinstance(array, np.ndarray):
            if copy:
                return th.tensor(array).to(self.device)
            return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(obs: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_obs(obs).astype(np.float32)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param n_envs: (int) Number of parallel environments
    :param optimize_memory_usage: (bool) Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        if isinstance(self.observation_space, spaces.Dict):
            self.observations = {k: np.zeros((self.buffer_size, self.n_envs,) + obs_shape, dtype=self.obs_dtype[k])
                                 for k, obs_shape in self.obs_shape.items()}
            if optimize_memory_usage:
                self.next_observations = None
            else:
                self.next_observations = \
                    {k: np.zeros((self.buffer_size, self.n_envs,) + obs_shape, dtype=self.obs_dtype[k])
                     for k, obs_shape in self.obs_shape.items()}

            obs_bytes = sum([v.nbytes for k, v in self.observations.items()])
        elif isinstance(self.observation_space, spaces.Tuple):
            self.observations = tuple(np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape[i],
                                               dtype=self.obs_dtype[i]) for i in range(len(self.obs_shape)))
            if optimize_memory_usage:
                self.next_observations = None
            else:
                self.next_observations = tuple(np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape[i],
                                               dtype=self.obs_dtype[i]) for i in range(len(self.obs_shape)))

            obs_bytes = sum([v.nbytes for v in self.observations])
        else:
            self.observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape,
                                         dtype=observation_space.dtype)
            if optimize_memory_usage:
                self.next_observations = None
            else:
                self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape,
                                                  dtype=observation_space.dtype)

            obs_bytes = self.observations.nbytes

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = obs_bytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                total_memory_usage += obs_bytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Union[np.ndarray, dict, tuple],
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray
    ) -> None:

        if isinstance(self.observation_space, spaces.Dict):
            for k, obs_array in obs.items():
                self.observations[k][self.pos] = np.array(obs_array).copy()
            if self.optimize_memory_usage:
                for k, obs_array in next_obs.items():
                    self.observations[k][(self.pos + 1) % self.buffer_size] = np.array(obs_array).copy()
            else:
                for k, obs_array in next_obs.items():
                    self.next_observations[k][self.pos] = np.array(obs_array).copy()
        elif isinstance(self.observation_space, spaces.Tuple):
            for i, obs_array in enumerate(obs):
                self.observations[i][self.pos] = np.array(obs_array).copy()
            if self.optimize_memory_usage:
                for i, obs_array in enumerate(next_obs):
                    self.observations[i][(self.pos + 1) % self.buffer_size] = np.array(obs_array).copy()
            else:
                for i, obs_array in enumerate(next_obs):
                    self.next_observations[i][self.pos] = np.array(obs_array).copy()
        else:
            self.observations[self.pos] = np.array(obs).copy()
            if self.optimize_memory_usage:
                self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
            else:
                self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if isinstance(self.observation_space, spaces.Dict):
            observation_samples = {k: self._normalize_obs(obs[batch_inds, 0], env)
                                   for k, obs in self.observations.items()}
            if self.optimize_memory_usage:
                next_observation_samples = {k: self._normalize_obs(obs[(batch_inds + 1) % self.buffer_size, 0], env)
                                            for k, obs in self.observations.items()}
            else:
                next_observation_samples = {k: self._normalize_obs(next_obs[batch_inds, 0], env)
                                            for k, next_obs in self.next_observations.items()}
        elif isinstance(self.observation_space, spaces.Tuple):
            observation_samples = (self._normalize_obs(obs[batch_inds, 0], env)
                                   for obs in self.observations)
            if self.optimize_memory_usage:
                next_observation_samples = (self._normalize_obs(obs[(batch_inds + 1) % self.buffer_size, 0], env)
                                            for obs in self.observations)
            else:
                next_observation_samples = (self._normalize_obs(next_obs[batch_inds, 0], env)
                                            for next_obs in self.next_observations)
        else:
            observation_samples = self._normalize_obs(self.observations[batch_inds, 0, :], env),
            if self.optimize_memory_usage:
                next_observation_samples = self._normalize_obs(self.observations[(batch_inds + 1) %
                                                                                 self.buffer_size, 0, :], env)
            else:
                next_observation_samples = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            observation_samples,
            self.actions[batch_inds, 0, :],
            next_observation_samples,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        if isinstance(self.observation_space, spaces.Dict):
            self.observations = {k: np.zeros((self.buffer_size, self.n_envs,) + obs_shape, dtype=np.float32)
                                 for k, obs_shape in self.obs_shape.items()}
        elif isinstance(self.observation_space, spaces.Tuple):
            self.observations = tuple(np.zeros((self.buffer_size, self.n_envs,) + obs_shape, dtype=np.float32)
                                      for obs_shape in self.obs_shape)
        else:
            self.observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_value: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, value: th.Tensor, log_prob: th.Tensor
    ) -> None:
        """
        :param obs: (np.ndarray) Observation
        :param action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Dict):
            for k, obs_array in obs.items():
                self.observations[k][self.pos] = np.array(obs_array).copy()
        elif isinstance(self.observation_space, spaces.Tuple):
            for i, obs_array in enumerate(obs):
                self.observations[i][self.pos] = np.array(obs_array).copy()
        else:
            self.observations[self.pos] = np.array(obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ["observations", "actions", "values", "log_probs", "advantages", "returns"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        if isinstance(self.observation_space, spaces.Dict):
            observation_samples = {k: obs[batch_inds] for k, obs in self.observations.items()}
        elif isinstance(self.observation_space, spaces.Tuple):
            observation_samples = (obs[batch_inds] for obs in self.observations)
        else:
            observation_samples = self.observations[batch_inds]

        data = (observation_samples,
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten())
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
