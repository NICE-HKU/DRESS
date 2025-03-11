"""
Some utility functions.
"""

import gymnasium as gym
import numpy as np

import time
import math

# from envs.MyFetchRobot import push, reach
# from envs.MyMujoco import ant_v4, humanoid_v4, humanoidstandup_v4, walker2d_v4


def classic_control_env_maker(env_id, seed=1, render=False):
    """
    Make the environment.
    :param env_id: the name of the environment
    :param seed: the random seed
    :param render: whether to render the environment
    :return: the environment
    """
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def robotics_env_maker(env_id, seed=1, render=False):
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # + transform the reward from {-1, 0} to {0, 1}
    env = gym.wrappers.TransformReward(env, lambda reward: reward + 1.0)
    # + flatten the dict observation space to a vector
    env = gym.wrappers.TransformObservation(env, lambda obs: np.concatenate(
        [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]))

    new_obs_length = env.observation_space['observation'].shape[0] + env.observation_space['achieved_goal'].shape[0] + \
                     env.observation_space['desired_goal'].shape[0]

    # redefine the observation of the environment, make it the same size of the flattened dict observation space
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_length,),
                                           dtype=np.float32)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env



def print_banner(s, separator="-", num_star=60):
	print(separator * num_star, flush=True)
	print(s, flush=True)
	print(separator * num_star, flush=True)


class Progress:
	def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()

	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		print('\n', end='')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
				(key, val)
				for key, val in params.items()
			])

		############
		# Position #
		############
		self._clear()

		###########
		# Percent #
		###########
		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		#########
		# Speed #
		#########
		speed = self._format_speed(self._step)

		##########
		# Params #
		##########
		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines

		description = '{} | {}{}'.format(percent, speed, params_string)
		print(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		print(position, end='')
		print(empty)
		print(position, end='')

	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i + n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0, '')
		padding = '\n' + ' ' * self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param):
		k, v = param
		return '{} : {}'.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			self._clear()
			print(string, end='\n')
			self._skip_lines = 1
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()


class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None


class EarlyStopping(object):
	def __init__(self, tolerance=5, min_delta=0):
		self.tolerance = tolerance
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False

	def __call__(self, train_loss, validation_loss):
		if (validation_loss - train_loss) > self.min_delta:
			self.counter += 1
			if self.counter >= self.tolerance:
				return True
		else:
			self.counter = 0
		return False
