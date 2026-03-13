import copy

import gymnasium as gym
import gymnasium.spaces.utils
from gymnasium.vector.utils import batch_space

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers.record import RecordEpisode

class RecordPianoEpisode(RecordEpisode):

	def step(self, action):
		if self.save_video and self._video_steps == 0:
			self.render_images.append(self.capture_image())
		obs, rew, terminated, truncated, info = super(RecordEpisode, self).step(action)

		if self.save_trajectory:
			state_dict = self.base_env.get_state_dict()
			if self.record_env_state:
			    self._trajectory_buffer.state = common.append_dict_array(
			        self._trajectory_buffer.state,
			        common.to_numpy(common.batch(state_dict)),
			    )
			self._trajectory_buffer.observation = common.append_dict_array(
			    self._trajectory_buffer.observation,
			    common.to_numpy(common.batch(obs)),
			)

			self._trajectory_buffer.action = common.append_dict_array(
			    self._trajectory_buffer.action,
			    common.to_numpy(common.batch(action)),
			)
			if self.record_reward:
			    self._trajectory_buffer.reward = common.append_dict_array(
			        self._trajectory_buffer.reward,
			        common.to_numpy(common.batch(rew)),
			    )
			self._trajectory_buffer.terminated = common.append_dict_array(
			    self._trajectory_buffer.terminated,
			    common.to_numpy(common.batch(terminated)),
			)
			self._trajectory_buffer.truncated = common.append_dict_array(
			    self._trajectory_buffer.truncated,
			    common.to_numpy(common.batch(truncated)),
			)
			done = terminated | truncated
			self._trajectory_buffer.done = common.append_dict_array(
			    self._trajectory_buffer.done,
			    common.to_numpy(common.batch(done)),
			)
			if "success" in info:
			    self._trajectory_buffer.success = common.append_dict_array(
			        self._trajectory_buffer.success,
			        common.to_numpy(common.batch(info["success"])),
			    )
			else:
			    self._trajectory_buffer.success = None
			if "fail" in info:
			    self._trajectory_buffer.fail = common.append_dict_array(
			        self._trajectory_buffer.fail,
			        common.to_numpy(common.batch(info["fail"])),
			    )
			else:
			    self._trajectory_buffer.fail = None

		if self.save_video:
			self._video_steps += 1
			if self.info_on_video:
				scalar_info = dict()
				scalar_info["reward"] = [float(rew) for rew in info["reward"]]
				for k in info["reward_log"]:
					scalar_info[k] = [float(v) for v in info["reward_log"][k]]
				image = self.capture_image(scalar_info)
			else:
				image = self.capture_image()

			self.render_images.append(image)
			if (
				self.max_steps_per_video is not None
				and self._video_steps >= self.max_steps_per_video
			):
			    self.flush_video()
		self._elapsed_record_steps += 1
		return obs, rew, terminated, truncated, info

class FlattenPianoActionSpaceWrapper(gym.ActionWrapper):
    """
    Flattens the action space. The original action space must be spaces.Dict
    """

    def __init__(self, env, control_mode=None, use_3d_finger_actions=False) -> None:
        super().__init__(env)
        self._orig_single_action_space = copy.deepcopy(
            self.base_env.single_action_space
        )
        self.single_action_space = gymnasium.spaces.utils.flatten_space(
            self.base_env.single_action_space
        )
        if self.base_env.num_envs > 1:
            self.action_space = batch_space(
                self.single_action_space, n=self.base_env.num_envs
            )
        else:
            self.action_space = self.single_action_space

        if "ee_script" in control_mode:
            self.ee_script = True
        else:
            self.ee_script = False
        self.use_3d_finger_actions = use_3d_finger_actions

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def action(self, action):
        if (
            self.base_env.num_envs == 1
            and action.shape == self.single_action_space.shape
        ):
            action = common.batch(action)

        # TODO (stao): This code only supports flat dictionary at the moment
        unflattened_action = dict()
        start, end = 0, 0
        for k, space in self._orig_single_action_space.items():
            end += space.shape[0]
            if self.ee_script:
                end -= 6
            if self.use_3d_finger_actions:
                end -= 4
            unflattened_action[k] = action[:, start:end]
            start += space.shape[0] 
            if self.ee_script:
                start -= 6
            if self.use_3d_finger_actions:
                start -= 4
        return unflattened_action