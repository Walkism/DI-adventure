"""
wrapper定义文件
"""
from typing import Union, List, Tuple, Callable
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv
import gym
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
import torch
from ding.torch_utils import to_ndarray
import os
import warnings
import copy


# 粘性动作wrapper
class StickyActionWrapper(gym.ActionWrapper):
    """
    Overview:
       A certain possibility to select the last action
    Interface:
        ``__init__``, ``action``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``p_sticky``: possibility to select the last action
    """

    def __init__(self, env: gym.Env, p_sticky: float=0.25):
        super().__init__(env)
        self.p_sticky = p_sticky
        self.last_action = 0

    def action(self, action):
        if np.random.random() < self.p_sticky:
            return_action = self.last_action
        else:
            return_action = action
        self.last_action = action
        return return_action


# 稀疏奖励wrapper
class SparseRewardWrapper(gym.Wrapper):
    """
    Overview:
       Only death and pass sparse reward
    Interface:
        ``__init__``, ``step``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        dead = True if reward == -15 else False
        reward = 0
        if info['flag_get']:
            reward = 15
        if dead:
            reward = -15
        return obs, reward, done, info


# 硬币奖励wrapper
class CoinRewardWrapper(gym.Wrapper):
    """
    Overview:
        add coin reward
    Interface:
        ``__init__``, ``step``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_coins = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += (info['coins'] - self.num_coins) * 10
        self.num_coins = info['coins']
        return obs, reward, done, info

# 修复版硬币奖励wrapper
class CustomCoinRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, coin_reward_scale: float = 200.0):
        super().__init__(env)
        self.num_coins = 0
        self.coin_reward_scale = coin_reward_scale

    def reset(self, **kwargs):
        # 【关键修复】每次重置环境时，必须清零金币计数
        self.num_coins = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 计算新增金币
        # 注意：info['coins'] 是当前累积金币数
        coin_diff = info['coins'] - self.num_coins
        
        if coin_diff > 0:
            # 【关键策略】大幅提升奖励。
            # 原始移动奖励一帧大概在0-2之间，stack4帧约10分左右。
            # 给 10 分太少，会被移动奖励淹没。
            # 建议给 100-200，强行告诉 AI “金币比命重要”。
            reward += coin_diff * self.coin_reward_scale
            # print(f"Agent ate a coin! Reward boost! Total coins: {info['coins']}") # 调试用
            
        self.num_coins = info['coins']
        return obs, reward, done, info


class GrowthRewardWrapper(gym.Wrapper):
    """
    只负责奖励“变大” (Small -> Tall/Fireball)
    """
    def __init__(self, env: gym.Env, growth_reward=500.0):
        super().__init__(env)
        self.growth_reward = growth_reward
        self.last_status = 'small'

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_status = 'small'
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 获取当前状态
        curr_status = info.get('status', 'small')

        # 逻辑：如果从小变大（或变火球），给予奖励
        if self.last_status == 'small' and curr_status in ['tall', 'fireball']:
            reward += self.growth_reward
            # print(f"wrapper: Growth! Reward +{self.growth_reward}")

        self.last_status = curr_status
        return obs, reward, done, info


class AttackRewardWrapper(gym.Wrapper):
    """
    只负责奖励“战斗/破坏” (踩怪、顶砖)。
    关键逻辑：从分数增量中剔除 [金币分数] 和 [吃蘑菇分数]。
    同时在 info 中记录当前 Episode 累计获得的战斗分数。
    """
    def __init__(self, env: gym.Env, attack_reward_scale=1.0):
        super().__init__(env)
        self.attack_reward_scale = attack_reward_scale
        # 状态记录
        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'
        # 【新增】用于累计一局内的战斗得分
        self.total_attack_score = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # 重置所有计数器
        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'
        self.total_attack_score = 0 # 【新增】清零
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        curr_score = info.get('score', 0)
        curr_coins = info.get('coins', 0)
        curr_status = info.get('status', 'small')

        # 1. 计算总分数增量
        score_diff = curr_score - self.last_score

        # 2. 计算金币带来的分数 (1 coin = 200 score)
        coins_diff = curr_coins - self.last_coins
        score_from_coins = coins_diff * 200

        # 3. 计算变大带来的分数 (Small -> Tall/Fireball = 1000 score)
        score_from_growth = 0
        if self.last_status == 'small' and curr_status in ['tall', 'fireball']:
            score_from_growth = 1000

        # 4. 剥离干扰项，计算“纯战斗/破坏分数” (Instant Attack Score)
        attack_score_delta = score_diff - score_from_coins - score_from_growth

        if attack_score_delta > 0:
            # 给 RL 模型奖励
            reward += attack_score_delta * self.attack_reward_scale
            # 【新增】累加到总战斗分
            self.total_attack_score += attack_score_delta

        # 更新历史状态
        self.last_score = curr_score
        self.last_coins = curr_coins
        self.last_status = curr_status
        
        # 【新增】将累计战斗分写入 info，方便 Evaluator 读取
        if done:            
            info['attack_score'] = self.total_attack_score

        return obs, reward, done, info

class GeneralScoreRewardWrapper(gym.Wrapper):
    """
    通用分数奖励 Wrapper。
    只要游戏的分数(Score)增加了，就给予奖励。
    这涵盖了：杀怪、吃金币、顶砖块、吃蘑菇等所有能增加分数的行为。
    """
    def __init__(self, env: gym.Env, score_reward_scale=0.1):
        super().__init__(env)
        self.scale = score_reward_scale
        self.last_score = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # 重置分数记录
        self.last_score = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 获取当前分数
        curr_score = info.get('score', 0)
        
        # 计算增量
        score_diff = curr_score - self.last_score
        
        if score_diff > 0:
            # 只有分数增加时才给奖励
            # 注意：Super Mario Bros 的分数膨胀很厉害（动不动几百几千）
            # 所以 scale 建议设小一点（例如 0.01 - 0.1），否则会把移动奖励(x_pos)完全淹没
            reward += score_diff * self.scale
            
        self.last_score = curr_score
        
        return obs, reward, done, info


# CAM相关，不需要了解
def dump_arr2video(arr, video_folder):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 6
    size = (256, 240)
    out = cv2.VideoWriter(video_folder + '/cam_pure.mp4', fourcc, fps, size)
    out1 = cv2.VideoWriter(video_folder + '/obs_pure.mp4', fourcc, fps, size)
    out2 = cv2.VideoWriter(video_folder + '/merged.mp4', fourcc, fps, size)
    for frame, obs in arr:
        frame = (255 * frame).astype('uint8').squeeze(0)
        frame_c = cv2.resize(cv2.applyColorMap(frame, cv2.COLORMAP_JET), size)
        out.write(frame_c)

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        out1.write(obs)

        merged_frame = cv2.addWeighted(obs, 0.6, frame_c, 0.4, 0)
        out2.write(merged_frame)
    # assert False


def get_cam(img, model):
    target_layers = [model.encoder.main[0]]
    input_tensor = torch.from_numpy(img).unsqueeze(0)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    return grayscale_cam


def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordCAM(gym.Wrapper):

    def __init__(
        self,
        env,
        cam_model,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        super(RecordCAM, self).__init__(env)
        self._env = env
        self.cam_model = cam_model

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum([x is not None for x in [episode_trigger, step_trigger]])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = []

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            warnings.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0

    def reset(self, **kwargs):
        observations = super(RecordCAM, self).reset(**kwargs)
        if not self.recording:
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = []

        self.recorded_frames = 0
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        time_step = super(RecordCAM, self).step(action)
        observations, rewards, dones, infos = time_step

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            self.video_recorder.append(
                (get_cam(observations, model=self.cam_model), copy.deepcopy(self.env.render(mode='rgb_array')))
            )
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > 10000:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones or infos['time'] < 250:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return time_step

    def close_video_recorder(self) -> None:
        if self.recorded_frames > 0:
            dump_arr2video(self.video_recorder, self.video_folder)
        self.video_recorder = []
        self.recording = False
        self.recorded_frames = 0

    def seed(self, seed: int) -> None:
        self._env.seed(seed)