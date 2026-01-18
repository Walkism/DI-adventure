# import time
# time.sleep(30)
"""
智能体评估函数
"""
import torch
from ding.utils import set_pkg_seed
from mario_dqn_config import mario_dqn_config, mario_dqn_create_config
from model import DQN
from policy import DQNPolicy
from ding.config import compile_config
from ding.envs import DingEnvWrapper
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv, RecordCAM, CustomCoinRewardWrapper, AttackRewardWrapper, GrowthRewardWrapper, GeneralScoreRewardWrapper # 新import自定义wrapper

action_dict = {2: [["right"], ["right", "A"]], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT}
action_nums = [2, 7, 12]


import gym
import os
import json
import numpy as np

# 自定义 JSON 编码器，处理 numpy 数据类型，用于保存info字典
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):  # <--- 新增这一行
            return bool(obj)
        return super(NpEncoder, self).default(obj)

# 定义自定义Wrapper，继承原有的DingEnvWrapper
class CustomDingEnvWrapper(DingEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._video_prefix = None

    def set_video_prefix(self, prefix):
        """设置视频文件名前缀"""
        self._video_prefix = prefix

    def reset(self):
        # 核心逻辑：在调用父类reset之前，手动处理RecordVideo的挂载
        # 这样可以绕过父类中强制使用 id(self) 命名的逻辑
        
        # 只有当启用了录像(path不为空) 且 设置了前缀时才进行自定义处理
        if self._replay_path is not None and self._video_prefix is not None:
            # 1. 手动挂载 RecordVideo，传入我们自定义的 name_prefix
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix=self._video_prefix
            )
            print(f"[CustomDingEnvWrapper] Video recording enabled. Prefix: {self._video_prefix}")
            
            # 2. 将 _replay_path 置为 None
            # 这样当随后调用 super().reset() 时，父类会认为不需要录像，从而跳过它那段“起坏名字”的逻辑，
            # 直接执行 self._env.reset() (此时 self._env 已经是我们包好 RecordVideo 的环境了)
            self._replay_path = None
            
        return super().reset()

# 修改 wrapped_mario_env 使用 CustomDingEnvWrapper
def wrapped_mario_env(model, cam_video_path, version=0, action=7, obs=1, crs=0.0, ars=0.0, gr=0.0, srs=0.0):
    return CustomDingEnvWrapper(
        JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v"+str(version)), action_dict[int(action)]),
        cfg={
            'env_wrapper': [
                lambda env: gym.wrappers.RecordVideo(env,
                    video_folder=cam_video_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix="original_obs"
                ), # 保存原速原尺寸视频
                lambda env: MaxAndSkipWrapper(env, skip=4),
                lambda env: WarpFrameWrapper(env, size=84),
                lambda env: ScaledFloatFrameWrapper(env),
                lambda env: FrameStackWrapper(env, n_frames=obs),
                lambda env: FinalEvalRewardEnv(env),
                lambda env: RecordCAM(env, cam_model=model, video_folder=cam_video_path),

                # 新增自定义奖励wrapper，和训练脚本对齐
                lambda env: CustomCoinRewardWrapper(env, coin_reward_scale=crs),
                lambda env: AttackRewardWrapper(env, attack_reward_scale=ars),  
                lambda env: GrowthRewardWrapper(env, growth_reward=gr),
                lambda env: GeneralScoreRewardWrapper(env, score_reward_scale=srs),
            ]
        }
    )


# 增加 video_prefix 参数以接收前缀
def evaluate(args, state_dict, seed, video_dir_path, eval_times, video_prefix):
    # 加载配置
    cfg = compile_config(mario_dqn_config, create_cfg=mario_dqn_create_config, auto=True, save_cfg=False)
    # 实例化DQN模型
    model = DQN(**cfg.policy.model)
    # 加载模型权重文件
    model.load_state_dict(state_dict['model'])
    # 生成环境
    env = wrapped_mario_env(model, args.replay_path, args.version, args.action, args.obs, 
                            crs=args.coin_reward_scale, ars=args.attack_reward_scale, gr=args.growth_reward, srs=args.score_reward_scale)
    # 实例化DQN策略
    policy = DQNPolicy(cfg.policy, model=model).eval_mode
    # 设置seed
    env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    # 保存录像
    env.enable_save_replay(video_dir_path)

    # --- 修改开始 ---
    # 使用 CustomDingEnvWrapper 新增的方法设置保存前缀
    if hasattr(env, 'set_video_prefix'):
        env.set_video_prefix(video_prefix)
    # --- 修改结束 ---

    eval_reward_list = []
    # 评估
    for n in range(eval_times):
        # 环境重置，返回初始观测
        obs = env.reset()
        eval_reward = 0
        while True:
            # 策略根据观测返回所有动作的Q值以及Q值最大的动作
            Q = policy.forward({0: obs})
            # 获取动作
            action = Q[0]['action'].item()
            # 将动作传入环境，环境返回下一帧信息
            obs, reward, done, info = env.step(action)
            eval_reward += reward
            if done or info['time'] < 0:
                print(info)
                eval_reward_list.append(eval_reward)
                
                # --- 新增：保存评估episode结束时的 info 字典为 JSON 文件 ---
                json_filename = f"{video_prefix}-episode-{n}-info.json"
                save_path = os.path.join(video_dir_path, json_filename)
                with open(save_path, 'w') as f:
                    json.dump(info, f, cls=NpEncoder, indent=4)
                
                break
        print('During {}th evaluation, the total reward your mario got is {}'.format(n, eval_reward))
    print('Eval is over! The performance of your RL policy is {}'.format(sum(eval_reward_list) / len(eval_reward_list)))
    print("Your mario video is saved in {}".format(video_dir_path))
    # print(f"eval_reward_list: {eval_reward_list}") # 测评时，由于mario环境确定性，DQN也为确定性，故每次奖励必固定，只需eval_times=1
    try:
        del env
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--checkpoint", "-ckpt", type=str, default='./exp/v0_1a_7f_seed0/ckpt/ckpt_best.pth.tar')
    parser.add_argument("--replay_path", "-rp", type=str, default='./eval_videos')
    parser.add_argument("--version", "-v", type=int, default=0, choices=[0,1,2,3])
    parser.add_argument("--action", "-a", type=int, default=7, choices=[2,7,12])
    parser.add_argument("--obs", "-o", type=int, default=1, choices=[1,4])

    # 新增奖励系数参数
    parser.add_argument("--coin_reward_scale", "-crs", type=float, default=0.0)
    parser.add_argument("--attack_reward_scale", "-ars", type=float, default=0.0)
    parser.add_argument("--score_reward_scale", "-srs", type=float, default=0.0)
    parser.add_argument("--growth_reward", "-gr", type=float, default=0.0)
    args = parser.parse_args()
    mario_dqn_config.policy.model.obs_shape=[args.obs, 84, 84]
    mario_dqn_config.policy.model.action_shape=args.action
    ckpt_path = args.checkpoint
    video_dir_path = args.replay_path
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # --- 新增：解析路径生成保存视频的前缀 ---
    # 假设路径结构: .../exp/EXP_NAME/ckpt/CKPT_NAME.pth.tar
    # 1. 获取文件名 (去除扩展名)
    ckpt_filename = os.path.basename(args.checkpoint)
    # 处理可能的双重后缀 .pth.tar
    while '.' in ckpt_filename:
        ckpt_filename = os.path.splitext(ckpt_filename)[0]
        
    # 2. 获取实验名 (假设是 ckpt 文件夹的上一级)
    # args.checkpoint 的 dirname 是 .../ckpt
    # 再 dirname 一次是 .../exp/v0_7a_1f_seed0_251123_173312
    exp_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    exp_name = os.path.basename(exp_dir_path)
    
    # 3. 组合前缀
    # 例如: v0_7a_1f_seed0_251123_173312-ckpt_best
    video_prefix = f"{exp_name}-{ckpt_filename}"
    print(f"Generated video prefix: {video_prefix}")

    # 4. 将录像保存路径设为实验目录
    video_dir_path = os.path.join(exp_dir_path, "videos")
    args.replay_path = video_dir_path # CAM的录像路径只和args.replay_path有关 也需改
    # -----------------------------

    # [12.21修改] 调用 evaluate 时传入 video_prefix
    evaluate(args, state_dict=state_dict, seed=args.seed, video_dir_path=video_dir_path, eval_times=1, video_prefix=video_prefix)
