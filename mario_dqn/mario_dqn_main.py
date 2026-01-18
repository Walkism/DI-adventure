"""
智能体训练入口，包含训练逻辑
"""
from tensorboardX import SummaryWriter
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import SyncSubprocessEnvManager, DingEnvWrapper, BaseEnvManager
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv, CoinRewardWrapper, StickyActionWrapper, SparseRewardWrapper,\
    CustomCoinRewardWrapper, AttackRewardWrapper, GrowthRewardWrapper, GeneralScoreRewardWrapper # 新import自定义wrapper
from ding.envs.env_wrappers import ObsTransposeWrapper
from policy import DQNPolicy
from model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from mario_dqn_config import mario_dqn_config
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from functools import partial
import os
import gym_super_mario_bros


# 动作相关配置
action_dict = {2: [["right"], ["right", "A"]], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT}
action_nums = [2, 7, 12]


# mario环境
def wrapped_mario_env(version=0, action=7, obs=1, crs=0.0, ars=0.0, gr=0.0, srs=0.0):
    print(f"===> Wrapped Mario Env: version {version}, action {action}, obs {obs}, coin_reward_scale {crs}, attack_reward_scale {ars}, growth_reward {gr}, score_reward_scale {srs}")
    return DingEnvWrapper(
        # 设置mario游戏版本与动作空间
        JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v"+str(version)), action_dict[int(action)]),
        cfg={
            # 添加各种wrapper
            'env_wrapper': [
                # 默认wrapper：跳帧以降低计算量
                # 1. 每skip个step执行同一action，取最后两帧的max_pool作为obs. obs_shape不变
                lambda env: MaxAndSkipWrapper(env, skip=4),
                # 默认wrapper：将mario游戏环境图片进行处理，返回大小为84X84的图片observation
                # 2. RGB转grey, 然后resize到84x84 obs_shape=(84,84)
                lambda env: WarpFrameWrapper(env, size=84),
                # 默认wrapper：将observation数值进行归一化
                # 3. 归一化到[0.0,1.0] dtype=np.float32
                lambda env: ScaledFloatFrameWrapper(env),
                # 默认wrapper：叠帧，将连续n_frames帧叠到一起，返回shape为(n_frames,84,84)的图片observation
                # 4. 每次返回的obs为最近n_frames帧的stack（用deque维护最近n_frames帧）
                lambda env: FrameStackWrapper(env, n_frames=obs),
                # 默认wrapper：在评估一局游戏结束时返回累计的奖励，方便统计
                
                # 以下是你添加的wrapper
                # 新增1. 金币奖励wrapper
                # lambda env: CoinRewardWrapper(env),
                lambda env: CustomCoinRewardWrapper(env, coin_reward_scale=crs), # 增强版CoinRewardWrapper
                
                # 新增2. 粘性动作wrapper
                # lambda env: StickyActionWrapper(env, p_sticky=0.25),
                # 新增3. 稀疏奖励wrapper
                # lambda env: SparseRewardWrapper(env),

                # 新增4. 只奖励攻击行为的wrapper
                lambda env: AttackRewardWrapper(env, attack_reward_scale=ars),  
                # 新增5. 只奖励成长行为的wrapper
                lambda env: GrowthRewardWrapper(env, growth_reward=gr),  
                # 新增6. 通用分数奖励wrapper
                lambda env: GeneralScoreRewardWrapper(env, score_reward_scale=srs),

                # 最后: 在info里累加reward info['final_eval_reward']
                lambda env: FinalEvalRewardEnv(env),
            ]
        }
    )


def main(cfg, args, seed=0, max_env_step=int(3e6)):
    # Easydict类实例，包含一些配置
    # cfg = compile_config(mario_dqn_config, create_cfg=mario_dqn_create_config, auto=True, save_cfg=False)
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        seed=seed,
        save_cfg=True
    )
    # 收集经验的环境数量以及用于评估的环境数量
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # 收集经验的环境，使用并行环境管理器
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_mario_env, version=args.version, action=args.action, obs=args.obs, 
                        crs=args.coin_reward_scale, ars=args.attack_reward_scale, gr=args.growth_reward, srs=args.score_reward_scale) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    # 评估性能的环境，使用并行环境管理器
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_mario_env, version=args.version, action=args.action, obs=args.obs, 
                        crs=args.coin_reward_scale, ars=args.attack_reward_scale, gr=args.growth_reward, srs=args.score_reward_scale) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    # 为mario环境设置种子
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    # 为torch、numpy、random等package设置种子
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # 采用DQN模型
    model = DQN(**cfg.policy.model)
    # 采用DQN策略
    policy = DQNPolicy(cfg.policy, model=model)

    # 设置学习、经验收集、评估、经验回放等强化学习常用配置
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    # 设置epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    # 训练以及评估
    while True:
        # 根据当前训练迭代数决定是否进行评估
        if evaluator.should_eval(learner.train_iter): # 默认eval_freq = 2000
            # stop条件: reward到达 stop_value=3000
            # 只负责保存 ckpt_best.pth.tar
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # 更新epsilon greedy信息
        eps = epsilon_greedy(collector.envstep)
        # 经验收集器从环境中收集经验
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        # 将收集的经验放入replay buffer
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # 采样经验进行训练
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
        if collector.envstep >= max_env_step:
            break


if __name__ == "__main__":
    from copy import deepcopy
    import argparse
    parser = argparse.ArgumentParser()
    # 种子
    parser.add_argument("--seed", "-s", type=int, default=0)
    # 游戏版本，v0 v1 v2 v3 四种选择
    parser.add_argument("--version", "-v", type=int, default=0, choices=[0,1,2,3])
    # 动作集合种类，包含[["right"], ["right", "A"]]、SIMPLE_MOVEMENT、COMPLEX_MOVEMENT，分别对应2、7、12个动作
    parser.add_argument("--action", "-a", type=int, default=7, choices=[2,7,12])
    # 观测空间叠帧数目，不叠帧或叠四帧
    parser.add_argument("--obs", "-o", type=int, default=1, choices=[1,4])

    parser.add_argument("--coin_reward_scale", "-crs", type=float, default=0.0)   # [新增] 金币奖励放大倍数
    parser.add_argument("--attack_reward_scale", "-ars", type=float, default=0.0) # [新增] 攻击奖励放大倍数
    parser.add_argument("--growth_reward", "-gr", type=float, default=0.0)        # [新增] 成长奖励系数
    parser.add_argument("--score_reward_scale", "-srs", type=float, default=0.0)  # [新增] 分数奖励放大倍数

    args = parser.parse_args()

    # [修改] 保存exp路径增加coin_reward_scale, attack_reward_scale, growth_reward, score_reward_scale信息
    mario_dqn_config.exp_name = 'exp/v'+str(args.version)+'_'+str(args.action)+'a_'+str(args.obs)+'f_seed'+str(args.seed) \
        +'_'+'crs_'+str(int(args.coin_reward_scale))+'_'+'ars_'+str(round(args.attack_reward_scale,2))+'_'+'gr_'+str(int(args.growth_reward))+'_'+'srs_'+str(int(args.score_reward_scale))
    mario_dqn_config.policy.model.obs_shape=[args.obs, 84, 84]
    # mario_dqn_config.policy.model.obs_shape=[3, 240, 256]  # 无任何特征空间处理的情况
    mario_dqn_config.policy.model.action_shape=args.action
    # main(deepcopy(mario_dqn_config), args, seed=args.seed)

    # [新增] 提高停止分数，适应新增奖励
    # 收敛目标: 原始3000 + 20个coin + 2000分来自踩扁Goomba + 2次growth + 4600分数(那个惊艳的吃金币ckpt的score)
    # 一般不会每个scale都同时激活。比如可能只激活coin奖励
    mario_dqn_config.env.stop_value = 3000 + 20*args.coin_reward_scale + 20000*args.attack_reward_scale + 2*args.growth_reward + 4600*args.score_reward_scale

    # v0_7a_1f_seed0_251123_173312 训练了3M步reward_mean还没到3000，最高只有2614。加大max_env_step！
    main(deepcopy(mario_dqn_config), args, seed=args.seed, max_env_step=int(3e7)) 