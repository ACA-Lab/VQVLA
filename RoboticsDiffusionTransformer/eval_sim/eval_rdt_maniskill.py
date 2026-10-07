import argparse
from collections import deque
import os
import random

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers ManiSkill task IDs with Gymnasium
import numpy as np
import torch
import yaml
from PIL import Image

from datalogger import DataLogger
from scripts.maniskill_model import create_model
from scripts.vqvla_rdt_tasks import TASK_INSTRUCTIONS
from tqdm import trange

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="StackCube-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parser.add_argument("--transition-height-threshold", type=float, default=0.12,
                        help="TCP height boundary that selects transition-state weights.")
    parser.add_argument("--stop-after-failures", type=int, default=0,
                        help="Stop after this many failures (0 disables early stopping).")
    parser.add_argument("--vq-4bit-archive", type=str, default=None,
                        help="Optional complete-model 4-bit GPTVQ archive.")
    parser.add_argument("--vq-3bit-archive", type=str, default=None,
                        help="Optional complete-model 3-bit GPTVQ archive.")
    parser.add_argument("--vq-mode", choices=("mixed", "4bit", "3bit"), default="mixed",
                        help="Use threshold-routed mixed weights or a fixed bit width.")
    parser.add_argument("--collect-hdiag-output", type=str, default=None,
                        help="Optionally save per-linear input second moments for GPTVQ calibration.")
    parser.add_argument("--hdiag-max-queries", type=int, default=0,
                        help="Stop collecting after this many policy queries; 0 collects all queries.")
    return parser.parse_args()

def accept_this_action(action, action_verified, last_action, obs):
    """Determine whether an action should be accepted"""
    l1_distance = torch.sum(torch.abs(torch.tensor(action_verified[:6] - action[:6])))
    threshold = 0.1
    delta = 0.025

    # if obs['extra']['is_grasped']:
    #     threshold -= delta
    #     diff = obs['extra']['tcp_pose'][0, :3] - obs['extra']['goal_pos']
    #     distance = torch.linalg.norm(diff)
    #     print(f"distance = {diff}")
    #     if distance < 0.1:
    #         threshold -= delta
    if last_action[-1] < 1:
        threshold -= delta

    return l1_distance < threshold

def at_exe_stage(action, last_action, obs, transition_height_threshold=0.12):
    if last_action is None:
        return 2
    if action[2] > 0 and last_action[2] > 0 and action[-1] == -1:
        return 0
    elif obs['extra']['tcp_pose'][0][2] > transition_height_threshold:
        return 1
    else:
        return 2

# set cuda
args = parse_args()
if args.hdiag_max_queries < 0:
    raise ValueError("--hdiag-max-queries must be non-negative")
if args.stop_after_failures < 0:
    raise ValueError("--stop-after-failures must be non-negative")
if args.collect_hdiag_output is not None and os.path.exists(args.collect_hdiag_output):
    raise FileExistsError(f"Refusing to overwrite {args.collect_hdiag_output}")
if (args.vq_4bit_archive is None) != (args.vq_3bit_archive is None):
    raise ValueError("both --vq-4bit-archive and --vq-3bit-archive must be provided together")
if args.collect_hdiag_output is not None and args.vq_4bit_archive is not None:
    raise ValueError("hdiag collection must use the original unquantized model")
# set random seeds
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

task2lang = TASK_INSTRUCTIONS

env_id = args.env_id
env = gym.make(
    env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode=args.render_mode,
    reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    sim_backend=args.sim_backend
)

config_path = 'configs/base.yaml'
with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
pretrained_path = args.pretrained_path
policy = create_model(
    args=config,
    dtype=torch.bfloat16,
    pretrained=pretrained_path,
    pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
)
hdiag_collector = None
if args.collect_hdiag_output is not None:
    from scripts.vqvla_rdt_hdiag import RdtHdiagCollector

    hdiag_collector = RdtHdiagCollector({
        "rdt": policy.policy,
        "siglip": policy.vision_model,
        "t5": policy.text_model,
    })
hdiag_active = hdiag_collector is not None
weight_router = None
if args.vq_4bit_archive is not None:
    from scripts.vqvla_rdt_vq import RdtMixedWeightRouter

    weight_router = RdtMixedWeightRouter.load(
        {
            "rdt": policy.policy,
            "siglip": policy.vision_model,
            "t5": policy.text_model,
        },
        args.vq_4bit_archive,
        args.vq_3bit_archive,
    )
    weight_router.set_execution_state(True)
    print(f"Loaded mixed 4/3-bit weights for {len(weight_router.routed_modules)} modules")
if os.path.exists(f'text_embed_{env_id}.pt'):
    # Preserve the original benchmark input path for raw and VQ evaluation.
    text_embed = torch.load(f'text_embed_{env_id}.pt')
    if hdiag_collector is not None:
        # Cached benchmark embeddings bypass T5; exercise it once for calibration.
        policy.encode_instruction(task2lang[env_id])
else:
    text_embed = policy.encode_instruction(task2lang[env_id])

MAX_EPISODE_STEPS = 400
total_episodes = args.num_traj
success_count = 0
failure_count = 0
episodes_completed = 0

base_seed = 20241201
logger = DataLogger()
hdiag_query_count = 0
policy_query_count = 0
transition_3bit_query_count = 0
for episode in trange(total_episodes):
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = episode + base_seed)
    policy.reset()

    img = env.render().squeeze(0).detach().cpu().numpy()
    obs_window.append(None)
    obs_window.append(np.array(img))
    proprio = obs['agent']['qpos'][:, :-1]
    # proprio = obs[:, :-1]

    global_steps = 0
    done = False
    gap = 8
    exe = False
    flag = 0
    total_step = 0
    exe_step = 0
    tran_step = 0
    stage = 2

    last_action = None

    while global_steps < MAX_EPISODE_STEPS and not done:
        image_arrs = []
        for window_img in obs_window:
            image_arrs.append(window_img)
            image_arrs.append(None)
            image_arrs.append(None)
        images = [Image.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        total_step += 1
        if exe:
            exe_step += 1
        else:
            tran_step += 1
        if weight_router is not None:
            if args.vq_mode == "4bit":
                selected_execution_state = True
            elif args.vq_mode == "3bit":
                selected_execution_state = False
            else:
                selected_execution_state = exe
            weight_router.set_execution_state(selected_execution_state)
            policy_query_count += 1
            transition_3bit_query_count += int(not selected_execution_state)
        actions = policy.step(proprio, images, text_embed, True, exe, False, 0.0).squeeze(0).cpu().numpy()
        if hdiag_active:
            hdiag_query_count += 1
            if args.hdiag_max_queries and hdiag_query_count >= args.hdiag_max_queries:
                hdiag_collector.close()
                hdiag_active = False
                print(f"Stopped hdiag collection after {hdiag_query_count} policy queries", flush=True)
        # Execute every eighth action from the predicted 64-step chunk.
        actions = actions[::gap, :]
        actual_shape = actions.shape[0]
        # if not exe:
        #     for idx in range(actual_shape):
        #         no_quant_actions = policy.step(proprio, images, text_embed, False, exe, False, 0.0).squeeze(0).cpu().numpy()
        #         action = actions[idx]
        #         no_quant_action = no_quant_actions[idx]
        #         print(f"quant_action = {action}\nno_quant_act = {no_quant_action}")
        #     print()

        # print(f"actions = {actions}")
        # if not exe:
        #     for idx in range(actual_shape):
        #         if actions[idx][2] > 0:
        #             actions[idx][2] *= 2.0
        # print(f"generate {64 // gap} actions")
        for idx in range(actual_shape):
            action = actions[idx]

            if flag == 0 and action[-1] == -1:
                flag = 1
            if flag == 1 and action[-1] == 1:
                flag = 2
            stage = at_exe_stage(
                action, last_action, obs, args.transition_height_threshold
            )
            # print(f"flag = {flag}")
            if stage == 0 and flag != 2:
                exe = True
            elif stage == 1 and flag != 2:
                exe = False
            else:
                exe = True

            obs, _, terminated, truncated, info = env.step(action)
            # pos = obs['extra']['tcp_pose'][0, :3].cpu().tolist()
            # action_for_print = action.tolist()
            # print(f"pos = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), action = ({action_for_print[0]:.4f}, {action_for_print[1]:.4f}, {action_for_print[2]:.4f}, {action_for_print[-1]:.4f})")
            # if action[-1] == 1:
            #     exe = False
            # else:
            #     exe = True
            # print(f"obs = {obs}")
            img = env.render().squeeze(0).detach().cpu().numpy()
            obs_window.append(img)
            proprio = obs['agent']['qpos'][:, :-1]
            global_steps += 1

            if terminated or truncated:
                assert "success" in info, sorted(info.keys())
                if info['success']:
                    done = True
                    break
    last_action = action
    episodes_completed += 1
    if info["success"]:
        success_count += 1
    else:
        failure_count += 1

    logger.log_step(total_step, exe_step, tran_step)
    print(total_step, exe_step, tran_step)
    print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")
    if weight_router is not None:
        ratio = 100.0 * transition_3bit_query_count / max(policy_query_count, 1)
        print(
            f"Cumulative transition-state 3-bit policy-query ratio: {ratio:.2f}% "
            f"({transition_3bit_query_count}/{policy_query_count})",
            flush=True,
        )
    if args.stop_after_failures and failure_count >= args.stop_after_failures:
        print(
            f"Stopped after {episodes_completed} episodes and {failure_count} failures",
            flush=True,
        )
        break


if hdiag_collector is not None:
    hdiag_collector.close()
    hdiag_collector.save(
        args.collect_hdiag_output,
        checkpoint=pretrained_path,
        observations=hdiag_query_count,
        suite=env_id,
    )
    print(f"Saved RDT hdiag summaries to {args.collect_hdiag_output}", flush=True)

success_rate = success_count / max(episodes_completed, 1) * 100
print(f"Success rate: {success_rate}% ({success_count}/{episodes_completed})")
if weight_router is not None:
    ratio = 100.0 * transition_3bit_query_count / max(policy_query_count, 1)
    print(
        f"Transition-state 3-bit policy-query ratio: {ratio:.2f}% "
        f"({transition_3bit_query_count}/{policy_query_count})"
    )
