import argparse
import json
import random

import wandb
wandb.login()

import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

from tensorboard_logger import configure

from models import models
from optimizers import optimizers
from training_funcs import train, validate, lr_multiplier_functor, _adjust_learning_rate

import os
import time
from dataclasses import dataclass

import gymnasium as gym
import tyro
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


# Check Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Experiment name', default='AUTO')
    # parser.add_argument('--model', type=str, help='Model name: resnet18', default='resnet18_cifar')
    parser.add_argument('--optim', type=str, help='Optimizer name: adagrad, sgd, koala-v/m...',
                        choices=list(optimizers.keys()), required=True)
    parser.add_argument('--env_id', type=str, help='Environment to run the experiment on', default='MountainCarContinuous-v0')
    parser.add_argument('--total-timesteps', type=int, help='total timesteps of the experiments', default=1000000)
    parser.add_argument('--num-envs', type=int, help='the number of parallel game environments', default=1)
    parser.add_argument('--num-steps', type=int, help='the number of steps to run in each environment per policy rollout', default=2048)
    parser.add_argument('--gamma', type=float, help='the discount factor gamma', default=0.99)
    parser.add_argument('--gae-lambda', type=float, help='the lambda for the general advantage estimation', default=0.95)
    # parser.add_argument('--batch-size', type=int, help='Batch size', default=32)
    parser.add_argument('--num-minibatches', type=int, help='Number of Mini Batch', default=32)     # original default: 32
    parser.add_argument('--update-epochs', type=int, help='the K epochs to update the policy', default=5) # original default: 10
    parser.add_argument('--norm-adv', type=bool, help='Toggles advantages normalization', default=True)
    parser.add_argument('--ent-coef', type=float, help='coefficient of the entropy', default=0.0)
    parser.add_argument('--vf-coef', type=float, help='coefficient of the value function', default=0.5)
    parser.add_argument('--num-gpus', type=int, help='Number of gpus', default=1)
    parser.add_argument('--num-epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--torch-deterministic', type=bool, help='no idea!', default=True)
    parser.add_argument('--capture-video', type=bool, help='whether to capture videos of the agent performances (check out `videos` folder)', default=False)
    parser.add_argument('--saving_freq', type=int, help='Frequency of model checkpoints', default=100)
    parser.add_argument('--resume-network', type=str, help='Path to model checkpoint to resume', default=None)
    parser.add_argument('--resume-epoch', type=int, help='Starting with epoch', default=None)
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset location')
    parser.add_argument('--warmup-epochs', type=int, help='Warmup epochs, set to 0 to disable', default=0)
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'step'], help='lr scheduler type') # original default: step
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for non-koala optimizers (note that the default is for SGD not Adam)')
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD's momentum")
    parser.add_argument('--step-gamma', type=float, default=0.2, help="step scheduler's gamma (lr*=gamma)")
    parser.add_argument('--seed', type=int, help='random seed', default=42)
    # KOALA specific args
    parser.add_argument('--r', type=float, help='None for adaptive', default=None)
    parser.add_argument('--sw', type=float, default=0.1)
    parser.add_argument('--sv', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--target-loss', type=float, default=-1.0)
    return parser.parse_args()


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)



def main():
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp == 'AUTO':
        args.exp = f'{args.env_id} {args.optim} {args.target_loss}' 
    
    wandb.init(
        project="koala-rl4", # project name 
        entity="hossein_abdi-the-university-of-manchester",
        name=args.exp,
        config=vars(args),                   # command line arguments
        monitor_gym=True,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    configure('runs/{}'.format(args.exp))

    with open(f'runs/{args.exp}/args.json', 'wt') as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(f"runs/{args.exp}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)

    # Manage extra params
    is_koala = args.optim.startswith('koala')
    extra_params = dict()
    if is_koala:
        extra_params['r'] = args.r
        if args.optim == 'koala-v':
            extra_params['sigma'] = args.sigma
        elif args.optim == 'koala-rl':
            extra_params['sigma'] = args.sigma
            extra_params['target_loss'] = args.target_loss
        else:
            extra_params['sw'] = args.sw
            extra_params['sv'] = args.sv
            extra_params['a'] = args.alpha
    else:
        extra_params['lr'] = args.lr
        if args.optim == 'sgd':
            extra_params['momentum'] = args.momentum

    # Setup optimizer
    optimizer = optimizers[args.optim](
        agent.parameters(),
        weight_decay=args.weight_decay,
        **extra_params)

    # Setup scheduler
    if args.scheduler == 'none':
        milestones = tuple()
    else:
        if args.num_epochs == 100:
            milestones = (30, 60, 90)
        else:
            raise NotImplementedError()
    calculate_lr = lr_multiplier_functor(args.num_envs * args.num_steps, base_lr=1.0 if args.optim.startswith('koala') else args.lr,
                                         warmup_iters=args.warmup_epochs * args.num_envs * args.num_steps, milestones=milestones,
                                         gamma=args.step_gamma)

    # Configure criterion
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        wandb.log({
                            "episodic_return": info["episode"]["r"],
                            "episodic_length": info["episode"]["l"]
                        }, step=global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # pg_loss = -(newlogprob * mb_advantages).mean()
                pg_loss = -(torch.exp(newlogprob-b_logprobs[mb_inds])* mb_advantages).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # compute gradient and perform update step
                if calculate_lr is not None:
                    _adjust_learning_rate(optimizer, calculate_lr(epoch, start))
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if is_koala:
                    loss_var = torch.mean(torch.pow(loss, 2))
                    optimizer.update(loss, loss_var)
                else:
                    optimizer.step()
                
                # wandb.log({
                #     "loss": loss.item(),
                #     "pg_loss": pg_loss.item(),
                #     "v_loss": v_loss.item(),
                #     "entropy_loss": entropy_loss.item(),
                #     "advantage": b_advantages[mb_inds].mean(),
                #     "return": b_returns[mb_inds].mean(),
                #     "value": b_values[mb_inds].mean(),
                # })


        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        wandb.log({
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "advantage": b_advantages.mean(),
            "return": b_returns.mean(),
            "value": b_values.mean(),
        }, step=global_step)

    # if args.save_model:
    #     model_path = f"runs/{args.exp}/{args.exp_name}.cleanrl_model"
    #     torch.save(agent.state_dict(), model_path)
    #     print(f"model saved to {model_path}")
    #     from cleanrl_utils.evals.ppo_eval import evaluate

    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=Agent,
    #         device=device,
    #         gamma=args.gamma,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         writer.add_scalar("eval/episodic_return", episodic_return, idx)


    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
