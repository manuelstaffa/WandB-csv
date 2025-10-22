from email import policy
from hackatari import HackAtari, HumanPlayable
import numpy as np
import cv2
import pygame
import torch
import gymnasium as gym
from ocatari.utils import load_agent
import os
import argparse
import json
from hackatari.utils import HackAtariArgumentParser
from stable_baselines3.common.atari_wrappers import (
    FireResetEnv,
    NoopResetEnv,
)
import rliable.metrics as rlm

import time

# Get the current time in a human-readable format
current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

# Disable graphics window (SDL) for headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Define human and random scores
atari_scores = {
    "alien": (227.8, 7127.7),
    "amidar": (5.8, 1719.5),
    "assault": (222.4, 742.0),
    "asterix": (210.0, 8503.3),
    "asteroids": (719.1, 47388.7),
    "atlantis": (12850.0, 29028.1),
    "bankheist": (14.2, 753.1),
    "battlezone": (2360.0, 37187.5),
    "beamrider": (363.9, 16926.5),
    "berzerk": (123.7, 2630.4),
    "bowling": (23.1, 160.7),
    "boxing": (0.1, 12.1),
    "breakout": (1.7, 30.5),
    "centipede": (2090.9, 12017.0),
    "choppercommand": (811.0, 7387.8),
    "crazyclimber": (10780.5, 35829.4),
    "defender": (2874.5, 18688.9),
    "demonattack": (152.1, 1971.0),
    "doubledunk": (-18.6, -16.4),
    "enduro": (0.0, 860.5),
    "fishingderby": (-91.7, -38.7),
    "freeway": (0.0, 29.6),
    "frostbite": (65.2, 4334.7),
    "gopher": (257.6, 2412.5),
    "gravitar": (173.0, 3351.4),
    "hero": (1027.0, 30826.4),
    "icehockey": (-11.2, 0.9),
    "jamesbond": (29.0, 302.8),
    "kangaroo": (52.0, 3035.0),
    "krull": (1598.0, 2665.5),
    "kungfumaster": (258.5, 22736.3),
    "montezumarevenge": (0.0, 4753.3),
    "mspacman": (307.3, 6951.6),
    "namethisgame": (2292.3, 8049.0),
    "phoenix": (761.4, 7242.6),
    "pitfall": (-229.4, 6463.7),
    "pong": (-20.7, 14.6),
    "privateeye": (24.9, 69571.3),
    "qbert": (163.9, 13455.0),
    "riverraid": (1338.5, 17118.0),
    "roadrunner": (11.5, 7845.0),
    "robotank": (2.2, 11.9),
    "seaquest": (68.4, 42054.7),
    "skiing": (-17098.1, -4336.9),
    "solaris": (1236.3, 12326.7),
    "spaceinvaders": (148.0, 1668.7),
    "stargunner": (664.0, 10250.0),
    "surround": (-10.0, 6.5),
    "tennis": (-23.8, -8.3),
    "timepilot": (3568.0, 5229.2),
    "tutankham": (11.4, 167.6),
    "upndown": (533.4, 11693.2),
    "venture": (0.0, 1187.5),
    "videopinball": (16256.9, 17667.9),
    "wizardofwor": (563.5, 4756.5),
    "yarsrevenge": (3092.9, 54576.9),
    "zaxxon": (32.5, 9173.3),
}


def calculate_hns(score, game):
    human_score = atari_scores[game.lower()][1]
    random_score = atari_scores[game.lower()][0]
    return (score - random_score) / (human_score - random_score)


def main():
    parser = HackAtariArgumentParser(description="HackAtari Experiment Runner")
    parser.add_argument(
        "-g", "--game", type=str, default="Seaquest", help="Game to be run"
    )
    parser.add_argument(
        "-obs",
        "--obs_mode",
        type=str,
        default="dqn",
        help="Observation mode (ori, dqn, obj)",
    )
    parser.add_argument(
        "-w", "--window", type=int, default=4, help="Buffer window size"
    )
    parser.add_argument(
        "-f",
        "--frameskip",
        type=int,
        default=4,
        help="Frames skipped after each action",
    )
    parser.add_argument(
        "-dp",
        "--dopamine_pooling",
        action="store_true",
        help="Enable dopamine-like frameskipping",
    )
    parser.add_argument(
        "-m", "--modifs", nargs="+", default=[], help="List of modifications to apply"
    )
    parser.add_argument(
        "-rf",
        "--reward_function",
        type=str,
        default="",
        help="Custom reward function path",
    )
    parser.add_argument(
        "-a",
        "--agents",
        nargs="+",
        required=True,
        help="List of trained agent model paths",
    )
    parser.add_argument(
        "-mo", "--game_mode", type=int, default=0, help="Alternative ALE game mode"
    )
    parser.add_argument(
        "-d", "--difficulty", type=int, default=0, help="Alternative ALE difficulty"
    )
    parser.add_argument(
        "-e", "--episodes", type=int, default=10, help="Number of episodes per agent"
    )
    parser.add_argument(
        "-wr", "--wrapper", type=str, default="", help="Use a masking wrapper"
    )
    parser.add_argument(
        "-out",
        "--output",
        type=str,
        default="results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=0,
        help="Epsilon that random actions is sampled",
    )

    args = parser.parse_args()

    run_name = f"{args.game}_{current_time}"

    env = HackAtari(
        args.game,
        args.modifs,
        args.reward_function,
        dopamine_pooling=args.dopamine_pooling,
        game_mode=args.game_mode,
        difficulty=args.difficulty,
        render_mode="None",
        obs_mode=args.obs_mode,
        mode="ram",
        hud=False,
        render_oc_overlay=True,
        buffer_window_size=args.window,
        frameskip=args.frameskip,
        repeat_action_probability=0.25,
        full_action_space=False,
    )

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = NoopResetEnv(env, noop_max=30)
    results = {}

    for agent_path in args.agents:
        policy = load_cleanrl_agent(agent_path, env, "cpu")
        print(f"Loaded agent from {agent_path}")

        rewards = []
        # Storage for custom metrics
        distances = []  # Kangaroo
        height = 180
        divers_collected = []  # Seaquest

        for episode in range(args.episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            # Per-episode custom metric accumulators
            ep_distance = 0
            ep_divers = 0

            while not done:
                if np.random.rand() < args.epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_t = (
                            torch.tensor(obs, dtype=torch.float32)
                            .unsqueeze(0)
                            .to("cpu")
                        )
                        action, _, _, _ = policy.get_action_and_value(obs_t)
                        action = action.item()

                obs, reward, terminated, truncated, info = env.step(action)  # type: ignore
                episode_reward += reward  # type: ignore
                done = terminated or truncated

                # -------------------------------
                # Custom metrics depending on game
                # -------------------------------
                if args.game.lower() == "kangaroo":
                    if hasattr(env.unwrapped, "objects"):
                        objects = env.unwrapped.objects
                        player = next(
                            (o for o in objects if o.category == "Player"), None
                        )
                        child = next(
                            (o for o in objects if o.category == "Child"), None
                        )

                        # print(player.y)

                        if player and child:
                            # Manhattan distance between player and child
                            ep_distance = abs(player.x - child.x) + abs(
                                player.y - child.y
                            )
                            # print(ep_distance)
                            distances.append(ep_distance)
                            # print(
                            #    f"Player: {player.x}, {player.y}, Child: {child.x}, {child.y}, Distance: {ep_distance}"
                            # )
                            if player.y < height:
                                height = player.y

                elif args.game.lower() == "seaquest":
                    if hasattr(env.unwrapped, "objects"):
                        objects = env.unwrapped.objects
                        # Count all Diver objects visible in this step
                        ep_divers += sum(1 for o in objects if o.category == "Diver")

            # Store episode results
            rewards.append(episode_reward)
            if args.game.lower() == "kangaroo":
                ep_distance = np.mean(distances) if distances else 0
                print(
                    f"Episode {episode + 1}: Reward={episode_reward}, Distance={ep_distance}"
                )
                print(f"Height Achieved: {height}")
            elif args.game.lower() == "seaquest":
                divers_collected.append(ep_divers)
                print(
                    f"Episode {episode + 1}: Reward={episode_reward}, Divers={ep_divers}"
                )
            else:
                print(f"Episode {episode + 1}: Reward={episode_reward}")

        hns_rewards = []
        for rew in rewards:
            hns_rewards.append(calculate_hns(rew, args.game))

        median_reward = np.median(rewards)
        hns_median = np.median(hns_rewards)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        hns_reward = np.mean(hns_rewards)
        iqm_reward = rlm.aggregate_iqm(np.array(rewards))
        hns_iqm = rlm.aggregate_iqm(np.array(hns_rewards))
        std_hns = np.std(hns_rewards)
        results = {}

        results[run_name] = {
            "model": agent_path,
            "episode_rewards": rewards,
            "hns_rewards": hns_rewards,
            "mean_reward": avg_reward,
            "std_reward": std_reward,
            "median_reward": median_reward,
            "iqm_reward": iqm_reward,
            "hns_mean": hns_reward,
            "hns_std": std_hns,
            "hns_median": hns_median,
            "hns_iqm": hns_iqm,
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "human": atari_scores[args.game.lower()][1],
            "random": atari_scores[args.game.lower()][0],
            "Args": vars(args),
        }

        print(f"\nSummary for {agent_path}:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"HNS: {hns_reward:.2f}")
        print(f"IQM Reward: {iqm_reward:.2f}")
        print(f"HNS (IQM): {hns_iqm:.2f}")
        print("--------------------------------------")

        try:
            with open(args.output, "r") as f:
                existing_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_results = []

        existing_results.append(results)

        with open(args.output, "w") as f:
            json.dump(existing_results, f, indent=4)

    env.close()


def load_cleanrl_agent(agent_path, env, device="cpu"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    checkpoint = torch.load(agent_path, map_location=device)
    args = checkpoint["args"]

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # Simple CNN policy/value network (CleanRL default for Atari PPO)
    class PPOAtariAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
            )
            self.actor = nn.Linear(512, n_actions)
            self.critic = nn.Linear(512, 1)

        def get_action_and_value(self, x, action=None):
            x = x / 255.0  # CleanRL normalizes obs
            h = self.network(x)
            logits = self.actor(h)
            probs = torch.distributions.Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(h)

        def get_value(self, x):
            x = x / 255.0
            h = self.network(x)
            return self.critic(h)

    # Alternative architecture for older models
    class PPOAtariAgentOld(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.mlp = nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
            )
            self.actor = nn.Linear(512, n_actions)
            self.critic = nn.Linear(512, 1)

        def get_action_and_value(self, x, action=None):
            x = x / 255.0  # CleanRL normalizes obs
            features = self.features(x)
            h = self.mlp(features)
            logits = self.actor(h)
            probs = torch.distributions.Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(h)

        def get_value(self, x):
            x = x / 255.0
            features = self.features(x)
            h = self.mlp(features)
            return self.critic(h)

    # Try loading with the current architecture first
    agent = PPOAtariAgent().to(device)
    try:
        agent.load_state_dict(checkpoint["model_weights"])
        print("Loaded model with current architecture")
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            print("Current architecture failed, trying alternative architecture...")
            # Try with the alternative architecture
            agent = PPOAtariAgentOld().to(device)
            agent.load_state_dict(checkpoint["model_weights"])
            print("Loaded model with alternative architecture")
        else:
            raise e

    agent.eval()
    return agent


if __name__ == "__main__":
    main()
