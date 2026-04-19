"""
train_continuous.py — PPO trainer for the continuous-action
Flu Vaccine Resource Management environment.

Supports multiple environment presets:
  - baseline
  - realistic

Expected environment factory:
  - envs.make_env_continuous.make_env_continuous(env_name)
    OR
  - adapt the import below to match your project structure.

Saves models as:
  flu_rl_model_continuous_baseline.zip
  flu_rl_model_continuous_realistic.zip

Typical usage:
  python train_continuous.py --env baseline
  python train_continuous.py --env realistic
  python train_continuous.py --env baseline --timesteps 500000
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Preferred: dedicated continuous env factory mirroring your existing make_env.py.
try:
    from envs.make_env_continuous import make_env_continuous
except ImportError:
    make_env_continuous = None

# Fallback: if you instead wire continuous creation into a different module,
# edit this block to match your project.
if make_env_continuous is None:
    try:
        from envs.flu_env_continuous import FluVaccineEnvContinuous
        from envs.configs import BASELINE_CONFIG, REALISTIC_CONFIG

        def make_env_continuous(env_name: str):
            if env_name == "baseline":
                return FluVaccineEnvContinuous(BASELINE_CONFIG)
            if env_name == "realistic":
                return FluVaccineEnvContinuous(REALISTIC_CONFIG)
            raise ValueError(f"Unknown env preset: {env_name}")

    except ImportError as exc:
        raise ImportError(
            "Could not import a continuous environment factory. "
            "Either create envs.make_env_continuous.make_env_continuous(), "
            "or ensure envs.flu_env_continuous and envs.configs are available."
        ) from exc



class RewardLoggerCallback(BaseCallback):
    """Lightweight callback to print rollout reward progress during training."""

    def __init__(self, print_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = int(print_freq)
        self.last_print_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_print_step >= self.print_freq:
            self.last_print_step = self.num_timesteps
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                print(
                    f"[train] steps={self.num_timesteps:,} "
                    f"mean_ep_reward={mean_reward:,.2f} "
                    f"mean_ep_len={mean_length:.1f}"
                )
            else:
                print(f"[train] steps={self.num_timesteps:,}")
        return True


class EpisodeTraceCallback(BaseCallback):
    """
    Optionally prints one deterministic rollout after training for sanity checking.
    """

    def __init__(self, env_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.env_name = env_name

    def _on_step(self) -> bool:
        return True

    def on_training_end(self) -> None:
        env = make_env_continuous(self.env_name)
        obs, info = env.reset(seed=123)
        done = False
        week = 0

        print("\n=== PPO Continuous Sanity Check (1 Episode) ===")
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(env.num_regions)
            displayed_orders = np.round(action * 600.0 / 10.0) * 10.0
            print(
                f"Week {week + 1}: "
                f"Orders={displayed_orders}, "
                f"Inventory={np.round(obs[:3], 1)}"
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            week += 1


def build_env(env_name: str, check: bool = False):
    """Create monitored vectorized env for SB3 PPO."""

    def _make_single_env():
        env = make_env_continuous(env_name)
        return Monitor(env)

    if check:
        raw_env = make_env_continuous(env_name)
        check_env(raw_env, warn=True)

    vec_env = DummyVecEnv([_make_single_env])
    return vec_env



def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on the continuous-action flu vaccine environment."
    )
    parser.add_argument(
        "--env",
        choices=["baseline", "realistic"],
        default="baseline",
        help="Environment preset to use.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO rollout steps per update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Stable-Baselines3 environment checks before training.",
    )
    parser.add_argument(
        "--show-final-episode",
        action="store_true",
        help="Print one deterministic PPO episode after training.",
    )
    args = parser.parse_args()

    env_name = args.env
    save_name = f"flu_rl_model_continuous_{env_name}"
    save_zip = f"{save_name}.zip"


    preview_env = make_env_continuous(env_name)
    print(
        f"\nContinuous environment preset: {env_name}"
        f"\nLead time: {preview_env.lead_time_weeks} week(s)"
        f"\nDemand noise std: {preview_env.demand_noise_std}"
        f"\nCatastrophic spike: {preview_env.use_catastrophic_spike}"
        f" (p={preview_env.catastrophic_spike_prob}, x{preview_env.catastrophic_spike_multiplier})"
    )
    print(
        f"Storage capacity: "
        f"Region 0={preview_env.storage_capacity[0]:.0f}  "
        f"Region 1={preview_env.storage_capacity[1]:.0f}  "
        f"Region 2={preview_env.storage_capacity[2]:.0f}"
    )
    print(
        f"Vulnerability weights: "
        f"Region 0={preview_env.vulnerability_weights[0]:.1f}x  "
        f"Region 1={preview_env.vulnerability_weights[1]:.1f}x  "
        f"Region 2={preview_env.vulnerability_weights[2]:.1f}x"
    )
    print(
        f"Action range per region: normalised "
        f"[{preview_env.action_space.low[0]:.0f}, {preview_env.action_space.high[0]:.0f}] scaled internally to doses"
    )
    print(f"Observation shape: {preview_env.observation_space.shape}")

    env = build_env(env_name, check=args.check_env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        verbose=1,
        device=args.device,
        tensorboard_log=f"./ppo_tensorboard/continuous_{env_name}/",
    )

    callbacks = [RewardLoggerCallback(print_freq=10_000)]
    if args.show_final_episode:
        callbacks.append(EpisodeTraceCallback(env_name))

    print(
        f"\nTraining PPO continuous agent for {args.timesteps:,} timesteps..."
        f"\nModel save path: {save_zip}\n"
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=f"ppo_flu_vaccine_continuous_{env_name}",
    )
    model.save(save_name)

    print(f"\nSaved PPO model to: {save_zip}")
    print("Next step:")
    print(f"  python evaluate_continuous.py --env {env_name}")


if __name__ == "__main__":
    main()
