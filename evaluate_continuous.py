"""
evaluate_continuous.py — Full policy comparison for continuous-action
Flu Vaccine Resource Management.

Supports multiple environment presets:
  - baseline
  - realistic

Runs policy comparisons over N episodes and prints:
  1. Full policy comparison
  2. Reward breakdown
  3. Stockout by region

Expected PPO model files:
  flu_rl_model_continuous_baseline.zip
  flu_rl_model_continuous_realistic.zip

Typical usage:
  python evaluate_continuous.py --env baseline
  python evaluate_continuous.py --env realistic
  python evaluate_continuous.py --env baseline --episodes 100 --show-ppo-behavior
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from envs.flu_env_continuous import FluVaccineEnvContinuous

try:
    from envs.configs import BASELINE_CONFIG, REALISTIC_CONFIG
except ImportError:
    # Fallback for projects that only defined BASELINE_CONFIG so far.
    from envs.configs import BASELINE_CONFIG
    REALISTIC_CONFIG = None


# ---------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------


def make_continuous_env(env_name: str) -> FluVaccineEnvContinuous:
    """Instantiate the continuous environment for a named preset."""
    if env_name == "baseline":
        return FluVaccineEnvContinuous(BASELINE_CONFIG)

    if env_name == "realistic":
        if REALISTIC_CONFIG is None:
            raise RuntimeError(
                "REALISTIC_CONFIG is not available in envs.configs. "
                "Add it first, or run with --env baseline."
            )
        return FluVaccineEnvContinuous(REALISTIC_CONFIG)

    raise ValueError(f"Unknown env preset: {env_name}")


# ---------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------


def run_episodes(env, policy_fn: Callable, label: str, n_episodes: int = 100) -> Dict:
    """Run n_episodes of a policy and collect summary metrics."""
    rewards: List[float] = []
    vaccinated_list: List[float] = []
    stockout_list: List[float] = []
    expired_list: List[float] = []
    weighted_penalty_list: List[float] = []
    avg_order_qty_list: List[float] = []
    stockout_per_region = np.zeros(env.num_regions, dtype=np.float64)

    reward_breakdown = {
        "vaccinated": [],
        "wtd_stockout": [],
        "expired": [],
        "storage": [],
    }

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False

        total_reward = 0.0
        total_vaccinated = 0.0
        total_stockout = 0.0
        total_expired = 0.0
        total_wtd = 0.0
        total_order_qty = 0.0
        total_steps = 0

        ep_vacc = 0.0
        ep_wtd = 0.0
        ep_exp = 0.0
        ep_stor = 0.0

        while not done:
            action = policy_fn(obs, info, env)
            action = np.asarray(action, dtype=np.float32).reshape(env.num_regions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_vaccinated += float(info["vaccinated"].sum())
            total_stockout += float(info["stockout"].sum())
            total_expired += float(info["expired"].sum())
            total_wtd += float(info["weighted_stockout_penalty"])
            stockout_per_region += info["stockout"]
            total_steps += 1

            if "orders_placed" in info:
                total_order_qty += float(np.sum(info["orders_placed"]))
            elif "requested_orders" in info:
                total_order_qty += float(np.sum(info["requested_orders"]))
            else:
                total_order_qty += float(np.sum(action))

            ep_vacc += float(info["vaccinated"].sum())
            ep_wtd += float(info["weighted_stockout_penalty"])
            ep_exp += float(info["expired"].sum()) * 1.5
            ep_stor += float(info["inventory"].sum()) * 0.01

        rewards.append(total_reward)
        vaccinated_list.append(total_vaccinated)
        stockout_list.append(total_stockout)
        expired_list.append(total_expired)
        weighted_penalty_list.append(total_wtd)
        avg_order_qty_list.append(total_order_qty / max(total_steps, 1))

        reward_breakdown["vaccinated"].append(ep_vacc)
        reward_breakdown["wtd_stockout"].append(ep_wtd)
        reward_breakdown["expired"].append(ep_exp)
        reward_breakdown["storage"].append(ep_stor)

    return {
        "label": label,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "vaccinated_mean": float(np.mean(vaccinated_list)),
        "vaccinated_std": float(np.std(vaccinated_list)),
        "stockout_mean": float(np.mean(stockout_list)),
        "stockout_std": float(np.std(stockout_list)),
        "expired_mean": float(np.mean(expired_list)),
        "expired_std": float(np.std(expired_list)),
        "wtd_penalty_mean": float(np.mean(weighted_penalty_list)),
        "wtd_penalty_std": float(np.std(weighted_penalty_list)),
        "avg_order_qty_mean": float(np.mean(avg_order_qty_list)),
        "avg_order_qty_std": float(np.std(avg_order_qty_list)),
        "stockout_per_region": stockout_per_region / n_episodes,
        "breakdown": {k: float(np.mean(v)) for k, v in reward_breakdown.items()},
    }


# ---------------------------------------------------------------------
# Learned policy
# ---------------------------------------------------------------------


def make_ppo_policy(model):
    """PPO policy loaded from Stable-Baselines3."""

    def policy(obs, info, env):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(env.num_regions)
        return action

    return policy


# ---------------------------------------------------------------------
# Continuous rule-based and naive policies
# ---------------------------------------------------------------------


def fixed_300_policy(obs, info, env):
    """Always order 300 doses per region."""
    return np.array([300.0, 300.0, 300.0], dtype=np.float32)



def fixed_100_policy(obs, info, env):
    """Always order 100 doses per region."""
    return np.array([100.0, 100.0, 100.0], dtype=np.float32)



def random_policy(obs, info, env):
    """Random continuous order quantity per region."""
    return env.action_space.sample().astype(np.float32)



def reorder_point_policy(obs, info, env):
    """
    Reorder point policy:
    order 300 when inventory drops below 200, else order 0.
    """
    inventory = info.get("inventory", np.array([300.0, 300.0, 300.0], dtype=np.float32))
    return np.array(
        [300.0 if inventory[i] < 200.0 else 0.0 for i in range(env.num_regions)],
        dtype=np.float32,
    )



def seasonal_schedule_policy(obs, info, env):
    """
    Pre-planned schedule aligned to the flu curve.

    Week index uses obs[-1].
    """
    schedule = [100.0, 100.0, 300.0, 600.0, 600.0, 300.0, 300.0, 100.0, 100.0, 0.0, 0.0, 0.0]
    week_idx = min(int(obs[-1]), 11)
    order = schedule[week_idx]
    return np.array([order, order, order], dtype=np.float32)



def vulnerability_first_policy(obs, info, env):
    """
    Prioritise the highest-vulnerability region when stock is tight.

    Keeps the same spirit as the discrete version, but outputs dose amounts.
    """
    inventory = info.get("inventory", np.array([300.0, 300.0, 300.0], dtype=np.float32))
    priority = np.argsort(env.vulnerability_weights)[::-1]
    action = np.array([100.0, 100.0, 100.0], dtype=np.float32)

    for rank, region in enumerate(priority):
        if inventory[region] < 150.0:
            action[region] = 600.0 if rank == 0 else (300.0 if rank == 1 else 100.0)
        elif inventory[region] < 250.0:
            action[region] = 300.0 if rank == 0 else 100.0
        else:
            action[region] = 0.0 if rank == 2 else 100.0

    return action.astype(np.float32)



def capacity_fill_policy(obs, info, env):
    """
    Top up toward each region's storage capacity, capped by max order quantity.

    This is a useful continuous baseline because it directly uses the storage signal.
    """
    inventory = info.get("inventory", np.array([300.0, 300.0, 300.0], dtype=np.float32))
    gap = np.maximum(env.storage_capacity - inventory, 0.0)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    return np.minimum(gap, high).astype(np.float32)


# ---------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------


def print_table(results: List[Dict], n_episodes: int, env_name: str):
    col = 36

    print(f"\n{'=' * 122}")
    print(f"  CONTINUOUS POLICY COMPARISON ({n_episodes} episodes, mean ± std) — env={env_name}")
    print(f"{'=' * 122}")
    print(
        f"{'Policy':<{col}} {'Type':<14} {'Reward':>16} {'Vaccinated':>14} "
        f"{'Stockout':>12} {'Expired':>12} {'Avg Order':>12}"
    )
    print(f"{'-' * 122}")

    type_map = {
        "PPO": "Learned",
        "Always order 300": "Naive",
        "Always order 100": "Naive",
        "Random": "Naive",
        "Reorder point": "Rule-based",
        "Seasonal schedule": "Rule-based",
        "Vulnerability first": "Rule-based",
        "Capacity fill": "Rule-based",
    }

    best_reward = max(r["reward_mean"] for r in results)

    for r in sorted(results, key=lambda x: x["reward_mean"], reverse=True):
        ptype = next((v for k, v in type_map.items() if k in r["label"]), "")
        marker = " <-- BEST" if abs(r["reward_mean"] - best_reward) < 1e-9 else ""
        print(
            f"{r['label']:<{col}} "
            f"{ptype:<14} "
            f"{r['reward_mean']:>7.1f} ±{r['reward_std']:>6.1f} "
            f"{r['vaccinated_mean']:>6.1f} ±{r['vaccinated_std']:>5.1f} "
            f"{r['stockout_mean']:>5.1f} ±{r['stockout_std']:>5.1f} "
            f"{r['expired_mean']:>5.1f} ±{r['expired_std']:>5.1f} "
            f"{r['avg_order_qty_mean']:>6.1f} ±{r['avg_order_qty_std']:>5.1f}"
            f"{marker}"
        )
    print(f"{'=' * 122}")

    print(f"\n{'=' * 88}")
    print("  REWARD BREAKDOWN (avg per episode)")
    print("  reward = +vaccinated − wtd_stockout − expired_cost − storage_cost")
    print(f"{'=' * 88}")
    print(f"{'Policy':<{col}} {'Vaccinated':>12} {'Wtd Stockout':>14} {'Expired Cost':>14} {'Storage':>10}")
    print(f"{'-' * 88}")
    for r in sorted(results, key=lambda x: x["reward_mean"], reverse=True):
        bd = r["breakdown"]
        print(
            f"{r['label']:<{col}} "
            f"{bd['vaccinated']:>12.1f} "
            f"{bd['wtd_stockout']:>14.1f} "
            f"{bd['expired']:>14.1f} "
            f"{bd['storage']:>10.1f}"
        )
    print(f"{'=' * 88}")

    print(f"\n{'=' * 78}")
    print("  STOCKOUT BY REGION (avg doses/episode)")
    print(f"{'=' * 78}")
    print(f"{'Policy':<{col}} {'R0':>10} {'R1':>10} {'R2':>10}")
    print(f"{'':>{col}} {'(urban)':>10} {'(town)':>10} {'(rural)':>10}")
    print(f"{'-' * 78}")
    for r in sorted(results, key=lambda x: x["reward_mean"], reverse=True):
        sr = r["stockout_per_region"]
        print(f"{r['label']:<{col}} {sr[0]:>10.1f} {sr[1]:>10.1f} {sr[2]:>10.1f}")
    print(f"{'=' * 78}")

    rl = next((r for r in results if "PPO" in r["label"]), None)
    if rl:
        rule_labels = [
            "Reorder point",
            "Seasonal schedule",
            "Vulnerability first",
            "Capacity fill",
        ]
        human = [r for r in results if any(lb in r["label"] for lb in rule_labels)]
        print("\nRL Agent (PPO) vs hand-designed continuous rules:")
        for h in human:
            diff = rl["reward_mean"] - h["reward_mean"]
            direction = "better" if diff > 0 else "worse"
            print(f"  vs {h['label']:<42} {direction} by {abs(diff):.1f} pts")



def show_policy_behavior(policy_fn: Callable, env, label: str = "Policy Behavior"):
    """Show one episode of order decisions for quick sanity checking."""
    obs, info = env.reset(seed=123)
    done = False
    week = 0

    print(f"\n=== {label} (1 Episode) ===")
    while not done:
        action = np.asarray(policy_fn(obs, info, env), dtype=np.float32).reshape(env.num_regions)
        print(
            f"Week {week + 1}: "
            f"Orders={np.round(action, 1)}, "
            f"Inventory={np.round(obs[:3], 1)}"
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        week += 1


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate continuous-action policies on the flu vaccine environment."
    )
    parser.add_argument(
        "--env",
        choices=["baseline", "realistic"],
        default="baseline",
        help="Environment preset to use.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per policy.",
    )
    parser.add_argument(
        "--show-ppo-behavior",
        action="store_true",
        help="Print one PPO episode trajectory for sanity checking.",
    )
    args = parser.parse_args()

    env_name = args.env
    n_episodes = args.episodes
    env = make_continuous_env(env_name)

    print("Eval action space:", env.action_space)


    print(
        f"\nContinuous environment preset: {env_name}"
        f"\nLead time: {env.lead_time_weeks} week(s)"
        f"\nDemand noise std: {env.demand_noise_std}"
        f"\nCatastrophic spike: {env.use_catastrophic_spike}"
        f" (p={env.catastrophic_spike_prob}, x{env.catastrophic_spike_multiplier})"
    )
    print(
        f"Storage capacity: "
        f"Region 0={env.storage_capacity[0]:.0f}  "
        f"Region 1={env.storage_capacity[1]:.0f}  "
        f"Region 2={env.storage_capacity[2]:.0f}"
    )
    print(
        f"Vulnerability weights: "
        f"Region 0={env.vulnerability_weights[0]:.1f}x  "
        f"Region 1={env.vulnerability_weights[1]:.1f}x  "
        f"Region 2={env.vulnerability_weights[2]:.1f}x"
    )
    print(
        f"Action range per region: "
        f"[{env.action_space.low[0]:.0f}, {env.action_space.high[0]:.0f}] doses"
    )

    policy_configs: List[Tuple[Callable, str]] = []

    # PPO
    ppo_model_path = f"flu_rl_model_continuous_{env_name}.zip"
    if os.path.exists(ppo_model_path):
        ppo_model = PPO.load(f"flu_rl_model_continuous_{env_name}")
        ppo_policy = make_ppo_policy(ppo_model)
        policy_configs.append((ppo_policy, "RL Agent (PPO continuous)"))
        print(f"  Loaded: PPO model ({ppo_model_path})")

        if args.show_ppo_behavior:
            behavior_env = make_continuous_env(env_name)
            show_policy_behavior(ppo_policy, behavior_env, "PPO Continuous Policy Behavior")
    else:
        print(
            f"  Skipped: {ppo_model_path} not found — "
            f"run train_continuous.py --env {env_name} first"
        )

    # Rule-based and naive policies
    policy_configs += [
        (fixed_300_policy, "Always order 300"),
        (fixed_100_policy, "Always order 100"),
        (random_policy, "Random"),
        (reorder_point_policy, "Reorder point (<200→300)"),
        (seasonal_schedule_policy, "Seasonal schedule"),
        (vulnerability_first_policy, "Vulnerability first (NACI)"),
        (capacity_fill_policy, "Capacity fill"),
    ]

    if not policy_configs:
        raise RuntimeError("No policies available to evaluate.")

    print(f"\nRunning {n_episodes} episodes per policy...\n")

    results = []
    for policy_fn, label in policy_configs:
        print(f"  Evaluating: {label}...")
        eval_env = make_continuous_env(env_name)
        results.append(run_episodes(eval_env, policy_fn, label, n_episodes))

    print_table(results, n_episodes, env_name)

    best = max(results, key=lambda x: x["reward_mean"])
    print(
        f"\nBest policy by avg reward: {best['label']} "
        f"({best['reward_mean']:.1f} ± {best['reward_std']:.1f})"
    )


if __name__ == "__main__":
    main()
