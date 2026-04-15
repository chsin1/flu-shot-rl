"""
evaluate.py — Full policy comparison for Flu Vaccine Resource Management

Supports multiple environment presets:
  - baseline
  - realistic

Runs policy comparisons over 100 episodes and prints:
  1. Full policy comparison
  2. Reward breakdown
  3. Stockout by region
"""

import argparse
import os
import pickle

import numpy as np
from stable_baselines3 import PPO

from envs.make_env import make_env


# ---------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------

def run_episodes(env, policy_fn, label, n_episodes=100):
    """Run n_episodes of a policy and collect summary metrics."""
    rewards = []
    vaccinated_list = []
    stockout_list = []
    expired_list = []
    weighted_penalty_list = []
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

        ep_vacc = 0.0
        ep_wtd = 0.0
        ep_exp = 0.0
        ep_stor = 0.0

        while not done:
            action = policy_fn(obs, info, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_vaccinated += info["vaccinated"].sum()
            total_stockout += info["stockout"].sum()
            total_expired += info["expired"].sum()
            total_wtd += info["weighted_stockout_penalty"]
            stockout_per_region += info["stockout"]

            ep_vacc += info["vaccinated"].sum()
            ep_wtd += info["weighted_stockout_penalty"]
            ep_exp += info["expired"].sum() * 1.5
            ep_stor += info["inventory"].sum() * 0.01

        rewards.append(total_reward)
        vaccinated_list.append(total_vaccinated)
        stockout_list.append(total_stockout)
        expired_list.append(total_expired)
        weighted_penalty_list.append(total_wtd)

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
        "stockout_per_region": stockout_per_region / n_episodes,
        "breakdown": {k: float(np.mean(v)) for k, v in reward_breakdown.items()},
    }


# ---------------------------------------------------------------------
# Learned policies
# ---------------------------------------------------------------------

def make_ppo_policy(model):
    """PPO policy loaded from Stable-Baselines3."""
    def policy(obs, info, env):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def make_qlearning_policy(q_table, all_actions):
    """
    Q-Learning policy using the same 324-state discretisation as qlearning.py.

    This abstraction is most appropriate for the baseline environment.
    """
    inv_thresholds = [150, 350]

    def discretise(v):
        if v < inv_thresholds[0]:
            return 0
        if v < inv_thresholds[1]:
            return 1
        return 2

    def state_to_index(obs):
        i0 = discretise(obs[0])
        i1 = discretise(obs[1])
        i2 = discretise(obs[2])
        week = min(int(obs[-1]), 11)
        return i0 * 3 * 3 * 12 + i1 * 3 * 12 + i2 * 12 + week

    def policy(obs, info, env):
        s = state_to_index(obs)
        action_idx = int(np.argmax(q_table[s]))
        return list(all_actions[action_idx])

    return policy


def make_vi_policy(vi_policy_array, all_actions):
    """
    Value Iteration policy using the same 324-state abstraction as vi_solver.py.

    This abstraction is most appropriate for the baseline environment.
    """
    inv_thresholds = [150, 350]

    def discretise(v):
        if v < inv_thresholds[0]:
            return 0
        if v < inv_thresholds[1]:
            return 1
        return 2

    def encode_state(i0, i1, i2, week):
        return (((i0 * 3 + i1) * 3 + i2) * 12 + week)

    def policy(obs, info, env):
        i0 = discretise(obs[0])
        i1 = discretise(obs[1])
        i2 = discretise(obs[2])
        week = min(int(obs[-1]), 11)

        state_idx = encode_state(i0, i1, i2, week)
        action_idx = int(vi_policy_array[state_idx])
        return list(all_actions[action_idx])

    return policy


# ---------------------------------------------------------------------
# Rule-based and naive policies
# ---------------------------------------------------------------------

def fixed_300_policy(obs, info, env):
    """Always order 300 doses per region."""
    return np.array([2, 2, 2])


def fixed_100_policy(obs, info, env):
    """Always order 100 doses per region."""
    return np.array([1, 1, 1])


def random_policy(obs, info, env):
    """Random order quantity per region."""
    return env.action_space.sample()


def reorder_point_policy(obs, info, env):
    """
    Reorder point policy:
    order 300 when inventory drops below 200, else order 0.
    """
    inventory = info.get("inventory", np.array([300, 300, 300], dtype=np.float32))
    return np.array([2 if inventory[i] < 200 else 0 for i in range(env.num_regions)])


def seasonal_schedule_policy(obs, info, env):
    """
    Pre-planned schedule aligned to the flu curve.

    Week index uses obs[-1].
    """
    schedule = [1, 1, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0]
    week_idx = min(int(obs[-1]), 11)
    order = schedule[week_idx]
    return np.array([order, order, order])


def vulnerability_first_policy(obs, info, env):
    """
    Prioritise the highest-vulnerability region when stock is tight.
    """
    inventory = info.get("inventory", np.array([300, 300, 300], dtype=np.float32))
    priority = np.argsort(env.vulnerability_weights)[::-1]
    action = np.array([1, 1, 1])

    for rank, region in enumerate(priority):
        if inventory[region] < 150:
            action[region] = 3 if rank == 0 else (2 if rank == 1 else 1)
        elif inventory[region] < 250:
            action[region] = 2 if rank == 0 else 1
        else:
            action[region] = 0 if rank == 2 else 1

    return action


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_qlearning(prefix="flu_ql"):
    """
    Load Q-table and action list saved by qlearning.py.

    Expected files:
      <prefix>_qtable.npy
      <prefix>_policy.pkl
    """
    qtable_path = f"{prefix}_qtable.npy"
    policy_path = f"{prefix}_policy.pkl"

    if not os.path.exists(qtable_path) or not os.path.exists(policy_path):
        return None, None

    q_table = np.load(qtable_path)
    with open(policy_path, "rb") as f:
        _ = pickle.load(f)

    all_actions = [
        (a0, a1, a2)
        for a0 in range(4)
        for a1 in range(4)
        for a2 in range(4)
    ]
    return q_table, all_actions


def load_vi(prefix="flu_vi"):
    """
    Load Value Iteration policy saved by vi_solver.py.

    Expected file:
      <prefix>_policy.pkl
    """
    policy_path = f"{prefix}_policy.pkl"
    if not os.path.exists(policy_path):
        return None

    with open(policy_path, "rb") as f:
        vi_policy = pickle.load(f)
    return vi_policy


# ---------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------

def print_table(results, n_episodes, env_name):
    col = 36

    print(f"\n{'=' * 108}")
    print(f"  FULL POLICY COMPARISON ({n_episodes} episodes, mean ± std) — env={env_name}")
    print(f"{'=' * 108}")
    print(f"{'Policy':<{col}} {'Type':<14} {'Reward':>16} {'Vaccinated':>14} {'Stockout':>12} {'Expired':>12}")
    print(f"{'-' * 108}")

    type_map = {
        "PPO": "Learned",
        "Q-Learning": "Learned",
        "Value Iteration": "Classical DP",
        "Always order 300": "Naive",
        "Always order 100": "Naive",
        "Random": "Naive",
        "Reorder point": "Rule-based",
        "Seasonal schedule": "Rule-based",
        "Vulnerability first": "Rule-based",
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
            f"{r['expired_mean']:>5.1f} ±{r['expired_std']:>5.1f}"
            f"{marker}"
        )
    print(f"{'=' * 108}")

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
        rule_labels = ["Reorder point", "Seasonal schedule", "Vulnerability first"]
        human = [r for r in results if any(lb in r["label"] for lb in rule_labels)]
        print("\nRL Agent (PPO) vs human-designed rules:")
        for h in human:
            diff = rl["reward_mean"] - h["reward_mean"]
            direction = "better" if diff > 0 else "worse"
            print(f"  vs {h['label']:<42} {direction} by {abs(diff):.1f} pts")

        classical = [r for r in results if any(lb in r["label"] for lb in ["Q-Learning", "Value Iteration"])]
        if classical:
            print("\nRL Agent (PPO) vs classical RL / DP methods:")
            for c in classical:
                diff = rl["reward_mean"] - c["reward_mean"]
                direction = "better" if diff > 0 else "worse"
                print(f"  vs {c['label']:<42} {direction} by {abs(diff):.1f} pts")


def show_policy_behavior(policy_fn, env, label="Policy Behavior"):
    """Show one episode of order decisions for quick sanity checking."""
    action_map = {0: 0, 1: 100, 2: 300, 3: 600}

    obs, info = env.reset(seed=123)
    done = False
    week = 0

    print(f"\n=== {label} (1 Episode) ===")
    while not done:
        action = policy_fn(obs, info, env)
        mapped_action = [action_map[int(a)] for a in action]

        print(
            f"Week {week + 1}: "
            f"Orders={mapped_action}, "
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
        description="Evaluate all policies on the flu vaccine environment."
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
    env = make_env(env_name)

    print(
        f"\nEnvironment preset: {env_name}"
        f"\nLead time: {env.lead_time_weeks} week(s)"
        f"\nDemand noise std: {env.demand_noise_std}"
        f"\nCatastrophic spike: {env.use_catastrophic_spike}"
        f" (p={env.catastrophic_spike_prob}, x{env.catastrophic_spike_multiplier})"
    )
    print(
        f"Vulnerability weights: "
        f"Region 0={env.vulnerability_weights[0]:.1f}x  "
        f"Region 1={env.vulnerability_weights[1]:.1f}x  "
        f"Region 2={env.vulnerability_weights[2]:.1f}x"
    )

    if env_name == "realistic":
        print(
            "\nNote: Q-Learning and Value Iteration still use the simplified "
            "324-state discretisation, so their comparison is most defensible "
            "on the baseline environment."
        )

    policy_configs = []

    # PPO
    ppo_model_path = f"flu_rl_model_{env_name}.zip"
    if os.path.exists(ppo_model_path):
        ppo_model = PPO.load(f"flu_rl_model_{env_name}")
        ppo_policy = make_ppo_policy(ppo_model)
        policy_configs.append((ppo_policy, "RL Agent (PPO)"))
        print(f"  Loaded: PPO model ({ppo_model_path})")

        if args.show_ppo_behavior:
            behavior_env = make_env(env_name)
            show_policy_behavior(ppo_policy, behavior_env, "PPO Policy Behavior")
    else:
        print(f"  Skipped: {ppo_model_path} not found — run train.py --env {env_name} first")

    # Q-Learning
    q_prefix = "flu_ql" if env_name == "baseline" else f"flu_ql_{env_name}"
    q_table, all_actions = load_qlearning(q_prefix)
    if q_table is not None:
        policy_configs.append((make_qlearning_policy(q_table, all_actions), "Q-Learning (scratch)"))
        print(f"  Loaded: Q-Learning Q-table ({q_prefix})")
    else:
        print(f"  Skipped: {q_prefix}_qtable.npy not found")

    # Value Iteration
    vi_prefix = "flu_vi" if env_name == "baseline" else f"flu_vi_{env_name}"
    vi_policy = load_vi(vi_prefix)
    if vi_policy is not None:
        all_actions_vi = [
            (a0, a1, a2)
            for a0 in range(4)
            for a1 in range(4)
            for a2 in range(4)
        ]
        policy_configs.append((make_vi_policy(vi_policy, all_actions_vi), "Value Iteration (DP)"))
        print(f"  Loaded: Value Iteration policy ({vi_prefix})")
    else:
        print(f"  Skipped: {vi_prefix}_policy.pkl not found")

    # Rule-based and naive policies
    policy_configs += [
        (fixed_300_policy, "Always order 300"),
        (fixed_100_policy, "Always order 100"),
        (random_policy, "Random"),
        (reorder_point_policy, "Reorder point (<200→300)"),
        (seasonal_schedule_policy, "Seasonal schedule"),
        (vulnerability_first_policy, "Vulnerability first (NACI)"),
    ]

    if not policy_configs:
        raise RuntimeError("No policies available to evaluate.")

    print(f"\nRunning {n_episodes} episodes per policy...\n")

    results = []
    for policy_fn, label in policy_configs:
        print(f"  Evaluating: {label}...")
        eval_env = make_env(env_name)
        results.append(run_episodes(eval_env, policy_fn, label, n_episodes))

    print_table(results, n_episodes, env_name)

    best = max(results, key=lambda x: x["reward_mean"])
    print(
        f"\nBest policy by avg reward: {best['label']} "
        f"({best['reward_mean']:.1f} ± {best['reward_std']:.1f})"
    )


if __name__ == "__main__":
    main()