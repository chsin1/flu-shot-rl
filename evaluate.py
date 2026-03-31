"""
evaluate.py — Full policy comparison for Flu Vaccine Resource Management
MMAI-845 Group Project

Runs 100 episodes per policy and prints three comparison tables:
  1. Full policy comparison (reward, vaccinated, stockout, expired)
  2. Reward breakdown (each reward term separately)
  3. Per-region stockout breakdown

Policies evaluated (9 total):
  Learned:      PPO (SB3), Q-Learning (scratch), Value Iteration (classical DP)
  Rule-based:   Reorder point, Seasonal schedule, Vulnerability first
  Naive:        Always order 300, Always order 100, Random
"""

from stable_baselines3 import PPO
from envs.flu_env import FluVaccineEnv, VULNERABILITY_WEIGHT, SEASONAL_CURVE
import numpy as np
import pickle
import os


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episodes(env, policy_fn, label, n_episodes=100):
    """Run n_episodes of a policy and collect metrics."""
    rewards            = []
    vaccinated_list    = []
    stockout_list      = []
    expired_list       = []
    weighted_penalty_list = []
    stockout_per_region   = np.zeros(env.num_regions)
    reward_breakdown   = {"vaccinated": [], "wtd_stockout": [], "expired": [], "storage": []}

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = ep_vacc = ep_wtd = ep_exp = ep_stor = 0
        total_vaccinated = total_stockout = total_expired = total_wtd = 0

        while not done:
            action = policy_fn(obs, info, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward     += reward
            total_vaccinated += info['vaccinated'].sum()
            total_stockout   += info['stockout'].sum()
            total_expired    += info['expired'].sum()
            total_wtd        += info['weighted_stockout_penalty']
            stockout_per_region += info['stockout']

            ep_vacc += info['vaccinated'].sum()
            ep_wtd  += info['weighted_stockout_penalty']
            ep_exp  += info['expired'].sum() * 1.5
            ep_stor += info['inventory'].sum() * 0.01

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
        "label":               label,
        "reward_mean":         np.mean(rewards),
        "reward_std":          np.std(rewards),
        "vaccinated_mean":     np.mean(vaccinated_list),
        "vaccinated_std":      np.std(vaccinated_list),
        "stockout_mean":       np.mean(stockout_list),
        "stockout_std":        np.std(stockout_list),
        "expired_mean":        np.mean(expired_list),
        "expired_std":         np.std(expired_list),
        "wtd_penalty_mean":    np.mean(weighted_penalty_list),
        "wtd_penalty_std":     np.std(weighted_penalty_list),
        "stockout_per_region": stockout_per_region / n_episodes,
        "breakdown":           {k: np.mean(v) for k, v in reward_breakdown.items()},
    }


# ── Policy definitions ────────────────────────────────────────────────────────

# --- Learned policies ---

def make_ppo_policy(model):
    """Policy 1: PPO agent trained via Stable-Baselines3."""
    def policy(obs, info, env):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def make_qlearning_policy(q_table, all_actions):
    """
    Policy 2: Q-Learning — implemented from scratch.
    Loads the trained Q-table and acts greedily.
    State is discretised: inventory → 3 levels, week = obs[-1].
    """
    INV_THRESHOLDS = [150, 350]

    def discretise(v):
        if v < INV_THRESHOLDS[0]:   return 0
        elif v < INV_THRESHOLDS[1]: return 1
        else:                        return 2

    def state_to_index(obs):
        i0   = discretise(obs[0])
        i1   = discretise(obs[1])
        i2   = discretise(obs[2])
        week = min(int(obs[-1]), 11)   # obs[-1] = week, always last element
        return i0 * 3 * 3 * 12 + i1 * 3 * 12 + i2 * 12 + week

    def policy(obs, info, env):
        s          = state_to_index(obs)
        action_idx = int(np.argmax(q_table[s]))
        return list(all_actions[action_idx])

    return policy


def make_vi_policy(vi_policy_array, all_actions):
    """
    Policy 3: Value Iteration — classical DP on discretised state.
    Expects:
      - vi_policy_array: NumPy array of shape (324,)
      - each entry is an action index into all_actions
    """
    INV_THRESHOLDS = [150, 350]

    def discretise(v):
        if v < INV_THRESHOLDS[0]:
            return 0
        elif v < INV_THRESHOLDS[1]:
            return 1
        else:
            return 2

    def encode_state(i0, i1, i2, week):
        """
        Flatten (i0, i1, i2, week) into a single index.
        State count = 3 * 3 * 3 * 12 = 324
        """
        return (((i0 * 3 + i1) * 3 + i2) * 12 + week)

    def policy(obs, info, env):
        i0   = discretise(obs[0])
        i1   = discretise(obs[1])
        i2   = discretise(obs[2])
        week = min(int(obs[-1]), 11)

        state_idx = encode_state(i0, i1, i2, week)

        # action index from VI table
        action_idx = int(vi_policy_array[state_idx])

        # convert action index to actual 3-region action
        action = all_actions[action_idx]

        return list(action)
        print("VI debug:", (i0, i1, i2, week), "->", state_idx, "->", action_idx, "->", action)

    return policy



# --- Rule-based policies ---

def fixed_300_policy(obs, info, env):
    """Policy 5: always order 300 doses per region."""
    return np.array([2, 2, 2])


def fixed_100_policy(obs, info, env):
    """Policy 6: always order 100 doses per region."""
    return np.array([1, 1, 1])


def random_policy(obs, info, env):
    """Policy 7: random order quantity per region."""
    return env.action_space.sample()


def reorder_point_policy(obs, info, env):
    """
    Policy 8: reorder point.
    Orders 300 when inventory drops below 200. Otherwise nothing.
    Classic inventory management threshold rule.
    """
    inventory = info.get("inventory", np.array([300, 300, 300]))
    return np.array([2 if inventory[i] < 200 else 0
                     for i in range(env.num_regions)])


def seasonal_schedule_policy(obs, info, env):
    """
    Policy 9: seasonal schedule.
    Pre-planned order calendar aligned with the 12-week flu curve.
    Orders heavily in weeks 3-5, tapers after week 8.

    Week:   1    2    3    4    5    6    7    8    9   10   11   12
    Doses: 100  100  300  600  600  300  300  100  100    0    0    0
    """
    schedule  = [1, 1, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0]
    week_idx  = min(int(obs[-1]), 11)   # obs[-1] = week, always last element
    order     = schedule[week_idx]
    return np.array([order, order, order])


def vulnerability_first_policy(obs, info, env):
    """
    Policy 10: vulnerability-first allocation.
    Prioritises highest-vulnerability region (Region 1, 2.0x) when
    stock is tight. Grounded in NACI 2024-2025 guidance.

    Priority: Region 1 (2.0x) > Region 2 (1.5x) > Region 0 (1.2x)
    """
    inventory = info.get("inventory", np.array([300, 300, 300]))
    priority  = np.argsort(VULNERABILITY_WEIGHT)[::-1]   # highest vuln first
    action    = np.array([1, 1, 1])

    for rank, region in enumerate(priority):
        if inventory[region] < 150:
            action[region] = 3 if rank == 0 else (2 if rank == 1 else 1)
        elif inventory[region] < 250:
            action[region] = 2 if rank == 0 else 1
        else:
            action[region] = 0 if rank == 2 else 1

    return action


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_qlearning(prefix="flu_ql"):
    """Load Q-table and action list saved by qlearning.py."""
    qtable_path = f"{prefix}_qtable.npy"
    policy_path = f"{prefix}_policy.pkl"
    if not os.path.exists(qtable_path):
        return None, None
    Q = np.load(qtable_path)
    with open(policy_path, "rb") as f:
        _ = pickle.load(f)   # policy dict (unused — we re-derive from Q)
    all_actions = [(a0, a1, a2)
                   for a0 in range(4)
                   for a1 in range(4)
                   for a2 in range(4)]
    return Q, all_actions


def load_vi(prefix="flu_vi"):
    """Load value iteration policy saved by vi_solver.py."""
    policy_path = f"{prefix}_policy.pkl"
    if not os.path.exists(policy_path):
        return None
    with open(policy_path, "rb") as f:
        vi_policy = pickle.load(f)
    return vi_policy


def print_table(results, n_episodes):
    col = 36

    # ── Main comparison ───────────────────────────────────────────────
    print(f"\n{'=' * 105}")
    print(f"  FULL POLICY COMPARISON ({n_episodes} episodes, mean ± std)")
    print(f"{'=' * 105}")
    print(f"{'Policy':<{col}} {'Type':<12} {'Reward':>16} {'Vaccinated':>14} {'Stockout':>12} {'Expired':>12}")
    print(f"{'-' * 105}")

    type_map = {
        "PPO":                    "Learned",
        "Q-Learning":             "Learned",
        "Value Iteration":        "Classical DP",
        "Always order 300":       "Naive",
        "Always order 100":       "Naive",
        "Random":                 "Naive",
        "Reorder point":          "Rule-based",
        "Seasonal schedule":      "Rule-based",
        "Vulnerability first":    "Rule-based",
    }

    best_reward = max(r['reward_mean'] for r in results)
    for r in sorted(results, key=lambda x: x['reward_mean'], reverse=True):
        ptype = next((v for k, v in type_map.items() if k in r['label']), "")
        marker = " <-- BEST" if abs(r['reward_mean'] - best_reward) < 0.01 else ""
        print(
            f"{r['label']:<{col}} "
            f"{ptype:<12} "
            f"{r['reward_mean']:>7.1f} ±{r['reward_std']:>5.1f} "
            f"{r['vaccinated_mean']:>6.1f} ±{r['vaccinated_std']:>4.1f} "
            f"{r['stockout_mean']:>5.1f} ±{r['stockout_std']:>4.1f} "
            f"{r['expired_mean']:>5.1f} ±{r['expired_std']:>4.1f}"
            f"{marker}"
        )
    print(f"{'=' * 105}")

    # ── Reward breakdown ──────────────────────────────────────────────
    print(f"\n{'=' * 85}")
    print(f"  REWARD BREAKDOWN (avg per episode)")
    print(f"  reward = +vaccinated − wtd_stockout − expired_cost − storage_cost")
    print(f"{'=' * 85}")
    print(f"{'Policy':<{col}} {'Vaccinated':>12} {'Wtd Stockout':>14} {'Expired Cost':>14} {'Storage':>10}")
    print(f"{'-' * 85}")
    for r in sorted(results, key=lambda x: x['reward_mean'], reverse=True):
        bd = r['breakdown']
        print(
            f"{r['label']:<{col}} "
            f"{bd['vaccinated']:>12.1f} "
            f"{bd['wtd_stockout']:>14.1f} "
            f"{bd['expired']:>14.1f} "
            f"{bd['storage']:>10.1f}"
        )
    print(f"{'=' * 85}")

    # ── Per-region stockout ───────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print(f"  STOCKOUT BY REGION (avg doses/episode)")
    print(f"{'=' * 76}")
    print(f"{'Policy':<{col}} {'R0 urban':>10} {'R1 town':>10} {'R2 rural':>10}")
    print(f"{'':>{col}} {'(1.2x)':>10} {'(2.0x)':>10} {'(1.5x)':>10}")
    print(f"{'-' * 76}")
    for r in sorted(results, key=lambda x: x['reward_mean'], reverse=True):
        sr = r['stockout_per_region']
        print(f"{r['label']:<{col}} {sr[0]:>10.1f} {sr[1]:>10.1f} {sr[2]:>10.1f}")
    print(f"{'=' * 76}")

    # ── RL vs human-designed rules ────────────────────────────────────
    rl = next((r for r in results if "PPO" in r['label']), None)
    if rl:
        rule_labels = ["Reorder point", "Seasonal schedule", "Vulnerability first"]
        human = [r for r in results if any(lb in r['label'] for lb in rule_labels)]
        print(f"\nRL Agent (PPO) vs human-designed rules:")
        for h in human:
            diff = rl['reward_mean'] - h['reward_mean']
            direction = "better" if diff > 0 else "worse"
            print(f"  vs {h['label']:<42} {direction} by {abs(diff):.1f} pts")

    # ── RL vs classical ───────────────────────────────────────────────
    if rl:
        classical = [r for r in results if any(lb in r['label']
                     for lb in ["Q-Learning", "Value Iteration"])]
        if classical:
            print(f"\nRL Agent (PPO) vs classical RL / DP methods:")
            for c in classical:
                diff = rl['reward_mean'] - c['reward_mean']
                direction = "better" if diff > 0 else "worse"
                print(f"  vs {c['label']:<42} {direction} by {abs(diff):.1f} pts")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    N_EPISODES = 100
    env        = FluVaccineEnv()

    print(f"\nVulnerability weights: "
          f"Region 0={VULNERABILITY_WEIGHT[0]:.1f}x  "
          f"Region 1={VULNERABILITY_WEIGHT[1]:.1f}x  "
          f"Region 2={VULNERABILITY_WEIGHT[2]:.1f}x")

    # ── Load models ───────────────────────────────────────────────────
    policy_configs = []

    # Policy 1: PPO
    if os.path.exists("flu_rl_model.zip"):
        ppo_model = PPO.load("flu_rl_model")
        policy_configs.append((make_ppo_policy(ppo_model), "RL Agent (PPO)"))
        print("  Loaded: PPO model")
    else:
        print("  Skipped: flu_rl_model.zip not found — run train.py first")

    # Policy 2: Q-Learning from scratch
    Q, all_actions = load_qlearning("flu_ql")
    if Q is not None:
        policy_configs.append((make_qlearning_policy(Q, all_actions), "Q-Learning (scratch)"))
        print("  Loaded: Q-Learning Q-table")
    else:
        print("  Skipped: flu_ql_qtable.npy not found — run qlearning.py first")

    # Policy 3: Value Iteration
    vi_policy = load_vi("flu_vi")
    if vi_policy is not None:
        all_actions_vi = [(a0, a1, a2)
                          for a0 in range(4)
                          for a1 in range(4)
                          for a2 in range(4)]
        policy_configs.append((make_vi_policy(vi_policy, all_actions_vi), "Value Iteration (DP)"))
        print("  Loaded: Value Iteration policy")
    else:
        print("  Skipped: flu_vi_policy.pkl not found — run vi_solver.py first")

    # Policies 5-10: rule-based and naive
    policy_configs += [
        (fixed_300_policy,          "Always order 300"),
        (fixed_100_policy,          "Always order 100"),
        (random_policy,             "Random"),
        (reorder_point_policy,      "Reorder point (<200→300)"),
        (seasonal_schedule_policy,  "Seasonal schedule"),
        (vulnerability_first_policy,"Vulnerability first (NACI)"),
    ]

    # ── Run all policies ──────────────────────────────────────────────
    print(f"\nRunning {N_EPISODES} episodes per policy...\n")
    results = []
    for policy_fn, label in policy_configs:
        print(f"  Evaluating: {label}...")
        results.append(run_episodes(env, policy_fn, label, N_EPISODES))

    # ── Print all tables ──────────────────────────────────────────────
    print_table(results, N_EPISODES)

    best = max(results, key=lambda x: x['reward_mean'])
    print(f"\nBest policy by avg reward: {best['label']} "
          f"({best['reward_mean']:.1f} ± {best['reward_std']:.1f})")


if __name__ == "__main__":
    main()