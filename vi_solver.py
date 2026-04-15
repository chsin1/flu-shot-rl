"""
Approximate Value Iteration baseline for the flu vaccine environment.

This solver uses a compact 324-state abstraction:
  3 inventory buckets per region × 12 weeks = 324 states

Supports:
  - baseline environment
  - realistic environment

Important note:
Value Iteration is most defensible on the baseline environment because
it relies on a simplified discretised abstraction. It can still be run
on the realistic preset, but results should be interpreted as a coarse
planning approximation rather than a full exact solution.
"""

import argparse
import pickle

import numpy as np

from envs.make_env import make_env


# ---------------------------------------------------------------------
# Constants for compact state abstraction
# ---------------------------------------------------------------------

GAMMA = 0.99
N_WEEKS_TOTAL = 12

INV_THRESHOLDS = [150, 350]
N_INV_LEVELS = 3
N_STATES = (N_INV_LEVELS ** 3) * N_WEEKS_TOTAL   # 324

REPRESENTATIVE_INVENTORY = [75.0, 250.0, 500.0]

ALL_ACTIONS = [
    (a0, a1, a2)
    for a0 in range(4)
    for a1 in range(4)
    for a2 in range(4)
]
N_ACTIONS = len(ALL_ACTIONS)


# ---------------------------------------------------------------------
# State abstraction helpers
# ---------------------------------------------------------------------

def discretise_inventory(inv_value: float) -> int:
    """Map continuous inventory to 0=low, 1=medium, 2=high."""
    if inv_value < INV_THRESHOLDS[0]:
        return 0
    if inv_value < INV_THRESHOLDS[1]:
        return 1
    return 2


def state_to_index(obs: np.ndarray) -> int:
    """
    Convert observation to compact discrete index using:
      - inventory in each region
      - week
    """
    i0 = discretise_inventory(obs[0])
    i1 = discretise_inventory(obs[1])
    i2 = discretise_inventory(obs[2])
    week = min(int(obs[-1]), N_WEEKS_TOTAL - 1)

    return (
        i0 * N_INV_LEVELS * N_INV_LEVELS * N_WEEKS_TOTAL
        + i1 * N_INV_LEVELS * N_WEEKS_TOTAL
        + i2 * N_WEEKS_TOTAL
        + week
    )


def index_to_state_components(state_idx: int):
    """
    Recover (i0, i1, i2, week) from flattened state index.
    """
    week = state_idx % N_WEEKS_TOTAL
    rem = state_idx // N_WEEKS_TOTAL
    i2 = rem % N_INV_LEVELS
    rem //= N_INV_LEVELS
    i1 = rem % N_INV_LEVELS
    i0 = rem // N_INV_LEVELS
    return i0, i1, i2, week


def representative_inventory(state_idx: int) -> np.ndarray:
    """
    Map discrete inventory buckets to representative continuous levels.
    """
    i0, i1, i2, _ = index_to_state_components(state_idx)
    return np.array(
        [
            REPRESENTATIVE_INVENTORY[i0],
            REPRESENTATIVE_INVENTORY[i1],
            REPRESENTATIVE_INVENTORY[i2],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------
# Environment initialisation from abstract state
# ---------------------------------------------------------------------

def initialise_env_from_state(env_name: str, state_idx: int, seed: int):
    """
    Build a fresh env instance and approximately initialise it from a
    compact discrete state.

    We set:
      - week
      - inventory
      - expiry tracker
      - demand level (approx seasonal baseline)

    For realistic environments with lead time, we initialise the pipeline
    to zeros so that planning remains consistent but approximate.
    """
    env = make_env(env_name)

    _, _, _, week = index_to_state_components(state_idx)
    inventory = representative_inventory(state_idx)

    env.reset(seed=seed)
    env.week = week
    env.inventory = inventory.copy()

    # Approximate mixed-age inventory by spreading doses across buckets
    env.expiry_tracker = np.zeros((env.expiry_window, env.num_regions), dtype=np.float32)
    env.expiry_tracker[:] = inventory / env.expiry_window

    # If lead time exists, initialise pipeline as empty
    if env.lead_time_weeks > 0:
        env.pipeline_orders = np.zeros(
            (env.lead_time_weeks, env.num_regions),
            dtype=np.float32,
        )

    # Approximate current demand using seasonal baseline (no noise here)
    seasonal_multiplier = env._seasonal_demand()
    # We do not want a stochastic call to overwrite demand directly here,
    # so instead reset demand to the already generated value from reset.
    # Then approximate week-specific demand using a fresh reset pattern.
    # Simpler and more stable approach: preserve env.demand_level from reset
    # if week == 0, otherwise use one extra seasonal sample.
    env.demand_level = seasonal_multiplier.copy()

    return env


# ---------------------------------------------------------------------
# Transition estimation
# ---------------------------------------------------------------------

def estimate_action_value(env_name: str, state_idx: int, action_idx: int, transition_samples: int):
    """
    Estimate expected immediate reward and transition probabilities for a
    given abstract state-action pair by Monte Carlo sampling from the real env.
    """
    _, _, _, week = index_to_state_components(state_idx)

    rewards = []
    transition_counts = {}

    for sample in range(transition_samples):
        env = initialise_env_from_state(
            env_name=env_name,
            state_idx=state_idx,
            seed=1000 * state_idx + sample,
        )

        obs_next, reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        rewards.append(reward)

        if terminated or truncated or week == N_WEEKS_TOTAL - 1:
            continue

        next_state = state_to_index(obs_next)
        transition_counts[next_state] = transition_counts.get(next_state, 0) + 1

    reward_mean = float(np.mean(rewards))
    transition_probs = {
        next_state: count / transition_samples
        for next_state, count in transition_counts.items()
    }
    return reward_mean, transition_probs


# ---------------------------------------------------------------------
# Value Iteration
# ---------------------------------------------------------------------

def solve(env_name: str, transition_samples: int, verbose: bool = True):
    """
    Solve the approximate DP problem over the 324-state abstraction.
    """
    values = np.zeros(N_STATES, dtype=np.float32)
    policy = np.zeros(N_STATES, dtype=np.int32)

    transition_cache = {}

    for week in reversed(range(N_WEEKS_TOTAL)):
        if verbose:
            print(f"Solving week {week + 1} / {N_WEEKS_TOTAL}")

        week_states = [
            state for state in range(N_STATES)
            if index_to_state_components(state)[3] == week
        ]

        for state_idx in week_states:
            best_value = -np.inf
            best_action = 0

            for action_idx in range(N_ACTIONS):
                reward_mean, transition_probs = estimate_action_value(
                    env_name=env_name,
                    state_idx=state_idx,
                    action_idx=action_idx,
                    transition_samples=transition_samples,
                )

                transition_cache[(state_idx, action_idx)] = (reward_mean, transition_probs)

                continuation = sum(
                    prob * values[next_state]
                    for next_state, prob in transition_probs.items()
                )
                q_value = reward_mean + GAMMA * continuation

                if q_value > best_value:
                    best_value = q_value
                    best_action = action_idx

            values[state_idx] = best_value
            policy[state_idx] = best_action

    return values, policy, transition_cache


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_policy(env_name: str, policy: np.ndarray, n_episodes: int = 100):
    """
    Evaluate the derived policy in the full environment.
    """
    rewards = []
    vaccinated = []
    stockouts = []
    expired = []

    for episode in range(n_episodes):
        env = make_env(env_name)
        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0.0
        total_vaccinated = 0.0
        total_stockout = 0.0
        total_expired = 0.0

        while not done:
            state_idx = state_to_index(obs)
            action = ALL_ACTIONS[int(policy[state_idx])]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_vaccinated += info["vaccinated"].sum()
            total_stockout += info["stockout"].sum()
            total_expired += info["expired"].sum()

        rewards.append(total_reward)
        vaccinated.append(total_vaccinated)
        stockouts.append(total_stockout)
        expired.append(total_expired)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "vaccinated_mean": float(np.mean(vaccinated)),
        "vaccinated_std": float(np.std(vaccinated)),
        "stockout_mean": float(np.mean(stockouts)),
        "stockout_std": float(np.std(stockouts)),
        "expired_mean": float(np.mean(expired)),
        "expired_std": float(np.std(expired)),
    }


# ---------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------

def save_results(values: np.ndarray, policy: np.ndarray, path_prefix: str):
    """Save value table and greedy policy."""
    np.save(f"{path_prefix}_values.npy", values)
    with open(f"{path_prefix}_policy.pkl", "wb") as handle:
        pickle.dump(policy, handle)

    print(f"Saved values -> {path_prefix}_values.npy")
    print(f"Saved policy -> {path_prefix}_policy.pkl")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Approximate Value Iteration for the flu vaccine environment."
    )
    parser.add_argument(
        "--env",
        choices=["baseline", "realistic"],
        default="baseline",
        help="Environment preset to use.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Transition samples per state-action pair.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes after solving.",
    )
    args = parser.parse_args()

    env_name = args.env

    probe_env = make_env(env_name)

    if env_name == "realistic":
        print(
            "Note: Value Iteration still uses the simplified 324-state abstraction.\n"
            "On the realistic environment, it should be interpreted as a coarse\n"
            "approximate planner rather than a full exact dynamic program."
        )
        print()

    print("=" * 72)
    print(f"Approximate Value Iteration — Flu Vaccine Resource Management ({env_name})")
    print("=" * 72)
    print(f"States:               {N_STATES}")
    print(f"Actions:              {N_ACTIONS}")
    print(f"Transition samples:   {args.samples}")
    print(f"Gamma:                {GAMMA}")
    print(f"Lead time:            {probe_env.lead_time_weeks}")
    print(f"Demand noise std:     {probe_env.demand_noise_std}")
    print("=" * 72)
    print()

    values, policy, _ = solve(
        env_name=env_name,
        transition_samples=args.samples,
        verbose=True,
    )

    save_prefix = f"flu_vi_{env_name}"
    save_results(values, policy, path_prefix=save_prefix)

    print(f"\nEvaluating policy over {args.eval_episodes} episodes...")
    results = evaluate_policy(
        env_name=env_name,
        policy=policy,
        n_episodes=args.eval_episodes,
    )

    print()
    print("=" * 72)
    print(f"VALUE ITERATION RESULTS ({args.eval_episodes} episodes, mean ± std)")
    print("=" * 72)
    print(f"Reward:     {results['reward_mean']:>8.1f} ± {results['reward_std']:.1f}")
    print(f"Vaccinated: {results['vaccinated_mean']:>8.1f} ± {results['vaccinated_std']:.1f}")
    print(f"Stockout:   {results['stockout_mean']:>8.1f} ± {results['stockout_std']:.1f}")
    print(f"Expired:    {results['expired_mean']:>8.1f} ± {results['expired_std']:.1f}")
    print("=" * 72)

    print("\nArtifacts saved:")
    print(f"  {save_prefix}_values.npy")
    print(f"  {save_prefix}_policy.pkl")


if __name__ == "__main__":
    main()