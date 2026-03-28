"""
Approximate value iteration baseline for the flu vaccine environment.

The full environment is continuous and stochastic, so this solver works on a
compact 324-state approximation:
  3 inventory buckets per region × 12 weeks = 324 states

Transitions are estimated by sampling one-step dynamics from the real
environment after mapping a discrete state to a representative inventory
profile. This keeps the baseline aligned with the actual project dynamics
without requiring an exact symbolic model of expiry and stochastic demand.
"""

import pickle

import numpy as np

from envs.flu_env import FluVaccineEnv, REGION_BASE_DEMAND, SEASONAL_CURVE


GAMMA = 0.99
N_WEEKS_TOTAL = 12

INV_THRESHOLDS = [150, 350]
N_INV_LEVELS = 3
N_STATES = (N_INV_LEVELS ** 3) * N_WEEKS_TOTAL

REPRESENTATIVE_INVENTORY = [75.0, 250.0, 500.0]
ALL_ACTIONS = [
    (a0, a1, a2)
    for a0 in range(4)
    for a1 in range(4)
    for a2 in range(4)
]
N_ACTIONS = len(ALL_ACTIONS)

TRANSITION_SAMPLES = 12


def discretise_inventory(inv_value):
    if inv_value < INV_THRESHOLDS[0]:
        return 0
    if inv_value < INV_THRESHOLDS[1]:
        return 1
    return 2


def state_to_index(obs):
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


def index_to_state_components(state_idx):
    week = state_idx % N_WEEKS_TOTAL
    rem = state_idx // N_WEEKS_TOTAL
    i2 = rem % N_INV_LEVELS
    rem //= N_INV_LEVELS
    i1 = rem % N_INV_LEVELS
    i0 = rem // N_INV_LEVELS
    return i0, i1, i2, week


def representative_inventory(state_idx):
    i0, i1, i2, _ = index_to_state_components(state_idx)
    return np.array(
        [
            REPRESENTATIVE_INVENTORY[i0],
            REPRESENTATIVE_INVENTORY[i1],
            REPRESENTATIVE_INVENTORY[i2],
        ],
        dtype=np.float32,
    )


def initialise_env_from_state(env, state_idx, seed):
    _, _, _, week = index_to_state_components(state_idx)
    inventory = representative_inventory(state_idx)

    env.reset(seed=seed)
    env.week = week
    env.inventory = inventory.copy()

    # Distribute doses across age buckets to approximate mixed-age inventory.
    env.expiry_tracker = np.zeros((env.expiry_window, env.num_regions), dtype=np.float32)
    env.expiry_tracker[:] = inventory / env.expiry_window

    demand_week = max(week - 1, 0)
    env.demand_level = (
        REGION_BASE_DEMAND * SEASONAL_CURVE[demand_week]
    ).astype(np.float32)
    return env


def estimate_action_value(state_idx, action_idx):
    _, _, _, week = index_to_state_components(state_idx)

    rewards = []
    transition_counts = {}

    for sample in range(TRANSITION_SAMPLES):
        env = initialise_env_from_state(FluVaccineEnv(), state_idx, seed=1000 * state_idx + sample)
        obs_next, reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
        rewards.append(reward)

        if terminated or truncated or week == N_WEEKS_TOTAL - 1:
            continue

        next_state = state_to_index(obs_next)
        transition_counts[next_state] = transition_counts.get(next_state, 0) + 1

    reward_mean = float(np.mean(rewards))
    transition_probs = {
        next_state: count / TRANSITION_SAMPLES
        for next_state, count in transition_counts.items()
    }
    return reward_mean, transition_probs


def solve():
    values = np.zeros(N_STATES, dtype=np.float32)
    policy = np.zeros(N_STATES, dtype=np.int32)

    transition_cache = {}

    for week in reversed(range(N_WEEKS_TOTAL)):
        print(f"Solving week {week + 1} / {N_WEEKS_TOTAL}")
        week_states = [state for state in range(N_STATES) if index_to_state_components(state)[3] == week]

        for state_idx in week_states:
            best_value = -np.inf
            best_action = 0

            for action_idx in range(N_ACTIONS):
                reward_mean, transition_probs = estimate_action_value(state_idx, action_idx)
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


def evaluate_policy(policy, n_episodes=100):
    env = FluVaccineEnv()

    rewards = []
    vaccinated = []
    stockouts = []
    expired = []

    for episode in range(n_episodes):
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


def save_results(values, policy):
    np.save("flu_vi_values.npy", values)
    with open("flu_vi_policy.pkl", "wb") as handle:
        pickle.dump(policy, handle)


def main():
    print("=" * 64)
    print("Approximate Value Iteration — Flu Vaccine Resource Management")
    print("=" * 64)
    print(f"States:   {N_STATES}")
    print(f"Actions:  {N_ACTIONS}")
    print(f"Samples per state-action: {TRANSITION_SAMPLES}")
    print()

    values, policy, _ = solve()
    save_results(values, policy)

    print("\nEvaluating policy over 100 episodes...")
    results = evaluate_policy(policy, n_episodes=100)

    print()
    print("=" * 64)
    print("VALUE ITERATION RESULTS (100 episodes, mean ± std)")
    print("=" * 64)
    print(f"Reward:     {results['reward_mean']:>8.1f} ± {results['reward_std']:.1f}")
    print(f"Vaccinated: {results['vaccinated_mean']:>8.1f} ± {results['vaccinated_std']:.1f}")
    print(f"Stockout:   {results['stockout_mean']:>8.1f} ± {results['stockout_std']:.1f}")
    print(f"Expired:    {results['expired_mean']:>8.1f} ± {results['expired_std']:.1f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
