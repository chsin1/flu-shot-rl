"""
Q-Learning from scratch — Flu Vaccine Resource Management
MMAI-845 Group Project

Implements tabular Q-Learning on a discretised version of the
FluVaccineEnv state space. Written entirely from scratch without
Stable-Baselines3 to satisfy the assignment requirement of
demonstrating RL algorithm understanding.

State discretisation:
  Continuous 13-dim state → compact discrete state index
  Each region's inventory mapped to 3 levels: low / medium / high
  Week kept as-is (0–11)
  Total states: 3^3 inventory levels × 12 weeks = 324 states

Action space:
  MultiDiscrete([4,4,4]) → flattened to single integer 0–63
  64 joint actions (4 order options × 3 regions)

Core Bellman update (written by hand, no library):
  Q[s][a] ← Q[s][a] + α · (r + γ · max_a' Q[s'][a'] − Q[s][a])
"""

import numpy as np
import pickle
from envs.flu_env import FluVaccineEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────

ALPHA        = 0.1      # learning rate
GAMMA        = 0.99     # discount factor — matches PPO for fair comparison
EPSILON_START = 1.0     # start fully exploratory
EPSILON_END   = 0.05    # minimum exploration
EPSILON_DECAY = 0.995   # multiply epsilon by this each episode
N_EPISODES   = 5000     # training episodes
N_WEEKS      = 12       # episode length (fixed horizon)

# ── State discretisation ──────────────────────────────────────────────────────

# Inventory thresholds for 3 levels per region
# low = 0–149,  medium = 150–349,  high = 350+
INV_THRESHOLDS = [150, 350]
N_INV_LEVELS   = 3   # low / medium / high
N_WEEKS_TOTAL  = 12
N_STATES       = (N_INV_LEVELS ** 3) * N_WEEKS_TOTAL   # 324

# ── Action space ──────────────────────────────────────────────────────────────

# All 64 joint actions: (a0, a1, a2) where each a_i ∈ {0,1,2,3}
# Precompute so we can index by a single integer
ALL_ACTIONS = [(a0, a1, a2)
               for a0 in range(4)
               for a1 in range(4)
               for a2 in range(4)]
N_ACTIONS = len(ALL_ACTIONS)   # 64


def discretise_inventory(inv_value):
    """Map a continuous inventory level to 0 (low), 1 (medium), or 2 (high)."""
    if inv_value < INV_THRESHOLDS[0]:
        return 0
    elif inv_value < INV_THRESHOLDS[1]:
        return 1
    else:
        return 2


def state_to_index(obs):
    """
    Convert the 16-dim continuous observation to a single integer state index.

    We use only inventory (dims 0-2) and week (dim 15) for the Q-table.
    Storage capacity and vulnerability weights are fixed constants — they
    add no information to the table. Expiry and demand are useful signals
    but discretising them would multiply the state space 9× (unmanageable
    for a tabular method at this project scale).

    Index formula:
      idx = i0 * (N_INV_LEVELS^2 * N_WEEKS)
          + i1 * (N_INV_LEVELS * N_WEEKS)
          + i2 * N_WEEKS
          + week
    """
    i0   = discretise_inventory(obs[0])
    i1   = discretise_inventory(obs[1])
    i2   = discretise_inventory(obs[2])
    week = int(obs[-1])
    week = min(week, N_WEEKS_TOTAL - 1)   # guard against edge case

    idx = (i0 * N_INV_LEVELS * N_INV_LEVELS * N_WEEKS_TOTAL
         + i1 * N_INV_LEVELS * N_WEEKS_TOTAL
         + i2 * N_WEEKS_TOTAL
         + week)
    return idx


# ── Q-Learning ────────────────────────────────────────────────────────────────

def train(env, verbose=True):
    """
    Train a Q-table using tabular Q-Learning with epsilon-greedy exploration.

    The Bellman update rule (implemented from scratch):
      Q[s][a] ← Q[s][a] + α · (r + γ · max_a' Q[s'][a'] − Q[s][a])

    Returns the trained Q-table and per-episode reward history.
    """

    # initialise Q-table to zeros: shape (N_STATES, N_ACTIONS)
    Q = np.zeros((N_STATES, N_ACTIONS))

    epsilon = EPSILON_START
    episode_rewards = []

    for episode in range(N_EPISODES):

        obs, info = env.reset()
        state = state_to_index(obs)

        total_reward = 0.0
        done = False

        while not done:

            # --- epsilon-greedy action selection ---
            if np.random.random() < epsilon:
                action_idx = np.random.randint(N_ACTIONS)          # explore
            else:
                action_idx = int(np.argmax(Q[state]))              # exploit

            action = list(ALL_ACTIONS[action_idx])

            # --- take step in environment ---
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = state_to_index(obs_next)

            # --- Bellman update (core Q-Learning equation) ---
            best_next_q = np.max(Q[next_state])
            td_target   = reward + GAMMA * best_next_q
            td_error    = td_target - Q[state][action_idx]
            Q[state][action_idx] += ALPHA * td_error

            # --- advance ---
            state        = next_state
            total_reward += reward

        # decay epsilon after each episode
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        # progress logging
        if verbose and (episode + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-500:])
            print(f"Episode {episode + 1:>5} / {N_EPISODES}"
                  f"  avg reward (last 500): {avg:>8.1f}"
                  f"  epsilon: {epsilon:.3f}")

    return Q, episode_rewards


def extract_policy(Q):
    """
    Extract the greedy policy from the trained Q-table.
    Returns a dict mapping state index → best action tuple.
    """
    policy = {}
    for s in range(N_STATES):
        best_action_idx = int(np.argmax(Q[s]))
        policy[s] = ALL_ACTIONS[best_action_idx]
    return policy


def evaluate(env, Q, n_episodes=100, verbose=False):
    """
    Evaluate the learned Q-table policy over n_episodes.
    Acts greedily (epsilon=0) — no exploration during evaluation.
    Returns mean and std of total reward, plus aggregate metrics.
    """
    rewards      = []
    vaccinated   = []
    stockouts    = []
    expired_list = []

    for ep in range(n_episodes):

        obs, info = env.reset()
        state = state_to_index(obs)

        done            = False
        total_reward    = 0.0
        total_vacc      = 0.0
        total_stockout  = 0.0
        total_expired   = 0.0

        while not done:

            # greedy action
            action_idx = int(np.argmax(Q[state]))
            action     = list(ALL_ACTIONS[action_idx])

            obs, reward, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            state = state_to_index(obs)

            total_reward   += reward
            total_vacc     += info['vaccinated'].sum()
            total_stockout += info['stockout'].sum()
            total_expired  += info['expired'].sum()

            if verbose:
                print(f"  Week {env.week:>2}  orders={action}"
                      f"  vacc={info['vaccinated'].sum():.0f}"
                      f"  stock={info['stockout'].sum():.0f}"
                      f"  reward={reward:.1f}")

        rewards.append(total_reward)
        vaccinated.append(total_vacc)
        stockouts.append(total_stockout)
        expired_list.append(total_expired)

    return {
        "reward_mean":     np.mean(rewards),
        "reward_std":      np.std(rewards),
        "vaccinated_mean": np.mean(vaccinated),
        "vaccinated_std":  np.std(vaccinated),
        "stockout_mean":   np.mean(stockouts),
        "stockout_std":    np.std(stockouts),
        "expired_mean":    np.mean(expired_list),
        "expired_std":     np.std(expired_list),
    }


def save(Q, policy, path_prefix="flu_ql"):
    """Save Q-table and policy to disk."""
    np.save(f"{path_prefix}_qtable.npy", Q)
    with open(f"{path_prefix}_policy.pkl", "wb") as f:
        pickle.dump(policy, f)
    print(f"Saved Q-table → {path_prefix}_qtable.npy")
    print(f"Saved policy  → {path_prefix}_policy.pkl")


def load(path_prefix="flu_ql"):
    """Load Q-table and policy from disk."""
    Q      = np.load(f"{path_prefix}_qtable.npy")
    with open(f"{path_prefix}_policy.pkl", "rb") as f:
        policy = pickle.load(f)
    return Q, policy


# ── Entry point ───────────────────────────────────────────────────────────────

def main():

    env = FluVaccineEnv()

    print("=" * 60)
    print("  Q-Learning from scratch — Flu Vaccine Resource Management")
    print("=" * 60)
    print(f"  States:      {N_STATES}  (3^3 inv levels × 12 weeks)")
    print(f"  Actions:     {N_ACTIONS}  (4^3 joint order combinations)")
    print(f"  Episodes:    {N_EPISODES}")
    print(f"  α (lr):      {ALPHA}")
    print(f"  γ (discount):{GAMMA}")
    print(f"  ε decay:     {EPSILON_START} → {EPSILON_END} over training")
    print("=" * 60)
    print()

    # train
    print("Training...")
    Q, episode_rewards = train(env, verbose=True)

    # extract and save policy
    policy = extract_policy(Q)
    save(Q, policy)

    # evaluate
    print("\nEvaluating over 100 episodes (greedy policy)...")
    results = evaluate(env, Q, n_episodes=100)

    print()
    print("=" * 60)
    print("  Q-LEARNING RESULTS (100 episodes, mean ± std)")
    print("=" * 60)
    print(f"  Reward:     {results['reward_mean']:>8.1f} ± {results['reward_std']:.1f}")
    print(f"  Vaccinated: {results['vaccinated_mean']:>8.1f} ± {results['vaccinated_std']:.1f}")
    print(f"  Stockout:   {results['stockout_mean']:>8.1f} ± {results['stockout_std']:.1f}")
    print(f"  Expired:    {results['expired_mean']:>8.1f} ± {results['expired_std']:.1f}")
    print("=" * 60)

    print()
    print("Compare against PPO baseline:")
    print("  PPO reward: 3434.8 ± 204.5")
    print(f"  Q-Learning: {results['reward_mean']:.1f} ± {results['reward_std']:.1f}")

    gap = 3434.8 - results['reward_mean']
    direction = "below" if gap > 0 else "above"
    print(f"  Gap:        {abs(gap):.1f} pts {direction} PPO")
    print()
    print("Note: gap is expected — Q-Learning uses a discretised state")
    print("space (324 states) vs PPO's full 16-dim continuous state.")
    print("This demonstrates why function approximation (DQN/PPO) is")
    print("needed for higher-dimensional inventory problems.")


if __name__ == "__main__":
    main()