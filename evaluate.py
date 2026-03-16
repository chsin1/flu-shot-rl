from stable_baselines3 import PPO
from envs.flu_env import FluVaccineEnv
import numpy as np
 
 
def run_episodes(env, policy_fn, label, n_episodes=100):
    rewards = []
    vaccinated_list = []
    stockout_list = []
    expired_list = []
 
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        total_vaccinated = 0
        total_stockout = 0
        total_expired = 0
 
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_vaccinated += info['vaccinated'].sum()
            total_stockout += info['stockout'].sum()
            total_expired += info['expired'].sum()
 
        rewards.append(total_reward)
        vaccinated_list.append(total_vaccinated)
        stockout_list.append(total_stockout)
        expired_list.append(total_expired)
 
    return {
        "label": label,
        "reward_mean":     np.mean(rewards),
        "reward_std":      np.std(rewards),
        "vaccinated_mean": np.mean(vaccinated_list),
        "vaccinated_std":  np.std(vaccinated_list),
        "stockout_mean":   np.mean(stockout_list),
        "stockout_std":    np.std(stockout_list),
        "expired_mean":    np.mean(expired_list),
        "expired_std":     np.std(expired_list),
    }
 
 
def main():
 
    N_EPISODES = 100
 
    env = FluVaccineEnv()
 
    # --- Policy 1: trained RL agent ---
    model = PPO.load("flu_rl_model")
    def rl_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
 
    # --- Policy 2: always order 300 per region ---
    def fixed_300_policy(obs):
        return np.array([2, 2, 2])
 
    # --- Policy 3: always order 100 per region ---
    def fixed_100_policy(obs):
        return np.array([1, 1, 1])
 
    # --- Policy 4: random policy ---
    def random_policy(obs):
        return env.action_space.sample()
 
    print(f"Running {N_EPISODES} episodes per policy...\n")
 
    results = []
    for policy_fn, label in [
        (rl_policy,        "RL Agent (PPO)"),
        (fixed_300_policy, "Baseline: always order 300"),
        (fixed_100_policy, "Baseline: always order 100"),
        (random_policy,    "Baseline: random"),
    ]:
        print(f"  Evaluating: {label}...")
        results.append(run_episodes(env, policy_fn, label, N_EPISODES))
 
    # --- comparison table ---
    col = 32
    print(f"\n{'=' * 85}")
    print(f"  COMPARISON SUMMARY ({N_EPISODES} episodes, mean ± std)")
    print(f"{'=' * 85}")
    print(f"{'Policy':<{col}} {'Reward':>18} {'Vaccinated':>18} {'Stockout':>18} {'Expired':>18}")  
    print(f"{'-' * 85}")  
    for r in results:
        print(
            f"{r['label']:<{col}} "
            f"{r['reward_mean']:>8.1f} ±{r['reward_std']:>6.1f} "
            f"{r['vaccinated_mean']:>8.1f} ±{r['vaccinated_std']:>6.1f} "
            f"{r['stockout_mean']:>8.1f} ±{r['stockout_std']:>6.1f} "
            f"{r['expired_mean']:>8.1f} ±{r['expired_std']:>6.1f}"
        )
    print(f"{'=' * 85}")
 
    # --- highlight best ---
    best = max(results, key=lambda x: x['reward_mean'])
    print(f"\nBest policy by avg reward: {best['label']} ({best['reward_mean']:.1f} ± {best['reward_std']:.1f})")
 
 
if __name__ == "__main__":
    main()