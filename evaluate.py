from stable_baselines3 import PPO
from envs.flu_env import FluVaccineEnv


def main():

    env = FluVaccineEnv()
    model = PPO.load("flu_rl_model")

    obs, info = env.reset()

    done = False
    total_reward = 0
    week = 0

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        week += 1
        total_reward += reward

        print("--------------")
        print(f"Week: {week}")
        print(f"Orders: {info['orders']}")
        print(f"Demand: {info['demand']}")
        print(f"Vaccinated: {info['vaccinated']}")
        print(f"Stockout: {info['stockout']}")
        print(f"Inventory: {info['inventory']}")
        print(f"Reward: {reward}")

    print("==============")
    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()