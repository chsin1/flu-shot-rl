from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.flu_env import FluVaccineEnv


def main():

    # create environment
    env = FluVaccineEnv()

    # check if environment follows Gym API
    check_env(env)

    # create RL model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
    )
 
    # train model
    model.learn(total_timesteps=500000)
 
    # save model
    model.save("flu_rl_model")
 
    print("Training complete. Model saved.")
 
 
if __name__ == "__main__":
    main()