import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.make_env import make_env


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on the flu vaccine environment."
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
        default=500_000,
        help="Total PPO training timesteps.",
    )
    args = parser.parse_args()

    env = make_env(args.env)

    # Validate Gymnasium API compliance
    check_env(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=f"./ppo_tensorboard/{args.env}/",
    )

    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=f"ppo_flu_vaccine_{args.env}",
    )

    save_path = f"flu_rl_model_{args.env}"
    model.save(save_path)
    print(f"Training complete. Model saved to: {save_path}.zip")


if __name__ == "__main__":
    main()