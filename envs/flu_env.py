import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FluVaccineEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.num_regions = 3

        # action: order option for each region
        self.order_options = [0, 100, 300, 600]
        self.action_space = spaces.MultiDiscrete([4, 4, 4])

        # state
        self.observation_space = spaces.Box(
            low=0,
            high=2000,
            shape=(6,),
            dtype=np.float32
        )

        self.inventory = np.array([300, 300, 300])
        self.week = 0

    def reset(self, seed=None, options=None):
        self.inventory = np.array([300, 300, 300])
        self.week = 0

        state = np.concatenate([
            self.inventory.astype(np.float32),
            np.array([self.week, 0, 0], dtype=np.float32)
        ])

        return state, {}

    def step(self, action):
        orders = [self.order_options[a] for a in action]

        self.inventory += orders

        demand = np.random.randint(50, 150, size=3)

        vaccinated = np.minimum(self.inventory, demand)

        self.inventory -= vaccinated

        stockout = demand - vaccinated

        reward = float(vaccinated.sum() - 2 * stockout.sum())

        self.week += 1
        done = self.week >= 12

        state = np.concatenate([
            self.inventory.astype(np.float32),
            np.array([self.week, 0, 0], dtype=np.float32)
        ])

        info = {
            "inventory": self.inventory.copy(),
            "orders": orders,
            "demand": demand,
            "vaccinated": vaccinated,
            "stockout": stockout
        }

        return state, reward, done, False, info

    def render(self):
        print("Inventory:", self.inventory)