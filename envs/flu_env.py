import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.configs import BASELINE_CONFIG, EnvConfig


# Seasonal demand curve: bell-shaped peak around weeks 6-7 of a 12-week flu season.
# Week index 0..11 maps to weeks 1..12.
SEASONAL_CURVE = np.array([
    0.4,   # week 1
    0.6,   # week 2
    0.8,   # week 3
    1.0,   # week 4
    1.2,   # week 5
    1.4,   # week 6
    1.4,   # week 7
    1.1,   # week 8
    0.9,   # week 9
    0.7,   # week 10
    0.5,   # week 11
    0.3,   # week 12
], dtype=np.float32)

# Base demand per region (doses/week at multiplier=1.0)
REGION_BASE_DEMAND = np.array([180.0, 130.0, 90.0], dtype=np.float32)

# Public-health weighting for stockout penalty
VULNERABILITY_WEIGHT = np.array([1.2, 2.0, 1.5], dtype=np.float32)


class FluVaccineEnv(gym.Env):
    """
    Flu vaccine allocation environment.

    Observation (13 dims):
      inventory(3) + expiring_soon(3) + storage_capacity(3) + demand_level(3) + week(1)

    Action:
      MultiDiscrete([4, 4, 4]) -> order option per region

    Reward:
      vaccinated
      - weighted stockout penalty
      - expiry penalty
      - storage holding cost
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig = BASELINE_CONFIG):
        super().__init__()

        self.config = config
        self.num_regions = config.num_regions
        self.order_options = list(config.order_options)
        self.storage_capacity = np.array(config.storage_capacity, dtype=np.float32)
        self.expiry_window = config.expiry_window
        self.n_weeks = config.n_weeks

        self.demand_noise_std = config.demand_noise_std
        self.use_catastrophic_spike = config.use_catastrophic_spike
        self.catastrophic_spike_prob = config.catastrophic_spike_prob
        self.catastrophic_spike_multiplier = config.catastrophic_spike_multiplier
        self.lead_time_weeks = config.lead_time_weeks

        self.vulnerability_weights = VULNERABILITY_WEIGHT.astype(np.float32)

        # Action: one discrete ordering choice per region
        self.action_space = spaces.MultiDiscrete(
            [len(self.order_options)] * self.num_regions
        )

        # Observation: 13-dimensional continuous vector
        self.observation_space = spaces.Box(
            low=0.0,
            high=2000.0,
            shape=(13,),
            dtype=np.float32,
        )

        # Internal state
        self.inventory = np.zeros(self.num_regions, dtype=np.float32)
        self.expiry_tracker = np.zeros(
            (self.expiry_window, self.num_regions), dtype=np.float32
        )
        self.demand_level = np.zeros(self.num_regions, dtype=np.float32)
        self.week = 0

        # For optional lead-time pipeline
        self.pipeline_orders = None

        # For reporting rare spikes
        self.last_spike_region = -1
        self.last_spike_multiplier = 1.0

    def _seasonal_demand(self) -> np.ndarray:
        """
        Generate stochastic weekly demand based on:
          base demand × seasonal multiplier + Gaussian noise

        Optionally applies a rare catastrophic spike to one region.
        """
        multiplier = SEASONAL_CURVE[self.week]
        base = REGION_BASE_DEMAND * multiplier
        noise = self.np_random.normal(
            loc=0.0,
            scale=self.demand_noise_std,
            size=self.num_regions,
        )
        demand = np.clip(base + noise, 10.0, 600.0).astype(np.float32)

        self.last_spike_region = -1
        self.last_spike_multiplier = 1.0

        if (
            self.use_catastrophic_spike
            and self.np_random.random() < self.catastrophic_spike_prob
        ):
            spike_region = int(self.np_random.integers(self.num_regions))
            demand[spike_region] = np.clip(
                demand[spike_region] * self.catastrophic_spike_multiplier,
                10.0,
                600.0,
            )
            self.last_spike_region = spike_region
            self.last_spike_multiplier = self.catastrophic_spike_multiplier

        return demand

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.inventory = np.array([300.0, 300.0, 300.0], dtype=np.float32)
        self.week = 0

        self.last_spike_region = -1
        self.last_spike_multiplier = 1.0

        # Oldest stock is at the last row; freshest at row 0 after roll-and-insert
        self.expiry_tracker = np.zeros(
            (self.expiry_window, self.num_regions), dtype=np.float32
        )
        self.expiry_tracker[0] = self.inventory.copy()

        if self.lead_time_weeks > 0:
            self.pipeline_orders = np.zeros(
                (self.lead_time_weeks, self.num_regions), dtype=np.float32
            )
        else:
            self.pipeline_orders = None

        self.demand_level = self._seasonal_demand()
        state = self._get_obs()

        info = {
            "inventory": self.inventory.copy(),
            "demand": self.demand_level.copy(),
            "storage_capacity": self.storage_capacity.copy(),
        }
        return state, info

    def step(self, action):
        # --- 1. interpret action as order placed this week ---
        orders_placed = np.array(
            [self.order_options[a] for a in action],
            dtype=np.float32,
        )

        # --- 2. determine what actually arrives this week ---
        if self.lead_time_weeks == 0:
            arriving_orders = orders_placed.copy()
        else:
            arriving_orders = self.pipeline_orders[0].copy()
            self.pipeline_orders = np.roll(self.pipeline_orders, shift=-1, axis=0)
            self.pipeline_orders[-1] = orders_placed

        # --- 3. cap arrivals by available storage ---
        space_available = self.storage_capacity - self.inventory
        arriving_orders = np.minimum(arriving_orders, space_available)
        arriving_orders = np.maximum(arriving_orders, 0.0)

        # --- 4. add arrivals into freshest bucket ---
        self.expiry_tracker = np.roll(self.expiry_tracker, shift=1, axis=0)
        self.expiry_tracker[0] = arriving_orders
        self.inventory += arriving_orders

        # --- 5. generate seasonal stochastic demand ---
        self.demand_level = self._seasonal_demand()

        # --- 6. vaccinate using oldest stock first (FIFO) ---
        vaccinated = np.zeros(self.num_regions, dtype=np.float32)
        remaining_demand = self.demand_level.copy()

        for age in range(self.expiry_window - 1, -1, -1):
            used = np.minimum(self.expiry_tracker[age], remaining_demand)
            self.expiry_tracker[age] -= used
            vaccinated += used
            remaining_demand -= used

        self.inventory = self.expiry_tracker.sum(axis=0)

        # --- 7. expire oldest doses ---
        expired = self.expiry_tracker[-1].copy()
        self.expiry_tracker[-1] = 0.0
        self.inventory = self.expiry_tracker.sum(axis=0)

        # --- 8. compute reward ---
        stockout = np.maximum(self.demand_level - vaccinated, 0.0)
        storage_cost = float(np.sum(self.inventory) * 0.01)
        weighted_stockout_penalty = float(
            np.sum(self.vulnerability_weights * stockout)
        )

        reward = float(
            vaccinated.sum()
            - weighted_stockout_penalty
            - 1.5 * expired.sum()
            - storage_cost
        )

        # --- 9. advance week ---
        self.week += 1
        terminated = self.week >= self.n_weeks
        truncated = False

        state = self._get_obs()

        info = {
            "inventory": self.inventory.copy(),
            "orders_placed": orders_placed,
            "arriving_orders": arriving_orders,
            "demand": self.demand_level.copy(),
            "vaccinated": vaccinated.copy(),
            "stockout": stockout.copy(),
            "expired": expired.copy(),
            "storage_capacity": self.storage_capacity.copy(),
            "seasonal_multiplier": float(SEASONAL_CURVE[self.week - 1]),
            "catastrophic_spike_region": self.last_spike_region,
            "catastrophic_spike_multiplier": float(self.last_spike_multiplier),
            "weighted_stockout_penalty": weighted_stockout_penalty,
        }

        return state, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        expiring_soon = self.expiry_tracker[-1].copy()
        return np.concatenate(
            [
                self.inventory.astype(np.float32),         # 3
                expiring_soon.astype(np.float32),          # 3
                self.storage_capacity.astype(np.float32),  # 3
                self.demand_level.astype(np.float32),      # 3
                np.array([self.week], dtype=np.float32),   # 1
            ]
        )

    def render(self):
        multiplier = SEASONAL_CURVE[min(self.week, self.n_weeks - 1)]
        print(f"Week {self.week} (seasonal multiplier: {multiplier:.1f}x)")
        print(f"  Inventory:        {self.inventory}")
        print(f"  Storage capacity: {self.storage_capacity}")
        print(f"  Demand level:     {self.demand_level}")
        print(f"  Expiring soon:    {self.expiry_tracker[-1]}")
        if self.lead_time_weeks > 0 and self.pipeline_orders is not None:
            print(f"  Pipeline orders:  {self.pipeline_orders}")