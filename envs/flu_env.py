import gymnasium as gym
from gymnasium import spaces
import numpy as np

 
# Seasonal demand curve: bell-shaped peak around weeks 6-7 of a 12-week flu season.
# Values are multipliers applied to each region's base demand.
# Week index 0..11 maps to weeks 1..12.
SEASONAL_CURVE = np.array([
    0.4,   # week 1  — season just starting, low uptake
    0.6,   # week 2
    0.8,   # week 3  — demand building
    1.0,   # week 4
    1.2,   # week 5
    1.4,   # week 6  — peak
    1.4,   # week 7  — peak
    1.1,   # week 8  — tapering
    0.9,   # week 9
    0.7,   # week 10
    0.5,   # week 11
    0.3,   # week 12 — season winding down
], dtype=np.float32)
 
# Base demand per region (doses/week at multiplier=1.0).
# Different regions reflect different population sizes / uptake rates.
# Region 0: large urban area  — high base demand
# Region 1: mid-size town     — moderate base demand
# Region 2: rural area        — lower base demand
REGION_BASE_DEMAND = np.array([180.0, 130.0, 90.0], dtype=np.float32)

# --- NEW: vulnerability weights per region ---
# Higher weight = higher penalty for stockout (public health priority)
# Region 0: urban, Region 1: town (elderly), Region 2: rural (access issues)
VULNERABILITY_WEIGHT = np.array([1.2, 2.0, 1.5], dtype=np.float32)
 
class FluVaccineEnv(gym.Env):
 
    def __init__(self):
 
        super().__init__()
 
        self.num_regions = 3
 
        # action: order option for each region
        self.order_options = [0, 100, 300, 600]
        self.action_space = spaces.MultiDiscrete([4, 4, 4])

        # penalise shortages more heavily in higher-priority regions
        self.vulnerability_weights = VULNERABILITY_WEIGHT
 
        # fridge storage capacity per region (fixed, different per region)
        self.storage_capacity = np.array([800.0, 600.0, 1000.0])
 
        # vaccines expire after this many weeks
        self.expiry_window = 4

        # rare catastrophic local outbreak: one region's demand doubles
        self.catastrophic_spike_prob = 0.05
        self.catastrophic_spike_multiplier = 2.0

        # state: inventory(3) + expiring_soon(3) + storage_capacity(3) + demand_level(3) + week(1) = 13
        self.observation_space = spaces.Box(
            low=0,
            high=2000,
            shape=(13,),
            dtype=np.float32
        )
 
        self.inventory = np.zeros(self.num_regions)
        self.expiry_tracker = None  # shape (expiry_window, num_regions)
        self.demand_level = np.zeros(self.num_regions)
        self.week = 0
        self.last_spike_region = -1
        self.last_spike_multiplier = 1.0
 
    def _seasonal_demand(self):
        """
        Generate demand for the current week using:
          demand = base_demand * seasonal_multiplier + gaussian_noise
 
        The seasonal curve creates a realistic bell-shaped flu season.
        Gaussian noise adds week-to-week variability so the agent cannot
        simply memorise a fixed schedule.
        """
        multiplier = SEASONAL_CURVE[self.week]
        base = REGION_BASE_DEMAND * multiplier
        noise = self.np_random.normal(loc=0.0, scale=20.0, size=self.num_regions)
        demand = np.clip(base + noise, 10, 600).astype(np.float32)

        self.last_spike_region = -1
        self.last_spike_multiplier = 1.0
        if self.np_random.random() < self.catastrophic_spike_prob:
            spike_region = int(self.np_random.integers(self.num_regions))
            demand[spike_region] = np.clip(
                demand[spike_region] * self.catastrophic_spike_multiplier,
                10,
                600,
            )
            self.last_spike_region = spike_region
            self.last_spike_multiplier = self.catastrophic_spike_multiplier

        return demand
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
 
        self.inventory = np.array([300.0, 300.0, 300.0])
        self.week = 0
        self.last_spike_region = -1
        self.last_spike_multiplier = 1.0
 
        # track doses by age: each row = age bucket (oldest first)
        self.expiry_tracker = np.zeros((self.expiry_window, self.num_regions))
        self.expiry_tracker[0] = self.inventory  # initial stock placed in oldest bucket
 
        # initialise demand using seasonal curve at week 0
        self.demand_level = self._seasonal_demand()
 
        state = self._get_obs()
        return state, {}
 
    def step(self, action):
 
        # --- 1. place orders (capped by storage capacity) ---
        orders = np.array([self.order_options[a] for a in action], dtype=np.float32)
        space_available = self.storage_capacity - self.inventory
        orders = np.minimum(orders, space_available)
 
        # add new doses into freshest bucket
        self.expiry_tracker = np.roll(self.expiry_tracker, shift=1, axis=0)
        self.expiry_tracker[0] = orders
 
        self.inventory += orders
 
        # --- 2. generate seasonal stochastic demand ---
        self.demand_level = self._seasonal_demand()
 
        # --- 3. vaccinate (use oldest stock first - FIFO) ---
        vaccinated = np.zeros(self.num_regions)
        remaining_demand = self.demand_level.copy()
 
        for age in range(self.expiry_window - 1, -1, -1):
            used = np.minimum(self.expiry_tracker[age], remaining_demand)
            self.expiry_tracker[age] -= used
            vaccinated += used
            remaining_demand -= used
 
        self.inventory = self.expiry_tracker.sum(axis=0)
 
        # --- 4. expire oldest doses ---
        expired = self.expiry_tracker[-1].copy()
        self.expiry_tracker[-1] = 0
        self.inventory = self.expiry_tracker.sum(axis=0)
 
        # --- 5. compute metrics ---
        stockout = np.maximum(self.demand_level - vaccinated, 0)
        expiring_soon = self.expiry_tracker[-1].copy()  # doses 1 step from expiry
 
        # --- 6. reward ---
        storage_cost = np.sum(self.inventory) * 0.01
                
        # weighted stockout penalty using vulnerability
        weighted_stockout_penalty = float(np.sum(self.vulnerability_weights * stockout))

        reward = float(
            vaccinated.sum()
            - weighted_stockout_penalty 
            - 1.5 * expired.sum()
            - storage_cost
        )
 
        # --- 7. advance week ---
        self.week += 1
        done = self.week >= 12
 
        state = self._get_obs()
 
        info = {
            "inventory": self.inventory.copy(),
            "orders": orders,
            "demand": self.demand_level,
            "vaccinated": vaccinated,
            "stockout": stockout,
            "expired": expired,
            "storage_capacity": self.storage_capacity,
            "seasonal_multiplier": SEASONAL_CURVE[self.week - 1],
            "catastrophic_spike_region": self.last_spike_region,
            "catastrophic_spike_multiplier": self.last_spike_multiplier,
            "weighted_stockout_penalty": weighted_stockout_penalty,
        }
 
        return state, reward, done, False, info
 
    def _get_obs(self):
        expiring_soon = self.expiry_tracker[-1].copy()  # doses 1 week from expiry
        state = np.concatenate([
            self.inventory.astype(np.float32),           # 3 — current stock per region
            expiring_soon.astype(np.float32),            # 3 — doses about to expire
            self.storage_capacity.astype(np.float32),    # 3 — max fridge capacity per region
            self.demand_level.astype(np.float32),        # 3 — last observed demand
            np.array([self.week], dtype=np.float32)      # 1 — week of flu season
        ])
        return state
 
    def render(self):
        multiplier = SEASONAL_CURVE[min(self.week, 11)]
        print(f"Week {self.week} (seasonal multiplier: {multiplier:.1f}x)")
        print(f"  Inventory:        {self.inventory}")
        print(f"  Storage capacity: {self.storage_capacity}")
        print(f"  Demand level:     {self.demand_level}")
        print(f"  Expiring soon:    {self.expiry_tracker[-1]}")
