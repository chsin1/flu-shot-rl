from dataclasses import dataclass
from typing import Tuple

@dataclass
class EnvConfig:

    """
    Configuration for the flu vaccine environment.

    This allows us to keep a single environment implementation while switching
    between a simpler baseline setting and a more realistic setting.
    """

    # Core environment structure
    num_regions: int = 3 # Number of regions in the simulation
    order_options: Tuple[int, ...] = (0, 100, 300, 600)  # Each index maps to a specific number of doses
    storage_capacity: Tuple[float, ...] = (800.0, 600.0, 1000.0) # Maximum storage capacity per region (fixed constraint)
    expiry_window: int = 4 # Number of weeks before vaccines expire (FIFO tracking window)
    n_weeks: int = 12

    # Demand uncertainty
    demand_noise_std: float = 20.0 # Standard deviation of demand noise (controls uncertainty)

    # Rare local outbreak shock
    use_catastrophic_spike: bool = True # Toggle catastrophic spikes on/off
    catastrophic_spike_prob: float = 0.05 # Probability of a rare demand spike (simulates outbreak shock)
    catastrophic_spike_multiplier: float = 2.0 # Multiplier applied when a spike occurs

    # Supply-chain realism
    # Lead time (in weeks) before orders arrive
    # 0 = immediate delivery (baseline), >0 = delayed supply (realistic)
    lead_time_weeks: int = 0


BASELINE_CONFIG = EnvConfig(
    # Matches the current assignment-friendly setup
    lead_time_weeks=0,
    demand_noise_std=20.0, 
    use_catastrophic_spike=True,
    catastrophic_spike_prob=0.05,
    catastrophic_spike_multiplier=2.0,
)

REALISTIC_CONFIG = EnvConfig(
    # Adds supply delay and stronger uncertainty
    lead_time_weeks=2,
    demand_noise_std=30.0,
    use_catastrophic_spike=True,
    catastrophic_spike_prob=0.10,
    catastrophic_spike_multiplier=2.0,
)