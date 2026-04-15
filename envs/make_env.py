from envs.configs import BASELINE_CONFIG, REALISTIC_CONFIG
from envs.flu_env import FluVaccineEnv


def make_env(env_name: str = "baseline") -> FluVaccineEnv:
    """
    Factory for environment presets.

    baseline  -> simplified version for assignment comparison
    realistic -> adds lead time and higher stochasticity
    """
    presets = {
        "baseline": BASELINE_CONFIG,
        "realistic": REALISTIC_CONFIG,
    }

    if env_name not in presets:
        raise ValueError(
            f"Unknown env_name '{env_name}'. Choose from: {list(presets.keys())}"
        )

    return FluVaccineEnv(config=presets[env_name])