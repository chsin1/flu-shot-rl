from envs.configs import BASELINE_CONFIG, REALISTIC_CONFIG
from envs.flu_env_continuous import FluVaccineEnvContinuous


def make_env_continuous(env_name: str = "baseline") -> FluVaccineEnvContinuous:
    presets = {
        "baseline": BASELINE_CONFIG,
        "realistic": REALISTIC_CONFIG,
    }

    if env_name not in presets:
        raise ValueError(
            f"Unknown env_name '{env_name}'. Choose from: {list(presets.keys())}"
        )

    return FluVaccineEnvContinuous(config=presets[env_name])