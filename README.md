# RL Flu Vaccine Resource Management

A reinforcement learning system that learns optimal vaccine ordering and distribution strategies across three simulated regions during a 12-week flu season.

The agent is trained to maximise vaccinations while minimising stockouts, vaccine expiry waste, and cold-chain storage costs — modelled as a Markov Decision Process (MDP).

---

## Project Structure

```
project/
├── envs/
│   └── flu_env.py        # Gymnasium environment (FluVaccineEnv)
├── train.py              # Train the PPO agent
├── evaluate.py           # Evaluate and compare policies over 100 episodes
├── flu_rl_model.zip      # Saved trained model (generated after training)
└── README.md
```

---

## Environment Overview

The environment simulates a centralised vaccine manager supplying **3 regions** over a **12-week flu season**.

### Regions

| Region | Type       | Base Demand (doses/week at peak) | Fridge Capacity |
|--------|------------|----------------------------------|-----------------|
| 0      | Urban      | 180                              | 800 doses       |
| 1      | Town       | 130                              | 600 doses       |
| 2      | Rural      | 90                               | 1,000 doses     |

### State Space (13 dimensions)

| Component             | Dimensions | Description                          |
|-----------------------|------------|--------------------------------------|
| Inventory             | 3          | Current stock per region             |
| Expiring soon         | 3          | Doses expiring next week per region  |
| Storage capacity      | 3          | Max fridge capacity per region       |
| Demand level          | 3          | Last observed demand per region      |
| Week                  | 1          | Current week of the flu season       |

### Action Space

`MultiDiscrete([4, 4, 4])` — each region independently chooses an order quantity:

| Action index | Doses ordered |
|--------------|---------------|
| 0            | 0             |
| 1            | 100           |
| 2            | 300           |
| 3            | 600           |

### Reward Function

```
reward = vaccinations_administered
       - 2.0 × stockout_doses
       - 1.5 × expired_doses
       - 0.01 × total_inventory  (storage holding cost)
```

### Seasonal Demand Curve

Demand follows a bell-shaped curve over the 12-week season, peaking at weeks 6–7. This models the active phase of a real flu vaccination campaign (approximately October–March in the Northern Hemisphere).

```
Week:       1    2    3    4    5    6    7    8    9   10   11   12
Multiplier: 0.4  0.6  0.8  1.0  1.2  1.4  1.4  1.1  0.9  0.7  0.5  0.3
```

Gaussian noise (std=20) is added each week to keep demand stochastic.

---

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd <project-folder>

# 2. Create and activate a virtual environment
python -m venv venv

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install gymnasium stable-baselines3 numpy
```

---

## Running the Project

### Train the agent

```bash
python train.py
```

This will:
- Validate the environment against the Gymnasium API
- Train a PPO agent for 500,000 timesteps
- Save the model as `flu_rl_model.zip`

Training takes approximately 5–10 minutes depending on your machine.

### Evaluate the agent

```bash
python evaluate.py
```

This will run 100 episodes for each of four policies and print a comparison table:

- RL Agent (PPO)
- Baseline: always order 300
- Baseline: always order 100
- Baseline: random

### Expected results (current baseline)

```
Policy                           Reward         Vaccinated     Stockout       Expired
RL Agent (PPO)            3434.8 ± 204.5   4090.9 ± 105.3    27.9 ± 31.7   346.8 ± 73.6
Baseline: always order 300  -173.3 ± 231.8  4115.0 ± 102.8     0.0 ±  0.0  2717.3 ± 91.5
Baseline: always order 100  2201.9 ± 237.6  3669.3 ±  96.7   439.5 ± 64.1   347.7 ± 80.6
Baseline: random             772.0 ± 790.2  3960.8 ± 241.1   158.7 ± 201.6 1812.0 ± 369.3
```

---

## Working on Your Own Experiment Branch

### Branch naming convention

```
experiment/<short-description>
```

Examples:
```
experiment/catastrophic-demand-spike
experiment/order-lead-time
experiment/venue-type-regions
```

### Workflow

```bash
# 1. Make sure you are on main and up to date
git checkout main
git pull

# 2. Create your experiment branch
git checkout -b experiment/your-experiment-name

# 3. Make your changes to flu_env.py (or other files)

# 4. Retrain the model — required after any env change
python train.py

# 5. Evaluate and record your results
python evaluate.py

# 6. Commit your changes with a descriptive message
git add .
git commit -m "experiment: <what you changed and key result>"

# 7. Push your branch
git push origin experiment/your-experiment-name
```

> **Important:** any change to `flu_env.py` (state shape, reward weights, demand logic) requires a full retrain before evaluating. Do not evaluate using a model trained on a different environment version.

### Merging back to main

Only merge your branch into main if your 100-episode evaluation shows a meaningful improvement in average reward compared to the current main baseline (3,434.8 ± 204.5). Discuss with the team lead before merging.

```bash
# After discussion and approval
git checkout main
git merge experiment/your-experiment-name
```

---

## Suggested Experiments (Weeks 4–5)

| Branch | Description | Effort | Priority |
|--------|-------------|--------|----------|
| `experiment/catastrophic-demand-spike` | 5% weekly chance of demand doubling in one region | Low | High |
| `experiment/order-lead-time` | Vaccines ordered this week arrive 2 weeks later | Medium | High |
| `experiment/venue-type-regions` | Relabel regions as hospital / clinic / pharmacy with different volatility | Low | Medium |
| `experiment/demographic-weights` | Weight demand by age/vulnerability profile per region | Medium | Low |

---

## Key Design Decisions

**Why PPO instead of Q-Learning / DQN?**
The original proposal specified Q-Learning and DQN. PPO was used because Stable-Baselines3 handles the MultiDiscrete action space (3 regions × 4 order options) more cleanly. Q-Learning and DQN can still be implemented as a comparison — see the proposal for details.

**Why 12 weeks?**
The environment models the active phase of a flu vaccination campaign, not the full calendar year. Week 1 = campaign launch, Weeks 6–7 = seasonal peak, Week 12 = end of campaign.

**Why FIFO expiry tracking?**
Real vaccine distribution uses first-in-first-out stock rotation to minimise expiry. The 4-week rolling age-bucket matrix in `flu_env.py` mirrors this.

**Why 100 episodes for evaluation?**
A single episode is affected by random demand draws. 100 episodes gives statistically reliable mean ± std results that can be compared fairly across policies.

---

## Notes for the Report

- Always cite results as **mean ± std over 100 episodes**
- The seasonal curve multipliers can be justified by referencing real CDC / WHO flu season data
- Rejected experiments are worth documenting — showing what you tried and why it didn't work demonstrates rigour
