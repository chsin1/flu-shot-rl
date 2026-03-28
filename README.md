# RL Flu Vaccine Resource Management

**MMAI-845 Group Project — March 2026**

A reinforcement learning system that learns optimal vaccine ordering and
distribution strategies across three simulated regions during a 12-week
flu season, modelled as a Markov Decision Process (MDP).

The objective is to maximise vaccinations while minimising stockouts,
vaccine expiry, and storage costs under stochastic seasonal demand.

---

## Table of Contents

1. [Assignment Requirements](#1-assignment-requirements)
2. [Project Structure](#2-project-structure)
3. [Environment Design](#3-environment-design)
4. [Algorithms](#4-algorithms)
5. [Setup and Installation](#5-setup-and-installation)
6. [Running the Project](#6-running-the-project)
7. [Policy Catalogue](#7-policy-catalogue)
8. [MDP Reference](#8-mdp-reference)
9. [Experiment Branches](#9-experiment-branches)
10. [Merging to Main](#10-merging-to-main)
11. [Team Plan and Timeline](#11-team-plan-and-timeline)
12. [Report Guide](#12-report-guide)

---

## 1. Assignment Requirements

| Requirement | Status | Evidence |
|-------------|:------:|----------|
| Gym-compatible environment | ✅ | `FluVaccineEnv` passes `check_env()` |
| Algorithm from Stable-Baselines3 | ✅ | PPO trained 500k timesteps |
| Algorithm matches state/action space | ✅ | PPO supports MultiDiscrete([4,4,4]) |
| RL algorithm implemented from scratch | ✅ | Q-Learning in `qlearning.py` — no SB3 |
| Clear business problem | ✅ | Flu vaccine resource management |
| Environment justification vs suggested envs | ✅ | See Section 3.1 |

### Responding to professor feedback

**On SB3 vs from scratch:**
Q-Learning is written from scratch (~50 lines, pure Python, Bellman update).
PPO and DQN use SB3 as a framework tool. The report clearly separates the two.
This is equivalent to using PyTorch rather than writing CUDA — the algorithm
understanding is demonstrated through Q-Learning.

**On custom vs third-party environment:**
We reviewed the OR-Gym Inventory environment and OR Library inventory management
environment from the MMAI-845 project list before building our own. See Section 3.1.

---

## 2. Project Structure

```
project/
├── envs/
│   └── flu_env.py        # Custom Gymnasium environment
├── train.py              # Train PPO (Stable-Baselines3)
├── train_dqn.py          # Train DQN — single-region wrapper
├── qlearning.py          # Q-Learning from scratch
├── vi_solver.py          # Value iteration (classical DP)
├── evaluate.py           # Compare all policies, 100 episodes
├── flu_rl_model.zip      # Saved PPO model (after training)
├── flu_dqn_model.zip     # Saved DQN model (after training)
└── README.md
```

---

## 3. Environment Design

### 3.1 Why custom — not the suggested inventory environments

Two environments from the MMAI-845 project list were reviewed:

- **OR-Gym Inventory** — single region, no expiry, uniform demand
- **OR Library inventory** — standard inventory control, single product

The flu vaccine problem requires three features neither environment provides:

1. Multi-region allocation with different storage capacities
2. Vaccine expiry tracking (4-week FIFO — doses spoil if unused)
3. Seasonal demand curve grounded in NACI 2024-2025 flu season data

`FluVaccineEnv` extends the standard inventory control structure with
these domain-specific additions while staying fully Gymnasium-compatible.
The base logic (order → receive → satisfy demand → penalise shortfall)
mirrors the OR-Gym approach. The extensions are our academic contribution.

The environment is intentionally scoped — 12 weeks, 3 regions, 4 order
options. The focus is on policy learning and algorithm comparison,
not building a large-scale simulator.

### 3.2 Regions

| Region | Type   | Base demand (peak) | Fridge cap | Vulnerability |
|--------|--------|--------------------|------------|---------------|
| 0      | Urban  | 180 doses/week     | 800 doses  | 1.2×          |
| 1      | Town   | 130 doses/week     | 600 doses  | 2.0×          |
| 2      | Rural  | 90 doses/week      | 1,000 doses| 1.5×          |

Vulnerability weights follow NACI 2024-2025 guidance (Canada.ca) —
elderly populations (Region 1) and limited healthcare access (Region 2)
face higher consequences from a missed dose.

### 3.3 State space (16 dimensions)

| Component            | Dims | Notes                              |
|----------------------|------|------------------------------------|
| Inventory            | 3    | Current stock per region           |
| Expiring soon        | 3    | Doses expiring next week           |
| Storage capacity     | 3    | Fixed fridge cap per region        |
| Demand level         | 3    | Last observed demand per region    |
| Vulnerability weight | 3    | NACI-grounded, fixed per region    |
| Week                 | 1    | 0–11                               |

### 3.4 Action space

`MultiDiscrete([4, 4, 4])` — each region independently selects an order:

| Index | Doses |
|:-----:|------:|
| 0     | 0     |
| 1     | 100   |
| 2     | 300   |
| 3     | 600   |

Total combinations: 4³ = 64 per week.

### 3.5 Reward function

```
reward = Σ vaccinated_i
       - Σ (vulnerability_i × stockout_i)   ← weighted by NACI priority
       - 1.5 × Σ expired_i
       - 0.01 × Σ inventory_i               ← holding cost
```

### 3.6 Seasonal demand

Bell-shaped curve peaking at weeks 6-7. Gaussian noise (std=20) prevents
the agent from memorising a fixed schedule.

```
Week:        1    2    3    4    5    6    7    8    9   10   11   12
Multiplier: 0.4  0.6  0.8  1.0  1.2  1.4  1.4  1.1  0.9  0.7  0.5  0.3
```

---

## 4. Algorithms

### Summary

| Algorithm | Implemented by | State space | Purpose |
|-----------|---------------|-------------|---------|
| Q-Learning | Us — from scratch | Discretised (324 states) | Demonstrates Bellman update |
| DQN | SB3 framework | Continuous (1 region) | Neural network Q-function |
| PPO | SB3 framework | Continuous (3 regions) | Main agent, full problem |
| Value iteration | Us — from scratch | Discretised (324 states) | Classical DP baseline |

### Q-Learning (from scratch)

Tabular implementation in `qlearning.py`. State space discretised to
3 inventory levels × 12 weeks = 324 states. Core update:

```python
Q[s][a] = Q[s][a] + alpha * (reward + gamma * max(Q[s_next]) - Q[s][a])
```

No SB3 used. This is the from-scratch implementation the assignment requires.

### DQN (SB3)

Extends Q-Learning with experience replay and a target network.
SB3's DQN requires `Discrete` (not `MultiDiscrete`), so we use a
single-region wrapper in `train_dqn.py` to demonstrate the algorithm.

### PPO (SB3)

Main agent. Handles `MultiDiscrete([4,4,4])` and the full 16-dim state.
Trained 500,000 timesteps. Uses actor-critic architecture with clipped
policy gradient to prevent unstable updates.

### Why this combination

Q-Learning breaks down at higher dimensions (324 states is already an
approximation). DQN scales better with neural networks but SB3 limits
it to single-action spaces here. PPO handles the full problem cleanly.
This progression — tabular → neural Q → actor-critic — is the standard
justification for using modern deep RL methods.

---

## 5. Setup and Installation

**Prerequisites:** Python 3.9+, pip

```bash
# Clone and enter the project
git clone <repo-url>
cd <project-folder>

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install gymnasium stable-baselines3 numpy
```

---

## 6. Running the Project

### Train PPO

```bash
python train.py
```
Trains 500k timesteps, saves `flu_rl_model.zip`. Takes ~5-10 minutes.

### Train DQN

```bash
python train_dqn.py
```
Single-region DQN wrapper, saves `flu_dqn_model.zip`.

### Run Q-Learning

```bash
python qlearning.py
```
Tabular Q-Learning on discretised state. Saves Q-table and policy.

### Evaluate all policies

```bash
python evaluate.py
```
100 episodes per policy. Prints three tables:
- Full policy comparison (reward, vaccinated, stockout, expired)
- Reward breakdown (each term separately)
- Per-region stockout

### Current baseline (PPO, 100 episodes)

```
Policy                    Reward           Vaccinated      Stockout     Expired
RL Agent (PPO)     3434.8 ± 204.5   4090.9 ± 105.3   27.9 ± 31.7  346.8 ± 73.6
Always order 300   -173.3 ± 231.8   4115.0 ± 102.8    0.0 ±  0.0  2717.3 ± 91.5
Always order 100   2201.9 ± 237.6   3669.3 ±  96.7  439.5 ± 64.1   347.7 ± 80.6
Random              772.0 ± 790.2   3960.8 ± 241.1  158.7 ± 201.6 1812.0 ± 369.3
```

### Train → freeze → retrain rule

Any change to `flu_env.py` invalidates the saved model.

```
1. Change flu_env.py
2. Commit to your branch
3. python train.py          ← full retrain
4. python evaluate.py       ← now valid
5. Record and commit results
```

Never evaluate using a model trained on a different environment version.

---

## 7. Policy Catalogue

| # | Policy | Type | Description |
|---|--------|------|-------------|
| 1 | RL Agent (PPO) | Learned — SB3 | Main agent, full state, 500k timesteps |
| 2 | Always order 300 | Naive baseline | Fixed high order regardless of demand |
| 3 | Always order 100 | Naive baseline | Fixed low order — undershoots at peak |
| 4 | Random | Naive baseline | Random each week — lower bound |
| 5 | Reorder point | Rule-based | Order 300 when inventory < 200 |
| 6 | Seasonal schedule | Rule-based | Pre-planned calendar matching the curve |
| 7 | Vulnerability first | Rule-based | Prioritises Region 1 — NACI grounded |
| 8 | Q-Learning | Learned — scratch | Tabular, discretised state |
| 9 | Value iteration | Classical DP | Approximate DP, discretised state |

Policies 2–4 are naive baselines. Policies 5–7 are human-designed rules
representing what a real health agency might implement. Policies 8–9 are
classical RL/DP methods. Policy 1 competes against all of them.

**Core research question:** does RL beat not just naive baselines but
also sensible human-designed rules?

---

## 8. MDP Reference

The problem is defined as **(S, A, T, R, γ)**.

```
S — 16-dim state: inventory, expiry, capacity, demand, vulnerability, week
A — MultiDiscrete([4,4,4]): 64 joint order combinations
T — stochastic: demand ~ Normal(base × seasonal(t), σ=20)
R — vaccinated − weighted stockout − 1.5×expired − 0.01×inventory
γ — 0.99 (high: early ordering decisions affect peak-week outcomes)
```

Bellman equation (what all algorithms are trying to solve):

```
Q*(s,a) = R(s,a) + γ · max_a' Q*(s_{t+1}, a')
```

Q-Learning solves this with a table. DQN approximates it with a neural
network. PPO estimates V*(s) and optimises the policy separately.

---

## 9. Experiment Branches

### Branch map

```
main  ← always stable, team lead owns
│
├── feat/env-state-expansion-seasonal-demand  ← merge to main now
│
├── experiment/qlearning-scratch              ← Member A, Week 4
├── experiment/catastrophic-demand-spike      ← Member B, Week 4
├── experiment/value-iteration                ← Member C, Week 5
├── experiment/order-lead-time                ← Member B, Week 5
└── experiment/venue-type-regions             ← stretch goal
```

### Branch reference

| Branch | What changes | Retrain? | Owner | Due |
|--------|-------------|:--------:|-------|-----|
| `experiment/qlearning-scratch` | `qlearning.py` + DQN wrapper, policies 8+9 | No | Member A | Wk 4 |
| `experiment/catastrophic-demand-spike` | 5% chance demand doubles in one region | Yes | Member B | Wk 4 |
| `experiment/value-iteration` | `vi_solver.py` — approx DP, 324 states | No | Member C | Wk 5 |
| `experiment/order-lead-time` | Orders arrive 2 weeks late, state expanded | Yes | Member B | Wk 5 |
| `experiment/venue-type-regions` | Regions as hospital/clinic/pharmacy | Yes | Member B | Stretch |

### Branch status: `experiment/catastrophic-demand-spike`

This branch is now implemented and retrained.

What changed:
- Added a rare catastrophic local outbreak event to `FluVaccineEnv`
- On each week, there is a 5% chance that one randomly selected region's demand doubles
- Spike metadata is exposed in `info` as `catastrophic_spike_region` and `catastrophic_spike_multiplier`
- PPO was fully retrained after the environment change, following the train-freeze-retrain rule

Implementation notes:
- The spike is applied after the normal seasonal demand plus Gaussian noise is generated
- The rest of the environment logic is unchanged: storage caps, FIFO usage, expiry, and reward calculation all remain consistent with the baseline branch
- The event is intentionally rare so it behaves like a stress-test scenario rather than a new normal demand pattern

Post-retrain evaluation:

```
Policy                                       Reward         Vaccinated           Stockout            Expired
RL Agent (PPO)                     3452.2 ± 248.9   4131.0 ± 137.2     49.8 ±  67.0    332.6 ±  81.1
Baseline: always order 300          -19.9 ± 334.6   4189.7 ± 152.4      0.0 ±   0.0   2665.1 ± 130.3
Baseline: always order 100         2190.8 ± 271.5   3700.3 ± 105.9    483.6 ± 102.5    317.8 ±  82.1
Baseline: random                    904.9 ± 875.2   4029.1 ± 230.3    142.4 ± 196.8   1789.1 ± 401.6
```

Interpretation:
- PPO remains the best policy by average reward under the catastrophic spike scenario
- The always-order-300 baseline still eliminates stockout, but performs poorly because expiry dominates the reward
- This branch is ready for team review or comparison against the baseline branch

---

## 10. Merging to Main

Main always holds the best stable version. Do not merge automatically.

**Merge if all three are true:**
- Average reward is meaningfully higher than current baseline
- The improvement is outside ±std overlap between the two versions
- Stockout and expiry are not significantly worse

**Do not delete failed branches.** A failed experiment that shows RL is
still relatively better than baselines is a valid result and belongs in
the report.

### Merge workflow

```bash
git checkout main && git pull
git merge experiment/your-branch
python train.py          # if env changed
python evaluate.py       # confirm improvement
git tag v1.x-description
git push origin main --tags
```

### Version tags

```
v1.0  initial PPO baseline
v1.1  post vulnerability-weighted-reward
v1.2  post catastrophic-demand-spike
v1.3  post order-lead-time
```

---

## 11. Team Plan and Timeline

### Roles

| Member | Owns |
|--------|------|
| Team lead | `main`, report writing, final decisions, merges |
| Member A | `qlearning.py`, `train_dqn.py`, algorithm section of report |
| Member B | `flu_env.py` experiments, environment section of report |
| Member C | `vi_solver.py`, presentation slides, Jupyter notebook |

### Week 4 — close the two gaps (parallel)

```
Team lead   Write env justification (Section 3.1 of report)
            Write proposal addendum — algorithm clarification
            Merge feat/ branch → main, brief team on codebase

Member A    Implement Q-Learning from scratch (qlearning.py)
            Add DQN single-region wrapper (train_dqn.py)
            Branch: experiment/qlearning-scratch

Member B    Add catastrophic demand spike to flu_env.py
            Retrain + evaluate
            Branch: experiment/catastrophic-demand-spike

Member C    Review OR-Gym + OR Library envs from prof list
            Document comparison vs our env for report
            Start presentation slide structure
```

### Week 5 — experiments + report drafting (parallel)

```
Team lead   Draft report: intro, env design, MDP section, results
            Review experiment results as they arrive

Member A    Run Q-Learning 100-episode evaluation
            Add policies 8+9 to evaluate.py
            Write algorithm section of report

Member B    Implement order-lead-time experiment
            Retrain + evaluate
            Write environment section of report

Member C    Implement value iteration (vi_solver.py)
            Build 7-slide presentation deck
            Write DP theory section of report
```

### Week 6 — finalise (parallel)

```
Team lead   Complete report — results + discussion
            Decide which branches merge to main
            Final proofread

Member A    Code cleanup + docstrings for qlearning.py
Member B    Code cleanup + docstrings for flu_env.py
Member C    Jupyter demo notebook + rehearsal coordination
```

### Minimum viable delivery

```
Must have:
  ✅ Custom Gym environment
  ✅ PPO via SB3
  ⚠️ Q-Learning from scratch      ← Member A, Week 4
  ⚠️ Env justification paragraph  ← Team lead, Week 4
  ⚠️ Report with MDP formulation  ← Week 5-6
  ⚠️ Presentation                 ← Week 6

Strong adds:
  DQN comparison, catastrophic spike, order lead time, 9-policy table

Distinction adds:
  Value iteration, Jupyter notebook, live simulation demo
```

---

## 12. Report Guide

### Section ownership

| Section | Owner |
|---------|-------|
| Introduction + business problem | Team lead |
| Environment design + justification | Member B |
| Algorithm description | Member A |
| MDP formulation | Team lead |
| Results and evaluation | Team lead |
| DP theory connection | Member C |
| Conclusion + future work | Team lead |

### Six things to state explicitly

1. Q-Learning was implemented from scratch — not via SB3
2. DQN and PPO use SB3 as a framework tool — cite the SB3 paper
3. Environment is custom — cite OR-Gym and OR Library and explain why they were insufficient
4. Seasonal curve is grounded in NACI 2024-2025 guidance — cite Canada.ca
5. All results are mean ± std over 100 episodes — never a single run
6. Correct spelling: Markov Decision Process (the proposal had a typo)

### Algorithm framing paragraph (use in report)

> Q-Learning was implemented from scratch using the Bellman update rule
> on a discretised state space, demonstrating understanding of the core
> RL update mechanism. DQN and PPO were implemented using Stable-Baselines3,
> which provides experience replay, target networks, and clipped policy
> gradient optimisation. This allows direct comparison between a hand-coded
> tabular method and two deep RL approaches on the same environment.

### Environment framing paragraph (use in report)

> The OR-Gym Inventory environment and OR Library inventory management
> environment from the MMAI-845 project list were reviewed prior to
> environment design. Both implement single-region, single-product
> inventory control without vaccine expiry or seasonal demand variation.
> FluVaccineEnv extends the standard inventory control formulation with
> multi-region allocation, FIFO expiry tracking, and a seasonal demand
> curve grounded in NACI 2024-2025 flu season data, while remaining
> fully Gymnasium-compatible.

---
