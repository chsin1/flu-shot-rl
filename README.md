# RL Flu Vaccine Resource Management
**MMAI-845 Group Project — Final Submission**  
**Team Spadina | April 2026**

## 1. Business Problem Overview
Seasonal influenza vaccination requires coordinated allocation of limited vaccine supply across regions with uneven demand patterns, strict storage constraints, and a limited shelf life.

Health agencies must balance several competing goals:
- maximize vaccinations
- minimize stockouts
- minimize expiry
- maintain fair service across regions, especially vulnerable ones

Because demand changes over time and today's ordering decisions affect future inventory, this is naturally a sequential decision problem under uncertainty.

## 2. Project Objective
This project models regional flu vaccine allocation as a Markov Decision Process (MDP) and investigates the following question:

**When do classical planning methods work well, and when does Reinforcement Learning become more effective?**

To answer this, we compare:
- Value Iteration (classical dynamic programming)
- Q-Learning (tabular reinforcement learning)
- PPO (deep reinforcement learning)
- Rule-based baselines

across:
- a discrete action setting
- a continuous action setting

and under two environments:
- Baseline: simpler and more predictable
- Realistic: delayed supply and higher uncertainty

## 3. Environment Design
We built custom Gymnasium environments simulating a 12-week flu season across three regions:
- Urban
- Town
- Rural

Key operational features:
- fixed storage capacity per region
- FIFO inventory usage
- 4-week vaccine expiry window
- seasonal demand variation
- stochastic demand noise
- occasional catastrophic demand spikes
- optional lead-time delay on deliveries

### Environment Presets

#### Baseline Environment
- immediate delivery (`lead_time_weeks = 0`)
- moderate demand volatility (`demand_noise_std = 20.0`)
- catastrophic spike probability = `0.05`
- catastrophic spike multiplier = `2.0`

#### Realistic Environment
- 2-week delivery delay (`lead_time_weeks = 2`)
- higher demand volatility (`demand_noise_std = 30.0`)
- catastrophic spike probability = `0.10`
- catastrophic spike multiplier = `2.0`

## 4. MDP Formulation

### State Space
The state is a 13-dimensional continuous vector:

- inventory `(3)`
- expiring soon `(3)`
- storage capacity `(3)`
- current demand `(3)`
- week index `(1)`

### Action Space

#### Discrete Setting
Each region chooses one shipment size from:
- `{0, 100, 300, 600}` doses

With 3 regions, this gives:
- `4^3 = 64` joint actions per week

#### Continuous Setting
The continuous environment uses:
- normalized action space `Box([0,1]^3)`

These actions are internally scaled to:
- `0..600` doses per region

## 5. Reward Function

### Discrete Reward
The discrete setting uses the original reward structure:

\[
R_t
=
\sum_{r=1}^{3} V_{r,t}
-
\sum_{r=1}^{3} w_r SO_{r,t}
-
1.5\sum_{r=1}^{3} E_{r,t}
-
0.01\sum_{r=1}^{3} I_{r,t}
\]

### Continuous Reward
The final continuous setting uses:

\[
R_t
=
\sum_{r=1}^{3} V_{r,t}
-
\sum_{r=1}^{3} w_r SO_{r,t}
-
1.5\sum_{r=1}^{3} E_{r,t}
-
0.01\sum_{r=1}^{3} I_{r,t}
-
\lambda \sum_{r=1}^{3} w_r \max(S_r - I_{r,t}, 0)
\]

Where:
- \(V_{r,t}\) = vaccinated doses
- \(SO_{r,t}\) = stockout
- \(E_{r,t}\) = expired doses
- \(I_{r,t}\) = ending inventory
- \(w_r\) = regional vulnerability weight
- \(S_r\) = safety-stock target
- \(\lambda\) = understock penalty coefficient

### Realistic Continuous Parameters
For the current realistic continuous setup:
- `w = [1.2, 1.8, 2.8]`
- `S = [180, 150, 220]`
- `λ = 0.6`

This reward redesign was introduced to discourage the learned policy from under-supplying the rural region.

## 6. Methods Evaluated

### Discrete Setting
- PPO
- Value Iteration
- Q-Learning
- Always Order 100
- Always Order 300
- Random
- Reorder Point
- Seasonal Schedule
- Vulnerability First

### Continuous Setting
- PPO (continuous)
- Always Order 100
- Always Order 300
- Random
- Reorder Point
- Seasonal Schedule
- Vulnerability First
- Capacity Fill

## 7. Important Implementation Notes

### Continuous Action Scaling
Initial continuous PPO results were poor because of action-scaling issues.

The continuous environment expects normalized actions in `[0,1]`, which are then internally scaled to doses. After fixing this, PPO learned meaningful ordering behavior.

### Rural Protection in Realistic Continuous Setting
Earlier PPO runs could under-supply the rural region. To address this, we:
- increased rural vulnerability weighting
- added a weighted understock penalty
- retrained PPO under the revised objective

This significantly improved realistic continuous PPO performance.

## 8. Final Results

## 8.1 Discrete Results

### Baseline Environment
| Policy | Type | Mean Reward |
| :--- | :--- | :--- |
| **Value Iteration (DP)** | Classical DP | **3516.4 ± 353.8** |
| **RL Agent (PPO)** | Learned | **3477.8 ± 267.2** |
| Reorder Point (<200→300) | Rule-based | 3414.1 ± 280.7 |
| Always Order 100 | Naive | 2507.0 ± 265.7 |
| Vulnerability First (NACI) | Rule-based | 2117.2 ± 291.5 |
| Seasonal Schedule | Rule-based | 2003.9 ± 319.4 |
| Q-Learning (scratch) | Tabular RL | 1080.6 ± 377.4 |
| Random | Naive | 1025.8 ± 717.2 |
| Always Order 300 | Naive | 32.7 ± 348.0 |

**Interpretation:**  
In the simple discrete setting, classical dynamic programming performs best, with PPO close behind.

### Realistic Environment
| Policy | Type | Mean Reward |
| :--- | :--- | :--- |
| **RL Agent (PPO)** | Learned | **2895.7 ± 379.3** |
| Seasonal Schedule | Rule-based | 1714.7 ± 412.3 |
| Random | Naive | 1573.2 ± 961.2 |
| Always Order 100 | Naive | 1560.1 ± 320.3 |
| Always Order 300 | Naive | 1307.9 ± 477.8 |
| Reorder Point (<200→300) | Rule-based | 1261.4 ± 598.0 |
| Vulnerability First (NACI) | Rule-based | 1156.4 ± 514.7 |
| **Value Iteration (DP)** | Classical DP | **-5087.0 ± 342.1** |

**Interpretation:**  
Under delay and uncertainty, PPO becomes the strongest discrete policy while Value Iteration breaks down badly.

## 8.2 Continuous Results

### Baseline Environment
| Policy | Reward | Stockout | Expired |
| :--- | ---: | ---: | ---: |
| Reorder Point | **3414.1** | 27.1 | 445.1 |
| PPO (continuous) | 3182.6 | 34.1 | 582.4 |
| Always Order 100 | 2507.0 | 511.2 | 319.9 |
| Vulnerability First | 2117.2 | 583.7 | 491.2 |
| Seasonal Schedule | 2003.9 | 8.8 | 1370.7 |
| Random | 611.4 | 14.4 | 2257.0 |
| Always Order 300 | 32.7 | 0.0 | 2649.1 |
| Capacity Fill | -632.2 | 0.0 | 3091.2 |

**Interpretation:**  
In the easier continuous setting, PPO performs strongly, but the reorder-point rule is still best.

### Realistic Environment
| Policy | Reward |
| :--- | ---: |
| **PPO (continuous)** | **1654.5** |
| Capacity Fill | 1474.9 |
| Random | 1229.8 |
| Always Order 300 | 1184.8 |
| Seasonal Schedule | 738.3 |
| Vulnerability First | -1285.0 |
| Reorder Point | -1762.5 |
| Always Order 100 | -2951.3 |

**Interpretation:**  
After correcting continuous action scaling and redesigning the reward to better protect rural inventory, PPO becomes the best-performing continuous policy in the realistic environment.

## 9. Main Findings

### 1. Classical Planning Wins Only in the Simplest Case
Value Iteration is strongest in the baseline discrete environment, where dynamics are predictable.

### 2. PPO Is the Most Robust Overall Method
PPO is:
- near-optimal in simple settings
- best in the realistic discrete environment
- best in the realistic continuous environment

### 3. Continuous Control Requires Correct Action Scaling
Continuous PPO performance depends heavily on correct normalization and internal scaling.

### 4. Reward Design Matters
Adding a weighted understock penalty changed learned behavior substantially and improved fairness across regions.

### 5. Strong Heuristics Still Matter
In simpler settings, strong inventory heuristics such as reorder-point control can remain highly competitive.

## 10. Limitations
This project is realistic enough to demonstrate meaningful policy differences, but it is not yet a deployment-grade real-world system.

Current limitations:
- only 3 regions
- stylized seasonal demand curve
- no upstream supplier shortages
- no cold-chain failures
- no epidemiological disease spread model
- fairness is still partly encoded through reward shaping

## 11. Setup and Installation

### Prerequisites
- Python `3.9+`
- `pip`

### Clone the Repository
```bash
git clone https://github.com/chsin1/rl-flu-vaccine-inventory.git
cd rl-flu-vaccine-inventory
```

### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 12. How to Run
### Train Discrete PPO
```bash
python train.py --env baseline
python train.py --env realistic
```

### Train Q-Learning
```bash
python qlearning.py
```

### Solve Value Iteration
```bash
python vi_solver.py --env baseline
python vi_solver.py --env realistic
```

### Evaluate Discrete Policies and Save Summary Results
```bash
python evaluate.py --env baseline --save-results discrete_baseline_results.json
python evaluate.py --env realistic --save-results discrete_realistic_results.json
```

### Evaluate Discrete Policies and Save Text Output
```bash
python evaluate.py --env baseline | tee discrete_baseline_output.txt
python evaluate.py --env realistic | tee discrete_realistic_output.txt
```

### Train Continuous PPO
```bash
python train_continuous.py --env baseline
python train_continuous.py --env realistic
```
### Evaluate Continuous Policies and Save Summary Results
```bash
python evaluate_continuous.py --env baseline --save-results continuous_baseline_results.json
python evaluate_continuous.py --env realistic --save-results continuous_realistic_results.json
```

### Evaluate Continuous Policies and Save Text Output
```bash
python evaluate_continuous.py --env baseline --save-output continuous_baseline_output.txt
python evaluate_continuous.py --env realistic --save-output continuous_realistic_output.txt
```

## 13. TensorBoard
PPO training logs can be visualized with TensorBoard.

### Discrete PPO Logs
- `ppo_tensorboard/baseline/`
- `ppo_tensorboard/realistic/`

### Continuous PPO Logs
- `ppo_tensorboard/continuous_baseline/`
- `ppo_tensorboard/continuous_realistic/`

Launch TensorBoard with:
```bash
tensorboard --logdir=ppo_tensorboard
```

Then open:
```text
http://localhost:6006
```

## 14. Final Conclusion
The overall pattern is clear:

- Classical planning can outperform RL in simple, stable environments
- PPO is the most robust method once realism and uncertainty increase
- Continuous control improves flexibility, but only when action scaling is handled correctly
- Reward design is as important as algorithm choice when fairness and operational priorities matter

**Final takeaway:**  
PPO was the strongest overall method across the realistic discrete and continuous environments, while classical or rule-based methods remained competitive in the simplest settings.

## 15. Repository
GitHub repository:  
https://github.com/chsin1/rl-flu-vaccine-inventory.git
