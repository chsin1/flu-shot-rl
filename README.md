# RL Flu Vaccine Resource Management
**MMAI-845 Group Project — Final Submission**
**Team Spadina | April 22, 2026**

## 1. Business Problem Overview
Seasonal influenza vaccination requires coordinated allocation of limited vaccine supply across regions with uneven demand patterns, strict storage capacity constraints, and a limited 4-week shelf life.

Health agencies must balance competing objectives:
- maximizing vaccinations (public health impact)
- minimizing stockouts (missed immunization opportunities)
- minimizing expiry (logistical waste)

This creates a sequential, stochastic decision problem under uncertainty, where demand fluctuates over time and across regions. Static, rule-based systems are insufficient because they cannot adapt dynamically to changing conditions.

## 2. Project Objective
This project models regional vaccine allocation as a Markov Decision Process (MDP) and investigates a key research question:

**When do classical decision-making methods break down, and when does Reinforcement Learning become necessary?**

To answer this, we design two environments:
- a simplified baseline environment where system dynamics are stable and predictable
- a realistic environment with stochastic demand, supply delays, and uncertainty

We evaluate:
- Classical Dynamic Programming (Value Iteration)
- Tabular RL (Q-Learning)
- Deep RL (PPO)
- Rule-based and heuristic policies

Across two environments:
- Baseline: simplified, predictable system
- Realistic: delayed supply + high uncertainty

## 3. Environment Design (`FluVaccineEnv`)
To test our algorithms, we built a custom Gymnasium environment simulating a 12-week flu season across three distinct regions (Urban, Town, Rural). The environment enforces universal global constraints, including strict 4-week FIFO expiry tracking. 

To evaluate algorithmic scalability, we established two testing configurations:

*   **Baseline Environment:** Simulates immediate fulfillment (0-week lead time), standard demand volatility (noise std=20.0), and a 5% probability of a 2.0x catastrophic demand spike.
*   **Realistic Environment:** Introduces severe real-world complexities, including a 2-week order lead time, higher demand volatility (noise std=30.0), and a doubled 10% probability of a 2.0x catastrophic demand shock.

## 4. MDP Formulation: State, Action, Reward
### State Space (13 dimensions)
The agent receives a continuously updated snapshot of all three regions:
*   Inventory (3 dimensions)
*   Expiring soon (3 dimensions)
*   Storage capacity (3 dimensions)
*   Demand level (3 dimensions)
*   Week index (1 dimension)

*Note: Regional vulnerability weights (1.2x, 2.0x, 1.5x) are fixed constants and are purposefully excluded from the state space to avoid redundant representation; they are encoded directly into the reward function instead.*

### Action Space: 
`Discrete (Baseline Model)`
- Options: {0, 100, 300, 600}
- Total combinations: 64 per week

`Continuous (Extension)`
- Rang: 0-600 doses per region
- Normalised to [0,1] for stable learning

### Reward Function
`Reward = Vaccinated - Weighted_Stockout - (1.5 × Expired) - (0.01 × Inventory)`
This aligns the agent with NACI public health guidelines by heavily penalizing stockouts in vulnerable regions, while applying a 1.5x penalty to expired doses to discourage over-ordering.

## 5. Algorithms Evaluated
*   **Discrete Action Setting:** 
- PPO (Deep RL)
- Value Iteration (DP)
- Q-Learning (Tabular)
- Rule-based policies

*   **Continuous Action Setting (Extension):** 
- PPO (continuous control)
- Continuous rule-based baselines

## 6. Setup and Installation

**Prerequisites:** Python 3.9+, pip

```bash
# Clone and enter the project
git clone https://github.com/chsin1/rl-flu-vaccine-inventory.git
cd <project-folder>

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

```

## 7. Key Results (100 Episodes)
### 1. Discrete Setting
The algorithms were evaluated across both the Baseline and Realistic environments to test breaking points.

### Baseline Environment (Simplified Logistics)
*In the simplified environment, classical model-based planning outperforms RL.*
| Policy | Type | Mean Reward |
| :--- | :--- | :--- |
| **Value Iteration (DP)**  | Classical DP | **3516.4 ± 353.8** (Best)  |
| **RL Agent (PPO)** | Learned | **3455.8 ± 278.9**  |
| Reorder Point (<200→300) | Rule-based | 3414.1 ± 280.7  |
| Always Order 100 | Naive | 2507.0 ± 265.7  |
| Q-Learning (scratch) | Tabular RL | 983.0 ± 427.4  |
| Always Order 300 | Naive | 32.7 ± 348.0  |

### Realistic Environment (Delayed Supply & High Volatility)
*Under real-world uncertainty, Value Iteration's transition model breaks down entirely, and PPO dominates.*
| Policy | Type | Mean Reward |
| :--- | :--- | :--- |
| **RL Agent (PPO)** | Learned | **2949.7 ± 405.0** (Best) |
| Seasonal Schedule | Rule-based | 1714.7 ± 412.3  |
| Always Order 100 | Naive | 1560.1 ± 320.3  |
| Always Order 300 | Naive | 1307.9 ± 477.8  |
| **Value Iteration (DP)** | Classical DP | **-5087.0 ± 342.1** (Fails)  |

### 2. Continuous Setting
Baseline Environment
| Policy | Reward | Stockout | Expired |
| :--- | :--- | :--- | :--- |
| PPO (continuous)  | 3158	| 31.6 | 901
| Reorder Point | 2373 | 13.7 | 1708

👉 PPO reduces expiry significantly while maintaining service levels.

Realistic Environment
| Policy |	Reward |
| :--- | :--- |
| Random | 1596 (Best) |
| PPO (continuous) | 971 |
| Reorder Point	| 825 |

👉 Continuous PPO performance drops under uncertainty.

## 8. Key Findings & Policy Insights
### 1. Value Iteration Wins in Simple Environments, Fails in Reality
Value Iteration achieved the highest reward (3516.4) in the Baseline because its discretized transition model was perfectly suited to immediate order fulfillment. However, when delayed supply (2-week lead times) and stochastic shocks (10%) were introduced, the transition model became completely unreliable, causing the algorithm to plummet to a catastrophic score of -5087.0.

### 2. PPO Learns Dynamically Robust Defenses
PPO's learned behavior organically adapted between environments. In the Baseline, it settled into a conservative rhythm, largely ordering `` and spiking only when inventory dropped. Conversely, in the Realistic environment, it learned that it could not react fast enough to 2-week delays. PPO organically shifted to a rigid `` structure almost every week, dynamically balancing the trade-off by actively building large buffers in the most volatile regions to survive demand shocks.

### 3. Continuous Control Requires Proper Scaling
Initial PPO attempts failed due to poor action scaling.
Normalization ([0,1] → doses) was critical for learning meaningful policies.

### 4. Flexibility vs Robustness Trade-off
Environment	Best Approach
Baseline	PPO (optimized)
Realistic	Random (robust)

👉  Insight:
PPO is efficient but fragile
Random is robust under uncertainty

### 5. Regional Failure Mode
PPO struggled under realistic settings:
Severe under-supply in rural region
High stockout due to delayed supply

## 9. The Path Forward
The MDP framework built here serves as foundational architecture for deployment-ready logistics AI.
*   **Phase 1:** Logistic Complexity (Achieved).
*   **Phase 2:** Epidemiological Fidelity (Integrating disease spread models and multi-agent coordination).
*   **Phase 3:** Real-World Integration (Ingesting live data for deployment).

## 10. Conclusion
Vaccine allocation is a complex sequential decision problem.
- Classical DP works in simple environments
- PPO scales better under uncertainty
- Continuous control improves efficiency
- BUT real-world deployment requires robustness to uncertainty

## 11. How to Run
### Discrete Models
```bash
# Train PPO
python train.py --env baseline
python train.py --env realistic

# Train Q-Learning (baseline environment only)
python qlearning.py

# Solve Value Iteration
python vi_solver.py --env baseline
python vi_solver.py --env realistic

# Evaluate all policies in baseline environment
python evaluate.py --env baseline

# Evaluate PPO + VI (and available policies) in realistic environment
python evaluate.py --env realistic
```
### Continuous Models
```bash
python train_continuous.py --env baseline 
python train_continuous.py --env realistic 
python evaluate_continuous.py --env baseline 
python evaluate_continuous.py --env realistic
```

## 12. Monitoring Training (TensorBoard)
PPO training logs are recorded for both baseline and realistic environments and can be visualized using TensorBoard.
```bash
# Launch TensorBoard
tensorboard --logdir=ppo_tensorboard
Then open in your browser:
http://localhost:6006


***
