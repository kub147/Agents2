# 🚗 Highway Driving RL Agent

## Project Overview
This project implements a Reinforcement Learning agent using **Stable-Baselines3** and the **Highway-Env** gymnasium environment. The goal is to train an autonomous vehicle to navigate highway traffic efficiently, balancing high speed with safety.

We have customized the environment rewards to create a "Balanced Driver" personality—one that drives fast but avoids crashes at all costs.

## 🚀 Current Status (What is done)
* **Environment Setup:** `highway-fast-v0` is configured and working.
* **Custom Wrapper:** `BalancedDriverWrapper` is implemented. It modifies the reward function to:
    * Penalize collisions heavily (-40).
    * Reward speed, but penalize driving too slow (traffic obstruction).
    * Reward survival/smooth driving.
* **Training Pipeline:** The PPO (Proximal Policy Optimization) algorithm is integrated and working.
* **Visualization:** Basic visual test after training is implemented.

---

## 🛠️ Installation

1.  **Clone the repository.**
2.  **Create a virtual environment** (recommended).
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

To train the agent and see the results, run:
```bash
python main.py
```

# To-Do List & Task Distribution (Remaining 2 Weeks)

We have a solid foundation, but to get a top grade, we need to expand the analysis and prepare the presentation. Please pick a task and update the group chat.

## 1. Code & Experimentation 🧪

### Create a "Load & Visualize" Script
- Currently, `main.py` trains the model every time. We need a separate script (e.g., `enjoy.py`) that simply loads a pre-trained `.zip` model and runs the visualization.
- **Goal:** To quickly demonstrate the agent during the presentation without waiting for training.

### Environment Tuning
- Experiment with `vehicles_count` and `vehicles_density` in the configuration. Find the "sweet spot" where the traffic is heavy enough to be impressive but not impossible to navigate.
- **Bonus:** Try adding obstacles or changing the road geometry if possible.

### The "3-Stage" Experiment (Crucial for Analysis)
Train 3 separate models with different `total_timesteps`:

1. **Short Training** (e.g., 10k steps) - "The Novice"
2. **Medium Training** (e.g., 50k steps) - "The Learner"
3. **Long Training** (e.g., 200k+ steps) - "The Expert"

Record statistics (Mean Reward) and save videos/gifs of their behavior for comparison.

## 2. Documentation & Presentation 📊

### Code Documentation
- Go through `main.py` and add detailed docstrings/comments explaining why we chose specific PPO hyperparameters and reward values.

### Presentation Slides
Create the final presentation. It must cover:

- Problem definition
- Our modifications (The Wrapper)
- Algorithm used (PPO)
- Results Analysis: Show the comparison charts of Short vs. Medium vs. Long training
- Visual demos (videos/live demo)

## ⚠️ Workflow & Deadlines

1. **Communication:** If you change any parameters or code, write on the group chat immediately.
2. **Git:** Push your changes to GitHub regularly. Do not keep files locally.
3. **Timing:** Please do not leave this for the last weekend. We need time to compile the final report/presentation.