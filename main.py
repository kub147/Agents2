import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os


# --- PART 1: CUSTOM ENVIRONMENT WRAPPER (Milestone 1) ---
class BalancedDriverWrapper(gym.Wrapper):
    """
    Custom Environment Modification: "Fast but Cautious".
    - High penalty for crashes remains (Safety first).
    - Reduced pressure to maintain maximum speed constantly.
    - Encourages smooth traffic flow.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Execute action in the original environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        speed = info['speed']
        custom_reward = 0.0

        # 1. Priority: SAFETY
        if info['crashed']:
            custom_reward = -40.0  # Heavy penalty, but allows agent to keep "hope"
            terminated = True  # Crash ends the episode

        else:
            # 2. Speed Reward
            # Encourage driving, but gently scaled
            custom_reward += (speed / 30.0) * 0.5

            # 3. REDUCED Speed Pressure
            # Only penalize for driving too slow (traffic obstruction)
            # Previously < 20, now < 10 to allow braking in traffic
            if speed < 10:
                custom_reward -= 0.1

            # 4. Survival/Smoothness Bonus
            # Reward for every step survived without crashing
            custom_reward += 0.2

        return obs, custom_reward, terminated, truncated, info


# --- PART 2: ENVIRONMENT CONFIGURATION (Milestone 1) ---
def create_env(render_mode="rgb_array"):
    """
    Creates and configures the Highway environment with balanced difficulty.
    """
    env = gym.make("highway-fast-v0", render_mode=render_mode)

    # Configuration: "The Golden Mean"
    env.unwrapped.configure({
        "lanes_count": 4,  # 4 lanes
        "vehicles_count": 30,  # Reduced from 50 (less chaotic traffic)
        "duration": 150,  # Optimal duration
        "initial_spacing": 4,  # Increased spacing (safer start)
        "vehicles_density": 1.5,  # Medium density (challenging but fair)
        "collision_reward": -1,  # (Overwritten by wrapper)
        "reward_speed_range": [20, 30],
        # Centering view - agent sees more around itself
        "centering_position": [0.3, 0.5],
    })

    # Apply custom reward logic
    env = BalancedDriverWrapper(env)
    return env


# --- PART 3: TRAINING & EVALUATION (Milestone 2 & 3) ---
def train_agent():
    # Setup directories
    models_dir = "models/PPO"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Create Training Environment
    env = create_env()

    # 2. Hyperparameters (Tuned for stability)
    learning_rate = 0.0003
    ent_coef = 0.01

    print("--- Initializing Agent ---")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        # tensorboard_log=log_dir # Uncomment if tensorboard is installed
    )

    print("--- Starting Training (Balanced Strategy) ---")
    # 30k steps is enough to see results with this balance
    TIMESTEPS = 70000
    model.learn(total_timesteps=TIMESTEPS)

    # Save Model
    model_path = f"{models_dir}/balanced_driver_ppo"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # 3. Evaluate Results
    print("--- Evaluating Trained Agent ---")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Trained Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model


# --- PART 4: VISUALIZATION ---
def test_agent_visual(model):
    print("--- Starting Visual Test ---")

    # Create environment for human viewing
    env = create_env(render_mode="human")

    # Configure display for smoother viewing
    env.unwrapped.configure({
        "simulation_frequency": 15,
        "policy_frequency": 2,  # Decisions every 2 frames (more human-like)
    })

    obs, _ = env.reset()

    for i in range(5):
        done = False
        total_score = 0
        print(f"--- Episode {i + 1} START ---")

        while not done:
            # Deterministic=True means use the best learned strategy
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            total_score += reward
            done = terminated or truncated

            if info.get('crashed'):
                print(f" -> Crash! Speed: {info['speed']:.1f}")

        print(f"Episode {i + 1} Finished. Score: {total_score:.2f}")
        obs, _ = env.reset()

    env.close()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    trained_model = train_agent()

    user_input = input("Training complete. Watch the agent drive? (y/n): ")
    if user_input.lower() == 'y':
        test_agent_visual(trained_model)