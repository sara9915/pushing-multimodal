import numpy as np
import time
from ..pushing_envs.pushing_env_lstm_categorical import PushingEnv
from sb3_contrib import RecurrentPPO

def select_shape():
    """Allow user to select which shape to test"""
    print("\n" + "="*50)
    print("SELECT SHAPE FOR TESTING")
    print("="*50)
    print("1. Box")
    print("2. Cylinder")
    print("3. Whale")
    print("="*50)
    while True:
        try:
            choice = int(input("Enter choice (1-3): "))
            if choice == 1:
                return 'box'
            elif choice == 2:
                return 'cylinder'
            elif choice == 3:
                return 'whale'
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Load model once
model = RecurrentPPO.load("pushing-multimodal/run_policies/policies/ppo_lstm_categorical")

# Test statistics
overall_successes = 0
overall_failures = 0
shape_stats = {'box': {'success': 0, 'total': 0}, 
               'cylinder': {'success': 0, 'total': 0},
               'whale': {'success': 0, 'total': 0}}

while True:
    # Select shape
    selected_shape = select_shape()
    
    # Create environment with fixed shape (no randomization)
    env = PushingEnv(graphics=True, randomize_shapes=False, shapes=[selected_shape])
    env.half_success_threshold()
    env.disturbances = False
    
    print(f"\nTesting {selected_shape.upper()} shape...")
    print("="*50)
    
    # Episode loop for selected shape
    episodes_tested = 0
    episodes_success = 0
    
    while episodes_tested < 10:  # Test 10 episodes per shape
        # Force the selected shape for this episode
        env.shape_type = selected_shape
        
        obs = env.reset()
        lstm_states = None
        num_envs = 1
        episode_starts = np.ones((num_envs,), dtype=bool)
        reward_ep = 0
        done = False
        episode_reward = 0
        
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_starts = done
            reward_ep += reward
            episode_reward = reward  # Keep track of the last reward
            time.sleep(1/30)
        
        episodes_tested += 1
        if episode_reward == 50:  # Success: final reward is 50
            episodes_success += 1
            overall_successes += 1
            print(f"Episode {episodes_tested}: SUCCESS (total reward: {reward_ep:.1f})")
        else:
            overall_failures += 1
            print(f"Episode {episodes_tested}: FAILED (total reward: {reward_ep:.1f})")
        
        shape_stats[selected_shape]['success'] += (1 if episode_reward == 50 else 0)
        shape_stats[selected_shape]['total'] += 1
    
    env.close()
    
    # Print statistics for this shape
    success_rate = (episodes_success / episodes_tested) * 100
    print(f"\n{selected_shape.upper()} Results: {episodes_success}/{episodes_tested} successful ({success_rate:.1f}%)")
    
    # Print overall statistics
    print("\n" + "="*50)
    print("OVERALL STATISTICS")
    print("="*50)
    for shape in ['box', 'cylinder', 'whale']:
        if shape_stats[shape]['total'] > 0:
            rate = (shape_stats[shape]['success'] / shape_stats[shape]['total']) * 100
            print(f"{shape.capitalize():12} {shape_stats[shape]['success']:2}/{shape_stats[shape]['total']:2} ({rate:5.1f}%)")
    
    print("="*50)
    
    # Ask if user wants to test another shape
    while True:
        response = input("\nTest another shape? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            break
    
    if response == 'n':
        print("Closing test. Goodbye!")
        break