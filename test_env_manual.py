#!/usr/bin/env python3
"""
Manual testing script for the pushing environment.
Use keyboard to control the pusher and test different shapes.
Requires: pip install pynput
"""

import sys
import os
import numpy as np
import threading
import time
from pynput import keyboard
from pushing_envs.pushing_env_lstm_categorical import PushingEnv

def print_controls():
    """Print control instructions."""
    print("\n" + "="*60)
    print("ENVIRONMENT MANUAL CONTROL")
    print("="*60)
    print("\nThis environment uses DISCRETE actions [0-10, 0-10]")
    print("which are converted to velocities: action*0.02 - 0.1")
    print("\nControls:")
    print("  W      - Forward velocity      -> action [10, 5]")
    print("  S      - Backward velocity     -> action [0, 5]")
    print("  A      - Left velocity         -> action [5, 10]")
    print("  D      - Right velocity        -> action [5, 0]")
    print("  SPACE  - No velocity           -> action [5, 5]")
    print("  R      - Reset episode")
    print("  Q      - Quit")
    print("="*60 + "\n")

def select_shape():
    """Allow user to select which shape to test."""
    print("\nSelect shape to test:")
    print("1. Box")
    print("2. Cylinder")
    print("3. Whale")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice == "1":
            return "box"
        elif choice == "2":
            return "cylinder"
        elif choice == "3":
            return "whale"
        else:
            print("Invalid choice. Please enter 1-3.")

class KeyboardListener:
    """Capture keyboard input without blocking."""
    
    def __init__(self):
        self.last_key = None
        self.listener = None
    
    def on_press(self, key):
        """Handle key press."""
        try:
            # Handle character keys
            self.last_key = key.char if hasattr(key, 'char') else str(key).lower()
        except AttributeError:
            # Handle special keys
            self.last_key = str(key).lower()
    
    def get_key(self):
        """Get the last pressed key and clear it."""
        key = self.last_key
        self.last_key = None
        return key
    
    def start(self):
        """Start listening to keyboard input."""
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
    
    def stop(self):
        """Stop listening to keyboard input."""
        if self.listener:
            self.listener.stop()
            # Wait for listener thread to fully stop
            self.listener.join(timeout=1.0)

def main():
    print("\n" + "="*60)
    print("PUSHING ENVIRONMENT MANUAL TEST")
    print("="*60)
    
    while True:
        # Select shape
        shape_type = select_shape()
        print(f"\nSelected shape: {shape_type}")
        print("Creating environment...")
        
        try:
            # Create environment with graphics enabled
            # Use randomize_shapes=True but set shapes to only contain the selected shape
            # This way the first reset will use that shape
            env = PushingEnv(
                graphics=True, 
                seed=42, 
                fps=30,
                randomize_shapes=True,    # Enable randomization
                shapes=[shape_type]        # But only with the selected shape
            )
            print("✓ Environment created successfully!")
            
        except Exception as e:
            print(f"✗ Error creating environment: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print_controls()
        
        # Start keyboard listener
        kb_listener = KeyboardListener()
        kb_listener.start()
        print("Keyboard listener started. Ready for input!")
        
        episode_num = 0
        step_num = 0
        done = False
        action = np.array([5, 5])  # Default action (neutral)
        quit_requested = False
        
        try:
            # Reset environment
            obs = env.reset()
            print(f"\nEpisode {episode_num} started")
            print(f"Initial observation shape: {obs.shape}")
            
            while not quit_requested:
                # Check for keyboard input
                key = kb_listener.get_key()
                
                if key is None:
                    # No key pressed, continue with last action
                    time.sleep(0.02)  # Small delay to prevent CPU spinning
                    continue
                
                # Handle special keys
                if key == "'key.q'" or key == 'q':
                    print("\nQuitting current environment. Select another shape or Q to exit completely...")
                    quit_requested = True
                    break
                
                if key == "'key.r'" or key == 'r':
                    print(f"\nResetting episode...")
                    obs = env.reset()
                    episode_num += 1
                    step_num = 0
                    action = np.array([5, 5])
                    print(f"Episode {episode_num} started")
                    continue
                
                # Action mapping with discrete actions [0-10, 0-10]
                # Formula: velocity = action*0.02 - 0.1
                # action=0 -> velocity=-0.1, action=5 -> velocity=0, action=10 -> velocity=0.1
                action_map = {
                    'w': np.array([10, 5]),   # Forward: positive X velocity, center Y
                    's': np.array([0, 5]),    # Backward: negative X velocity, center Y
                    'a': np.array([5, 10]),   # Left: center X, positive Y velocity
                    'd': np.array([5, 0]),    # Right: center X, negative Y velocity
                    ' ': np.array([5, 5]),    # No action: center both axes
                }
                
                if key in action_map:
                    action = action_map[key]
                    
                    try:
                        # Step environment
                        obs, reward, done, info = env.step(action)
                        step_num += 1
                        
                        # Print step information
                        print(f"[Episode {episode_num}, Step {step_num}] Action: {action} | Reward: {reward:.4f} | Done: {done}")
                        
                        if done:
                            print("Episode finished!")
                            print(f"Final step: {step_num}")
                            obs = env.reset()
                            episode_num += 1
                            step_num = 0
                            action = np.array([5, 5])
                            print(f"Episode {episode_num} started")
                            
                    except Exception as e:
                        print(f"\n✗ Error during step: {e}")
                        import traceback
                        traceback.print_exc()
                        quit_requested = True
                        break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            quit_requested = True
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            quit_requested = True
        finally:
            kb_listener.stop()
            print("\nClosing environment...")
            env.close()
            print("✓ Done!")
            time.sleep(0.5)  # Small delay to ensure listener is fully stopped
        
        # Ask if user wants to try another shape
        print("\n" + "="*60)
        response = input("Do you want to try another shape? (y/n): ").strip().lower()
        if response == 'y':
            print("Restarting...\n")
            continue
        else:
            print("Exiting...")
            break
    
    print("\n✓ Goodbye!")

if __name__ == "__main__":
    main()
