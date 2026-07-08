import gymnasium as gym
import numpy as np
import time

# --- 1. SETUP ENVIRONMENT AND HYPERPARAMETERS ---

# Use the updated v1 environment
env = gym.make('FrozenLake-v1')

# Get total number of states and actions
STATES = env.observation_space.n
ACTIONS = env.action_space.n

# Initialize the Q-Table with zeros
Q = np.zeros((STATES, ACTIONS))

# Training configuration
EPISODES = 1500  
MAX_STEPS = 100  

LEARNING_RATE = 0.81  
GAMMA = 0.96          

RENDER = False  
epsilon = 0.9

# --- 2. MAIN TRAINING LOOP ---

rewards = []  

for episode in range(EPISODES):
    
    # Gymnasium reset returns a tuple: (initial_state, info)
    state, info = env.reset()
    
    for _ in range(MAX_STEPS):
        
        if RENDER:
            env.render()
            
        # Epsilon-Greedy Action Selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = np.argmax(Q[state, :])    # Exploit
            
        # Gymnasium step returns 5 values instead of 4
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Q-Table update formula (Bellman Equation)
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])
        
        state = next_state
        
        if done:
            rewards.append(reward)      
            epsilon -= 0.001            # Decay exploration rate
            break                       

# --- 3. EVALUATION AND RESULTS ---

print("Final Q-Table:")
print(Q)

print(f"\nAverage reward: {sum(rewards)/len(rewards)}")