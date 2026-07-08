import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt  # Added for plotting

# --- 1. SETUP ENVIRONMENT AND HYPERPARAMETERS ---
env = gym.make('FrozenLake-v1')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

EPISODES = 1500  
MAX_STEPS = 100  

LEARNING_RATE = 0.81  
GAMMA = 0.96          

RENDER = False  
epsilon = 0.9

# --- 2. MAIN TRAINING LOOP ---
rewards = []  

for episode in range(EPISODES):
    state, info = env.reset()
    
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()
            
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state, :])    
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])
        
        state = next_state
        
        if done:
            rewards.append(reward)      
            epsilon -= 0.001            
            break                       

# --- 3. EVALUATION AND RESULTS ---
print("Final Q-Table:")
print(Q)
print(f"\nOverall Average reward: {sum(rewards)/len(rewards)}")

# --- 4. PLOT TRAINING PROGRESS ---
# Helper function to compute averages
def get_average(values):
    return sum(values) / len(values)

avg_rewards = []

# Chop rewards into blocks of 100 episodes and average them
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i+100]))

# Create and display the plot
plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel("episodes (100's)")
plt.title('Agent Training Progress')
plt.show()
