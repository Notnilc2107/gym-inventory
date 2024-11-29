import pickle
# from tqdm import tqdm
import gym-inventory
import gymnasium as gym
import numpy as np
from collections import defaultdict
import time

class TabularQLearningAgent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor = 0.95):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        # qTable is defined as a defaultdict. A defaultdict is just a dictionary that can take in any key.
        # The default value is np.zeros(env.action_space.n). you can change it to anything, even a string.
        # The reason why the size of qTable is only the size of the action space (as opposed to the state-action space) is because we use
        # the observations as keys. The reason why it is done this way is because the action space is usually smaller than the state space.
        
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
    def get_epsilon_greedy_action(self, obs):
        if np.random.random() < self.epsilon: # random action
            return self.env.action_space.sample()
        else:                                 # greedy action
            return int(np.argmax(self.q_values[obs]))

    def updateQvalues(self, obs, action, reward, terminated, next_obs):
        greedy_next_q_value = (not terminated) * np.max(self.q_values[obs]) # np.max() is maximizing over the actions because self.q_values is a 
                                                                            # default dict.
        temporal_difference = (
            self.discount_factor * greedy_next_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * ( reward + temporal_difference )
        )

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

"Training the agent"
learning_rate = 0.01
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make('gym-inventory/Inventory-v0')

agent = TabularQLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

start = time.time()

episodicreturns = []
for episode in  (range(n_episodes)):
    obs, info = env.reset()
    done = False
    current_episodic_return = 0

    # play one episode
    while not done:
        action = agent.get_epsilon_greedy_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.updateQvalues(obs, action, reward, terminated, next_obs)
        
        done = terminated or truncated
        obs = next_obs
        current_episodic_return += reward

    episodicreturns += [current_episodic_return]
    agent.decay_epsilon()
    if episode % 10 == 0:
        print("episode: "+str(episode)+" total reward: "+str(episodicreturns[episode])+" epsilon: "+str(agent.epsilon))

end = time.time()

print('\n---------------------------------\n')
print('total training time: ', end-start)
with open('q_values.pkl', 'wb') as f:
    pickle.dump({k:v for k,v in agent.q_values.items()}, f)

# with open('q_values.pkl', 'rb') as f:
#     qvals = pickle.load(f)

print('optimal action for each state:')
# {'I='+str(k[0]):np.argmax(v) for k,v in dict(sorted(agent.q_values.items())).items()}
for k, v in dict(sorted(agent.q_values.items())).items():
    print('when state='+str(k),'the optimal action=',int(np.argmax(v)))
