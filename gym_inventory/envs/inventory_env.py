import gymnasium as gym
from gymnasium import spaces
from gymnasium import utils
import numpy as np

import logging
logger = logging.getLogger(__name__)


class InventoryEnv(gym.Env, utils.EzPickle):
    """Inventory control with lost sales environment

    Originally by Paul Hendricks. Updated to the current version of Gymnasium by Clinton Nguyen.

    This environment corresponds to the version of the inventory control
    with lost sales problem described in Example 1.1 in Algorithms for
    Reinforcement Learning by Csaba Szepesvari (2010).
    https://sites.ualberta.ca/~szepesva/RLBook.html
    """

    def __init__(self, initial_inventory_level=20,
                 n=20, # maximum inventory level
                 k=5, # fixed cost of ordering a non-zero number of items
                 c=2, # cost of ordering a single item
                 h=2, # cost of holding a single item in storage
                 p=3, # revenue from selling a single item
                 lam=8 # average demand within a single timestep
                ):
        self.n = n
        self.action_space = spaces.Discrete(n + 1) # actions are the number of items ordered during the evening. the store is closed at night. + 1 because index starts at 0.
        self.observation_space = spaces.Discrete(n + 1) # states are the number of items that are being held in storage by the evening
        self.max = n
        self.state = initial_inventory_level
        self.k = k
        self.c = c
        self.h = h
        self.p = p
        self.lam = lam

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, x, a, d):
        m = self.max
        return max(min(x + a, m) - d, 0)

    def reward(self, x, # current inventory level
                     a, # order quantity
                     y  # observed demand
                    ):
        k = self.k
        m = self.max
        c = self.c
        h = self.h
        p = self.p
        
        fixed_cost = k * (a > 0)
        order_cost =  c * max(min(x + a, m) - x, 0)
        holding_cost = h * x
        revenue = p * max(min(x + a, m) - y, 0)
        
        r = fixed_cost + order_cost + holding_cost - revenue
        r = r * -1 # take negative because RL algorithms maximize reward but we want to minimize cost. maximizing negative is minimizing positive.
        return r

    def step(self, action):
        assert self.action_space.contains(action)
        observation = self.state
        demand = self.demand()
        new_observation = self.transition(observation, action, demand)
        self.state = new_observation
        reward = self.reward(observation, action, new_observation)
        truncated = False
        terminated = False
        return new_observation, reward, terminated, truncated, {}

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        return self.state, {} # nothing fancy is done here because demand is stochastic so initial state will be stochastic. other environments usually sample initial state randomly but that's already done.
