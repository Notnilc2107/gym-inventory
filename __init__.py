# This registers the inventory_env.py file to the gymnasium library.
# Based off of the code in train_tql_agent.py, you might be wondering why we use the gymnasium library at all if env.step() and env.reset() are already defined.
# The reason is because the gymnasium library has code for parallelisation. Look up vectorized environments if you're curious.
from gymnasium.envs.registration import register

register(
    id='gym-inventory/Inventory-v0',
    entry_point='gym_inventory.envs:InventoryEnv',
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)
