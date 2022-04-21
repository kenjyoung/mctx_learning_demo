import jax as jx
import jax.numpy as jnp
from jax import jit
from functools import partial
# import operator

#0: no_op, 1: left, 2: down, 3: right, 4: up
move_map = jnp.asarray([[0, 0], [-1,0], [0,-1], [1,0], [0,1]])

class Asterix:
    def __init__(self, grid_size=10, spawn_rate=0.5, action_failure_prob=0.1):
        self._num_actions = 5
        self.grid_size = grid_size
        self.gold_prob = 1/3
        self.spawn_rate = spawn_rate
        self.action_failure_prob = action_failure_prob
        self.channels ={
            'player':0,
            'enemy':1,
            'left':2,
            'right':3,
            'gold':4
        }

    @partial(jit, static_argnums=(0,))
    def step(self, action, env_state):
        pos, is_enemy, is_gold, movement_dir, entity_x, key = env_state
        reward = 0
        terminal = False

        # Move player
        key, subkey = jx.random.split(key)
        action_failure = jx.random.bernoulli(subkey, p=self.action_failure_prob)
        pos = jnp.where(action_failure,pos,jnp.clip(pos+move_map[action], 0, self.grid_size-1))

        # Move gold and enemies, remove if moved off-screen
        entity_x = jnp.clip(jnp.where(movement_dir, entity_x-1,entity_x+1),-1, self.grid_size)
        is_gold = jnp.where(jnp.logical_or(entity_x<0,entity_x>self.grid_size-1),False,is_gold)
        is_enemy = jnp.where(jnp.logical_or(entity_x<0,entity_x>self.grid_size-1),False,is_enemy)

        # Give reward and remove gold if player touches gold, terminate if they touch an enemy
        reward = jnp.logical_and(entity_x[pos[1]]==pos[0],is_gold[pos[1]]).astype(int)
        is_gold = is_gold.at[pos[1]].set(jnp.where(entity_x[pos[1]]==pos[0], False, is_gold[pos[1]]))
        terminal = jnp.logical_and(entity_x[pos[1]]==pos[0],is_enemy[pos[1]])

        # Maybe spawn new gold or enemy
        free_slot = jnp.any(jnp.logical_not(jnp.logical_or(is_enemy,is_gold)))
        key, subkey = jx.random.split(key)
        spawn_entity = jnp.logical_and(jx.random.bernoulli(subkey, p=self.spawn_rate), free_slot)
        key, subkey = jx.random.split(key)
        spawn_gold = jnp.logical_and(jx.random.bernoulli(subkey, p=self.gold_prob), spawn_entity)
        spawn_enemy = jnp.logical_and(jnp.logical_not(spawn_gold), spawn_entity)
        key, subkey = jx.random.split(key)
        spawn_right = jx.random.bernoulli(subkey, p=0.5)
        spawn_x = jnp.where(spawn_right, self.grid_size-1, 0)

        key, subkey = jx.random.split(key)
        proposed_slots = jx.random.permutation(subkey, self.grid_size)

        #if there are not free slots, nothing should spawn anyways so whatever this returns should be ok
        first_free_slot = proposed_slots[jnp.argwhere(jnp.logical_not(jnp.logical_or(is_enemy,is_gold))[proposed_slots], size=1)]

        is_gold = is_gold.at[first_free_slot].set(jnp.where(spawn_gold,True,is_gold[first_free_slot]))
        is_enemy = is_enemy.at[first_free_slot].set(jnp.where(spawn_enemy,True,is_enemy[first_free_slot]))
        entity_x = entity_x.at[first_free_slot].set(jnp.where(spawn_entity, spawn_x, entity_x[first_free_slot]))
        movement_dir = movement_dir.at[first_free_slot].set(jnp.where(spawn_entity, spawn_right, movement_dir[first_free_slot]))

        env_state = (pos, is_enemy, is_gold, movement_dir, entity_x, key)
        return env_state, self.get_observation(env_state), reward, terminal, {}

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        pos = jnp.array((self.grid_size//2,self.grid_size//2))
        is_enemy = jnp.zeros(self.grid_size,dtype=bool)
        is_gold = jnp.zeros(self.grid_size,dtype=bool)
        #false: left, true: right
        movement_dir = jnp.zeros(self.grid_size,dtype=bool)
        entity_x = -jnp.ones(self.grid_size,dtype=int)
        env_state = (pos, is_enemy, is_gold, movement_dir, entity_x, key)
        return env_state

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        pos, is_enemy, is_gold, movement_dir, entity_x, key = env_state
        obs = jnp.zeros((self.grid_size, self.grid_size, len(self.channels)), dtype=bool)
        obs = obs.at[pos[0],pos[1],self.channels['player']].set(True)
        for i in range(self.grid_size):
            obs = obs.at[entity_x[i],i,self.channels['enemy']].set(jnp.where(is_enemy[i],True, False))
            obs = obs.at[entity_x[i],i,self.channels['gold']].set(jnp.where(is_gold[i],True, False))
            is_entity = jnp.logical_or(is_enemy[i], is_gold[i])
            obs = obs.at[entity_x[i],i,self.channels['right']].set(jnp.where(is_entity,movement_dir[i],False))
            obs = obs.at[entity_x[i],i,self.channels['left']].set(jnp.where(is_entity,jnp.logical_not(movement_dir[i]),False))
        return obs

    def num_actions(self):
        return self._num_actions
