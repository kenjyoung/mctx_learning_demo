import jax as jx
import jax.numpy as jnp
from jax import jit, vmap
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

class ProcMaze:
    def __init__(self, grid_size=10, goal_reward=False, timeout=None):
        self.move_map = jnp.asarray([[0, 0], [-1,0], [0,-1], [1,0], [0,1]])
        self.timeout = timeout

        self._num_actions = 5
        self.grid_size = grid_size
        self.channels ={
            'player':0,
            'goal':1,
            'wall':2,
            'empty':3
        }

    @partial(jit, static_argnums=(0,))
    def step(self, action, env_state):
        goal, wall_grid, pos, t = env_state

        # Move if the new position is on the grid and open
        new_pos = jnp.clip(pos+self.move_map[action], 0, self.grid_size-1)
        pos = jnp.where(jnp.logical_not(wall_grid[new_pos[0], new_pos[1]]), new_pos, pos)

        # Treminated if we reach the goal
        terminal = jnp.array_equal(pos, goal)
        if(self.timeout is not None):
            terminal = jnp.logical_or(terminal, t>=self.timeout)

        reward = -1

        t+=1

        env_state = goal, wall_grid, pos, t

        return env_state, self.get_observation(env_state), reward, terminal, {}

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        def push(stack, top, x):
            stack= stack.at[top].set(x)
            top+=1
            return stack, top

        def pop(stack, top):
            top-=1
            return stack[top], top

        #takes and flattened index, returns neighbours as (x,y) pair
        def neighbours(cell):
            coord_tuple = jnp.unravel_index(cell, (self.grid_size, self.grid_size))
            coord_array = jnp.stack(list(coord_tuple))
            return coord_array+self.move_map

        #takes (x,y) pair
        def can_expand(cell, visited):
            # A neighbour can be expanded as long as it is on the grid, it has not been visited, and it's only visited neighbour is the current cell
            flat_cell = jnp.ravel_multi_index((cell[0],cell[1]),(self.grid_size,self.grid_size),mode='clip')
            not_visited = jnp.logical_not(visited[flat_cell])
            ns = neighbours(flat_cell)
            ns_on_grid = jnp.all(jnp.logical_and(ns>=0,ns<=self.grid_size-1), axis=1)
            flat_ns = jnp.ravel_multi_index((ns[:,0],ns[:,1]),(self.grid_size,self.grid_size),mode='clip')
            # Only count neighbours which are actually on the grid
            only_one_visited_neighbor = jnp.equal(jnp.sum(jnp.logical_and(visited[flat_ns],ns_on_grid)),1)
            on_grid = jnp.all(jnp.logical_and(cell>=0,cell<=self.grid_size-1))
            return jnp.logical_and(jnp.logical_and(not_visited,only_one_visited_neighbor),on_grid)
        can_expand = vmap(can_expand, in_axes=(0,None))

        wall_grid = jnp.ones((self.grid_size, self.grid_size), dtype=bool)

        #Visited node map
        visited = jnp.zeros(self.grid_size*self.grid_size, dtype=bool)

        #big enough to hold every location in the grid, indices should be flattened to store here
        stack = jnp.zeros(self.grid_size*self.grid_size, dtype=int)
        top = 0

        #Location of current cell being searched
        key, subkey = jx.random.split(key)
        curr = jx.random.choice(subkey, self.grid_size, (2,))
        flat_curr = jnp.ravel_multi_index((curr[0],curr[1]),(self.grid_size,self.grid_size),mode='clip')
        wall_grid = wall_grid.at[curr[0], curr[1]].set(False)

        visited = visited.at[flat_curr].set(True)
        stack, top = push(stack,top, flat_curr)

        def cond_fun(carry):
            visited, stack, top, wall_grid, key = carry
            #continue until stack is empty
            return top!=0

        def body_fun(carry):
            visited, stack, top, wall_grid, key = carry
            curr, top = pop(stack,top)
            ns = neighbours(curr)
            flat_ns = jnp.ravel_multi_index((ns[:,0],ns[:,1]),(self.grid_size,self.grid_size),mode='clip')

            expandable = can_expand(ns,visited)

            has_expandable_neighbour = jnp.any(expandable)

            # This will all be used only conditioned on has_expandable neighbor
            _stack, _top = push(stack, top, curr)
            key, subkey = jx.random.split(key)
            selected = jx.random.choice(subkey, flat_ns,p=expandable/jnp.sum(expandable))
            _stack, _top = push(_stack, _top, selected)
            _wall_grid = wall_grid.at[jnp.unravel_index(selected, (self.grid_size, self.grid_size))].set(False)
            _visited = visited.at[selected].set(True)

            stack = jnp.where(has_expandable_neighbour, _stack, stack)
            top = jnp.where(has_expandable_neighbour, _top, top)
            wall_grid = jnp.where(has_expandable_neighbour, _wall_grid, wall_grid)
            visited = jnp.where(has_expandable_neighbour, _visited, visited)
            return (visited, stack, top, wall_grid, key)



        key, subkey = jx.random.split(key)
        carry = (visited, stack, top, wall_grid, subkey)
        visited, stack, top, wall_grid, key = jx.lax.while_loop(cond_fun, body_fun, carry)

        flat_open = jnp.logical_not(jnp.ravel(wall_grid))

        key, subkey = jx.random.split(key)
        pos = jx.random.choice(subkey, self.grid_size*self.grid_size, p=flat_open/jnp.sum(flat_open))
        pos = jnp.stack(list(jnp.unravel_index(pos, (self.grid_size, self.grid_size))))
        key, subkey = jx.random.split(key)
        goal = jx.random.choice(subkey, self.grid_size*self.grid_size, p=flat_open/jnp.sum(flat_open))
        goal = jnp.stack(list(jnp.unravel_index(goal, (self.grid_size, self.grid_size))))

        wall_grid = wall_grid.at[goal[0], goal[1]].set(False)
        wall_grid = wall_grid.at[pos[0], pos[1]].set(False)
        t=0
        env_state = goal, wall_grid, pos, t
        return env_state

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        goal, wall_grid, pos, t = env_state
        obs = jnp.zeros((self.grid_size, self.grid_size, len(self.channels)), dtype=bool)
        obs = obs.at[pos[0],pos[1],self.channels['player']].set(True)
        obs = obs.at[goal[0],goal[1],self.channels['goal']].set(True)
        obs = obs.at[:,:,self.channels['wall']].set(wall_grid)
        obs = obs.at[:,:,self.channels['empty']].set(jnp.logical_not(wall_grid))
        return obs

    def num_actions(self):
        return self._num_actions
