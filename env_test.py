from jax_environments import Asterix
import jax.numpy as jnp
import jax as jx

env = Asterix()
key = jx.random.PRNGKey(0)
num_actions = env.num_actions()

last_is_gold = None
last_is_enemy = None
last_entity_x = None
last_movement_dir = None

returns=[]
for i in range(100):
	G=0
	key, subkey = jx.random.split(key)
	env_state = env.reset(subkey)
	terminal = False
	while(not terminal):
		key, subkey = jx.random.split(key)
		action = jx.random.choice(subkey,num_actions)

		key, subkey = jx.random.split(key)
		env_state, obs, reward, terminal, _ = env.step(subkey, action, env_state)
		pos, is_enemy, is_gold, movement_dir, entity_x, key = env_state

		assert(not jnp.any(jnp.logical_and(is_enemy,is_gold)))
		if(reward>0 and pos[0]>0 and pos[0]<env.grid_size-1):
			assert(last_is_gold[pos[1]])
			assert(last_entity_x[pos[1]]+(-1 if last_movement_dir[pos[1]] else 1)==pos[0])
		last_is_gold= is_gold
		last_is_enemy=is_enemy
		last_entity_x = entity_x
		last_movement_dir =movement_dir
		G+=reward
	if(pos[0]>0 and pos[0]<env.grid_size-1):
		assert(last_is_enemy[pos[1]])
		assert(last_entity_x[pos[1]])
	returns+=[G]

print(jnp.mean(jnp.asarray(returns)))
