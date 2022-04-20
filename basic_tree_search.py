import mctx
import haiku as hk
import jax as jx

from jax import jit, vmap, grad

import functools

from optimizers import adamw

from jax import numpy as jnp
import jax_environments
import argparse

import json

from tqdm import tqdm

import pickle as pkl


############################################################
# This file implements a naive tree search using the mctx
# codebase. I believe here stochasticity is handled using
# m=1 samples for chance nodes, which is probably not ideal.
############################################################

#Copied from: https://stackoverflow.com/a/23689767
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

key = jx.random.PRNGKey(0)

Environment = getattr(jax_environments, "Asterix")

env = Environment()

activation_dict = {"relu": jx.nn.relu, "silu": jx.nn.silu, "elu": jx.nn.elu}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--output", "-o", type=str, default="basic_tree_search")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.config, 'r') as f:
    config = dotdict(json.load(f))

config.update({"agent_type":"basic_tree_search", "seed":args.seed})

Environment = getattr(jax_environments, config.environment)

env_config = config.env_config

class V_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config.num_hidden_units
        self.num_hidden_layers = config.num_hidden_layers
        self.activation_function = activation_dict[config.activation]

    def __call__(self, obs):
        x = jnp.ravel(obs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        V = hk.Linear(1)(x)[0]
        return V

class pi_function(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config.num_hidden_units
        self.num_hidden_layers = config.num_hidden_layers
        self.activation_function = activation_dict[config.activation]
        self.num_actions = num_actions

    def __call__(self, obs):
        x = jnp.ravel(obs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        pi_logit = hk.Linear(self.num_actions)(x)
        return pi_logit

# this assumes the agent has access to the exact environment dynamics
def get_recurrent_fn(env, V_func, pi_func):
    batch_step = vmap(env.step, in_axes=(0,0,0))
    batch_pi_func = vmap(pi_func,in_axes=(None,0))
    batch_V_func = vmap(V_func,in_axes=(None,0))
    def recurrent_fn(params, key, actions, env_states):
        V_params = params["V"]
        pi_params = params["pi"]
        key, subkey = jx.random.split(key)
        subkeys = jx.random.split(subkey, num=config.batch_size)
        env_states, obs, rewards, terminals, _ = batch_step(subkeys, actions, env_states)
        V = batch_V_func(V_params, obs.astype(float))
        pi_logit = batch_pi_func(pi_params, obs.astype(float))
        recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=rewards,
        discount=(1.0-terminals)*config.discount,
        prior_logits=pi_logit,
        value=V)
        return recurrent_fn_output, env_states
    return recurrent_fn

def get_init_fn(env):
    batch_reset = vmap(env.reset)

    def init_fn(key):
        dummy_state = env.reset(key)
        obs = env.get_observation(dummy_state)
        dummy_obs = obs.astype(float)

        key, subkey = jx.random.split(key)
        subkeys = jx.random.split(subkey, num=config.batch_size)
        env_states = batch_reset(subkeys)
        num_actions = env.num_actions()

        V_net = hk.without_apply_rng(hk.transform(lambda obs: V_function(config)(obs.astype(float))))
        key, subkey = jx.random.split(key)
        V_params = V_net.init(subkey, dummy_obs)
        V_func = V_net.apply

        pi_net = hk.without_apply_rng(hk.transform(lambda obs: pi_function(config, num_actions)(obs.astype(float))))
        key, subkey = jx.random.split(key)
        pi_params = pi_net.init(subkey, dummy_obs)
        pi_func = pi_net.apply

        V_opt_init, V_opt_update, get_V_params = adamw(config.V_alpha, eps=config.eps_adam, b1=config.b1_adam, b2=config.b2_adam, wd=config.wd_adam)
        V_opt_state = V_opt_init(V_params)

        pi_opt_init, pi_opt_update, get_pi_params = adamw(config.pi_alpha, eps=config.eps_adam, b1=config.b1_adam, b2=config.b2_adam, wd=config.wd_adam)
        pi_opt_state = pi_opt_init(pi_params)

        return env_states, V_func, pi_func, V_opt_state, pi_opt_state, V_opt_update, pi_opt_update, get_V_params, get_pi_params
    return init_fn

def get_AC_loss(pi_func, V_func):
    def AC_loss(pi_params, V_params, pi_target, V_target, obs):
        pi_logits = pi_func(pi_params, obs.astype(float))
        V = V_func(V_params, obs.astype(float))

        pi_loss = jnp.sum(pi_target*(jnp.log(pi_target)-jx.nn.log_softmax(pi_logits)))
        V_loss = (V_target-V)**2

        return jnp.sum(pi_loss+V_loss)
    return AC_loss

def get_agent_environment_interaction_loop_function(env, V_func, pi_func, recurrent_fn, V_opt_update, pi_opt_update, get_V_params, get_pi_params, num_actions, iterations):
    batch_loss = lambda *x: jnp.mean(vmap(get_AC_loss(pi_func, V_func), in_axes=(None,None,0,0,0))(*x))
    loss_grad = grad(batch_loss, argnums=(0,1))
    batch_step = vmap(env.step, in_axes=(0,0,0))
    batch_obs = vmap(env.get_observation)
    batch_reset = vmap(env.reset)
    batch_V_func = vmap(V_func,in_axes=(None,0))
    batch_pi_func = vmap(pi_func,in_axes=(None,0))

    def agent_environment_interaction_loop_function(S):
        def loop_function(S, data):
            obs = batch_obs(S["env_states"])
            pi_logits = batch_pi_func(get_pi_params(S["pi_opt_state"]), obs.astype(float))
            V = batch_V_func(get_V_params(S["V_opt_state"]), obs.astype(float))

            root = mctx.RootFnOutput(
              prior_logits=pi_logits,
              value=V,
              embedding=S["env_states"]
            )

            S["key"], subkey = jx.random.split(S["key"])
            policy_output = mctx.gumbel_muzero_policy(
              params={"V":get_V_params(S["V_opt_state"]), "pi":get_pi_params(S["pi_opt_state"])},
              rng_key=subkey,
              root=root,
              recurrent_fn=recurrent_fn,
              num_simulations=config.num_simulations,
              max_num_considered_actions=num_actions,
              qtransform=functools.partial(
                  mctx.qtransform_completed_by_mix_value,
                  use_mixed_value=config.use_mixed_value
              ),
            )

            # tree search derived targets for policy and value function
            search_policy = policy_output.action_weights
            if(config.value_target=='maxq'):
                search_value = policy_output.search_tree.qvalues(policy_output.action)
            elif(config.value_target=='nodev'):
                search_value = policy_output.search_tree.node_values[:, policy_output.search_tree.ROOT_INDEX]
            else:
                raise ValueError("Unknown value target.")

            # compute loss gradient compared to tree search targets and update parameters
            pi_grads, V_grads = loss_grad(get_pi_params(S["pi_opt_state"]), get_V_params(S["V_opt_state"]), search_policy, search_value, obs)
            S["pi_opt_state"] = pi_opt_update(S["opt_t"], pi_grads, S["pi_opt_state"])
            S["V_opt_state"] = V_opt_update(S["opt_t"], V_grads, S["V_opt_state"])

            # always take action recommended by tree search
            actions = policy_output.action

            S["key"], subkey = jx.random.split(S["key"])
            subkeys = jx.random.split(subkey, num=config.batch_size)
            S["env_states"], obs, reward, terminal, _ = batch_step(subkeys, actions, S["env_states"])

            # reset environment if terminated
            S["key"], subkey = jx.random.split(S["key"])
            subkeys = jx.random.split(subkey, num=config.batch_size)
            S["env_states"] = jx.tree_multimap(lambda x,y: jnp.where(jnp.reshape(terminal,[terminal.shape[0]]+[1]*(len(x.shape)-1)), x,y), batch_reset(subkeys), S["env_states"])

            # update statistics for computing average return
            S["episode_return"] += reward
            S["avg_return"] = jnp.where(terminal, S["avg_return"]*config.avg_return_smoothing+S["episode_return"]*(1-config.avg_return_smoothing), S["avg_return"])
            S["episode_return"] = jnp.where(terminal, 0, S["episode_return"])
            S["num_episodes"] = jnp.where(terminal, S["num_episodes"]+1, S["num_episodes"])
            return S, None

        S["key"], subkey = jx.random.split(S["key"])
        S, _ = jx.lax.scan(loop_function, S, None, length=iterations)

        return S
    return agent_environment_interaction_loop_function


opt_t = 0
time_step = 0
avg_return = jnp.zeros(config.batch_size)
episode_return = jnp.zeros(config.batch_size)
num_episodes = jnp.zeros(config.batch_size)

env = Environment(**env_config)
num_actions = env.num_actions()

key, subkey = jx.random.split(key)
env_states, V_func, pi_func, V_opt_state, pi_opt_state, V_opt_update, pi_opt_update, get_V_params, get_pi_params = get_init_fn(env)(subkey)

recurrent_fn = get_recurrent_fn(env, V_func, pi_func)

agent_environment_interaction_loop_function = jit(get_agent_environment_interaction_loop_function(env, V_func, pi_func, recurrent_fn, V_opt_update, pi_opt_update, get_V_params, get_pi_params, num_actions, config.eval_frequency))

# run_state contains all information to be maintained and updated in agent_environment_interaction_loop
run_state_names = ["env_states", "V_opt_state", "pi_opt_state", "opt_t", "avg_return", "episode_return", "num_episodes", "key"]
var_dict = locals()
run_state = {name:var_dict[name] for name in run_state_names}

avg_returns = []
times = []
for i in tqdm(range(config.num_steps//config.eval_frequency)):
    # perform a number of iterations of agent environment interaction including learning updates
    run_state = agent_environment_interaction_loop_function(run_state)

    # avg_return is debiased, and only includes batch elements wit at least one completed episode so that it is more meaningful in early episodes
    valid_avg_returns = run_state["avg_return"][run_state["num_episodes"]>0]
    valid_num_episodes = run_state["num_episodes"][run_state["num_episodes"]>0]
    avg_return = jnp.mean(valid_avg_returns/(1-config.avg_return_smoothing**valid_num_episodes))
    print("Running Average Return: "+str(avg_return))
    avg_returns+=[avg_return]

    time_step+=config.eval_frequency
    times+=[time_step]

with open(args.output+".out", 'wb') as f:
    pkl.dump({
        'config': dict(config),
        'avg_returns': avg_returns,
        'times': times
    }, f)

with open(args.output+".params", 'wb') as f:
    pkl.dump({
        'V' : get_V_params(V_opt_state),
        'pi' : get_pi_params(pi_opt_state)
    }, f)
