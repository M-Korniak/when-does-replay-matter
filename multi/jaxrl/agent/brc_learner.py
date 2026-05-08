import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agent.update import build_actor_input, update_actor, update_critic, update_target_critic, update_temperature
from jaxrl.agent.metrics import param_norms

from jaxrl.networks import NormalTanhPolicy, DoubleCriticTest, Temperature
from jaxrl.utils import Model, PRNGKey, Batch


@functools.partial(jax.jit, static_argnames=('discount', 'target_entropy', 'num_bins', 'v_max', 'multitask'),)
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, None, None, None, None, None))
def _get_infos(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    target_entropy: float,
    num_bins: int,
    v_max: float,
    multitask: bool,
):
    is_weights = jnp.ones(batch.rewards.shape)
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    _, critic_info = update_critic(critic_key, actor, critic, target_critic, temp, batch, is_weights, discount, num_bins, v_max, multitask)
    _, actor_info = update_actor(actor_key, actor, critic, temp, batch, num_bins, v_max, multitask) 
    _, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)


    plasticity_critic_norms = param_norms(critic, prefix='critic')
    plasticity_actor_norms = param_norms(actor, prefix='actor')
    

    critic_info.pop('critic_loss_per_sample', None)

    return {
        **critic_info,
        **actor_info,
        **alpha_info,
        **plasticity_critic_norms,
        **plasticity_actor_norms,
    }

@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0))
def _get_critic_q_values(critic: Model, batch: Batch):
    q_logits = critic(batch.observations, batch.actions, batch.task_ids)
    q_probabilities = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
    return q_probabilities

@jax.jit
def _get_temperature(temp):
    temp_val = temp()
    return temp_val

@jax.jit
def _sample_actions(
    rng: PRNGKey,
    actor: Model,
    inputs: np.ndarray,
    temperature: float = 1.0,
):
    dist = actor(inputs, temperature)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    return rng, actions

def _update(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    is_weights: jnp.ndarray,
    discount: float,
    tau: float,
    target_entropy: float,
    num_bins: int,
    v_max: float,
    multitask: bool,
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    new_critic, critic_info = update_critic(critic_key, actor, critic, target_critic, temp, batch, is_weights, discount, num_bins, v_max, multitask)
    new_target_critic = update_target_critic(new_critic, target_critic, tau)
    new_actor, actor_info = update_actor(actor_key, actor, new_critic, temp, batch, num_bins, v_max, multitask) 
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info,
    }

@functools.partial(jax.jit, static_argnames=('use_priority_sampling', 'discount', 'tau', 'target_entropy', 'num_bins', 'v_max', 'multitask', 'num_updates'))
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batches: Batch,
    is_weights: jnp.ndarray,
    discount: float,
    tau: float,
    target_entropy: float,
    num_bins: int,
    v_max: float,
    multitask: bool,
    step: int,
    num_updates: int,
    use_priority_sampling: bool,
):
    _, batch_size = batches.rewards.shape
    priorities = jnp.zeros((num_updates, batch_size))

    def one_step(i, state):
        step, rng, actor, critic, target_critic, temp, info, priorities = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update(
            rng,
            actor,
            critic,
            target_critic,
            temp,
            jax.tree.map(lambda x: jnp.take(x, i, axis=0), batches),
            jax.tree.map(lambda x: jnp.take(x, i, axis=0), is_weights),
            discount,
            tau,
            target_entropy,
            num_bins,
            v_max,
            multitask,
        )
        if use_priority_sampling:
            critic_loss_per_sample = info.pop('critic_loss_per_sample', jnp.zeros(batch_size))
            priorities = priorities.at[i, :].set(critic_loss_per_sample)
        else:
            info.pop('critic_loss_per_sample', None)

        return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, info, priorities

    step, rng, actor, critic, target_critic, temp, info, priorities = one_step(0, (step, rng, actor, critic, target_critic, temp, {}, priorities))
    return jax.lax.fori_loop(1, num_updates, one_step, (step, rng, actor, critic, target_critic, temp, info, priorities))

class BRC(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        num_tasks: int,
        embedding_size: int = 32,
        ensemble_size: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 0.1,
        updates_per_step: int = 10,
        width_critic: int = 512,
        width_actor: int = 256,
        num_bins: int = 101,
        v_max: float = 10.0,
    ) -> None:

        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.seed = seed
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.num_bins = num_bins
        self.v_max = v_max

        self.num_tasks = num_tasks
        self.embedding_size = embedding_size
        self.task_ids = jnp.arange(num_tasks, dtype=jnp.int32)
        
        task_embedding_init = jnp.zeros((1, embedding_size))
        task_ids_init = self.task_ids[:1]
        self.multitask = True if num_tasks > 1 else False
        
        actor_init = jnp.concatenate((observations, task_embedding_init), axis=-1) if self.multitask else observations
        
        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
            actor_def = NormalTanhPolicy(action_dim=action_dim, hidden_dims=width_actor)
            critic_def = DoubleCriticTest(num_tasks=num_tasks, embedding_size=embedding_size, ensemble_size=2, hidden_dims=width_critic, depth=2, output_nodes=num_bins, multitask=self.multitask)
            
            actor = Model.create(actor_def, inputs=[actor_key, actor_init], tx=optax.adamw(learning_rate=actor_lr))
            critic = Model.create(critic_def, inputs=[critic_key, observations, actions, task_ids_init], tx=optax.adamw(learning_rate=critic_lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions, task_ids_init])
            temp = Model.create(Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5))
            return actor, critic, target_critic, temp, rng

        self.init_models = jax.jit(_init_models)
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seed)
        self.step = 1

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0):
        inputs = build_actor_input(self.critic, observations, self.task_ids, self.multitask)
        rng, actions = _sample_actions(self.rng, self.actor, inputs, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, num_updates: int, env_step: int, is_weights: Optional[np.ndarray], replay_buffer=None):

        use_priority_sampling = is_weights is not None

        if not use_priority_sampling:
            is_weights = jnp.ones(batch.rewards.shape)

        step, rng, actor, critic, target_critic, temp, info, priorities = _do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            is_weights,
            self.discount,
            self.tau,
            self.target_entropy,
            self.num_bins,
            self.v_max,
            self.multitask,
            self.step,
            num_updates,
            use_priority_sampling,
        )
        if use_priority_sampling:
            replay_buffer.update_last_batch_priorities(priorities)

        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        return info
    
    def get_infos(self, batch: Batch):
        return _get_infos(
                    self.rng,
                    self.actor,
                    self.critic,
                    self.target_critic,
                    self.temp,
                    batch,
                    self.discount,
                    self.target_entropy,
                    self.num_bins,
                    self.v_max,
                    self.multitask,
                )
    
    def get_temperature(self):
        return _get_temperature(self.temp)
    
    def get_critic_outputs(self, batch: Batch):
        return _get_critic_q_values(self.critic, batch)

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seeds)
        
    def save(self, path):
        self.actor.save(f'{path}/actor.txt')
        self.critic.save(f'{path}/critic.txt')
        self.target_critic.save(f'{path}/target_critic.txt')
        self.temp.save(f'{path}/temp.txt')
        
    def load(self, path):
        self.actor = self.actor.load(f'{path}/actor.txt')
        self.critic = self.critic.load(f'{path}/critic.txt')
        self.target_critic = self.target_critic.load(f'{path}/target_critic.txt')
        self.temp = self.temp.load(f'{path}/temp.txt')
