import os

os.environ['MUJOCO_GL'] = 'egl'

from absl import app, flags

from jaxrl.agent.brc_learner import BRC
from jaxrl.replay_buffer import ParallelReplayBuffer, TruncGeomReplayBuffer, ForgettingParallelReplayBuffer, PrioritySamplingReplayBuffer, EREBuffer
from jaxrl.envs import ParallelEnv
from jaxrl.normalizer import RewardNormalizer
from jaxrl.logger import EpisodeRecorder
from jaxrl.env_names import get_environment_list
from jaxrl.saving import load_checkpoint, save_checkpoint, save_resume_script

ERE_BUFFER_EPISODE = 1000

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1000000), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(1000000), 'Replay buffer size.')
flags.DEFINE_integer('start_training', int(5000),'Number of training steps to start training.')
flags.DEFINE_string('env_names', 'cheetah-run', 'Environment name.')
flags.DEFINE_boolean('log_to_wandb', True, 'Whether to log to wandb.')
flags.DEFINE_boolean('offline_evaluation', True, 'Whether to perform evaluations with temperature=0.')
flags.DEFINE_boolean('render', True, 'Whether to log the rendering to wandb.')
flags.DEFINE_string('sampling_method', 'uniform', 'Sampling method: uniform, per, geom, ere.')
flags.DEFINE_integer('updates_per_step', 2, 'Number of updates per step.')
flags.DEFINE_integer('width_critic', 4096, 'Width of the critic network.')
flags.DEFINE_float('per_alpha', 0.6, 'Alpha parameter for prioritized sampling (per only).')
flags.DEFINE_float('ere_decay_constant', 0.996, 'Decay constant for ERE buffer (ere only).')
flags.DEFINE_float('trunc_geom_alpha', 10.0, 'Exponential sampling parameter (geom only).')

        
def main(_):
    resume = os.environ.get('RESUME_RUN_ID', None)
    if resume is not None:
        run_id = os.environ.get('RESUME_RUN_ID', None)
    else:
        run_id = os.environ.get('SLURM_JOB_ID', None)

    if FLAGS.sampling_method != 'per':
        FLAGS.per_alpha = None

    if FLAGS.sampling_method != 'ere':
        FLAGS.ere_decay_constant = None

    if FLAGS.sampling_method != 'geom':
        FLAGS.trunc_geom_alpha = None

    if FLAGS.log_to_wandb:
        import wandb
        wandb.init(
            config=FLAGS,
            entity='wartoslaw-uw',
            project='ReplayBufferMultitask',
            group=f'{FLAGS.env_names}',
            name=f'{FLAGS.seed}',
            id=f'{run_id}',
            resume="allow",
        )
    env_names = get_environment_list(FLAGS.env_names)
    env = ParallelEnv(env_names, seed=FLAGS.seed)
    if FLAGS.offline_evaluation:
        if FLAGS.env_names == "METAWORLD_ALL":
            eval_env = ParallelEnv(env_names, seed=FLAGS.seed)
        else:
            eval_env = ParallelEnv(env_names, seed=FLAGS.seed+42)
    else:
        eval_env = None
        
    eval_interval = FLAGS.eval_interval if FLAGS.offline_evaluation else 5000
        
    # Kwargs setup
    kwargs = {}
    kwargs['updates_per_step'] = FLAGS.updates_per_step
    kwargs['width_critic'] = FLAGS.width_critic
    
    num_tasks = len(env.envs)

    agent = BRC(
        FLAGS.seed,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        **kwargs,
    )

    if FLAGS.sampling_method == 'per':
        replay_buffer = PrioritySamplingReplayBuffer(
            env.observation_space,
            env.action_space.shape[-1],
            FLAGS.replay_buffer_size,
            num_tasks=num_tasks,
            alpha_sampling=FLAGS.per_alpha,
            beta=0.4,
            max_steps=FLAGS.max_steps
        )
    elif FLAGS.sampling_method == 'ere':
        replay_buffer = EREBuffer(
            env.observation_space,
            env.action_space.shape[-1],
            FLAGS.replay_buffer_size,
            num_tasks=num_tasks,
            decay_const=FLAGS.ere_decay_constant,
            total_updates=FLAGS.updates_per_step * ERE_BUFFER_EPISODE,
            min_recent_samples=FLAGS.batch_size * 10
        )
    elif FLAGS.sampling_method == 'geom':
        replay_buffer = TruncGeomReplayBuffer(
            env.observation_space,
            env.action_space.shape[-1],
            FLAGS.replay_buffer_size,
            num_tasks=num_tasks,
            trunc_geom_alpha=FLAGS.trunc_geom_alpha
        )
    else:  # uniform
        replay_buffer = ParallelReplayBuffer(
            env.observation_space,
            env.action_space.shape[-1],
            FLAGS.replay_buffer_size,
            num_tasks=num_tasks
        )
    
    evaluation_replay_buffer_size = (FLAGS.max_steps // eval_interval) * FLAGS.eval_episodes * 1000
    evaluation_replay_buffer = ParallelReplayBuffer(
        env.observation_space,
        env.action_space.shape[-1],
        evaluation_replay_buffer_size,
        num_tasks=num_tasks
    )

    forgetting_replay_buffer = ForgettingParallelReplayBuffer(
        env.observation_space,
        env.action_space.shape[-1],
        FLAGS.batch_size,
        num_tasks=num_tasks
    )
    start_step = 1
    if resume is not None: 
        checkpoint_path = f'checkpoints/checkpoint_{run_id}'
        start_step = load_checkpoint(checkpoint_path, agent, replay_buffer, evaluation_replay_buffer, forgetting_replay_buffer)

    reward_normalizer = RewardNormalizer(num_tasks, agent.target_entropy, discount=agent.discount, max_steps=None) #change max_steps according to env
    statistics_recorder = EpisodeRecorder(num_tasks)
    
    observations = env.reset()

    for i in range(start_step, FLAGS.max_steps + 1):
        print(f'Step {i}/{FLAGS.max_steps}', end='\r')
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions(observations, temperature=1.0)
        next_observations, rewards, terms, truns, goals = env.step(actions)
        reward_normalizer.update(rewards, terms, truns)
        statistics_recorder.update(rewards, goals, terms, truns)
        masks = env.generate_masks(terms, truns)
        overriden_sample = replay_buffer.insert(observations, actions, rewards, masks, next_observations, return_overriden=i >= FLAGS.replay_buffer_size)
<<<<<<< HEAD
        
        modulo = int((FLAGS.max_steps / FLAGS.batch_size) // 10)
        if modulo == 0:
            modulo = 1
        if overriden_sample is not None and i % modulo == 0:
=======

        if overriden_sample is not None and (FLAGS.max_steps / FLAGS.batch_size) // 10 != 0 and i % int((FLAGS.max_steps / FLAGS.batch_size) // 10) == 0:
>>>>>>> main
            q_probabilities = agent.get_critic_outputs(overriden_sample)
            forgetting_replay_buffer.insert(
                overriden_sample.observations,
                overriden_sample.actions,
                overriden_sample.rewards,
                overriden_sample.masks,
                overriden_sample.next_observations,
                q_probabilities
            )

        observations = next_observations
        observations, terms, truns = env.reset_where_done(observations, terms, truns)
        if i >= FLAGS.start_training:
            if FLAGS.sampling_method != 'ere':
                if FLAGS.sampling_method == 'per':
                    replay_buffer.anneal_priority_beta(i)
                    batches, is_weights = replay_buffer.sample(FLAGS.batch_size, FLAGS.updates_per_step)
                else:
                    batches = replay_buffer.sample(FLAGS.batch_size, FLAGS.updates_per_step)
                batches = reward_normalizer.normalize(batches, agent.get_temperature())
                _ = agent.update(batches, FLAGS.updates_per_step, i, is_weights if FLAGS.sampling_method == 'per' else None, replay_buffer if FLAGS.sampling_method == 'per' else None)
            else:
                if i % ERE_BUFFER_EPISODE == 0:
                    for update_count in range(FLAGS.updates_per_step * ERE_BUFFER_EPISODE):
                        batches = replay_buffer.sample(batch_size=FLAGS.batch_size, num_batches=1, update_count=update_count)
                        batches = reward_normalizer.normalize(batches, agent.get_temperature())
                        _ = agent.update(batches, 1, i, None, None)

            if i % eval_interval == 0 and i >= FLAGS.start_training:  
                info_dict = statistics_recorder.log(FLAGS, agent, replay_buffer, evaluation_replay_buffer, forgetting_replay_buffer, reward_normalizer, i, env_names, eval_env, render=FLAGS.render)
                save_checkpoint(
                    f'checkpoints/checkpoint_{run_id}',
                    i + 1,
                    agent,
                    replay_buffer,
                    evaluation_replay_buffer,
                    forgetting_replay_buffer
                )
                save_resume_script(f'checkpoints/checkpoint_{run_id}', run_id)

    checkpoint_path = f'checkpoints/checkpoint_{run_id}'
    if os.path.exists(checkpoint_path):
        import shutil
        shutil.rmtree(checkpoint_path)
        print(f'Checkpoint {checkpoint_path} deleted after training completion.')
            
if __name__ == '__main__':
    app.run(main)
