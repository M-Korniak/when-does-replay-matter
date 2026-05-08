import numpy as np
import wandb

from jaxrl.env_names import HB_HANDS
from jaxrl.env_normalize import normalize

def log_to_wandb(step: int, infos: dict, env_names : list, suffix: str = ''):
    dict_to_log = {'timestep': step}
    mean_dict = {}
    for info_key in infos:
        if info_key != 'renders':
            mean_dict[f'mean/{info_key}{suffix}'] = 0
        for seed, value in enumerate(infos[info_key]):
            env_name = env_names[seed]
            if info_key not in {'return', 'return_online'} and env_names[0] == 'h1hand-walk-v0':
                value = value
            else: 
                if env_name in HB_HANDS:
                    value = normalize(value, env_name)  #Normalize return and return online for humanoidbench
                else:
                    value = value
 
            dict_to_log[f'seed{seed}/{info_key}{suffix}'] = value

            if info_key != 'renders':
                mean_dict[f'mean/{info_key}{suffix}'] += value

        if info_key != 'renders':
            mean_dict[f'mean/{info_key}{suffix}'] /= len(infos[info_key])
    dict_to_log = {**dict_to_log, **mean_dict}
    wandb.log(dict_to_log, step=step)
    
def get_wandb_video(renders: np.ndarray, fps: int = 15):
    videos = []
    for i in range(renders.shape[0]):
        video = wandb.Video(renders[i], fps=fps, format='mp4')
        videos.append(video)
    return videos
    
class EpisodeRecorder:
    
    def __init__(self, num_seeds: int):
        self.returns_online = np.zeros(num_seeds)
        self.goals_online = np.zeros(num_seeds)
        self.counts = np.zeros(num_seeds)
        self.returns_online_episode = np.zeros(num_seeds)
        self.goals_online_episode = np.zeros(num_seeds)
        self.num_seeds = num_seeds
        
    def update(self, rewards: np.ndarray, goals: np.ndarray, terminals: np.ndarray, truncates: np.ndarray):
        self.returns_online_episode += rewards
        self.goals_online_episode += goals
        if terminals.any() or truncates.any():
            done = np.logical_or(terminals, truncates)
            self.counts = np.where(done, self.counts+1, self.counts)
            self.goals_online_episode[self.goals_online_episode > 0.0] = 1.0
            self.goals_online = np.where(done, self.goals_online+self.goals_online_episode, self.goals_online)
            self.returns_online = np.where(done, self.returns_online+self.returns_online_episode, self.returns_online)
            self.returns_online_episode = np.where(done, 0, self.returns_online_episode)
            self.goals_online_episode = np.where(done, 0, self.goals_online_episode)
        
    def _get_scores(self):
        self.counts = np.where(self.counts==0.0, 1e-8, self.counts)
        infos_online_eval = {'goal_online': self.goals_online/self.counts, 'return_online': self.returns_online/self.counts}
        self.returns_online = np.zeros(self.num_seeds)
        self.goals_online = np.zeros(self.num_seeds)
        self.counts = np.zeros(self.num_seeds)
        print(infos_online_eval)
        return infos_online_eval
    
    def log(self, FLAGS, agent, replay_buffer, evaluation_replay_buffer, forgetting_replay_buffer, reward_normalizer, step, env_names, eval_env=None, render=False):
        batches_info = replay_buffer.sample_task_batches()
        batches_info = reward_normalizer.normalize(batches_info, agent.get_temperature())
        infos = agent.get_infos(batches_info)
        infos_online_eval = self._get_scores()
        infos = {**infos, **infos_online_eval}
        if evaluation_replay_buffer.size > 32:
            eval_batches_info = evaluation_replay_buffer.sample_task_batches()
            eval_batches_info = reward_normalizer.normalize(eval_batches_info, agent.get_temperature())
            eval_infos = agent.get_infos(eval_batches_info)
            eval_infos = {k: v for k, v in eval_infos.items() if 'loss' in k}
            eval_infos = {f'eval_{k}': v for k, v in eval_infos.items()}
            infos = {**infos, **eval_infos}
        if forgetting_replay_buffer.is_full():
            batches_old, q_probabilities = forgetting_replay_buffer.sample_task_batches(batch_size_per_task=FLAGS.batch_size)
            q_probabilities_now = agent.get_critic_outputs(batches_old)
            kl_divs = np.mean(np.sum(q_probabilities * (np.log(q_probabilities + 1e-8) - np.log(q_probabilities_now + 1e-8)), axis=-1), axis=1)
            forgetting_info = {'forgetting_kl_div': kl_divs}
            infos = {**infos, **forgetting_info}
        if FLAGS.offline_evaluation:
            eval_stats = eval_env.evaluate(agent, num_episodes=FLAGS.eval_episodes, temperature=0.0, render=render, evaluation_replay_buffer=evaluation_replay_buffer)
            if render:
                eval_stats['renders'] = get_wandb_video(eval_stats['renders'])
            infos = {**infos, **eval_stats}
        if FLAGS.log_to_wandb:
            log_to_wandb(step, infos, env_names)
        return infos
    