import os
import pickle

from absl import flags

from jaxrl.agent.brc_learner import BRC
from jaxrl.replay_buffer import ParallelReplayBuffer 


def save_checkpoint(
    save_path: str,  # e.g., "checkpoints/checkpoint_{job_id}"
    step: int,
    agent: BRC,
    replay_buffer: ParallelReplayBuffer,
    evaluation_replay_buffer: ParallelReplayBuffer,
    forgetting_replay_buffer: ParallelReplayBuffer
):
    ckpt_dir = save_path
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save step
    with open(os.path.join(ckpt_dir, 'step.pkl'), 'wb') as f:
        pickle.dump(step, f)
    
    # Save agent
    agent_dir = os.path.join(ckpt_dir, 'agent')
    os.makedirs(agent_dir, exist_ok=True)
    agent.save(agent_dir)
    
    # Save replay buffers using their own save() methods
    buffer_dir = os.path.join(ckpt_dir, 'buffers')
    os.makedirs(buffer_dir, exist_ok=True)

    replay_buffer.save(os.path.join(buffer_dir, 'replay_buffer'))
    evaluation_replay_buffer.save(os.path.join(buffer_dir, 'evaluation_replay_buffer'))
    forgetting_replay_buffer.save(os.path.join(buffer_dir, 'forgetting_replay_buffer'))

    print(f"Checkpoint saved at step {step} to {ckpt_dir}")


def load_checkpoint(
    save_path: str,  # e.g., "checkpoints/checkpoint_{job_id}"
    agent: BRC,
    replay_buffer: ParallelReplayBuffer,
    evaluation_replay_buffer: ParallelReplayBuffer,
    forgetting_replay_buffer: ParallelReplayBuffer
):
    ckpt_dir = save_path
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist")
    
    # Load step
    step_path = os.path.join(ckpt_dir, 'step.pkl')
    if not os.path.exists(step_path):
        raise FileNotFoundError(f"Step file not found in {ckpt_dir}")
    with open(step_path, 'rb') as f:
        step = pickle.load(f)
    
    # Load agent
    agent_dir = os.path.join(ckpt_dir, 'agent')
    if not os.path.exists(agent_dir):
        raise FileNotFoundError(f"Agent directory not found in {ckpt_dir}")
    agent.load(agent_dir)
    
    # Load replay buffers
    buffer_dir = os.path.join(ckpt_dir, 'buffers')
    if not os.path.exists(buffer_dir):
        raise FileNotFoundError(f"Buffers directory not found in {ckpt_dir}")
    
    replay_buffer.load(os.path.join(buffer_dir, 'replay_buffer'))
    evaluation_replay_buffer.load(os.path.join(buffer_dir, 'evaluation_replay_buffer'))
    forgetting_replay_buffer.load(os.path.join(buffer_dir, 'forgetting_replay_buffer'))

    print(f"Checkpoint loaded from {ckpt_dir}, resuming from step {step}")
    return step

def save_resume_script(save_path: str, resume_run_id: str = None):
    FLAGS = flags.FLAGS
    script_path = os.path.join(save_path, f"0.sh")
    
    # Whitelist of flags to include in resume script
    allowed_flags = {
        "seed",
        "eval_episodes",
        "eval_interval",
        "batch_size",
        "start_training",
        "max_steps",
        "replay_buffer_size",
        "render",
        "env_names",
        "offline_evaluation",
        "sampling_method",
        "updates_per_step",
        "width_critic",
        "per_alpha",
        "ere_decay_constant",
        "trunc_geom_alpha",
    }

    flag_lines = []
    for key, value in FLAGS.flag_values_dict().items():
        if key not in allowed_flags:
            continue

        if key == "per_alpha" and value is None:
            value = 0.0

        val_str = str(value)
        flag_lines.append(f"    --{key}={val_str} \\")
    
    # Remove trailing backslash on last flag
    if flag_lines:
        flag_lines[-1] = flag_lines[-1].rstrip(" \\")
    
    flags_str = "\n".join(flag_lines)
    
    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
#
# Load environment
source ~/miniconda3/bin/activate
conda activate humanoidbench_multi

module load cuDNN/8.9.2.26-CUDA-12.2.0

# Environment variables for resuming
export RESUME_RUN_ID={resume_run_id if resume_run_id is not None else ""}

# Load WANDB key from environment
export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')}

# Training flags from current Python FLAGS
python3 train.py \\
{flags_str}
""")
    print(f"Resume script saved to {script_path}")
