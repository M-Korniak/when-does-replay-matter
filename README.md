# Replay Buffer Strategies for Deep RL

This repository studies the effect of **replay buffer sampling strategies** across three modern deep RL codebases:

| Sub-project | Algorithm | Benchmarks |
|---|---|---|
| [`FastTD3/`](FastTD3/) | FastTD3 (TD3 variant) | HumanoidBench, MuJoCo Playground, IsaacLab, MTBench |
| [`multi/`](multi/) | BRC — Bigger, Regularized, Categorical | DMControl, multi-task settings |
| [`Simba/`](Simba/) | SimbaV2 (SAC + hyperspherical normalisation) | MuJoCo, DMControl, MyoSuite, HumanoidBench |

The same four sampling strategies are implemented in each codebase, allowing direct comparison of their effect across algorithms and environments.

---

## Sampling Strategies

### Uniform (baseline)
Samples transitions uniformly at random from the entire replay buffer.

### GEOM — Geometric / Exponential Recency Bias
Samples transitions with a truncated geometric (or exponential) distribution that favours recently collected experience. A single hyperparameter controls the strength of the recency bias.

| Codebase | Flag / config key | Default |
|---|---|---|
| FastTD3 | `--sampling_method GEOM --geom_alpha_sampling` | `10.0` |
| BRC | `--sampling_method=geom --trunc_geom_alpha` | `10.0` |
| SimbaV2 | `buffer.buffer_type=exponential buffer.alpha_exponential_sampling` | `10.0` |

### ERE — Emphasizing Recent Experience
Performs batched gradient updates over an adaptively shrinking window of recent transitions. The window contracts each update according to a decay constant, focusing learning on fresh data while still seeing older experience.

| Codebase | Flag / config key | Default decay |
|---|---|---|
| FastTD3 | `--sampling_method ERE --ere_decay_constant` | `0.996` |
| BRC | `--sampling_method=ere --ere_decay_constant` | `0.996` |
| SimbaV2 | `buffer=numpy_ere buffer.decay_const` | `0.966` |

### PER — Prioritized Experience Replay
Samples transitions proportional to their TD-error magnitude (raised to exponent α), with importance-sampling weights to correct the resulting bias (β annealed from 0.4 → 1 over training).

| Codebase | Flag / config key | Default α |
|---|---|---|
| FastTD3 | `--sampling_method PER --per_alpha` | `0.6` |
| BRC | `--sampling_method=per --per_alpha` | `0.6` |
| SimbaV2 | `buffer=numpy_prioritized buffer.alpha_sampling` | `0.6` |

---

## Repository Layout

```
ReplayBufferProject/
├── FastTD3/      # FastTD3 — PyTorch, single/multi-GPU, humanoid control
├── multi/        # BRC — JAX, single- and multi-task DMControl
└── Simba/        # SimbaV2 — JAX, broad continuous-control benchmarks
```

Each sub-directory contains its own README with full installation and usage instructions.

---

## Quick Start

### FastTD3

```bash
cd FastTD3
conda create -n fasttd3_hb -y python=3.10 && conda activate fasttd3_hb
pip install --editable git+https://github.com/carlosferrazza/humanoid-bench.git#egg=humanoid-bench
pip install -r requirements/requirements.txt

# Uniform (default)
python fast_td3/train.py --env_name h1hand-hurdle-v0 --seed 1
# PER
python fast_td3/train.py --env_name h1hand-hurdle-v0 --sampling_method PER --seed 1
# ERE
python fast_td3/train.py --env_name h1hand-hurdle-v0 --sampling_method ERE --seed 1
# GEOM
python fast_td3/train.py --env_name h1hand-hurdle-v0 --sampling_method GEOM --seed 1
```

See [FastTD3/README.md](FastTD3/README.md) for all benchmarks and hyperparameter details.

### BRC (multi-task)

```bash
cd multi

# Single task — uniform (default)
python3 train.py --env_names=dog-run
# Multi-task
python3 train.py --env_names=dog-run,dog-walk,dog-stand

# PER
python3 train.py --env_names=dog-run --sampling_method=per --per_alpha=0.6
# ERE
python3 train.py --env_names=dog-run --sampling_method=ere --ere_decay_constant=0.996
# GEOM
python3 train.py --env_names=dog-run --sampling_method=geom --trunc_geom_alpha=10.0
```

See [multi/README.md](multi/README.md) for full usage.

### SimbaV2

```bash
cd Simba
pip install -e .   # or: conda env create -f deps/environment.yaml

# Uniform
python run_online.py 'buffer.buffer_type=uniform'
# Exponential recency bias
python run_online.py 'buffer.buffer_type=exponential' 'buffer.alpha_exponential_sampling=10.0'
# ERE
python run_online.py buffer=numpy_ere
# PER
python run_online.py buffer=numpy_prioritized
```

See [Simba/README.md](Simba/README.md) for full usage.

---

## Citations

**FastTD3**
```bibtex
@article{seo2025fasttd3,
  title={FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control},
  author={Seo, Younggyo and Sferrazza, Carmelo and Geng, Haoran and Nauman, Michal and Yin, Zhao-Heng and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2505.22642},
  year={2025}
}
```

**BRC**
```bibtex
@article{nauman2025bigger,
  title={Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners},
  author={Nauman, Michal and Cygan, Marek and Sferrazza, Carmelo and Kumar, Aviral and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2505.23150},
  year={2025}
}
```

**SimbaV2**
```bibtex
@article{lee2025hyperspherical,
  title={Hyperspherical Normalization for Scalable Deep Reinforcement Learning},
  author={Lee, Hojoon and Lee, Youngdo and Seno, Takuma and Kim, Donghu and Stone, Peter and Choo, Jaegul},
  journal={arXiv preprint arXiv:2502.15280},
  year={2025}
}
```
