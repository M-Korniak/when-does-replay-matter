# Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners

https://arxiv.org/pdf/2505.23150

This branch contains the implementation of the BRC algorithm.

## Example usage

To run the BRC algorithm in a single task mode, just pass a single task name to the `env_names` variable:

`python3 train.py --env_names=dog-run`

By passing a list of task names, multi-task mode will be enabled.

## Replay Buffer Sampling Strategies

The default sampling strategy is uniform. Three alternatives are available via `--sampling_method`.

### GEOM — Geometric (recency-biased) sampling

Samples transitions with a truncated geometric distribution that favours recent experience. Controlled by `--trunc_geom_alpha` (higher = stronger recency bias).

```bash
python3 train.py \
    --env_names=dog-run \
    --sampling_method=geom \
    --trunc_geom_alpha=10.0
```

| Flag | Default | Description |
|---|---|---|
| `--sampling_method=geom` | — | Enable geometric sampling |
| `--trunc_geom_alpha` | `10.0` | Shape of the truncated geometric distribution |

### ERE — Emphasizing Recent Experience

Prioritises recent transitions by concentrating samples in an adaptively shrinking recent window. Updates are batched every 1000 environment steps.

```bash
python3 train.py \
    --env_names=dog-run \
    --sampling_method=ere \
    --ere_decay_constant=0.996
```

| Flag | Default | Description |
|---|---|---|
| `--sampling_method=ere` | — | Enable ERE sampling |
| `--ere_decay_constant` | `0.996` | Controls how fast the sampling window shrinks toward recent data |

### PER — Prioritized Experience Replay

Samples transitions proportionally to their TD-error magnitude and corrects for the resulting bias with importance-sampling weights.

```bash
python3 train.py \
    --env_names=dog-run \
    --sampling_method=per \
    --per_alpha=0.6
```

| Flag | Default | Description |
|---|---|---|
| `--sampling_method=per` | — | Enable prioritized experience replay |
| `--per_alpha` | `0.6` | Priority exponent — 0 = uniform, 1 = fully proportional |

## Citation

If you find this repository useful, feel free to cite our paper using the following bibtex.

```
@article{nauman2025bigger,
  title={Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners},
  author={Nauman, Michal and Cygan, Marek and Sferrazza, Carmelo and Kumar, Aviral and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2505.23150},
  year={2025}
}
```
