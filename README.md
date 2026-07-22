# PPO from Scratch

An iterative, first-principles implementation of PPO, built up one Jupyter notebook at a
time on `Acrobot-v1` (Gymnasium). Each notebook adds exactly one new idea on top of the
previous one, so the series doubles as a walkthrough of *why* PPO looks the way it does,
not just *what* it looks like.

## The series

| Notebook | Adds | Greedy eval return |
|---|---|---|
| [`V0_vanilla_pg.ipynb`](V0_vanilla_pg.ipynb) | REINFORCE + reward-to-go | −421 ± 157 |
| [`V1_value_baseline.ipynb`](V1_value_baseline.ipynb) | Learned value baseline (critic) | −88.6 ± 28.5 |
| [`V2_buffer_bootstrap.ipynb`](V2_buffer_bootstrap.ipynb) | Fixed-horizon buffer + bootstrapping | −85.0 ± 30.1 |
| [`V3_gae.ipynb`](V3_gae.ipynb) | Generalized Advantage Estimation (GAE-λ) | −88.0 ± 23.5 |
| [`V4_ppo_clip.ipynb`](V4_ppo_clip.ipynb) | PPO clipped surrogate objective + multiple update passes | −85.3 ± 12.3 |
| [`V5_kl_advnorm.ipynb`](V5_kl_advnorm.ipynb) | KL early-stopping + advantage normalization | −84.6 ± 21.6 |
| [`V6_0_parallel_rollout.ipynb`](V6_0_parallel_rollout.ipynb) | Vectorized environments, batched rollout collection (data half) | — |
| [`V6_1_minibatch_ppo.ipynb`](V6_1_minibatch_ppo.ipynb) | Minibatch PPO updates over the vectorized rollout (training half) | −83.2 ± 9.6 |

Greedy eval return is the mean ± std episode return of the deterministic (argmax) policy
on `Acrobot-v1` after training, evaluated at a fixed 15-epoch compute budget so the
versions are directly comparable.

Each notebook that produced a trained policy also has a matching rollout GIF
(`vN_baseline.gif`) showing the greedy policy in action.

## Why V6 is split in two

V0–V5 are about the *algorithm* — going from vanilla policy gradients to PPO. V6 is a
separate axis: reshaping the same algorithm for throughput, the way real large-scale PPO
implementations (e.g. CleanRL) run — many environments stepped in lockstep, flattened into
minibatches, with several epochs of updates per rollout. V6_0 builds the vectorized
rollout collector; V6_1 builds the minibatch PPO update on top of it. On CPU alone this
gets a 7–8x throughput improvement over the single-environment loop from V0–V5.

## Continuous control (MuJoCo)

Two forks take the finished V6 pipeline off `Acrobot`'s discrete actions onto continuous
MuJoCo control, changing as little as possible each step:

| Notebook | Env | Adds | Greedy eval return |
|---|---|---|---|
| [`V6_continuous_pendulum.ipynb`](V6_continuous_pendulum.ipynb) | `InvertedPendulum-v5` | Gaussian `Independent(Normal, 1)` policy, action clipping, `ACT_DIM` grids | ~1000 (solved) |
| [`V6_cheetah.ipynb`](V6_cheetah.ipynb) | `HalfCheetah-v5` | From-scratch observation + return normalization (running mean/std, ±10 clip) | ~3900 ± 90 |

The Gaussian head keeps the PPO update byte-identical to V6_1. `HalfCheetah` is the first
env where input/return normalization is decisive (without it the policy barely leaves zero);
it also never terminates, so *every* episode boundary is a 1000-step time-limit and the GAE
bootstrap is always `V(final_obs)`. `V6_cheetah` records in-progress GIFs at epochs 1 / 25 /
50 / 100 so you can watch the gait emerge.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` installs the CPU build by default; see [pytorch.org](https://pytorch.org/get-started/locally/)
for a CUDA build if you have a GPU.

## Running

```bash
jupyter notebook
```

Open any notebook and run all cells top to bottom. Notebooks are self-contained (config,
model, training loop, eval, and GIF rendering all live in the same file) — no shared
module to install.
