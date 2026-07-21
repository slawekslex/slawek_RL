# PPO implementation notes — V6_1 vs. canonical implementations

A gap analysis of this repo's `V6_1_minibatch_ppo.ipynb` against the reference PPO
implementations and the empirical "implementation details" literature, plus a bag of standard
tricks to reach for on harder environments. Scope is **classic PPO implementation details** (the
CleanRL / SpinningUp / "37 details" knobs), not LLM-scale variants.

V6_1 is correct and lands ≈ −83 greedy on `Acrobot-v1` — this is **not** a bug list. Everything
below is standard *hardening*: knobs that mostly tighten variance on Acrobot (already near the
env's practical floor) but start mattering a lot on harder envs.

Reference points:
- **CleanRL** `ppo.py` (discrete, classic-control) and `ppo_continuous_action.py` (MuJoCo)
- **OpenAI SpinningUp** `ppo`
- **"The 37 Implementation Details of PPO"** — Huang et al., ICLR 2022 Blog Track
- **"What Matters for On-Policy RL?"** — Andrychowicz et al., 2020/2021 (large-scale ablation)
- **"Implementation Matters in Deep RL"** — Engstrom et al., 2020 (PPO vs TRPO code-level study)

---

## What V6_1 already does right

- **GAE-λ with correct per-lane autoreset bootstrap** (goal→0, time-limit→V(final_obs)). This is
  the part most from-scratch implementations get wrong; here it's verified by a unit test.
- **λ-return as the value target** (`returns = adv + value`) — CleanRL's choice, not SpinningUp's
  reward-to-go.
- **Advantage normalization per-minibatch** — matches CleanRL's default (`norm_adv=True`).
- **KL early-stopping** with a 1.5× target brake — SpinningUp-style.
- **Entropy bonus** (`ENT_COEF=0.01`).
- **Separate policy/value networks and separate optimizers** with different LRs — SpinningUp's
  structure (`pi_lr`/`vf_lr`). A valid fork from CleanRL's shared-Adam.
- **Tanh activations** — the "What Matters" paper found tanh > ReLU for on-policy control.
- **Persistent env state across collections** (episodes outlive a rollout) — matches CleanRL.
- **Last-step bootstrap at the rollout boundary** is correct because `next_value` always carries
  the right bootstrap, so `advantages[-1] = deltas[-1]` needs no special-casing.

---

## Part A — Meaningful improvements (apply-now menu, ranked by expected value)

Cheap, standard, low-risk. Ranked most-impactful first.

### A1. Orthogonal initialization with layer-specific gains  ⭐ top pick
**Now:** default PyTorch `nn.Linear` init (Kaiming-uniform).
**Canonical:** orthogonal init, gain `√2` on hidden layers, **`0.01` on the policy output head**,
`1.0` on the value output head, biases zero. The small policy-head gain makes the initial policy
near-uniform (high entropy).
**Evidence:** Engstrom found orthogonal init is one of the code-level optimizations that
materially separates PPO from TRPO. Andrychowicz is more precise: *"apart from the last-layer
scaling, the init scheme does not matter much"* — the decisive piece is scaling the **last policy
layer ~100× smaller**. So the one thing to cherry-pick is the `0.01` gain on the policy head;
orthogonality is just the cheap, standard way to get there.
**Cost:** ~8 lines (a `layer_init` helper applied in `MyPolicy`/`MyCritic.__init__`).

### A2. Global gradient clipping (`max_grad_norm = 0.5`)
**Now:** none.
**Canonical:** `nn.utils.clip_grad_norm_(params, 0.5)` between `.backward()` and `.step()`, for
both optimizers. Universal in CleanRL; standard guard against the occasional large-advantage
minibatch.
**Cost:** 2 lines.

### A3. Better KL estimator (k3) + clipfrac logging
**Now:** `compute_approx_kl = (logp_old - current_logp).mean()` — the **k1** estimator. Unbiased
but high-variance and can go negative, so the 1.5×-target brake fires on a noisy signal.
**Canonical (CleanRL):** with `logratio = new_logp - logp_old`, `ratio = logratio.exp()`:
- `approx_kl = ((ratio - 1) - logratio).mean()`  ← **k3**, always ≥ 0, much lower variance
- track `clipfrac = ((ratio - 1).abs() > clip_ratio).float().mean()` as a diagnostic.
k3 is Schulman's low-variance unbiased estimator (joschu.net/blog/kl-approx.html). (CleanRL breaks
the whole update the moment KL crosses target and defaults `target_kl=None` — off; V6_1 breaks at
walk boundaries with the brake always on. Coarser but reasonable.)
**Cost:** rewrite one helper; add clipfrac to the log line.

### A4. Learning-rate annealing (linear to 0)
**Now:** fixed `PI_LR`/`VALUE_LR` for all 15 collections.
**Canonical:** anneal both LRs linearly to 0 over the run (`frac = 1 - epoch/EPOCHS`). CleanRL
default (`anneal_lr=True`). Reliable end-of-training stabilizer; tends to tighten final eval
variance.
**Cost:** a few lines in the train loop setting `param_group['lr']` each collection.

### A5. Adam `eps = 1e-5`
**Now:** Adam default `eps=1e-8`. **Canonical:** CleanRL sets `eps=1e-5`. Minor, free alignment.
**Cost:** 1 arg on each `Adam(...)`.

> **Suggested apply-now bundle:** A1 + A2 + A4 have the strongest empirical backing and near-zero
> risk; A3 improves the reliability of the existing brake; A5 is a one-token alignment. None change
> the algorithm's shape — they harden it. On Acrobot expect *tighter variance and cleaner curves*
> rather than a big mean jump.

---

## Part B — Bag of tricks for later (context-dependent; mostly harder/unbounded envs)

Not worth adding to Acrobot now — listed with the condition under which each pays off.

### B1. Observation normalization (running mean/std) + clip to ±10
`gym.wrappers.NormalizeObservation` + `TransformObservation` clip. **Andrychowicz ranks this among
the most important details.** Big on MuJoCo / unbounded observations; limited on Acrobot because
its obs are already bounded (sin/cos + clipped velocities). Reach for it on leaving classic
control. Freeze stats at eval time.

### B2. Reward / return normalization (running std of discounted returns) + clip
`gym.wrappers.NormalizeReward` — divides rewards by a running std of the *discounted return*. Also
high-impact in the ablations. Caveat: interacts with GAE and γ; it's return-scaling, not naive
reward-scaling, and getting that subtlety wrong is a common bug source.

### B3. Value-function loss clipping (PPO2-style)
CleanRL implements it (`clip_vloss=True`): clip `V` to `V_old ± clip_ratio` (reusing `clip_coef=0.2`)
and take the max of clipped/unclipped squared error, ×0.5. **But this is the trick the empirical
papers are most skeptical of** — Andrychowicz found *no* benefit (and suggests ~0.25 if used at
all), Engstrom found it inconsistent and sometimes harmful. The cleanest example of "in the
reference implementation" ≠ "actually helps."

### B4. Advantage normalization scope: minibatch vs whole-batch
V6_1 normalizes per-minibatch. Alternative: normalize once over the whole rollout before splitting.
Andrychowicz found it doesn't matter much; CleanRL uses per-minibatch.

### B5. Shared policy/value trunk + `vf_coef`
One network, two heads, single optimizer, combined loss `pi_loss + vf_coef·v_loss - ent_coef·entropy`
(`vf_coef=0.5`). Cheaper and standard for pixel/LLM backbones; can couple the two objectives badly.
V6_1's separate-nets design sidesteps that and is legitimate — this is the alternative to know.

### B6. Clip-range and entropy annealing
Anneal `clip_ratio` and/or `ent_coef` toward 0 over training. Minor, env-dependent.

### B7. Huber / smooth-L1 value loss
Blunts value-target outliers on noisy-reward envs. Situational.

### B8. Reward-to-go value target (SpinningUp)
Regress the critic on Monte-Carlo reward-to-go instead of `adv + value`. Lower bias, higher
variance. Worth an A/B if value learning ever looks unstable.

---

## Quick reference — default hyperparameters

| Knob | V6_1 | CleanRL `ppo.py` | SpinningUp |
|---|---|---|---|
| γ | 0.99 | 0.99 | 0.99 |
| GAE λ | 0.95 | 0.95 | 0.97 |
| clip ε | 0.2 | 0.2 | 0.2 |
| update epochs | 4 (KL-braked) | 4 | 80 pi / 80 v iters, KL-braked |
| minibatches | 10 (4096 each) | 4 | full-batch |
| ent coef | 0.01 | 0.01 (Atari) / 0.0 (MuJoCo) | 0 |
| vf coef | n/a (separate opt) | 0.5 | n/a (separate opt) |
| max grad norm | **none** | 0.5 | none |
| LR anneal | **none** | linear→0 | none |
| orthogonal init | **none** | yes (√2 / 0.01 / 1.0) | yes |
| obs norm | none | none (on for MuJoCo) | none |
| Adam eps | 1e-8 | 1e-5 | 1e-8 |
| KL estimator | k1 | k3 (+clipfrac) | k1-ish |
| target KL | 0.015 (1.5×0.01) | None (off by default) | 0.015 |

*Plain CleanRL `ppo.py` (CartPole/classic-control) does **not** normalize obs/reward — those knobs
live in `ppo_continuous_action.py` (MuJoCo) and `ppo_atari.py`, both clipping to ±10. MuJoCo also
uses 10 epochs / 32 minibatches vs Atari's 4 / 4.*

---

## Sources

- CleanRL — https://github.com/vwxyzjn/cleanrl · https://docs.cleanrl.dev/rl-algorithms/ppo/
- "37 Implementation Details of PPO" (Huang et al., ICLR 2022 Blog) — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- SpinningUp PPO — https://spinningup.openai.com/en/latest/algorithms/ppo.html
- "What Matters in On-Policy RL" (Andrychowicz et al.) — https://arxiv.org/abs/2006.05990
- "Implementation Matters in Deep RL" (Engstrom et al.) — https://arxiv.org/abs/2005.12729
- Schulman, "Approximating KL Divergence" (k1/k3) — http://joschu.net/blog/kl-approx.html
