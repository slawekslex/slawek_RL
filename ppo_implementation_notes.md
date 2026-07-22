# PPO implementation notes — V6_1 vs. canonical implementations

A gap analysis of this repo's `V6_1_minibatch_ppo.ipynb` against the reference PPO
implementations and the empirical "implementation details" literature, plus a bag of standard
tricks to reach for on harder environments. Scope is **classic PPO implementation details** (the
CleanRL / SpinningUp / "37 details" knobs), not LLM-scale variants.

V6_1 is correct, and the reference-implementation *hardening* in Part A is being folded in as this
doc evolves. **A1–A4 (orthogonal init, per-net gradient clipping, k3 KL + clipfrac, LR annealing)
are now implemented; only A5 (Adam eps) remains.** A1–A2 clearly earned their keep — they killed a
mid-training value blowup (v_loss 89 → 23). The effect on the *final* greedy number, though, is
inside Acrobot's eval noise: single-seed final eval across these runs has ranged from ≈ −79 ± 8 to
≈ −96 ± 70, the large std coming from an unlucky final-epoch snapshot (a near-zero-KL update can
still flip enough argmax actions to time out a few eval episodes). **Average over ≥3 seeds before
trusting any single greedy figure.** None of this is a bug list: these knobs mostly tighten variance
on Acrobot (already near the env's practical floor) but start mattering a lot on harder envs.

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
- **Orthogonal weight init** with per-layer gains (√2 hidden, 0.01 policy head, 1.0 value head)
  via a `layer_init` helper on both nets — *A1, now implemented*.
- **Per-net gradient clipping** to `MAX_GRAD_NORM = 0.5` between each `backward()` and `step()` —
  *A2, now implemented*.
- **k3 KL estimator** driving the early-stop brake, plus a **clipfrac** diagnostic in the log —
  *A3, now implemented*.
- **Linear LR annealing** — both optimizers decay to 1% of their initial LR over the run via
  `LinearLR` — *A4, now implemented*.

---

## Part A — Implementation-detail hardening (ranked by expected value)

Cheap, standard, low-risk. **A1–A3 are implemented; A4–A5 remain.**

### A1. Orthogonal initialization with layer-specific gains  ✅ implemented
**Was:** default PyTorch `nn.Linear` init (Kaiming-uniform).
**Now:** orthogonal `layer_init` on both nets — gain `√2` on hidden layers, **`0.01` on the policy
output head**, `1.0` on the value output head, biases zero. The small policy-head gain makes the
initial policy near-uniform (high entropy).
**Evidence:** Engstrom found orthogonal init is one of the code-level optimizations that
materially separates PPO from TRPO. Andrychowicz is more precise: *"apart from the last-layer
scaling, the init scheme does not matter much"* — the decisive piece is scaling the **last policy
layer ~100× smaller**. So the one thing to cherry-pick is the `0.01` gain on the policy head;
orthogonality is just the cheap, standard way to get there.
**Cost:** ~8 lines (a `layer_init` helper applied in `MyPolicy`/`MyCritic.__init__`).

### A2. Global gradient clipping (`max_grad_norm = 0.5`)  ✅ implemented
**Was:** none.
**Now:** `nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)` on the critic and the policy
*separately* (two nets, two optimizers), each between its `.backward()` and its `.step()`. This is
what tamed the epoch-8 value blowup (v_loss 89 → 23, no greedy crater) in testing.

### A3. Better KL estimator (k3) + clipfrac logging  ✅ implemented
**Was:** `compute_approx_kl = (logp_old - current_logp).mean()` — the **k1** estimator. Unbiased
but high-variance and can go negative, so the 1.5×-target brake fired on a noisy signal.
**Now (matches CleanRL):** with `logratio = new_logp - logp_old`, `ratio = logratio.exp()`:
- `approx_kl = ((ratio - 1) - logratio).mean()`  ← **k3**, always ≥ 0, much lower variance
- track `clipfrac = ((ratio - 1).abs() > clip_ratio).float().mean()` as a diagnostic.
k3 is Schulman's low-variance unbiased estimator (joschu.net/blog/kl-approx.html). (CleanRL breaks
the whole update the moment KL crosses target and defaults `target_kl=None` — off; V6_1 breaks at
walk boundaries with the brake always on. Coarser but reasonable.) The `clipfrac`
(`(|ratio − 1| > clip_ratio).float().mean()`) is now logged per collection — healthy runs sit ~0.05–0.2.

### A4. Learning-rate annealing (linear to 0)  ✅ implemented
**Was:** fixed `PI_LR`/`VALUE_LR` for all 15 collections.
**Now:** a `torch.optim.lr_scheduler.LinearLR(start_factor=1.0, end_factor=0.01, total_iters=EPOCHS)`
on each optimizer, `.step()`ed once per collection — both LRs decay to 1% of initial. (CleanRL
anneals fully to 0; 1% is close enough.) **Watch-out:** by the final epochs the LR floor makes
updates tiny (KL ≈ 0, clipfrac ≈ 0), so the last snapshot is whatever the policy froze into — which
on a noisy env can be an unlucky one. Its stabilizing benefit is real but easily swamped by
Acrobot's eval variance at single-seed granularity.

### A5. Adam `eps = 1e-5`  ⬜ remaining
**Now:** Adam default `eps=1e-8`. **Canonical:** CleanRL sets `eps=1e-5`. Minor, free alignment.
**Cost:** 1 arg on each `Adam(...)`.

> **Status:** A1–A4 implemented. A1–A2 clearly helped (killed the value blowup, v_loss 89 → 23).
> A3–A4's effect on the *final* greedy number is inside Acrobot's eval noise — runs have landed
> anywhere from ≈ −79 ± 8 to ≈ −96 ± 70, the spread driven by an unlucky final-epoch snapshot, not
> by the algorithm. **Remaining:** A5 (Adam eps), a one-token alignment. **Recommended next:**
> average over ≥3 seeds (and/or report a best-of-last-K-snapshots eval) before treating any single
> greedy figure as the score — single-seed comparison at this granularity is unreliable.

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
| max grad norm | 0.5 | 0.5 | none |
| LR anneal | linear→0.01 | linear→0 | none |
| orthogonal init | yes (√2 / 0.01 / 1.0) | yes (√2 / 0.01 / 1.0) | yes |
| obs norm | none | none (on for MuJoCo) | none |
| Adam eps | **1e-8** | 1e-5 | 1e-8 |
| KL estimator | k3 (+clipfrac) | k3 (+clipfrac) | k1-ish |
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
