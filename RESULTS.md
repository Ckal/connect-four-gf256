# Results — GF(256) Connect Four

Every published number, the command that produces it, and the artifact it
comes from. This file contains **no identity or infrastructure details** —
only numbers, commands and artifacts.

**Gate:** if a number is not in this file, it is not publishable. The paper,
the README, the model card and the Hugging Face card all cite this file and
nothing else.

Recorded at release.

---

## 1. The release artifact

| Key | Value | Source |
|---|---|---|
| `artifact_step` | **18000** | `best_hard.pt` top-level `step` |
| `artifact_bytes` | 33,830,949 | `stat` |
| `parameters` | 8,422,496 | `sum(p.numel for p in model.parameters)` |
| `artifact_sha256` | `134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2` | `sha256sum` of the frozen copy |
| `transcript_version` | v2 | checkpoint meta |
| `context_width` | 192 | trainer `--ctx` |
| `batch_size` | 16 | trainer `--batch` |
| `seed` | 32 | trainer `--seed` |
| `lags` | 0–10, 20 | checkpoint meta |
| `table_bias` | 0.0 | trainer `--table-bias` |
| `inference` | CPU-only | `exp_t`/`log_t` are CPU tensors |

**Where the checkpoint lives**

```bash
curl -L -o best_hard.pt \
 https://huggingface.co/Chris4K/connect-four-gf256/resolve/main/best_hard.pt
sha256sum best_hard.pt
# 134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2
```

The checkpoint is **not** committed to the git repository: it is already
hosted on Hugging Face, and committing a 33.8 MB binary makes every clone
pay for it and every retrain grow the history.

---

## 2. The authoritative numbers

### 2.1 Label agreement

| Key | Value | Source |
|---|---|---|
| `label_agreement_train` | **59.55 %** | `tools/measure_promotion_scorer.py --corpus-run runs/_probe/v24`; also `metrics.jsonl` `perply_agreement_pct` at step 18000 |
| `label_agreement_holdout` | **41.02 %** | `metrics.jsonl` `holdout_perply_agreement_pct` at step 18000 (n=512) |
| `solver_holdout_coverage` | 560 / 1,500 = **37.3 %** | bounded-endgame-solver holdout audit |

> **Terminology.** The teacher is a **negamax alpha-beta search policy** (and a
> 2-ply lookahead policy). It is *not* an exact solver: Connect Four is a
> solved game, but this model imitates a bounded search. The bounded endgame
> solver (`src/c4/exact_solver.py`, `max_empty=7`) is used to *audit* holdout
> positions, not to label the corpus.

### 2.2 Training corpus

| Key | Value | Source |
|---|---|---|
| `corpus_games` | 320,000 | trainer preflight row count |
| `corpus_labels` | 4,462,571 | per-ply label count |
| `corpus_boards_scanned` | 5,100,175 | corpus integrity scan |
| `corpus_violations` | 0 | preflight |

### 2.3 Game strength — the 600-game × 3-seed matrix

Canonical harness `src.c4.duel`: model greedy (ε = 0), opponent explores
(ε = 0.3) so trajectories are genuinely distinct.

| Key | Value | Result |
|---|---|---|
| **strength vs random** | **85.17 %** (511-88-1, n=600) | model wins |
| **strength vs centre_policy (head-to-head)** | **90.50 %** (543-57-0, n=600) | model wins |
| **strength vs random, centre_policy** | **91.00 %** (546-54, n=600) | centre_policy scores *higher* against random than the model does |
| **strength vs 1-ply heuristic** | **37.17 %** (223-376-1, n=600) | model **loses** |
| **strength vs 2-ply lookahead** | **24.33 %** (146-452-2, n=600) | model **loses** |
| **strength vs minimax** | **41.83 %** (251-343-6, n=600) | model **loses** |
| `best_constant_baseline` | `centre_policy` (546-54 vs random) | — |
| `model_vs_centre_policy_fisher_p` | **0.00236** | centre_policy is significantly stronger than the model when both are measured against random |
| vs. random, aggregate | **W1460 L337 D3** of 1800 | p ≈ 3.2 × 10⁻¹⁶⁶ |

**The one-line truth:** the model reaches **85.2 % vs random**, **beats the
centre-column policy head-to-head (90.5 %)**, but is **net negative against a
1-ply heuristic (37.2 %)** and is **significantly weaker than the
centre-column policy when both are measured against random** (85.2 % vs.
91.0 %, p = 0.00236). It is not a strong Connect Four player.

The Fisher implementation is self-tested against the known value
[[1,9],[11,3]] = 0.00275946.

### 2.4 Per-column recall

| Column | Train | Holdout |
|---|---:|---:|
| c0 | 63.7 % | — |
| c1 | 56.5 % | 42.1 % |
| c2 | 56.2 % | — |
| c3 | 73.9 % | — |
| c4 | 50.1 % | — |
| c5 | 43.3 % | 18.6 % |
| c6 | 43.7 % | 7.1 % |

**The c1–c4 finding.** The model is measurably better on the **centre column
(c3, 73.9 % recall)** than on the flanks, and its worst columns are the
**outer ones** (c5 43.3 %, c6 43.7 % on train; 18.6 % and 7.1 % on holdout).
This is the residual of the historical "always-c3" defect: the repair fixed
the *hard* centre bias, but a soft centre preference survives. The model's
weakest play is on the outer columns — exactly where a centre-column policy
is never challenged.

### 2.5 The convergence curve

Source: `metrics.jsonl`, 4000-sample holdout, 512-position holdout per-ply
subset, 48-game ladder vs. random and heuristic.

| Step | Label % (train) | Label % (holdout) | vs heuristic W-L | vs random W-L |
|---:|---:|---:|---:|---:|
| 0 | 16.3 | 11.9 | 9-39 | 36-12 |
| 2000 | 40.2 | 27.9 | 10-38 | 37-11 |
| 6000 | 50.9 | 36.7 | 15-33 | 38-10 |
| 10000 | 56.0 | 40.8 | 19-29 | 37-11 |
| 14000 | 57.5 | 38.1 | 23-25 | 44-4 |
| 18000 | **59.5** | 41.0 | 16-32 | 41-7 |
| 20000 | 59.0 | 38.9 | 15-32 | 37-11 |
| 26000 | 58.7 | 39.1 | 15-33 | 38-10 |
| 30000 | 59.4 | 42.0 | 18-30 | 38-10 |
| 34000 | 58.5 | 41.8 | 14-34 | 43-5 |

**Reading it correctly:**

- **Label agreement rose from 16.3 % → ~59 %.** A real, large improvement
 over the corrected short-run baseline.
- **Label agreement has plateaued** since roughly step 14000–18000 at
 57–60 % train / 38–42 % holdout. More steps are buying very little.
- **The model beats random decisively** (85.2 % win rate).
- **The model is NET NEGATIVE against the 1-ply heuristic** for the entire
 run (14-34, 18-30, 23-25 at the best points). It never reaches parity.

### 2.6 Checkpoint selection is evidence-based, not "newest wins"

Head-to-head, promoted step 18000 vs. pinned step 20000:

```
W208 L192 (52.0 %, one-sided p = 0.45)
82.5 % same-column agreement
first divergence at ply 16
```

Statistically inconclusive on the direct duel. The strength matrix resolves
it — step 18000 is ahead of step 20000 on **all five** opponent arms:

| Opponent | step 18000 | step 20000 | Better |
|---|---:|---:|---|
| random | **85.17 %** | 84.67 % | 18000 |
| centre_policy | **90.50 %** | 86.67 % | 18000 |
| 1-ply heuristic | **37.17 %** | 30.17 % | 18000 |
| 2-ply lookahead | **24.33 %** | 24.00 % | 18000 (tie) |
| minimax | **41.83 %** | 41.33 % | 18000 (tie) |

Both the label metric and actual play agree — and they agree in the same
direction.

---

## 3. Reproduce it

```bash
git clone https://github.com/Ckal/connect-four-gf256.git
cd connect-four-gf256

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Train (the exact V24 configuration)
V24_TAG=v24 bash run_v24.sh

# Score a checkpoint against its owner corpus
python -u tools/measure_promotion_scorer.py \
 --corpus-run runs/_probe/v24 \
 runs/_probe/v24/checkpoints/best_hard.pt

# Serve the game locally
python -m src.c4.ui_server --model best_hard.pt --port 7860 --device cpu

# Full regression suite
bash tools/run_regression_sweep.sh
```

The scorer **refuses to guess** which corpus belongs to a checkpoint: pass
`--corpus-run` explicitly, or let it find an unambiguous ancestor owner run.
Without one it exits non-zero rather than scoring against the wrong data.

### 3.1 Regression suite

| Property | Evidence |
|---|---|
| Full regression suite | **27 PASS, 0 FAIL** — reproduced by independent sweeps |
| Live UI parity | 40/40 played columns, next-move distribution L1 = **0.000e+00** |
| Atomic checkpoint publication | 9/9 checks; 0 corrupt reads in 769 reads |
| Bit-exact resume | max weight Δ **0.000e+00**, identical Adam moments (12/12 checks) |
| Transcript stamping | 3/3 checks |
| Session isolation | 47/47 checks |
| Embedding + origin policy | 18/18 checks |

Component suites: live parity v3/v4/v5 · UI hot reload 14/14 · resume state
19/19 · resume equivalence 12/12 · single-server UI 8/8 · atomic save 9/9 ·
transcript stamp 3/3 · watcher 12/12.

### 3.2 The public deployment gate

```bash
python scripts/verify_public_isolation.py https://c4.tensorgarten.de
# PASSED: 19 FAILED: 0
```

The gate runs **against the deployed process over public HTTPS** — not
against a local server.

---

## 4. Bugs that had to be fixed first

Each of these produced a *plausible-looking* number that was wrong.

| # | Bug | Impact |
|---|---|---|
| 1 | Terminal-state check truthy on terminal boards | Wrong game-over detection |
| 2 | Corrupt trailing field in **126,537 / 320,000** rows (39.5 %) | 1,567,958 affected plies; trainer now rejects corrupt corpora |
| 3 | **Sidecar read by batch position instead of sampled corpus row** | ~41.3 % of supervised positions had all-zero targets |
| 4 | Promotion key = lowest byte-bpb | Selected step 1500 over the final step 3923 |
| 5 | Final exports omitted transcript version | Read as v1 → scored 24.4 % instead of 91.3 % |
| 6 | Historical per-ply index arithmetic | Early positions `t < k` assigned to row `k−1` |

**What this invalidates.** The project's earlier conclusions — *"dense
per-ply supervision is worse"*, *"mirror consistency is net negative"*,
*"symmetry is not worth it"* — were all drawn **before** the row-index
repair. They are not evidence about the corrected model and are **not**
cited in the paper.

Fixing this also invalidated the old "always play column 3" control: it had
actually been a centre-first *staircase*, not a constant policy.

---

## 5. Limitations (must appear in every public artifact)

1. **Net-negative vs. the 1-ply heuristic** — the headline negative result.
2. **Teacher-agreement ≠ game strength** — the central finding.
3. **Holdout solver coverage is 37.3 %** (560/1,500). The other 940
 positions have no proved label and are excluded, not guessed.
4. **~18 pp train/holdout gap** indicates memorisation in the headline
 number.
5. **Architecture-level superiority is unproven** — the controlled A/B arms
 reached only 0.30 epoch-equivalents and are provisional.
6. **Mirror/symmetry questions remain open** for the corrected model.
7. The teacher is a **bounded search policy**, not a proof of optimality.

---

## 6. Artifacts

| Artifact | Where |
|---|---|
| Source | `https://github.com/Ckal/connect-four-gf256` |
| Checkpoint (SHA-256 verified) | `https://huggingface.co/Chris4K/connect-four-gf256` |
| Playable game | `https://c4.tensorgarten.de` |
| Hugging Face Space (static) | `https://huggingface.co/spaces/Chris4K/connect-four-gf256` |
| Deployment gate | `scripts/verify_public_isolation.py` |
| The manuscript | `CONNECT_FOUR_PAPER.md` |
| The model card | `MODEL_CARD.md` |
