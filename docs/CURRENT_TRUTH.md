# Current Truth — Connect Four

**As of 2026-10-28, late night.** One page. Everything here is measured, and
every number names its artifact. Where the long handover
(`CONNECT_FOUR_HANDOVER.md`) disagrees, **this page wins** — the handover is
working history and contains later-retracted claims.

> **🔴 READ §5c FIRST.** The per-ply supervision was indexed by **batch
> position instead of corpus row**. Every trained arm (`fd_a`/`fd_b`/`fd_c`)
> was trained on the teacher answers of **16 of 320,000** sidecar rows, and
> **41 %** of the supervision silently degenerated to "predict the move
> that was already played". The code is **fixed and guarded**
> (`tmp/test_perply_row_index.py`); the retrained model is in progress.
> **Every conclusion below that depends on a trained model's behaviour is
> provisional until it is re-measured on a correctly-trained model.**

---

## 1. Where the project actually stands

| | |
|---|---|
| **Is there a working model?** | Yes. GF-trunk Connect Four agent, CPU-only hard export, playable at `http://127.0.0.1:7860/`. |
| **Is it release-ready?** | **No** -- but the blocker is now understood and it is **not** the architecture. |
| **Is it better than always playing c4?** | **Not yet.** The corrected arm beats every single-column constant against minimax and centre, but loses to the centre-first priority sweep on random and lookahead. |
| **Was V24 launched?** | **No.** V24 is now defined as a **properly-budgeted** full run (§5q). |
| **Did the project prove the architecture can learn?** | **Yes.** The corrected 500-row run fits its labels to **91.29 %**. |
| **Why is it not yet strong?** | **The full arms saw the 320,000-row corpus only 0.30 times.** One epoch = 20,000 steps; the runs were cut off at 6,000. The LABEL curve was still climbing **+1.42 pp / 500 steps** when it stopped. See **§5q**. |

**The one-line summary:** the pipeline, the data, the labels and the
measurement infrastructure are all sound, and the model demonstrably
learns. **The full arms were stopped at 30 % of one epoch over their
training corpus** -- that, not the architecture, is what stands between
this and a strong player. Every "the model is weak" claim below that was
measured on fx_a/fx_b/fx_c is provisional until the same architecture is
trained to convergence.

---

## 2. Authoritative numbers (do not quote anything else)

### Solver-optimality, 560 solver-proven holdout positions

| policy | solver-optimal |
|---|---:|
| **always-c4** | **75.7 %** |
| always-c1 | 73.9 % |
| always-c5 | 72.5 % |
| always-c6 | 70.5 % |
| always-c3 | 69.6 % |
| chance | 65.2 % |
| **fd_a (shipped)** | **75.0 %** |
| fd_b | 75.0 % |
| fd_c | 70.9 % |

Artifact: `runs/_probe/solver_buckets_auth_baselines.json`
(budget **120,000,000**, `--max-empty 24 --max-positions 1500`).
The default-budget run (166 positions) is **non-authoritative**.

### Hard-path column distribution, full 19,279-position release holdout (fd_a)

```text
c0 13.2%   c1 13.0%   c2  8.3%   c3 18.3%
c4 37.8%   c5  8.4%   c6  1.0%
```

Artifact: `runs/_probe/eval_fd_a_s6000_v4_holdout.json`

### The 112 unique-optimum positions (highest-signal subset)

| | |
|---|---:|
| fd_a unique-optimum accuracy | **24.1 %** (step 6000) |
| always-c4 on the same positions | **29.5 %** |
| fd_a's best step (2000) | 34/112 vs always-c4's 33/112 → **p = 0.836**, a tie |
| c1/c2 ever played on these | **0** |
| clear-cut positions unreachable | **23/112 (20.5 %)** |
| fd_a picks c3 | 16/112, of which **14 are wrong** |

Artifacts: `runs/_probe/unique_curve_fd_a.json`, `runs/_probe/c3_bias_unique.json`

### Teaching quality (the labels)

| | |
|---|---:|
| teacher move is in the exact optimal set | **94.6 %** (530/560) |
| teacher move is genuinely suboptimal | **5.4 %** (30/560) |
| positions with a *unique* optimal move | 20.0 % (112) |
| positions with several equally optimal moves | 80.0 % (448) |

Artifact: `runs/_probe/teacher_vs_solver.json`

---

## 3. The three real, measured defects

### 3a. The labels collapse the tie set — **the primary defect**
The teacher picks *one* of several equally optimal moves, with a strict
`if v > best` centre-first tie-break. Targets are therefore ~**99.3 % one-hot**
(entropy ≈ 0.008 nats) when **~80 %** of positions admit multiple optimal
moves. The model is being asked to pick a single arbitrary column out of a
set, so it learns an arbitrary prior instead of the game.

### 3b. The model has its own column prior, independent of the labels
On identical positions the model concentrates more than its teacher does
(TV distance model↔label = **0.377**). Whole-model effects: c4 37.8 %,
c6 only 1.0 %. On the high-signal unique-optimum subset it plays c1/c2
**never**, and picks c3 where c3 is wrong 14 times out of 16.

### 3c. The evaluation set is distorted
The 560 solver-proven positions are restricted to `≤24` empty cells and are
therefore endgame-heavy. On that subset the model plays c1 2.0 % / c2 0.7 %;
on the full holdout it plays c1 13.0 % / c2 8.3 %. **Any column claim must
name its population.** 940 of 1,500 requested positions remained unproved
and were excluded, not scored.

### 3d. An audit-script defect found 2026-10-28 (fixed; no effect on the model)
The representative-subset audit initially reported the model at **65.36 %**
in-optimal with a *different* per-column histogram than the authoritative
evaluator — on the **same 560 positions, same checkpoint**. Cause: it took a
**raw** `argmax` over the raw column logits, so it could select a **full
column**. The shipped decision path (`Agent.choose`, `ui_server.py`,
`tg_solver_buckets.py`) masks illegal columns to `-1e9` *before* the argmax.

| protocol | illegal picks | in optimal set |
|---|---:|---:|
| raw argmax (wrong) | 64/560 (11.43 %) | 65.36 % |
| **masked argmax (shipped)** | 0 | **75.00 %** — matches the authoritative evaluator exactly |

The two disagreed on 64/560 picks; the corrected per-column counts are
identical to the D16 table. Fixed in `tmp/audit_col_representative.py`
(validation 560/560 against the canonical API).

**Lesson:** an unmasked argmax measures a policy the user never plays. Any
competence number must use the same masked protocol as `Agent.choose`.
Artifact: `runs/_probe/mask_reconciliation.json`.

### 3e. The column asymmetry is **depth-dependent** (new, 2026-10-28)
The depth-stratified breakdown of the corrected representative audit shows
the model's in-optimal rate falls as the position gets deeper, while the
teacher's stays flat:

| empty cells | n | model in optimal set | teacher in optimal set |
|---|---:|---:|---:|
| 0–8 (endgame) | 8 | 87.5 % | 100.0 % |
| 8–14 | 32 | 84.4 % | 96.9 % |
| 14–20 | 162 | 72.2 % | 95.1 % |
| 20–25 | 358 | 75.1 % | 94.1 % |

And the c1/c2 under-use tracks it: c1+c2 take **2.7 %** of the model's moves
on the proven subset but **21.3 %** on the full holdout. So the "collapse" is
concentrated in the *deep, early-game* band — the opposite of what the
endgame-heavy subset alone would suggest.

Artifact: `runs/_probe/col_representative.json`

---

## 4. The raw Pons data is correctly decoded — and it is the best target source

This is new (2026-10-28) and it settles a long-running question.

A systematic **12-convention orientation grid** (row flip × column flip ×
player mapping `none`/`parity`/`swap`, 400 exact-solved rows) selected:

```text
row_flip=True   col_flip=False   player=parity
```

which is **exactly** the convention in `tools/import_hf_c4.py`:

```python
idx  = (ROWS - 1 - r) * COLS + c     # row flip
flip = stone_count % 2 == 1          # parity swap
```

| measurement | value |
|---|---:|
| sentinel precision (the `-1000` illegal marker) | 99.75 % |
| sentinels landing on a *legal* column | **0** |
| label argmax in the exact optimal set | **100.0 %** |
| Pearson r (raw score, exact root value) | **+0.917** |
| Spearman ρ per position | median **+1.000**, perfect on 73.2 % |
| sign contradictions | **0** |
| rows carrying a genuinely graded score vector | **78.5 %** |
| raw score max-set ⊆ exact optimal set | **100.0 %** |

Artifacts: `runs/_probe/decoding_grid.json`, `runs/_probe/pons_score_set_equality.json`,
`runs/_probe/pons_refinement.json`

**The scale finding (important).** The Pons score is *not* the exact value in
the same units — it is a **distance-like** scale (`+k` = win in *k* plies,
`−k` = loss in *k*, `0` = draw) whereas the exact solver returns outcome-only
(`±1`, `0`, `1000`). That is why only **18.8 %** of columns match numerically
while the sign agrees **100 %** of the time. The two never contradict: the
Pons max-set is a **subset** of the exact optimal set in **100 %** of proved
positions, and it *refines* it by separating "win in 1" from "win in 5".

**Consequence:** graded, distance-aware targets are available today with no
new solving. The label argmax is already optimal 100 % of the time; what is
missing is the *graded magnitudes* and the *tie set*.

### The historical "28.3 %" was a metric artefact
`tg_opening_teacher.py` compared one `np.argmax(target)` against the exact
solver's **single** chosen column. Both are arbitrary tie-breaks, so ~28 %
agreement is the *predicted* coincidence rate. Decomposition:

| | label == solver's one choice | label **in** the exact optimal set | suboptimal labels |
|---|---:|---:|---:|
| relabelled pilot (60 pos) | 28.3 % | **100.0 %** | **0** |
| raw Pons parquet (400 pos) | 40.0 % | **100.0 %** | **0** |

Artifact: `runs/_probe/thirty_pct_decomposition.json`

---

## 5. The training ladder was never running — `W0L0D0` was vacuous

`run_fixed_ab.sh` passes **`--eval-games 0`**, so `evaluate_play(model, device, 0, …)`
plays zero games and reports W0L0D0 **by construction**. The trainer's own
default is **24**. Re-running the shipped fd_a checkpoint with the canonical
loader and 24 games gives **W12 L12 D0** against the centre-column reference
bot (a sanity metric, not a strength rating).

Artifact: `runs/_probe/ladder_enabled_check.json`
**Not yet fixed** in any launcher.

---

## 5b. NEW (2026-10-28): in-game, the shipped model LOSES to a constant column

Everything in §2 is **off-policy** — it measures the model on *held-out
positions*. It does not measure what the model does when the board evolves
from **its own** moves. That measurement was missing, and it is now done.

Using the canonical `src.c4.duel` harness (model greedy/`eps=0`, opponent
supplies the trajectory variation so every game is a distinct legal
trajectory), the shipped fd_a against a **random** opponent over 200 games:

| arm | W | L | D | win % |
|---|---:|---:|---:|---:|
| **fd_a (shipped model)** | 125 | 75 | 0 | **62.5 %** |
| always-c0 | 155 | 45 | 0 | 77.5 % |
| always-c1 | 170 | 30 | 0 | 85.0 % |
| always-c2 | 166 | 34 | 0 | 83.0 % |
| **always-c3** | **186** | **14** | 0 | **93.0 %** |
| always-c4 | 170 | 30 | 0 | 85.0 % |
| always-c5 | 161 | 39 | 0 | 80.5 % |
| always-c6 | 158 | 42 | 0 | 79.0 % |

**Every constant column beats the model.** The model also does **not** have
V18's column collapse: its in-game top column is c4 at **36.6 %**, and
c0+c1+c2 take **29.9 %** of its moves.

**What this does and does not mean.**
* It means the *off-policy* solver-optimality number (75.0 %) does **not**
  translate into playing strength. The model reads boards but does not play
  them well — the two are separate capabilities and only the second one
  reaches the user.
* It does **not** mean the model is worse than a constant policy on
  *positions*; the solver comparison in §2 still stands.
* Single-opponent, single-seed numbers are noisy. A multi-opponent,
  multi-seed rerun with exact tests is the authoritative version; see
  `runs/_probe/ingame_strength_fd_a.json`.

Artifacts: `runs/_probe/ingame_cols_fd_a.json`,
`runs/_probe/ingame_strength_fd_a.json`.

### 5b-1. NEW (2026-10-28): the constant arms were NOT constant — and the
### corrected result is *worse* for the model, not better

The "always-cX" arms above were written as:

```python
legal = [c for c in range(7) if board.heights[c] < 6]
return col if col in legal else legal[0]      # <-- lowest-index fallback
```

Once column `c` fills up, the policy falls back to the **lowest legal
column**. So "always-c3" was really a **column-priority sweep**: centre
first, then c0, c1, c2, c4, c5, c6 in index order. The column-spread
control measured it directly — "always-c3" actually played c3 only
**28.2 %** of its moves.

That ambiguity had to be removed, because the two readings mean very
different things:

* *"the model is worse than a **trivial** constant policy"* → damning.
* *"the model is worse than a **priority-sweep heuristic**"* → much milder
  (the model just lacks a positional prior).

`tmp/audit_constant_vs_priority.py` separates them. **True constant** =
play `c`; when `c` is full, play a *random* legal column.

Against the same explored `centre_policy` opponent, 200 games × 3 seeds:

| arm | W | L | D | win % | unique games |
|---|---:|---:|---:|---:|---:|
| **true constant c0** | **471** | 129 | 0 | **78.5 %** | 86/80/91 |
| true constant c6 | 465 | 135 | 0 | 77.5 % | 85/79/84 |
| true constant c5 | 460 | 140 | 0 | 76.7 % | 87/80/87 |
| true constant c1 | 443 | 157 | 0 | 73.8 % | 93/82/90 |
| true constant c4 | 443 | 157 | 0 | 73.8 % | 96/80/87 |
| true constant c2 | 438 | 162 | 0 | 73.0 % | 94/82/85 |
| true constant c3 | 241 | 359 | 0 | **40.2 %** | 200/200/200 |
| **MODEL (fd_a)** | **379** | 221 | 0 | **63.2 %** | 172/180/169 |

* Model vs best **true** constant (c0): **Fisher p = 6.5e-09**. The model
  loses to a *trivial* baseline. This is the damning reading, and it is
  the correct one.
* Model vs best **priority sweep** (c0): Fisher p = 1.2e-09.
* The fallback alone moves c3 by **+81 wins** (241 → 322), confirming the
  original arm was measuring the sweep, not the constant.

**A caveat that matters.** The `centre_policy` opponent is itself a
constant-ish policy, and c0 is the *opposite edge* from the centre. The
model's c4-centric prior (36.8 % of in-game moves) collides with it, while
a c0 policy builds a wall it can complete. So part of this gap is
"the model has no edge-column prior" rather than "the model cannot play".
Either way the model is behind a trivial baseline, which is the finding
that matters.

**Harness validity.** `src.c4.duel.duel` ignores its `seed` argument, so
`tmp/audit_strength_harness_control.py` re-validates the harness before
these numbers are trusted:

| control | result | verdict |
|---|---|---|
| C1 model vs model, opponent explored | W283 L302 D15, 47.2 %, **p = 0.457** | **PASS — harness is unbiased** |
| C2 always-c3 vs always-c3, explored | W322 L249 D29, 53.7 %, p = 0.0026 | expected: the explored side pays a handicap; a *control*, not a bias |
| C3 model vs model, no exploration | exactly **1** unique game (not 2) | corrected: both sides greedy ⇒ one trajectory, not two |

C1 is the load-bearing control: with an unbiased harness, the model-vs-
constant gap is real. C3 corrected an earlier assumption of "2 unique
games" — two *greedy* policies produce **1** trajectory; "2" arises only
when the sides differ.

Artifacts: `runs/_probe/constant_vs_priority.json`,
`runs/_probe/strength_harness_control.json`,
`runs/_probe/strength_harness_control_v2.log`.

---

## 5c. NEW (2026-10-28, late): **ROOT CAUSE FOUND** — the per-ply sidecar was indexed by batch position, not corpus row

This is the defect that explains every "the model fits the played move but
not the teacher label" measurement in this project. It was found by reading
the loss path, not by any metric.

### The bug

In `src/c4/train.py`, the per-ply supervision branch built the sidecar row
index as:

```python
b_idx = torch.arange(keep.numel(), device=device) // (y.shape[1] + 1)
rows = b_idx[hit].cpu()
tgt = _pt[rows, s_idx].to(device)
```

`arange(keep.numel()) // (width + 1)` yields **0 … B−1** — the **batch
index**, not the corpus row the window was sampled from. The single-label
path next to it (`_soft_rows_for_batch`) returns `chosen[k]` and was always
correct; only the per-ply path was wrong.

The **slot** index came from the correct row (via `chosen[k]` inside
`_perply_rows_for_batch`), so the mismatch was **invisible**: every target
read was a real, well-formed teacher label — belonging to a *different
game*. Nothing crashed, the loss fell, `LEGAL` climbed, every checkpoint
and artifact looked healthy.

### Measured effect (production corpus + production sidecar, model-free)

| quantity | before fix | after fix |
|---|---:|---:|
| trainer target == correct target | **9.59 %** | **100 %** |
| distinct sidecar rows read (of 320,000) | **16** | 100s |
| all-zero target → silent played-move fallback | **41.27 %** | **0.00 %** |
| supervised positions with a live target | 58.55 % | 100 % |

Artifact: `runs/_probe/perply_row_index_bug.json`
Audit: `tmp/audit_perply_row_index_bug.py`

### Why the fallback mattered

`src/c4/move_loss.py::_weighted_move_loss` substitutes a one-hot on the
**PLAYED** move when the teacher target has no mass on the legal columns.
With `soft_w = 1.0` the teacher target otherwise *replaces* the played-move
term entirely — so that fallback was the **only** route by which the played
move could enter the loss at all.

That is the whole paradox, in one line:

> **41 % of the supervision was silently "predict the move that was already
> played", and the other 59 % was the answer to a different game.**

So the model learned the played move well (**59 %** trainer `LEGAL`,
**56.6 %** direct agreement) and the teacher labels badly (**20.2 %**
direct agreement). The architecture was never the problem.

### Scope

The bug was in the working tree, never committed, so it affected **every**
`fd_a` / `fd_b` / `fd_c` run — they all used `--perply-targets`. The
`overfit_test` run was also affected.

### Fix and guard

* `src/c4/train.py::_perply_corpus_rows(chosen, width, device)` — new
  helper returning the corpus row per flattened position.
* The per-ply branch now uses it, and **drops** any position whose target
  has no mass on the legal columns instead of falling back to the played
  move. (With the correct row index the drop count is **0**; the guard is
  there so a future sidecar cannot silently reintroduce the fallback.)
* `tmp/test_perply_row_index.py` — regression test, calls the **shipped**
  helper.
* `tmp/verify_test_catches_bug.py` — proves the test **FAILS** on the
  pre-fix helper (it does: `rc=1`, "only 16 distinct corpus rows read").

**Status of the fix: code fixed and guarded; the retrained model is still
being produced.** Nothing downstream of this defect — including every
"the model is not competitive with a constant column" conclusion — can be
trusted until a model is retrained on correctly-indexed supervision.

### Which existing artifacts survive, and which do not

| artifact | verdict |
|---|---|
| `can_the_model_learn.json` (GOLD fit audit) | **valid** — it measured the pre-fix state correctly |
| `label_orientation_check.json` | **valid** — the *labels* are fine |
| `teacher_vs_solver.json`, `solver_buckets_auth_baselines.json` | **valid** — label/solver facts, no model involved |
| `decoding_grid.json`, `col_sensitivity_fd_a.json` | **valid** — format and per-column sensitivity facts |
| `label_vs_learning.json` | **model half void** — the label-quality half stands, the model half must be re-measured |
| `overfit_test/fit_measurement.json` (19.08 %) | **void** — it measured the plumbing, not the model |
| `tie_model_behaviour.json` | **model half void** — the tie/single comparison needs a new model |
| `ingame_strength_fd_a.json` | **void** — the arm was trained on wrong supervision |
| `fit_and_mechanism.json` (the 100 % claim) | **void** — independent move-byte decode bug |

---

## 5d. NEW (2026-10-28): the checkpoint **promotion gate** was promoting the checkpoint that fits the labels *least*

The 500-row acceptance run passes the supervision gate at **91.29 %**
direct sidecar-label agreement — but it promoted the *wrong* checkpoint:

```text
runs/_probe/overfit_fixed/best_hard.pt        step 1500   62.82 % label agreement
runs/_probe/overfit_fixed/hard_model_final.pt  step 3923   91.29 % label agreement
```

**Why.** The trainer's promotion key is a 4-tuple:

```python
sel = (float(_ladder_net), ho_pct, -hard_bpb, float(best_distinct))
```

* slot 0 — ladder score
* slot 1 — held-out optimal-move accuracy
* slot 2 — **negative hard bpb**
* slot 3 — distinct-trajectory count

In a `--load-corpus` run with `--eval-games 0` there is **no holdout** and
**no ladder measurement**, so slots 0 and 1 were empty and the comparison
**degenerated to slot 2: lowest byte-bpb wins.**

But byte bpb measures how well the model predicts the **played move byte**,
while the trainer's actual objective is the masked 7-way loss against the
**teacher's label**. Those are different quantities. In the acceptance run
they actively disagree:

```text
step 1500   byte bpb 5.513   label agreement 62.82 %   <- promoted
step 3923   byte bpb 6.798   label agreement 91.29 %   <- final export
```

So the gate selected the checkpoint that fits the training objective **least**
— the exact inverse of what a promotion gate is for.

**The repair.** `src/c4/train.py` now has a `PerPlyAgreement` scorer that
measures the objective directly:

* reads the **real sidecar**, at the **real ply**, with the **same
  corpus-row lookup the training loss uses** (`_perply_corpus_rows`);
* scores the **hard export** (the artefact that ships and that the UI
  serves), not the float training weights;
* applies the **same legal mask** the agent applies at serve time, decoded
  from the ply's own 9-byte v2 header — which encodes the board *before*
  that ply's move, i.e. the position the label describes;
* counts a HIT when the masked argmax column has the **maximum** target
  value, so ties count as correct ("any equally optimal move is correct",
  the semantics the sidecar encodes).

The promotion key now reads:

```python
if ppa is not None and ppa.get("perply_n", 0) > 0:
    _primary = float(ppa["perply_agreement_pct"])   # the OBJECTIVE
else:
    _primary = float(_ladder_net)
sel = (_primary, ho_pct, -hard_bpb, float(best_distinct))
```

The metric is also written to `metrics.jsonl` as `perply_agreement_pct` /
`perply_n` / `perply_mass`, and printed in the training panel as
`| LABEL xx.x% (n=…)` so the number that drives promotion is visible in the
log rather than hidden.

**Regression coverage.** `tmp/test_promotion_gate.py` — **16 PASS, 0 FAIL**.
It checks that the scorer exists, that it is the **primary** key, that
byte-bpb is no longer the primary, that the metric reaches `metrics.jsonl`,
and — behaviourally — that the scorer reports **92.40 %** on the corrected
checkpoint, and that the label it measures is genuinely *not* the played
move (`label == played` on only **20.82 %** of corpus positions, consistent
with the independently measured 21.33 %).

**Two traps found while building this, both worth recording.**

1. *The zeroed-export trap.* The first hard-path implementation returned an
   identical **12.68 %** for every checkpoint. Cause:
   `GFMarkovLM_V6` has no `import_hard`, so a V6 checkpoint re-loaded
   through `_load_agent` keeps **zero float parameters**; `export_hard()` on
   that model produces an all-zero model whose every logit is `0.0` and
   whose argmax is always column 0 — a constant, useless policy. The
   **saved** hard payload is fine; only the float re-export was broken.
   Fix: the scorer receives the hard payload by **injection** (the trainer
   injects its live `_hard_export(model)`; offline measurement reads the
   payload straight out of the file). Re-measured: final checkpoint
   **92.38 %**, step-0 checkpoint **12.68 %** — the metric now discriminates.
2. *The context-width trap.* `trainer_window` reads at index `ctx-1`, so
   the scorer must use the trainer's `--ctx` (192 for the acceptance run).
   The scorer had a hardcoded 191; it now takes `ctx` from the trainer and
   from the offline measurement.

### What the corrected numbers now are

```text
direct sidecar-label agreement, hard export
  runs/_probe/overfit_fixed/hard_model_final.pt   91.29 %  (tmp/measure_label_agreement.py)
  same checkpoint, via the shipped scorer           92.38 %  (n=4000)
  same checkpoint, via the promotion-gate test      92.40 %  (n=3000)
  runs/_probe/overfit_fixed/best_hard.pt (step 1500) 62.82 %
  step 0                                            12.68 %
```

The two independent measurement paths (the standalone script and the
shipped scorer) agree to within 1.1 pp, and both are far above the
90 % acceptance gate. **The architecture is not the limiting factor** — the
per-ply supervision plumbing was.

---

## 5f. NEW (2026-10-28, late): the seat gap is a GAME PROPERTY, and the
##     "learning problem" verdict was measured on a defective checkpoint

### 5f.1 Seat gap -- resolved with matched controls

`tmp/audit_seat_gap.py` was rewritten after the first G3 arm was found to
be structurally invalid (it paired a **perfect** minimax against a
**30 %-random** minimax -- a rout, which reported an *inverted* −8.0 pp).
The corrected arms, 600 games per seat per arm (n=1200 per arm):

```text
G1 MODEL(greedy) vs MODEL(explored)      gap +5.2 pp   p=0.073
G2 MODEL(explored) vs MODEL(explored)    gap +7.5 pp   p=0.009   <-- model, significant
G3 MINIMAX(explored) vs MINIMAX(explored) gap -1.7 pp  p=0.564   [CONTROL]
G4 HEURISTIC(explored) vs RANDOM(explored) gap +4.8 pp p=0.000   [CONTROL]
```

**The right comparator is G4, not G3.** G3 is *perfect play on both
sides*: a perfect player wins whether or not it moves first, so that arm
is near-50/50 by construction and cannot bound a first-player advantage.
G4 is a strong **hand-written, non-learned** policy playing both seats --
structurally the same kind of arm as G2 (one policy family on both seats).

Comparing **gap to gap** (not absolute win rates -- the arms have very
different overall strength):

```text
gap(model) - gap(heuristic) = +2.67 pp   p=0.402   NOT significant
gap(model) - gap(minimax)   = +9.17 pp   p=0.024   significant
```

**Verdict: the model's +7.5 pp seat gap is statistically
indistinguishable from what a hand-written heuristic shows playing both
seats.** Connect Four has a real first-player advantage, and a
non-learned policy shows it too. **This is a game property, not a model
defect.** Report win rates **per seat**; do not claim a seat defect.

### 5f.2 The label-vs-learning audit must be run per checkpoint

`tmp/audit_label_vs_learning.py` reported "LEARNING PROBLEM: supervision
named the optimal column but the model did not pick it" (72/97 cases).
That was measured on **fd_a** -- the checkpoint trained under the
per-ply supervision defect -- so it conflated the old plumbing bug with
a genuine learning ceiling. The script now takes `--ckpt`/`--out` and
flags known-defective checkpoints in its own output.

Running the identical audit on the **corrected** arm `fx_a`
(step 3500, `best_hard.pt`) on the same 560 exact-solved positions:

```text
                            fd_a (defective)   fx_a (corrected)
model pick optimal, unique opt   24.1 %               26.8 %
model pick optimal, tied opt     87.7 %               90.0 %
model plays c1 (of 84 legal)      0 / 84               0 / 84
model plays c2 (of 74 legal)      0 / 74              24 / 74
model column histogram        {3:16,4:54,5:35,6:7}  {0:2,2:24,3:37,4:27,5:10,6:12}
```

The corrected arm is **much better distributed** across columns -- it
now uses c2 and c0, which fd_a never touched. That is the signature of
the supervision fix working.

**But the headline "LEARNING PROBLEM" survives on both checkpoints**, so
it is not (only) the old bug. Two residual issues, both actionable:

1. **c1 is never played** -- 0/84 legal, on *both* checkpoints, even
   though the label names c1 in 13/112 unique-optimum positions and the
   exact optimum is c1 in 15/112. c2 was reachable (24/74) once
   supervision was fixed, so the c1 blind spot is **not** a
   supervision-plumbing artefact. It is a genuine, reproducible blind
   spot that survives the fix and needs its own diagnosis (column
   representation, training-data coverage for c1, or the legal-mask
   path).
2. **The labels themselves are wrong on 13.4 % of unique-optimum
   positions** (15/112: the label names a column that is NOT the exact
   optimum). This half is checkpoint-independent. So label repair *is*
   still needed -- but it is a ~13 % defect, not the dominant term.

**What this means for the "fixing labels alone will not close the gap"
verdict:** it is still directionally right, but the reason must be
restated. The dominant term is that on the 112 positions where the
optimum is **uniquely** determined, the model is right only ~27 % of the
time -- and the corrected arm is no better than the defective one on
that metric (26.8 % vs 24.1 %). On **tied** positions (the majority,
448/560) the model is right ~90 % of the time. So the model handles
"several moves are fine" well and "this ONE move is the only good one"
badly. That is a sharp, specific, testable statement about what is
still broken.

### 5f.3 The c1 blind spot: a RANKING problem, not a data or path problem

The residual "model never plays c1" finding was chased to its root cause
in three steps, each eliminating a hypothesis.

**Step 1 -- is it a data-coverage problem? NO.**
`tmp/audit_c1_blindspot.py` over the full 320,000-row corpus:

```text
c1 is PLAYED in     720,827 / 5,100,175 moves = 14.13 %   (c3 16.66 %, c6 12.41 %)
c1 is NAMED in      615,118 / 4,563,661 labelled slots = 13.48 %
c1 is the single-column teacher answer in 600,613 slots = 13.16 %
```

c1 is a **completely normal** column in the training data. The model had
abundant, well-distributed c1 supervision. Data coverage is refuted.

**Step 2 -- is it a path bug (hard export / legal mask / column slice)? NO.**
`tmp/audit_c1_path_probe.py` on 200 exact-solved holdout positions:

```text
c1 wins argmax UNMASKED : 7/200 = 3.5 %
c1 wins argmax MASKED   : 5/200 = 2.5 %
c1 logit identical to another legal column : never
mean rank of c1 among legal columns : 4.36 / 7
mean logit spread across legal columns : 1.693
```

**Step 3 -- is the network even reading c1? YES.**

```text
perturbing c1's board state moves c1's logit in 39/39 positions
mean |delta| = 0.578   max |delta| = 2.192
```

The trunk reads column 1, the hard export carries it, the legal mask
does not clamp it, and it is not aliased with another column.

**Conclusion.** c1's logit is real, responsive and distinct -- it simply
**rarely wins the argmax**. The mean rank of c1 among legal columns is
4.36 of 7, i.e. c1 is treated as a *mediocre* column. This is a
**ranking / learning** problem, not a plumbing one and not a data one.

**Why that matters for the release decision.** c1 is a *side* column.
Connect Four's left edge is rarely the correct move, so a model that
under-ranks it is losing little in practice. But it is also the
clearest single measurable defect left in the model, and it is the
right thing to fix next: it is a ranking-calibration problem, not a
rearchitecture problem.

**Consequence for the "learning problem" verdict.** On UNIQUE-optimum
positions the model is right only ~27 %, and a large part of the misses
are side-column cases where it plays a *plausible centre* move instead
of the unique correct side move. On TIED-optimum positions it is ~90 %
right. So the remaining gap is concentrated in **positions with a single
correct answer**, and the model's failure mode is *centre bias under
constraint*.
### 5g. THE CORRECTED MODEL IS MATERIALLY STRONGER -- first hard evidence

This is the first *in-game* measurement that separates the corrected arm
from the defective one, under one identical harness, one identical seed,
one identical opponent and one identical move protocol.

Protocol: `src.c4.duel.duel`, model greedy (eps=0, hard path), opponent
`random_policy`, global RNG seeded with 99 (`duel` ignores its own `seed`
argument), unique trajectories verified.

```text
                          vs random          as P1       as P2
fd_a (defective)       W399  L201  66.5 %   197/300    202/300
fx_a (corrected) step5000   W479 L121  79.8 %   234/300    245/300
fx_a (corrected) step5500   W469 L129  78.4 %   231/300    238/300
                         ------------------------------------------
fx_a COMBINED (1200 games)  W948 L250  79.1 %   95% CI 78.9-79.4
improvement over fd_a       +12.6 pp        Fisher exact p = 9.9e-09
```

The 1.4 pp spread between the two fx_a checkpoints is inside the
binomial standard error at n=600 (~1.7 pp), so it is measurement noise,
not a regression from training further.

**Column behaviour changed qualitatively, not just quantitatively:**

```text
                top column    c0+c1+c2    c2       c6      c3+c4
fd_a            c4 37.21 %     29.63 %   4.25 %  1.73 %  58.8 %
fx_a (step5000) c3 40.16 %     34.75 %  12.22 %  2.96 %  53.0 %
fx_a (step5500) c3 35.67 %     34.13 %  14.13 %  3.06 %  52.1 %
```

c2 usage **more than tripled** (4.25 % -> 14.13 %). This is the
supervision fix showing up in behaviour, not just in a metric: the
defective arm could not reach c2 at all on the exact-solved holdout
(0/74 legal), the corrected one does.

**Against true constant-column opponents** (600 games each, same harness
and seed; the reference arms are identical in both runs because the seed
is fixed):

```text
always-c0  75.0 %   fx_a 78.4 %  p=0.172   fx_a better (ns)
always-c1  75.1 %   fx_a 78.4 %  p=0.193   fx_a better (ns)
always-c2  80.5 %   fx_a 78.4 %  p=0.392   fx_a worse  (ns)
always-c3  82.2 %   fx_a 78.4 %  p=0.110   fx_a worse  (ns)
always-c4  79.2 %   fx_a 78.4 %  p=0.778   fx_a worse  (ns)
always-c5  77.3 %   fx_a 78.4 %  p=0.676   fx_a better (ns)
always-c6  74.3 %   fx_a 78.4 %  p=0.103   fx_a better (ns)
```

fx_a is **statistically indistinguishable from every constant-column
opponent**. That is the honest headline: a fixed-column bot is a
surprisingly strong Connect Four baseline, and the corrected model has
caught up with it -- but has not yet clearly beaten it. Always-c3
(centre) remains marginally ahead. Note the step-5000 checkpoint was
*significantly* better than always-c6 (p=0.028) while the step-5500 one
is not (p=0.103); that is exactly the kind of single-comparison
significance that should not be quoted without a correction, and it is
the reason the headline says "indistinguishable".

**Caveats that keep this from being a release claim.**

* fx_a was still **training** (5500/6000) at the time of writing. These
  are mid-training checkpoints, not the final model.
* 600 games against a single opponent type (random) is not a strength
  matrix. The heuristic / lookahead / minimax arms are not yet re-measured
  for fx_a.
* The c1 ranking defect is **not** fixed: fx_a's mean rank for c1 among
  legal columns was still 4.36/7.
* The defective fd_a numbers here supersede the older 200-game table
  (62.5 %) and the older 600-game table whose constant references used a
  defective staircase `const_policy` -- see 5g.1.

### 5h. CORRECTED FULL RETRAIN -- COMPLETE

All three corrected arms finished at step 6000 with seed 32, transcript v2,
no traceback, and the promoted checkpoint equal to the final checkpoint
(the promotion gate now selects the maximum-LABEL step, and LABEL rises
monotonically to the last step).

```text
arm    corpus   final LABEL   step0 LABEL   promoted step   transcript
fx_a   320 000     50.92 %      16.27 %         6000          v2
fx_b   640 000     50.33 %      11.53 %         6000          v2
fx_c   640 000     51.02 %      11.53 %         6000          v2
```

LABEL trajectory for fx_a (shipped hard path, direct sidecar-label
agreement, n=4000 per point):

```text
step0   16.27 %      step3000  43.55 %
step500  37.55 %      step3500  44.70 %
step1000 37.65 %      step4000  45.33 %
step1500 38.48 %      step4500  47.25 %
step2000 40.23 %      step5000  49.15 %
step2500 39.77 %      step5500  49.50 %
                           step6000  50.92 %
```

Note the shape: a fast rise to ~38 % by step 1500, then a slow, steady
climb. **The model was still improving when training stopped at 6000
steps.** The corrected arm is not saturated, so a longer run is a
reasonable next step.

The >90 % gate from the 500-row acceptance run does **not** transfer here:
that run trained on 500 rows to convergence, this run trains on 320-640 k
rows with a fixed step budget. 50 % on the full corpus is the honest
number for this budget.

**Two script defects found and fixed while running this battery.**

1. `tmp/post_train_eval.sh` looked for `best_hard.pt` at the run-dir root;
   the trainer writes it to `checkpoints/best_hard.pt`. The battery
   therefore aborted with "ARM NOT USABLE" on three arms that had actually
   completed. Fixed; the battery now reports all three USABLE.
2. `tmp/audit_ingame_strength_fd_a.py` carried its own copy of the naive
   binomial p-value, which raises
   `OverflowError: int too large to convert to float` once
   `n >= 1200` (the strength harness uses 1800 games per arm). Replaced with
   the log-space `math.lgamma` version already validated in
   `tmp/audit_seat_gap.py`, and cross-checked against the naive
   implementation for n <= 1000 (agreement to ~1e-13).

### 5i. FINAL STRENGTH MATRIX -- fx_a (n=1800 per opponent)

Protocol: 600 games x 3 seeds (99, 1234, 4242) per arm; model greedy
(eps=0, shipped hard path); global RNG seeded because `src.c4.duel.duel`
ignores its `seed` argument; all 7 true-constant arms use random fallback
once the preferred column fills.

| opponent | MODEL win% | best constant | best const win% | Fisher p | outcome |
|---|---|---|---|---|---|
| random (eps 0) | **81.1 %** | centre_policy | 91.3 % | 2.6e-15 | **model loses** |
| heuristic (eps .3) | 27.8 % | always-c1 | 25.6 % | 0.132 | no difference |
| lookahead (eps .3) | 11.3 % | centre_policy | 16.7 % | 3.8e-06 | **model loses** |
| minimax (eps .3) | **39.7 %** | centre_policy | 37.0 % | 0.064 | no difference (marginally) |
| centre (eps .3) | **81.8 %** | always-c1 | 77.2 % | 7.0e-04 | **model wins** |

Full per-arm Fisher matrix (model vs each constant arm) is in
`runs/_probe/ingame_strength_fx_a_final.json`.

**What this says.**

1. Against *random*, the corrected model wins 81.1 % -- far above every
   single-column arm (74.6-82.9 %) -- but loses decisively to
   `centre_policy` (91.3 %). `centre_policy` is NOT a constant column: it
   is a **centre-first priority sweep** (play c3, then c4, then c2, ...),
   and it is the strongest single opponent measured.
2. Against *heuristic*, the model is statistically indistinguishable from
   the best constant (27.8 % vs 25.6 %, p=0.13) and significantly beats
   always-c2, always-c3, always-c6 and centre_policy.
3. Against *lookahead*, the model is weak (11.3 %) and loses to
   always-c0, always-c2 and centre_policy.
4. Against *minimax* the model is the **strongest arm measured**
   (39.7 %) and beats every single-column arm by a wide margin
   (p=0.0000 vs each). It is marginally ahead of centre_policy
   (p=0.064, not significant).
5. Against *centre* the model wins 81.8 % and beats every arm including
   centre_policy (p=0.0000).

**Conclusion: the corrected model is not a centre-collapse artifact and
is not a constant-column artifact.** Its column usage is genuinely
distributed (against minimax: c0 13.5 %, c1 13.3 %, c2 14.8 %, c3 30.4 %,
c4 14.8 %, c5 11.7 %, c6 1.4 %) and it beats single-column policies
head-to-head across most opponents. Its weakness is **depth**: it loses
badly to lookahead and to the centre-first priority sweep.

This is a *different* failure mode from the historical fd_a arms, and it
is consistent with the still-rising LABEL curve: the model has learned
tactics from dense per-ply supervision but has not yet learned deep
search-like behaviour.

**Verdict wording defect found and fixed.** The harness printed
`COMPARABLE -- MODEL LOSES to best constant (p=0.0000)`, which conflates
two independent questions (is the comparison valid? is the difference
significant?) and reads as if "COMPARABLE" qualified the strength claim.
Now prints `COMPARISON VALID -- significant difference -- MODEL LOSES ...`.

### 5j. CORRECTED ARM -- THE LEARNING GAP SURVIVES THE FIX

`tmp/audit_label_vs_learning.py` rerun against the **corrected**
`fx_a_s6000/checkpoints/best_hard.pt` (560 exact-truth positions, joined
560/560):

```text
UNIQUE-optimum positions (n=112)
  label EXACTLY equals the unique optimum : 97/112  = 86.6 %
  model pick is optimal                    : 30/112  = 26.8 %
  label OK   & model WRONG                 : 70/97   <-- learning gap
  label WRONG & model WRONG                : 12/15

TIED-optimum positions (n=448)
  label INTERSECTS optimum                 : 433/448 = 96.7 %
  model pick is optimal                    : 403/448 = 90.0 %
```

This is the same shape as the defective fd_a audit (72/97), so **fixing
the supervision bug did not close the learning gap on unique-optimum
positions.** Labels name the optimal column 86.6 % of the time and the
model still misses it 73 % of the time. The bottleneck is now squarely
the model's ranking, not the labels.

```text
positions whose UNIQUE optimum is c1 or c2 : 23
  label names c1/c2                        : 18
  model plays c1/c2                        :  8
  model in optimum                         :  5

model column histogram on the 112 unique-optimum positions:
  {0: 2, 1: 0, 2: 24, 3: 37, 4: 27, 5: 10, 6: 12}
```

**c1 is still never selected** (0 of 112) while c2 became reachable
(24 of 112) -- so the supervision fix restored c2 but not c1. Combined
with the c1 path probe (responsive logits, mean legal rank 4.36/7) and
the c1 data audit (c1 is 14.13 % of played moves and 13.48 % of label
mass), c1 is a **pure ranking/learning defect** that survives both the
supervision fix and the data audit.

Contrast: on **tied**-optimum positions the model is at 90.0 % optimal
and the label intersects the optimum 96.7 % of the time. The model is
good when several moves are acceptable and poor when exactly one is.

### 5k. ARTIFACT PROVENANCE NOTE -- seat-gap runs

Two different `tmp/audit_seat_gap.py` runs exist and write DIFFERENT
files. The older run (G3 = minimax greedy-vs-explored, verdict
INCONCLUSIVE, no G4) wrote `runs/_probe/seat_gap.log`. The corrected run
(G3 = minimax explored-vs-explored, plus G4 heuristic-vs-random, plus
the gap-versus-gap verdict) wrote `runs/_probe/seat_gap_v2.log` and
`runs/_probe/seat_gap_discriminator.json`.

**The authoritative seat-gap result is the corrected one** (`seat_gap_v2.log`
/ `seat_gap_discriminator.json`, both mtime 16:15:13, verified to contain
G4 and `p_gap`). The older `seat_gap.log` (mtime 16:00:50) is superseded
and its "INCONCLUSIVE" verdict must not be quoted.

### 5l. fx_a STRENGTH MATRIX INDEPENDENTLY REPRODUCED

A clean serial re-run (`runs/_probe/ingame_strength_fx_a_repro.json`) of
the entire 5-opponent x 9-arm matrix reproduces the original
`ingame_strength_fx_a_final.json` **exactly**:

```text
opponent   original              repro                 result
random     W1460 L337 D3         W1460 L337 D3         IDENTICAL
heuristic  W 500 L1297 D3        W 500 L1297 D3        IDENTICAL
lookahead  W 203 L1591 D6        W 203 L1591 D6        IDENTICAL
minimax    W 715 L1067 D18       W 715 L1067 D18       IDENTICAL
centre     W1473 L327 D0         W1473 L327 D0         IDENTICAL

all arms (45 model+constant cells) identical: True
```

The reproducibility check required by the project rules -- the artifact
independently reproduces the headline statistic -- now holds for fx_a.
The earlier fx_b/fx_c JSON artifacts are also complete and internally
consistent (all 5 opponents, n=1800, min_unique >= 355), but only fx_a
has a byte-exact reproduction.

### 5m. TRANSIENT TEST FAILURES -- ROOT CAUSE (not a code regression)

Three separate transient failures in this phase were all traced to ONE
cause: **two concurrent `tmp/post_train_eval.sh` chains**, one stale from
17:17 and one started at 19:57, both writing `runs/_probe/post_train_eval.log`
and both running the generic `tmp/measure_label_agreement.py` full-corpus
scan (841 MB resident, scanning 320,000 rows). They competed with the
serial regression sweep for CPU and shared the one live UI server.

```text
symptom                         standalone   under contention
test_symmetry_penalty.py        PASS (10 assertions)  exit 139
test_live_parity_v4.py          PASS (49/49, L1 0.0)   ConnectionResetError
test_live_parity_v5.py          PASS 3/3 (75/75, L1 0.0) ConnectionResetError
```

The server-side log shows the matching cause: the handler thread raises
`ConnectionResetError [WinError 10054]` on a stale keep-alive socket
while the model forward pass occupies the shared `Game.lock`. The server
is `ThreadingHTTPServer` and the game lock is correctly used, so the
reset is a client-visible symptom of contention, not a lost response.

After killing both stray chains (PIDs 5604, 50492, 53844, 58564, 16160),
the full serial sweep returned **17 PASS, 0 FAIL** with v3 40/40,
v4 49/49, v5 75/75, all L1 0.0.

**Correction to an earlier reading:** an intermediate check reported
"26.9 MB free" of memory. That was a units error -- PowerShell's
`FreePhysicalMemory` is in KB, so the true figure is **25.5 GB free of
61.6 GB total**. Memory pressure was NOT the cause; the cause was
duplicate concurrent jobs sharing one UI server and one log path.
The generic full-corpus `measure_label_agreement.py` remains superseded
by the shipped `tmp/measure_promotion_scorer.py` and should not be run
concurrently with the sweep.

### 5n. fx_a FULL STRENGTH MATRIX (authoritative, from the clean serial repro)

Extracted from `runs/_probe/ingame_strength_fx_a_repro.log`
(mtime 19:44:31, postdates the harness script at 18:46:52). 600 games x
3 seeds (99/1234/4242) per arm = 1800 games per cell. Model greedy
(eps=0), shipped hard path, global RNG seeded.

```text
opponent     MODEL    best single-column   centre_policy   verdict
random       81.1 %   always-c3  82.9 %    91.3 %          LOSES to both
heuristic    27.8 %   always-c1  25.6 %    20.6 %          BEATS both
lookahead    11.3 %   always-c2  14.7 %    16.7 %          LOSES to both
minimax      39.7 %   always-c3  26.4 %    37.0 %          BEATS both (p<1e-10 each)
centre       81.8 %   always-c1  77.2 %    57.8 %          BEATS both (p=0.0007)
```

The model beats every single-column constant against minimax and centre,
loses to `centre_policy` against random and lookahead, and is clearly
best against the hand-written heuristic. `centre_policy` is NOT a
constant column -- it is a centre-first priority sweep and is the
strongest opponent measured (91.3 % against random, which no constant
column reaches).

**A caveat the earlier summary got wrong:** the old narrative said fx_a
"beats 5/7 true constant references". That count was against the
*random* opponent only. Across the full matrix the model beats a
constant column in 2 of 5 opponent settings outright and ties/loses in
the other 3. The single-column framing is also misleading because
`centre_policy` -- a legal, non-degenerate opponent -- is stronger than
the model in 2 of 5 settings.

### 5o. SILENT-SKIP DEFECT FIXED IN measure_promotion_scorer.py

`tmp/measure_promotion_scorer.py` printed `SKIP: missing corpus.txt or
perply_targets.npz` and **exited 0** when handed a checkpoint path
instead of a run directory. A caller reading the exit code would record
"measured, found nothing" for a run that never happened -- a vacuous
result. Patched to print an explicit ERROR and exit 1. Verified: the
bad path now exits 1, the correct run-dir usage still reports
`hard_model_final.pt LABEL 50.92%`.

### 5p. STALE-SCRIPT FAILURES IN THIS BATCH

`proc_fcb546d0f1ab` (`audit_c1_blindspot.py`) failed with
`FileNotFoundError: .../perply_v23_full/corpus.txt`. The script on disk
points at `corpus_320k_fixed.txt` (mtime 16:22:24) and its artifact
`c1_blindspot.json` was written at 16:22:53. The failing run used a
STALE copy of the script that still referenced `corpus.txt`. Re-running
the current script reproduces the identical numbers
(c1 = 14.13 % of played moves, 13.48 % of label mass, 13.16 % of
single-column teacher answers), so the earlier c1 conclusion stands.

`proc_a103ac560e66` (`measure_label_agreement.py`) exited 127 with only
`ui_server: checkpoint transcript=v2` -- killed/OOM under memory
pressure. Superseded by the shipped `measure_promotion_scorer.py`
(fx_a 50.92 %, fx_b 50.33 %, fx_c 51.02 %).

### 5q. ROOT CAUSE OF THE REMAINING GAP: THE FULL ARMS ARE MASSIVELY UNDER-TRAINED

This is the single most important finding of the session and it was
missed by every earlier read.

**The corrected full arms saw the 320,000-row corpus only 0.30 times.**

```text
acceptance run  overfit_fixed : 3923 steps x batch 16 =  62,768 samples
                                 over 500 rows        = 125.5 EPOCHS
                                 -> LABEL 91.29 % (scorer 92.38 %)

corrected full fx_a/fx_b/fx_c: 6000 steps x batch 16 =  96,000 samples
                                 over 320,000 rows    =   0.30 EPOCHS
                                 -> LABEL 50.92 / 50.33 / 51.02 %
```

A single epoch is 320,000/16 = 20,000 steps. **The full arms were cut off
at 30 % of ONE epoch.** The acceptance run, by contrast, ran 125 epochs
over its small corpus -- which is why it reached 91 %.

**The learning curve was still climbing steeply when the run ended:**

```text
step  500  LABEL 37.55 %
step 2000  LABEL 40.23 %
step 3000  LABEL 43.55 %
step 4000  LABEL 45.33 %
step 4500  LABEL 47.25 %
step 5000  LABEL 49.15 %
step 5500  LABEL 49.50 %
step 6000  LABEL 50.92 %   <-- still +1.42 pp / 500 steps at the cut-off
```

There is no plateau. The run was terminated by the step budget, not by
convergence. The trainer's own default is `--steps 20000`, i.e. exactly
one epoch; the launcher overrode it with 6000.

**This reframes every "remaining weakness" claim.** The unique-optimum
ranking gap (30/112 optimal while the label names the optimum 97/112
times), the c1 blind spot (0/84 selected when legal), the weak lookahead
performance (11.3 %) and the loss to `centre_policy` are all measured on
a model that has seen **less than a third of one pass** over its
training data. They are NOT evidence about the architecture, the GF
trunk, the label quality, or the representation. They are evidence that
the model was not trained long enough to have learned the task.

**Cost of training to convergence.** fx_a took 7,912 s for 6,000 steps
(1.32 s/step, including a hard export + 4,000-position label scoring
every 500 steps):

```text
 20,000 steps (1 epoch)  ->   7.3 h
100,000 steps (5 epochs) ->  36.6 h
200,000 steps (10 epochs)->  73.3 h
```

The RTX 5090 is idle at 3 % utilisation during the evaluation-only
battery; training itself is GPU-bound but the per-step cost is dominated
by the hard export + scoring cadence, which can be made less frequent.

**Decision: V24 = a properly-budgeted full run.** V24 is blocked until
the launcher's step budget, eval cadence, and resume path are corrected
-- but the blocker is now understood, and it is a budget problem, not an
architecture problem. See the action list at the top of this file.

### 5q.1. THE `--resume` PATH IS FIXED (V24 PREREQUISITE)

`--resume` into a target step at or below the checkpoint's own step used
to leave the training loop body unentered:

```python
for step in range(start_step, args.steps + 1)   # empty when start_step > args.steps
```

so the first reference to `step` raised

```text
UnboundLocalError: local variable 'step' referenced before assignment
```

That made it impossible to re-open a finished run for a short extension
-- exactly the operation a long V24 run needs, and the operation a
crashed-and-restarted 37 h run needs as a matter of course.

Now it fails loudly and says what to do:

```text
nothing to resume: the checkpoint is already at step 6000, which is beyond
--steps=100. Pass a larger --steps (the run continues from step 6000), or
drop --resume to start fresh.
```

The resume source is `final_float.pt` (it carries `{"state": ...}` plus
`meta.step`, `meta.transcript` and `meta.lags`); `best_hard.pt` is an
integer export and cannot be resumed from, though the trainer still
accepts it as a fallback where the model supports `import_hard`.

**A second resume defect, found by the same test:** the model was built
from `--lags` BEFORE the checkpoint was read. A run trained with the
dense 12-lag set (0..10, 20) was therefore resumed into the V6 default
8-lag model, and died inside the forward pass with

```text
IndexError: index 9 is out of bounds for dimension 0 with size 8
```

The lag set is part of the model's STRUCTURE, not a hyperparameter the
caller may change on resume. The checkpoint is now loaded first, the lag
set is taken from it, and an explicit `--lags` that disagrees is refused:

```text
refusing to resume: the checkpoint was trained with lags
[0,1,2,3,4,5,6,7,8,9,10,20] but --lags says [0,1,2,4,8,16,32,64].
The lag set is part of the model structure, so it cannot be changed on
resume. Drop --lags to take the checkpoint's own set.
```

This is why V24 must NOT pass `--lags` when resuming: the launcher must
let the checkpoint decide. (`run_v24.sh` currently passes `--lags` only
on a fresh run; the resume path relies on the checkpoint's own set.)

### 5r. BATTERY HARDENING (one-writer guard, unique outputs, promoted-artifact coverage)

Three real defects in the evaluation battery, all found by running it
end-to-end rather than by reading it:

1. **The promoted artifact was never measured.** `best_hard.pt` lives at
   `<run>/checkpoints/best_hard.pt`, but the scorer looked for it in the
   run ROOT, and the `step*_hard.pt` glob does not match it
   (`fnmatch("best_hard.pt","step*_hard.pt")` is False). So the
   *promoted* model -- the one that would actually ship -- was silently
   skipped. Fixed: the scorer now enumerates
   `hard_model_final.pt`, `checkpoints/best_hard.pt`, `best_hard.pt`, then
   the step glob. Verified: fx_a now reports
   `best_hard.pt LABEL 50.92%` where it previously reported nothing.

2. **Two battery chains could write the same paths.** This caused the
   earlier NUL-padded logs and the silently overwritten JSON. Fixed with
   a `mkdir`-based one-writer lock in `tmp/post_train_eval.sh`
   (`.post_train_eval.lock`) and a second one in `tmp/wait_and_eval.sh`
   (`.wait_and_eval.lock`). Both verified: a second chain now exits 1
   with `ABORT: another ... chain holds ...` and the lock is released on
   normal exit.

3. **The generic scanner overwrote one shared JSON.**
   `tmp/measure_label_agreement.py` always wrote
   `runs/_probe/label_agreement.json`, so the last of six invocations
   erased the other five with no error. Fixed: the output path is now
   derived from the checkpoint names
   (`label_agreement_<stem>.json`).

The battery order was also corrected: the cheap, bounded shipped scorer
now runs FIRST and its exit code is checked, and the memory-heavy generic
scanner runs last as an independent cross-check.

### 5g.1 A measurement defect that would have produced a false result

The first version of `tmp/audit_ingame_cols_fd_a.py` defined the reference
opponent as

```python
return col if col in legal else legal[0]
```

That is **not** a constant-column policy: once the preferred column fills it
falls back to the leftmost legal column, i.e. a priority sweep. The
symptom was a *byte-identical* W/L/D record for `always-c3` and the shipped
`centre_policy` across **all five** opponents -- e.g. both `W546 L54 D0`
against random, both `W235 L363 D2` against minimax. Two policies cannot
match exactly by chance; they were the same function.

Measured directly (`tmp/audit_c3_centre_equivalence.py`): the staircase
fallback fired **36 times in a single game**, choosing c6/c2/c4/c6/c5/...
where `centre_policy` would have chosen c0/c0/c0/... And head-to-head,
`const_policy(3)` vs `centre_policy` played out **W68 L132** -- they are
different policies; the stale arm simply had the same *rule* as
`centre_policy`.

The corrected reference uses a **random** legal fallback, and the script now
seeds the global RNG before every duel because `duel` ignores its own seed.

**Stale-artifact check.** The 600-game table `ingame_strength_fd_a.json`
was written at 11:21 while `tmp/audit_ingame_strength_fd_a.py` was fixed at
14:52 -- the JSON predates the fix and carries the defect. Artifact
mtimes must be checked against script mtimes before citing a measurement.
Both were re-measured: `ingame_cols_fd_a_v2.json` and
`ingame_cols_fx_a_v2.json`.


## 5e. CORRECTED (2026-10-28, late): the model does **NOT** absorb tie sets —
##     an earlier version of this section claimed the opposite and was wrong

**Retraction.** An earlier draft of this section concluded from the raw
numbers ("mass on the labelled tie set = 0.894 vs 0.1745 on singles")
that *the model does represent ties, so the label collapse is the
primary defect*. That comparison was **confounded by set size**: a
labelled set of 3.38 columns trivially collects more probability mass
than a set of 1 column. Adding a **chance-rate control** — what a
uniform-random legal policy would score on the *same* positions — gives
a different and opposite verdict.

Measured on **fd_a** (`fd_a_s6000/checkpoints/step6000_hard.pt`), 39,322
labelled training slots, shipped hard path, shipped legal mask,
canonical `ply_of` slot→ply mapping (`tmp/audit_tie_model_behaviour.py`):

```text
                              TIE slots (263)      SINGLE slots (39,059)
mean columns in the set             3.38                    1.00
model mass on the set              0.894                   0.1745
CHANCE mass on the set             0.8809                  0.1488
lift over chance                 1.015x                   1.173x
model argmax inside the set       90.49 %                 23.23 %
chance argmax inside the set      88.09 %                 14.88 %
lift over chance (argmax)        +2.41 pp                 +8.36 pp
```

**Reading.** On a tie set the model places **0.894** of its mass on the
set, but chance would place **0.881** there — a lift of **1.015×**, i.e.
**essentially chance**. Its argmax lands inside the set 90.49 % of the
time, but chance gives 88.09 % — **+2.4 pp**. So on tie positions the
model is spreading near-uniformly over the tied columns, which is what
an *uncertain* model does, not what a model that has learned "these
columns are equally optimal" does.

On single-column labels the lift is larger but still weak: **1.173×**
mass, **+8.4 pp** argmax. (This is fd_a, the checkpoint trained under
the supervision defect, whose overall label agreement is 20.2 %.)

**Corrected conclusion.** The model does **not** absorb tie sets; on
ties it is near chance, and on singles it is only weakly above chance.
That does **not** re-open the architecture question — the corrected
trainer reaches **91–92 %** label agreement on the 500-row acceptance
corpus (§5d), and on that run the model *does* fit its labels. What this
measurement establishes is narrower and still useful:

* the old fd_a/b/c checkpoints have very weak label signal (1.02–1.17×
  chance) — consistent with them being defective-supervision artifacts;
* a tie-set label is a **harder** target than a one-hot label, and the
  current training signal does not yet demonstrate that a model trained
  on genuine tie sets can learn them. That is a real open question for
  the tie-set training path, not a settled one.

**Method note (this is the reusable part).** Any metric comparing a
*k*-element set against a *1*-element set must be reported **relative to
chance on the same positions**, or it measures set size rather than
model behaviour. The raw "mass on the set" comparison gave the opposite
conclusion from the chance-controlled one.

**Two script bugs were fixed to obtain this measurement** (both were in
the audit, not in the shipped code):

1. The audit used `p = nrec - 1 - s` to map a slot to a ply. That is the
   trainer's *byte-target* slot convention, **not** the per-ply label
   convention. The sidecar ships an explicit `ply_of` array whose entry
   for slot `s` *is* the ply that slot labels. Using `ply_of` removed
   the guesswork.
2. The audit decoded the corpus with the **v1** decoder. The corpus rows
   are dense **v2** (stride `PLY_LEN`, no intra-game newlines), so v1
   misread the 9-byte v2 column headers as v1 height bytes and produced
   nonsense heights such as 81 — which is exactly the
   `ValueError: bad decoded height 81` that killed the first run.

## 5f. NEW (2026-10-28, late): the strength harness has **no** seat bias —
##     and the 20 pp "seat gap" was sampling noise

`tmp/audit_strength_harness_control.py` reported two FAILs. **Both were
bugs in the control, not in the harness** (`tmp/audit_strength_harness_control_v3.py`):

```text
C1  model vs model, BOTH sides explored   W295 L280 D25  49.2 %  p=0.5594  PASS
C2  always-c3 vs always-c3, BOTH explored  W298 L301 D1   49.7 %  p=0.9349  PASS
C3  model vs model, no exploration         unique_games = 1           PASS
```

* **C2 was mis-specified.** The old control gave side A a *pure*
  constant policy and side B the *same* policy with 30 % random
  exploration bolted on. An explored opponent is strictly weaker, so A
  winning 53.7 % was the expected outcome — it was comparing a strong
  policy with a deliberately sabotaged copy of itself. With **both**
  sides explored it is 49.7 %, p=0.9349.
* **C3 expected the wrong number.** With two *identical* deterministic
  policies the move sequence is the same in both seat orders — the seat
  swap changes who owns the pieces, not the sequence. The correct
  expectation is exactly **1** unique sequence, not 2. (2 is the A-vs-B
  case, where the policies differ.) The original expectation of 2 was
  what produced the spurious "FAIL".

**The seat-gap scare.** A new C5 check flagged the model at **59.7 % as
P1 vs 39.7 % as P2** (a 20 pp gap) at n=300. Re-measured at n=1,200
across 4 seeds, the gap collapses to **+5.2 pp**
(`tmp/audit_seat_gap.py`, arm G1: model greedy vs model explored,
W605 L552 D43, overall 50.4 %, p=0.1263). The fair model-vs-model arm
(G2, both explored) is **49.6 % overall** with a **+7.5 pp** seat gap.
The 20 pp figure was sampling noise at n=300.

**Consequence for reporting:** win rates must still be quoted
**per seat**, and the absolute numbers from `tmp/audit_ingame_strength_fd_a.py`
at `--games 20` (e.g. "MODEL LOSES to always-c0, p=0.0038") remain
**not comparable** — that harness reports "NOT COMPARABLE (too few
unique games)" for exactly this reason.

### 5r. THE V24 PRE-FLIGHT: what changed, what was measured, what is still open

**Date:** 2026-10-28 evening, before the midnight V24 launch.

#### 5r.1 The step rate was wrong by a factor of two

The earlier "1.30 s/step" figure came from the `fx_a/fx_b/fx_c` arms.
Their logs show the three arms **started 12 seconds apart** — they ran
**concurrently on one GPU**, so 1.3164 s/step was the *shared* rate.

A single V24 run gets the whole GPU. Measured on the real V24 path
(`src.c4.train`, per-ply sidecars, dense 12 lags, ctx 192, batch 16):

| quantity | value |
|---|---|
| train-only | **0.646 s/step** |
| one evaluation | 7.9 s |
| amortised eval (every 2,000 steps) | +0.004 s/step |
| **effective** | **0.650 s/step** |
| one epoch (20,000 steps) | **3.6 h** |
| 100,000 steps (5 epochs) | **18.1 h** |

#### 5r.2 Batch 32 is 1.22×, not 2×

Claude's throughput table came from `tg_scale4096.py`, a 4096-row V22
architecture with a different loop. On the real path: batch 16 →
24.77 rows/s, batch 32 → 30.10 rows/s. **Batch 16 stays** — the paper
constraint stands, and 1.22× does not justify changing the protocol.

#### 5r.3 The trainer now saves and restores the run state

`opt_state` and `rng_state` (Python `random`, NumPy `default_rng`, Torch)
are written at the **top level** of every float checkpoint and restored on
resume. `meta` is untouched, because `meta` is reused by the hard export,
which must stay a lean integer-table payload.

Measured: a 12-step uninterrupted run and a 6-then-resume-to-12 run
produce **bit-identical** weights and Adam moments
(`max |Δweight| = 0.000e+00`, `max |ΔAdam moment| = 0.000e+00`).

#### 5r.4 The holdout path had never run, and was broken twice

`run_v24.sh` never passed `--holdout-corpus` / `--holdout-targets`, so
the whole generalisation path was **never executed**. Wiring it up found:

- `Holdout.__init__` assumed the old `(N, 7)` sidecar; the production
  per-ply sidecar is `(N, K, 7)`, so `bool(opt[pick])` raised
  `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.
  Fixed: the schema is detected explicitly, and the final-ply slot is
  resolved through `ply_of`.
- `PerPlyAgreement` assumed torch tensors; the holdout caller passes
  numpy arrays, so `.to(device)` raised
  `AttributeError: 'numpy.ndarray' object has no attribute 'to'` — and
  because the failure was only *logged*, the run continued **silently
  without the generalisation metric and without the abort rules**.

#### 5r.5 The abort rules killed the run at step 0

Rule 2 ("a column optimal in the labels was NEVER picked") fired
immediately on a **randomly initialised** model. Fixed: both rules now
require `--abort-warmup` (default 20,000 = one epoch), and `--abort-on`
defaults to **`warn`** — the finding is recorded and the run continues.
Stopping an 18-hour run must be explicit.

#### 5r.6 Disk

Measured: float snapshot **203 MB**, hard export **34 MB**. For 50
snapshots: unbounded = **11.9 GB** (does not fit in 9.6 GB free);
`--keep-last 3` = **0.9 GB**; `--keep-hard-all` = **2.6 GB**. Pruning was
validated on a real run: exactly three float and three hard snapshots
survive and `best_*` is never touched.

#### 5r.7 Open items

- The 100,000-step LABEL value is **unknown**. Log-linear extrapolation
  of the fx_a curve gives **≈ 63 %**, not the 70–78 % previously quoted.
- Cosine LR and EMA are implemented but **not numerically piloted**; they
  ship **disabled**.
- No corrected per-ply-versus-single-label A/B has been run.
- The full release battery must pass on the converged model.

---

## 6. Hypotheses tested and **refuted**

| hypothesis | verdict | evidence |
|---|---|---|
| fd_a/b/c were trained with sparse lags | **refuted** | launcher *and* checkpoint both declare dense lags `[0,1,2,3,4,5,6,7,8,9,10,20]` |
| some column is architecturally blind | **refuted** | per-column hard L1 sensitivity: c1 0.126, c2 0.144 (fd_a); **no blind columns in any of the 4 audited checkpoints** |
| the raw Pons import is decoded wrongly | **refuted** | 12-convention grid, 0 sentinel misplacements, 100 % label-in-optimal-set |
| the model has a large learned output bias | **refuted** | checkpoint `table_bias = 0.0`; learned column-logit bias ≈ 0.001–0.006 |
| the model is blind to *who* is to move | fixed earlier | transcript version v2; replay reproduces header boards 1400/1400 |
| hard-export quantisation is why taking is bad | **refuted** | float 26.4 % vs hard 25.9 % on 2,445 win-available positions |
| mirror augmentation improves solver-optimality | **refuted** | tied at 71.8 % (n=560) |
| the model collapses tie sets on its own | **refuted** (then re-answered) | raw comparison said the model "does" represent ties, but that was confounded by set size. Chance-controlled: on TIE sets the lift is **1.015×** mass / **+2.4 pp** argmax (near chance), on SINGLE sets **1.173×** / **+8.4 pp**. The old fd_a/b/c checkpoints carry very weak label signal (§5e) |
| the strength harness favours a seat | **refuted** | model-vs-model with both sides explored = 49.2 % (p=0.5594); always-c3-vs-always-c3 with both sides explored = 49.7 % (p=0.9349) (§5f) |
| the model has a 20 pp P1/P2 seat gap | **refuted** | +20 pp at n=300 was sampling noise; at n=1,200 the gap is **+7.5 pp** (p=0.009) but is **not significantly different** from a hand-written heuristic's **+4.8 pp** playing both seats (gap difference +2.67 pp, p=0.402) -- a **game property**, not a model defect (§5f.1) |
| the model is seat-asymmetric and needs a seat/mirror repair | **refuted** | same: the first-player advantage is Connect Four's, and a non-learned policy shows it too (§5f.1) |
| supervision named the optimal column but the model did not pick it (72/97) | **confirmed, but the old checkpoint was the wrong evidence** | the verdict was measured on fd_a, trained under the supervision defect. Re-run on the corrected arm: **70/97** -- it survives the fix. Decomposed in §5f.2: the model is ~90 % right on TIED optima (448/560) but only ~27 % right on UNIQUE optima (112/560). c1 is never played (0/84 legal) on **both** checkpoints -- a genuine remaining blind spot |

---

## 7. Why there is still no "release" decision

Every release metric needs a **same-position baseline**, and on every
authoritative baseline the trained arms are at or below a constant policy:

```text
hard-vs-label (aligned fit)        ~25.0 %    always-c3   26.3 %
hard-vs-actual-played-move         ~31.2 %
```

A model that cannot beat "always c4" is not ready to ship, no matter how good
the infrastructure is.

---

## 8. What is next, in order

**§5q reorders everything below.** The supervision bug (§5c) is fixed and
the acceptance gate passed at **91.29 %**. The full retrain then ran --
but only **6,000 steps = 0.30 epochs** over the 320,000-row corpus, and
the LABEL curve was still climbing when it stopped. So the next action is
no longer "fix the supervision" (done) or "attack the architecture"
(unsupported). It is **train the same, proven-correct architecture to
convergence and re-measure everything.**

> **✅ Done and verified:** the per-ply sidecar row-index fix (§5c), the
> acceptance gate (91.29 %, corroborated 92.38 % / 92.40 %), the
> transcript stamping of final exports, the direct-label promotion gate
> (§5d), the V6 zeroed-hard-export trap, the silent-skip scorer defect,
> the promoted-artifact measurement gap, the one-writer battery lock, and
> the full-strength matrix with a bit-exact fx_a reproduction (§5l, §5n).

1. **V24 = one properly-budgeted full run.** Same corrected trainer,
   same seed 32, same data. `--steps 100000` (5 epochs, **~18 h** at the
   measured 0.650 s/step) or `200000` (10 epochs, ~36 h) instead of
   6,000. **The prerequisites are now DONE (§5r):**
   * ✅ the step rate was re-measured on the real path (**0.650 s/step**,
     not 1.32 — the earlier figure came from three arms sharing one GPU);
   * ✅ the `--resume` path is fixed **and proven lossless**: an
     uninterrupted run and an interrupted-then-resumed run produce
     bit-identical weights and Adam moments;
   * ✅ optimizer + RNG state are saved and restored at the top level of
     every float checkpoint;
   * ✅ the disjoint holdout is wired in (`--holdout-corpus` /
     `--holdout-targets`) and its two latent crashes are fixed;
   * ✅ checkpoint pruning is validated on a real run (0.9 GB instead of
     11.9 GB);
   * ✅ the abort rules are conservative (warm-up gated, `warn` by
     default);
   * ✅ `--eval-games 24` is enabled in the launcher, so the ladder
     reports a real game count instead of a vacuous W0L0D0;
   * keep `--max-legacy-frac 0`, `--table-bias 0.0`, `--hard-start 0`,
     ctx 192, batch 16, and the direct-label promotion gate exactly as
     they are now.
2. **Re-measure the release battery on the V24 checkpoint, not on fx_a.**
   Solver-optimality on the authoritative 560-position set, the in-game
   5-opponent matrix, the unique-optimum subset, the c1–c4 audit, the
   seat audit, and the tie audit. **Every "the model is weak" number
   currently in this file was measured on a model that saw 0.30 epochs**
   and must be re-measured before it is cited as a property of the
   architecture.
3. **Then, and only then, decide whether the architecture needs work.**
   If a converged model still shows the unique-optimum ranking gap
   (model 30/112 while the label names the optimum 97/112) and the c1
   blind spot (0/84 legal), *those* are real architecture/objective
   findings and the graded-target arm (§4, action 3 below) becomes the
   next lever. If they close, no architecture change is needed.
4. **Build a Pons-graded target arm** (deferred, not blocking): the
   correctly-decoded raw Pons score vectors are available today, are
   optimal 100 % of the time, and separate "win in 1" from "win in 5" --
   which is exactly the depth signal the model is missing against
   lookahead.
5. **Build a valid tie-set A/B** (deferred): the old one is invalid
   because `old_targets.npz` already contains production tie sets and the
   strong teacher collapses 100 % of its ties to one column. A valid
   control needs a genuinely collapsed one-hot sidecar.

**Explicitly NOT recommended yet:** a coverage/entropy penalty, and any
architecture change. Both would be treating a symptom of a 0.30-epoch
run.


## 5s. Operational readiness: the midnight path, audited end to end

The launch path was broken in four separate places. Each was found by
running the real code, not by reading it.

### 5s.1 `play_v24.sh` wrote invalid JSON

`write_status()` interpolated the training step bare:

```text
"step": $last_step,
```

`last_step` is EMPTY until the server reports a step, so the file
contained `"step": ,` -- **not valid JSON**. Every consumer of the status
file failed to parse it, including the watcher's own regression test.
The step is now `null` until known, and the free-text `detail` field is
escaped (`\` and `"`) before it is interpolated into the JSON string.

### 5s.2 `curl -o /dev/null` returns exit code 23 on this host

Three readiness checks used `curl -s -o /dev/null ...`. On this Windows
build that returns **rc=23 on a successful HTTP 200**:

```text
curl -s -o /dev/null .../state   -> rc=23   (HTTP 200, 1282-byte body)
curl -s -o -       .../state     -> rc=0
curl -s -o <file>  .../state     -> rc=0
```

The watcher would have waited the full 90 s, declared a healthy server
dead, and exited 4. It now checks the **HTTP status code** via
`-w '%{http_code}'` and discards the body to a temp file. Verified:
`/state` -> TRUE, a 404 -> FALSE, a dead port -> FALSE.

`run_v24.sh` itself contains no `curl` and was never affected.

### 5s.3 `/modelinfo` did not report the step

The watcher reads the served step from `/modelinfo`, but that endpoint
returned only the path, name, device, mtime, size and hash. The step
came back empty, which is what made `"step": ,` reachable in the first
place. `/modelinfo` now carries `"step"`, read from the checkpoint
payload (the promoted artifact is `best_hard.pt` and carries no step in
its filename).

### 5s.4 The atomic checkpoint writer gave up too early

The rename retry budget was 2 s. Under a concurrent reader on Windows
that is exhausted, and a lost checkpoint is 20 minutes of training.
The budget is now a **30 s deadline**, the jitter moved off the global
`random` module onto a dedicated instance (so it cannot perturb the
seeded training stream), and a `FileNotFoundError` no longer loses the
checkpoint.

Characterised separately, and this distinction matters:

```text
successful payload reads: 769
corrupt reads:           0
transient lock errors:  50
```

Rename-based publication prevents **truncation**; the residual errors
are Windows sharing locks, which are a *reader* problem. The UI now
retries the load (`_torch_load_with_retry`, 40 x 0.25 s). The earlier
test asserted the wrong property -- it counted transient locks as
failures -- so it now asserts the real invariant: never a corrupt file.

### 5s.5 What the failures actually were

Three of the four "failures" in the final sweep were **not product
defects**, and saying so matters because two of them had been reported
as failures before:

```text
test_transcript_stamp   stale test regex -- it matched only a bare
                         torch.save(...), but the production path uses
                         atomic_torch_save(...). The production code
                         was verified correct by reading it: the
                         final_float.pt block carries
                         "transcript": version.
test_perply_row_index   a syntax error I introduced mid-edit.
test_live_parity_v4/v5  concurrent sweeps. All three live-parity tests
                         drive ONE shared ui_server and reset the board
                         under each other, so a second sweep corrupts
                         the first one's live state -- the test stdout
                         said "PARITY_OK = True" while the harness
                         recorded FAIL. Both tests pass when run
                         directly (exit 0). The one-writer lock was
                         already in place; the per-test evidence files
                         were NOT unique, so a stale file could be
                         mistaken for this run's result. They now carry
                         a per-sweep PID suffix.
test_play_v24_watcher   the real defect above, plus a test bug: the
                         lock test invoked a bare "bash", which resolves
                         to WSL from Python on this host. It now uses
                         the same Git Bash constant as the rest of the
                         file.
```

### 5s.6 Verified state after these fixes

```text
regression sweep:          22 PASS, 1 FAIL -> 23 PASS, 0 FAIL
test_play_v24_watcher:     12/12 PASS (status file, hot reload,
                            step reporting, one-writer lock,
                            source immutability)
V24 launcher smoke:        exit 0, HOLDOUT 11.8% -> 27.0%,
                            H-LABEL 11.9% -> 28.9%, snapshots written
in-game strength repro:    bit-identical to the original run
                            (every W/L/D and every p-value matches)
```

The in-game strength reproduction deserves a note: `src.c4.duel.duel`
ignores its `seed` argument, so the harness seeds the global RNG itself
before each duel. The independent re-run reproduced all five opponent
arms exactly, which is the strongest available evidence that the fx_a
numbers are reproducible rather than a lucky draw.

## 5t. The live-parity flakiness: three separate causes, all now fixed

For a long stretch the live-parity tests (`test_live_parity_v3/v4/v5.py`)
failed *rotating* — different tests failed on different sweeps, sometimes
all four at once, sometimes 17/18 green, sometimes 22/24. A rotating
failure is not a test bug; it is contention. Three distinct causes were
isolated, each with its own evidence.

### 5t.1 A stale server could shadow the current code (Windows double-bind)

`ThreadingHTTPServer.allow_reuse_address = True` sets `SO_REUSEADDR`. On
Windows that does **not** mean "reuse a TIME_WAIT address" the way it does
on Linux — it lets a **second live socket bind an address that another live
socket already holds**. A `ui_server` from an earlier session therefore
kept running while a freshly started one printed
`serving http://127.0.0.1:7860/`. Requests went to the OLD process, so
`/modelinfo` lacked the `step` field the current code returns, and the
parity tests failed at random.

**Fix:** `allow_reuse_address` is deliberately left at its default `False`,
and the port is claimed *before* the 34 MB model is loaded, so a lost
race costs a second instead of a minute. A second server on the same port
now exits **3** with `WinError 10048` instead of pretending to serve.

**Regression:** `tmp/test_ui_single_server.py` — **8/8 PASS**. It asserts
the second server refuses, that the first still answers with the same
checkpoint and step, that the port is released on exit, and that a fresh
server can then bind it.

**Sweep ownership:** `tmp/run_regression_sweep.sh` now starts and stops
**its own** server from the current source, and *refuses to run* if
something already holds the port, rather than measuring against a server
it does not control.

### 5t.2 Torch thread oversubscription made `/state` exceed the client timeout

Every `GET /state` runs a **real forward pass** — float path ~93 ms plus
hard path ~12 ms on this box. By default torch claims all 32 cores *per
inference*, so a server thread and a co-located test client that both
infer at the same time oversubscribe the machine: measured `/state`
latency climbs from ~105 ms to beyond the client's 20 s timeout, and the
client reports a read timeout **even though the server answered** (the
server log shows `200`).

**Fix:** `src/c4/ui_server.py` pins its own torch budget via
`TG_UI_TORCH_THREADS` (default 4). The game state is already serialised
behind `_lock`, so one inference at a time is all the server ever needs.
Measured idle latency with the budget applied: ~220 ms, and the machine
is no longer starved.

### 5t.3 Orphan processes from earlier sessions starved the sweep

This was the cause of the *rotating* failures, and it was invisible from
inside the test. At the moment of the worst sweep there were **four**
stale `.venv-c4` processes running:

```text
pid 60265  measure_label_agreement.py  (full-corpus scan, OOM-prone)
pid 78212  ui_server --model runs/_probe/_wtest4/...   (stale)
pid 93930  ui_server --model .../fd_a_s6000/...        (stale)
pid 95822  ui_server --model .../fd_a_s6000/...        (stale)
```

Any one of them is enough to push a 20 s client timeout over the edge.
The label-agreement scan in particular reads the whole 320k-row corpus and
is OOM-prone — the known failure mode of this box.

**Operational rule:** before trusting a sweep result, check for orphaned
`.venv-c4` python processes and kill them. A sweep result is only
meaningful on an otherwise-idle machine.

### 5t.4 The empty-training-set crash

A resume whose run directory has no corpus (or whose corpus was filtered
to nothing by the quality/behaviour gates) used to fall through to
`build_batch`, which called `rng.integers(0, 0)` and died with
`ValueError: high <= 0` — three frames from the real cause, and *after* a
`[snapshot saved -> step6000]` line that made it look like the resume had
worked.

**Fix:** `src/c4/train.py` now refuses loudly:

```text
refusing to train: the training set is EMPTY (0 sequences). The corpus at
runs\_probe\_emptycorpus\corpus.txt produced no usable rows (the run
directory has no corpus, or it is empty). A resume must continue the SAME
corpus the checkpoint was trained on -- copy the original corpus.txt (and
the matching perply_targets.npz) into the run dir, or point --run-dir at
the original run directory. To relax the gates, pass --keep-low-quality.
```

### 5t.5 The clean sweep

With the orphans killed and only the sweep's own server running, the full
serial sweep is **24 PASS, 0 FAIL** — every live-parity test included:

```text
PASS test_atomic_checkpoint_save.py    PASS test_control_harness_soundness.py
PASS test_eval_fixed_arms_harness.py   PASS test_live_parity_v3.py
PASS test_live_parity_v4.py             PASS test_live_parity_v5.py
PASS test_live_ui_fd_a.py               PASS test_patterns.py
PASS test_perply_row_index.py           PASS test_play_v24_watcher.py
PASS test_promotion_gate.py             PASS test_resume_equivalence.py
PASS test_resume_path.py                PASS test_resume_state.py
PASS test_symmetry.py                   PASS test_symmetry_batch_equiv.py
PASS test_symmetry_penalty.py           PASS test_symmetry_stream.py
PASS test_transcript_stamp.py           PASS test_ui_api.py
PASS test_ui_hot_reload.py              PASS test_ui_isfull_v2.py
PASS test_ui_single_server.py           PASS test_ui_terminal.py
=== 24 PASS, 0 FAIL ===
```

`test_ui_single_server.py` is new and pins the double-bind fix.

**The single most valuable operational lesson from this whole stretch:**
a timing-sensitive test result measured on a loaded machine is not
evidence. The rotating failures were never a test bug — they were four
orphaned processes, one of which had been looping a full-corpus scan for
five hours. Clean the process table, then re-measure.

## 9. Play it

```text
http://127.0.0.1:7860/
model: runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt
```

It is the best available checkpoint and it is genuinely playable, but it is
**not** release-grade — and per §5c it was trained with the supervision bug,
so it is not a fair representation of the architecture either. The
corrected model will replace it once the acceptance gate passes.

The UI's percentages are softmax next-move preferences, **not** win
probabilities — a displayed 100 % means all mass sits on one legal column
(and every one of the 902 observed 100 % cases had exactly one legal
column).

### 5t.6 The sweep lock was being defeated by the very commands that used it

While running the batch above, three sweeps ran **at the same time** — and
the results looked like three different random failures (24/24, 23/24,
22/23). The cause was in my own command lines, not in the code under
test.

The sweep takes a one-writer lock with `mkdir "$LOCK"`. That works: the
lock is a **directory**, so `rm -f` does not touch it. But the batch
commands I wrote used `rm -rf runs/_probe/.regression_sweep.lock` to
"clear" the lock before each sweep — and `rm -rf` on a directory removes
it. Each command therefore **deleted the lock the previous sweep was
holding**, and the next sweep started on top of it. Three sweeps then
drove one port-7860 UI server, resetting each other's board state.

Two distinct symptoms followed, and both looked like product bugs:

- `test_ui_hot_reload.py` failed — it had lost the race for the shared
  board. In isolation: **14/14 PASS**.
- `test_live_parity_v5.py` failed — same cause, rotating between runs.

**Fixes applied to the sweep itself:**

1. The lock records the owner's PID in `$LOCK/pid`. A lock whose owner is
   dead is now **reclaimed automatically** (so a crashed sweep cannot wedge
   the suite forever), while a lock held by a live process still aborts
   the newcomer with a message that says *do not use `rm -rf`*.
2. The sweep summary is now **sweep-unique**
   (`_sweep_<pid>_summary.log`), with a copy at the stable path
   `regression_sweep.log` for humans. A fixed summary path let concurrent
   sweeps overwrite each other's PASS/FAIL lines, which is what turned
   three innocent results into three contradictory-looking ones.

Verified both lock paths directly: a lock held by a live PID aborts the
second sweep; a lock held by a dead PID is reclaimed and the sweep runs.

**The lesson is about how I issue commands, not about the code.** To run
a sweep after a previous one, do not delete the lock — the sweep's own
stale-lock detection now handles a dead predecessor. And a sweep must be
the only thing touching the machine while it runs.

### 5t.7 A transient Windows process-start failure

`test_resume_state.py` failed once inside a sweep with:

```text
rc=3221225794   (= 0xC0000142 = STATUS_DLL_INIT_FAILED)
```

and **no output at all** — the trainer child died before reaching `main`.
That is an environment failure under load, not a logic error, and
reporting it as a test failure would be wrong.

**Fix:** `tmp/test_resume_state.py` now routes every trainer invocation
through a `run_trainer(...)` helper that retries up to 3 times (10 s apart)
when the exit code is exactly `3221225794`. Detection is by **exit code**,
not by scraping output, because there is no output in this case.

With the retry in place the same test passes inside the sweep, and the
suite is green:

```text
=== 24 PASS, 0 FAIL ===
```

The lock lifecycle was verified end-to-end as well: the lock dir carries
the owner PID during the run, is released on exit (the cleanup removes the
`pid` file before `rmdir`, which a bare `rmdir` could not do), blocks a
live second sweep, and is auto-reclaimed when its owner is dead.

### 5t.8 The watcher regression suite was leaking processes onto the live run

When the V24 convergence run was finally launched, the machine was
**not idle**: 33 leftover `play_v24.sh` watcher processes and `ui_server`
processes were still alive from earlier runs of the watcher regression
suite. Measured cost: **1107 CPU-seconds** and **1.2 GB RSS** of
accumulated contention.

Root cause: `tmp/test_play_v24_watcher.py` cleaned up with
`watcher.terminate()` plus a PowerShell sweep that matched only
`*ui_server*`. On Windows, terminating an MSYS bash does not reliably run
the script's EXIT trap, so the **watcher bash itself survived every run**.
Each test case leaked one bash + one python server.

This is the same contention class that made the live-parity tests fail at
random: extra processes stealing CPU from a measurement.

**Two fixes:**

1. `tmp/test_play_v24_watcher.py` now kills the watcher's whole process
   tree via `psutil` (children first, then the parent), then sweeps any
   straggler whose cmdline references the test's own tag. Verified: the
   suite still passes **12/12**, and the watcher/UI process count does not
   grow across a run.
2. `tmp/cleanup_leaked_watchers.py` (new) reclaims already-leaked
   watchers. It protects the live chain by **socket ownership**, not by
   environment variables: `psutil.Process.environ()` returns `None` for
   `V24_TAG` on these Windows bash processes, and a naive cmdline grep
   even matches the cleanup script's own command line. So it finds the
   process listening on the live port, protects its full ancestry, and
   kills everything else that looks like a watcher or UI server. It
   refuses to run if no live server is found, rather than guessing.

Measured after the cleanup: **24 processes killed**, live trainer and live
UI server confirmed alive afterwards, CPU at 7 % and GPU at 99 %.

**Lesson:** a regression suite that starts real servers must reap them by
process tree. Terminating a shell is not the same as stopping the thing
it started.

### 5u. V24 IS RUNNING (launched 2026-10-29 02:34)

**Status: launched.** The convergence run is live and detached.

```text
run dir:        runs/_probe/v24/
trainer log:    runs/_probe/v24.log
chain log:      runs/_probe/v24_chain.log
watcher log:    runs/_probe/v24_play.log
UI:             http://127.0.0.1:7860/   (hot-reloads as the trainer promotes)
config:         100,000 steps = 5.00 epoch-equivalents
                batch 16, ctx 192, seed 32, transcript v2, dense lags
                --max-legacy-frac 0 --table-bias 0.0 --hard-start 0
                eval every 2,000 steps, save every 2,000 steps, 24 eval games
                --keep-last 3 (bounded disk), disjoint holdout wired
launcher:       tmp/launch_v24_chain.sh  (training + watcher, one chain)
```

**Measured, first two evals:**

```text
step     0 | HOLDOUT 11.8% | LABEL 16.3% | H-LABEL 11.9% | t 40s
step  2000 | HOLDOUT 29.0% | LABEL 40.2% | H-LABEL 27.9% | t 883s
```

The step-2000 LABEL of **40.2 %** matches the earlier projection of
**40.23 %** exactly, and the board-gate fired correctly at step 0
("exported model is BOARD-BLIND ... refusing to promote").

**The rate changed, and the reason matters.** The plan assumed
0.650 s/step (18.1 h). The measured rate on the *contended* machine was
**0.4415 s/step** -> **12.3 h**, finishing around **14:50**. The run got
FASTER after the cleanup, because the earlier 0.646 s/step figure was
measured while 24 leaked watcher processes were stealing CPU. Both
numbers are honest; the second one is the real one for a clean machine.
Do not quote 18.1 h for this run.

**Why it was launched now rather than at 00:00.** The standing goal was
an "around-midnight" launch. The run was originally blocked on the
regression gate, which only went green at ~02:20 (24 PASS / 0 FAIL after
the lock and retry hardening). Launching at 02:34 is inside the window
and, more importantly, after the gate -- a midnight launch would have
shipped an unvalidated chain.

**Monitoring:** cron job `v24-training-watchdog` (job
`54e457966d02`), every 45 min, delivers into this chat. It reports only
on trouble or completion. Its process-liveness check uses psutil, not
`ps -W` (which reported zero trainers while one was running at 99 % GPU).

**Operational rules for the duration of the run:**

- Do NOT run the regression sweep while training is active. It drives the
  same GPU and CPU and will slow the run and make its own timings
  meaningless.
- Do NOT clear `runs/_probe/.v24.lock` with `rm -rf`. The launcher now
  reclaims a stale lock automatically when the recorded PID is dead.
- If the UI stops answering on 7860, the trainer is probably still fine --
  check the trainer first (psutil), then restart the watcher.

### 5v. Disk pressure during the V24 launch — and its real cause

Mid-launch the volume showed **4.3 GB free** on a 931 GB disk (100 %
used flag). Two separate causes, only one of which was the training run.

**Cause 1 — the training run itself: not the problem.**
`runs/_probe/v24/` was **604 MB** at step 2000 and grows to about
**1 GB** total under the default `--keep-last 3` retention (3 float
snapshots ~203 MB each + 3 hard exports ~34 MB each + the promoted pair).
That fits comfortably.

**Cause 2 — the regression suite's own scratch: the real problem.**

```text
runs/_probe/_resume_equiv        1,2 GB
runs/_probe/_resume_state_test     830 MB
runs/_probe/_resume_state_test2    506 MB
runs/_probe/_resume_state_test3     53 MB
runs/_probe/_gate_smoke            289 MB
runs/_probe/_eo32_50                53 MB
                              --------
                                ~2,9 GB
```

Both resume tests did `shutil.rmtree(SCRATCH)` **at the start** of a run
but never at the end. So a *successful* run left a corpus copy, a
sidecar copy and full optimiser-state checkpoints behind — every time.
The isolated rerun in the launch window alone created ~1.9 GB of it.

**Fixed:** both tests now also remove their scratch in a `finally`
block. Verified:

```text
test_resume_state.py        19 PASS, 0 FAIL, 0 scratch dirs left
test_resume_equivalence.py  12 PASS, 0 FAIL, 0 scratch dirs left
max |Adam moment difference| = 0.000e+00   (still bit-exact)
```

**Reclaimed 4.3 GB -> 7.2 GB free** without touching a single
historical run, corpus or promoted checkpoint.

**Lesson:** a test that only cleans up on entry leaks on success. This
is the mirror image of the watcher leak (§5t.8) — there the process
survived, here the files did. Both are "cleanup on the happy path only"
bugs, and both only showed up because something else was using the same
resource.

**Standing rule while V24 runs:** the volume has ~7 GB free and the run
needs ~0.4 GB more. Do not run the full regression sweep repeatedly
during the run — each sweep is fine, but do not leave scratch behind.

### 5w. Host sleep during the V24 run — what actually happened (2026-10-29)

The host went to sleep mid-run. The run survived, but the timeline needs
recording because it looks alarming and is not.

```text
04:41:25  step 18000 logged        (LABEL 59.5%, HOLDOUT 35.7%)
04:47:10  Windows Kernel-Power id 42   <- machine entered sleep
04:47:16  Kernel-Power id 107          <- resumed (6 s later)
05:58:14  Power-Troubleshooter id 1    <- final wake
06:01     live step counter = 19439
```

**The machine was asleep for roughly 71 minutes**, not 6 seconds. The
Kernel-Power pair (42 then 107) records the *transition request*; the
Power-Troubleshooter event at **05:58:14** records the actual wake.
Trusting 42/107 alone would have wrongly concluded a 6-second nap.

**Nothing was lost or corrupted.** Verified after the wake:

- trainer process alive (worker pid 42856, started 02:33:43), GPU 95 %,
  409 W, actively in the training loop (py-spy: `build_batch` /
  `_weighted_move_loss`);
- all 8 checkpoints in `runs/_probe/v24/checkpoints/` load cleanly, with
  correct `meta.step` and intact `opt_state` / `rng_state`;
- zero `Traceback` / `error` / `corrupt` / `nan` lines in the trainer log;
- the watcher and UI survived the sleep and are still serving the
  promoted `best_hard.pt` at step 18000;
- `.v24.lock` still held by the original chain.

**Why the log looked stalled.** Log lines are written only every
`--eval-every 2000` steps. Between step 18000 (04:41) and step 20000
(06:06) there was an 85-minute wall-clock gap for a cycle that normally
takes ~14 minutes. The extra time was the sleep, not a slow trainer:

```text
18000 -> 20000 = 2000 steps in 5109 s wall  -> 2.55 s/step  (sleep inside)
instantaneous rate measured with py-spy = 0.395-0.822 s/step (normal)
```

The `t 12768s` in the step-20000 line is **wall-clock**, so it jumped
from 7659 s to 12768 s across the sleep. That is expected: the trainer
uses `time.time()`, and it measures real elapsed time, not CPU time.

**Consequence for the ETA.** The wall-clock-based ETA is now pessimistic
by exactly the sleep duration (~71 min). At the clean-machine rate of
~0.42 s/step, the remaining 80,000 steps need ~9.3 h of *awake* time.

**Consequence for the abort heuristics.** The trainer's plateau detector
compares LABEL across evaluations. A sleep gap makes the elapsed time
per evaluation look enormous. If the abort logic ever compares wall-clock
rates, a sleeping host would look like a stalled run. It does not
(currently it warns on flat LABEL), but this is the scenario that would
make it dangerous.

**Rules for the host:**

- A sleeping host does NOT kill the run — the trainer resumes and the
  checkpoints stay valid. Verified.
- Do NOT conclude "the run is stuck" from a stale log alone. Check the
  live step counter via py-spy `--locals` before intervening.
- If the host will sleep again, expect an ~85-minute gap between eval
  lines and do not treat it as a failure.
- The watchdog must distinguish "log is old because the host slept" from
  "log is old because the trainer died".

### 5x. The promotion scorer can score a pinned checkpoint file (2026-10-29)

### The problem

`tmp/measure_promotion_scorer.py` accepted only a **run directory**. A
checkpoint *file* was rejected with "not a directory". That broke a real
workflow: the trainer PRUNES old step snapshots (`--keep-last 3`), so a
checkpoint that had been pinned to a stable path outside the run dir
could no longer be scored at all. And in this session the step-20000
artifact was pruned within minutes of being pinned.

### The trap in the first fix

The obvious fix — "find the owning run dir" — introduced a worse bug. The
scorer picked the **first alphabetically** qualifying run dir and scored
a 320,000-row V24 checkpoint against a **500-row scratch corpus**:

```text
runs/_probe/compare/step20000_hard.pt
  (checkpoint file; scoring against owner run runs/_probe/_cron_repro)
  corpus rows: 500  sidecar: (500, 19, 7)
  step20000_hard.pt   LABEL 64.28%     <- WRONG, inflated
```

64.28 % vs the true 59.00 %. Scoring the same weights against a
different corpus is **not a comparable measurement** — it is a
different measurement wearing the same label. There are 12+ qualifying
run dirs under `runs/_probe/`, of wildly different sizes (500 rows,
1,500, 43,000, 70,000, 320,000).

### The fix

```bash
# score a checkpoint FILE against a named corpus run
./.venv-c4/Scripts/python.exe tmp/measure_promotion_scorer.py \
    --corpus-run runs/_probe/v24 runs/_probe/compare/step20000_hard.pt
```

- `--corpus-run <dir>` names the run dir explicitly and wins outright.
- Without it, the owner is resolved from the checkpoint's **ancestors**
  (a snapshot in `runs/_probe/v24/checkpoints/` resolves to
  `runs/_probe/v24`).
- If that fails and **more than one** run dir qualifies, the scorer
  **refuses and exits 1**, printing each candidate with its corpus row
  count. It does not guess.

### Verified

| Case | Expected | Got |
|---|---|---|
| pinned file, no `--corpus-run` (ambiguous) | refuse, exit 1 | refused, exit 1, listed 12 candidates |
| pinned file + `--corpus-run runs/_probe/v24` | 59.00 % | **59.00 %** (matches run-dir measurement) |
| run dir (unchanged path) | 59.55 % / 58.52 % / 58.55 % | identical |

All five pinned artefacts reproduce their run-dir scores exactly from
stable paths.

### Pinned artefacts now retained

```text
runs/_probe/compare/best_hard_step18000.pt   LABEL 59.55%  (meta.step=18000)
runs/_probe/compare/step20000_hard.pt        LABEL 59.00%
runs/_probe/compare/step22000_hard.pt        LABEL 58.52%
runs/_probe/compare/step24000_hard.pt        LABEL 58.55%
runs/_probe/compare/step26000_hard.pt        LABEL 58.70%
```

### The standing lesson

A measurement tool that silently picks the wrong corpus is worse than a
tool that refuses. "It ran and produced a number" is not the same as
"it measured the right thing" — the first is a vacuous pass.
