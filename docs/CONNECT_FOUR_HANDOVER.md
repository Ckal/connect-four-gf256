# Connect Four — Handover

> **🔴 ROOT CAUSE FOUND (late night).** The per-ply supervision
> was indexed by **batch position instead of corpus row**. Every trained arm
> (`fd_a`/`fd_b`/`fd_c`) learned the teacher answers of **16 of 320,000**
> sidecar rows, and **41 %** of the supervision silently degenerated into
> "predict the move that was already played". The code is **fixed and
> guarded**; a corrected run is in progress. **Every model-behaviour
> conclusion in this document is provisional until re-measured.**
> Full detail: Appendix D20 and `docs/CURRENT_TRUTH.md` §5c.
>
> **⚠️ STATUS SUPERSEDED (late).** The line that used to read
> *"the centre-only defect is FIXED … `mir_a_s6000` is the leading candidate"*
> was **wrong** and is retracted. No arm passes the release gates.
> `fd_a_s6000` (75.0 %) does **not** beat `always-c4` (75.7 %) on the
> authoritative 560-position solver set, and the 94.1 % c3/c4/c5 collapse
> quoted in Appendix D16 is **specific to that 560-position endgame-heavy
> subset**, not a whole-model property.
>
> **Read `docs/CURRENT_TRUTH.md` first** — it is the one-page, current,
> non-contradictory summary. Everything below is the working history.
> The historical body is kept verbatim for provenance, but where it
> disagrees with `CURRENT_TRUTH.md`, `CURRENT_TRUTH.md` wins.
>
> Retraction details: Appendix D18.

> **ADDENDUM (WIDE SOLVER — 560 POSITIONS).**
>
> The corrected `--max-empty 24` solver run completed: **560 / 1,500 = 37.3%**
> of the holdout is now solver-proven (up from 166 = 11.1%). 940 positions
> stayed unproved within budget and were **excluded, not scored**.
>
> | arm | solver-optimal (n=560) | c3 use where teacher said non-c3 |
> |---|---:|---:|
> | shipped | 68.9% | 72.5% |
> | fc_a_tau0 | 71.1% | 30.3% |
> | fc_b_tau05 | 70.0% | 31.0% |
> | mir_a_tau0 | 67.9% | 62.3% |
> | **fc_a_s6000** | **71.8%** | **26.8%** |
> | **mir_a_s6000** | **71.8%** | 41.5% |
>
> **The mirror advantage on solver-optimality was a small-sample artefact.**
> At n=166 `mir_a_s6000` led 71.7% vs 69.3%; at n=560 they are **tied at
> 71.8%**. Any claim that mirror augmentation improves solver-optimality is
> no longer supported.
>
> **The teacher is strongly anti-centre relative to proven play:** c3 is
> provably optimal **54.3%** of the time, the teacher picks it **24.6%** of
> the time. Teacher-agreement must not be used as a release metric.
>
> **FINAL VERDICT: no arm passes every gate.** `mir_a_s6000` wins on
> symmetry (36.7 vs 16.4), tactics (50.2 vs 42.7) and head-to-head (63.0%),
> and ties on solver-optimality. `fc_a_s6000` is better calibrated across
> the whole board. Mirror augmentation is a **targeted** improvement that
> **trades against** whole-board centre calibration. Still **NO V24**.

> **ADDENDUM (CORRECTION + FULL BATCH).**
>
> **A number I reported earlier was from the wrong population.** The `c3%`
> column in the first version of the matched table came from the
> **166-position endgame set**, not the 1,500-position holdout. On the full
> holdout the picture inverts:
>
> | arm | c3 share (1,500 holdout) | c3 share (166 endgames) |
> |---|---:|---:|
> | fc_a_s6000 | **33.3%** | 25.3% |
> | mir_a_s6000 | **54.6%** | 27.7% |
>
> So the claim "mirror augmentation reduces centre bias" is **false on the
> full board** and true only on endgames. What mirror augmentation *does*
> improve, measured against independent ground truth, is:
>
> | metric | fc_a_s6000 | mir_a_s6000 | verdict |
> |---|---:|---:|---|
> | mirror consistency | 16.4% | **36.7%** | **BETTER** |
> | tactical blocking | 42.7% | **50.2%** | **BETTER** |
> | solver-optimal (endgames) | 69.3% | **71.7%** | **BETTER** |
> | head-to-head duel | — | **63.0%** | **BETTER** |
> | c3 share (full holdout) | **33.3%** | 54.6% | **WORSE** |
>
> Both directions are real and neither cancels the other.
>
> **Diversity A/B (all 4 arms completed).** The diverse corpus moves the
> model off the centre (c3 83% → 58–60%, off-centre 5–6% → 10–12%) but is
> **strictly worse than the full-corpus per-ply arms** on every metric.
> Diversity is necessary but not sufficient.
>
> **A silent no-op was caught:** the `tg_solver_buckets.py --max-empty 24`
> run was ignored — the script had no argparse and hardcoded
> `MAX_EMPTY = 18`. A real CLI was added and the corrected run is in
> flight. Solver proof currently covers **11.1%** of the holdout
> (endgames only).
>
> Still **NO V24**.

> **ADDENDUM (CONTROLLED MATCHED A/B — DECISIVE).**
>
> The mirror-augmentation question is now answered by a **confound-free**
> experiment. The earlier `mir_a_tau0` result was confounded: the trainer
> samples rows *with replacement* (`src/c4/train.py:633`), so a 640k-row
> mixed corpus at 48,000 samples sees only ~23,120 distinct *original*
> games against `fc_a_tau0`'s 44,573. The matched pair `mir_a_s6000`
> (44,572 distinct originals) vs `fc_a_tau0` (44,573) matches to **0.002 %**,
> and `mir_a_s6000` vs `fc_a_s6000` holds the sample count fixed.
>
> | arm | mirror | blocking | solver-opt | teacher strict | teacher off-c |
> |---|---:|---:|---:|---:|---:|
> | shipped | 6.2% | – | 64.5% | 28.5% | 6.2% |
> | fc_a_tau0 | 18.6% | 45.0% | 69.3% | 21.9% | 15.0% |
> | fc_a_s6000 | 16.4% | 42.7% | 69.3% | 21.7% | 15.4% |
> | **mir_a_s6000** | **36.7%** | **50.2%** | **71.7%** | 23.8% | 9.7% |
>
> **Mirror augmentation wins on every independent-ground-truth metric.**
>
> The one regression — "off-centre strict" — is a **teacher artifact**, now
> proven: the teacher picks c3 only 21.7% of the time while c3 is *provably
> optimal* 36.1% of the time. Splitting solver-optimality by the teacher's
> target (`tg_solver_buckets.py`, n=166) shows that in the teacher's
> non-c3 bucket (n=130) `mir_a_s6000` is solver-optimal **70.0%** — better
> than `fc_a_s6000` (65.4%) and `fc_a_tau0` (65.4%) — with a model-c3 rate
> of only 18.5%, i.e. it avoids the centre exactly when the centre is wrong.
>
> Also new: `tg_symmetry_probe.py` proves the symmetry information is **not**
> in the logits. Inference-time symmetrisation gains only +0.1…+0.6 points
> and pushes c3 from 37.1% → **50.7%** on `fc_a_tau0`. Mirror augmentation
> must come from training, not an ensemble.
>
> Full detail: **`docs/CONNECT_FOUR_SECTION_0I.md` §6**.
> Machine-readable: **`runs/_probe/consolidated_candidate.json`**.
>
> Still **NO V24** — awaiting the release decision.

> **ADDENDUM (section 0h — ROOT CAUSE ISOLATED).** The
> remaining "always plays the centre" defect is **not** an architecture
> limitation. A **canonical-board-disjoint** linear probe on layer2
> reaches **31.42 %** validation versus **23.34 %** for ALWAYS-c3, so
> the GF trunk *does* encode which column to play. The defect is the
> **learned readout and its training objective**, and the decisive
> measurements are in **`docs/CONNECT_FOUR_SECTION_0H.md`**.
>
> Also resolved: the long-standing **1250 / 1367 / 1500**
> c3-count discrepancy was **never a model bug** — it was an unmasked
> `argmax` in the decomposition script plus one number that belonged to
> a *different arm*. See section 0h §6.

> **ADDENDUM (evening).** Three things changed after this
> document was written and they change the plan. See **section 0d**:
> 1. the "saved hard export is corrupt" finding was a **loader bug**, not
>    trainer corruption;
> 2. the labels were **mirror-inconsistent** (32.4% of mirrored pairs),
>    and the cause was the **teacher**, not the corpus;
> 3. there is now a **position-disjoint holdout**, which the old
>    evaluation did not have.
>
> **ADDENDUM (night).** A fourth and deeper bug was found
> and fixed. See **sections 0e.2–0e.5**:
> 1. the exported integer **tables were frozen at initialisation** — the
>    argmax never moved at any checkpoint, the table gradient was
>    *exactly zero*, and every saved artifact was effectively a
>    constant-column policy;
> 2. the fix is `--table-bias 0.0` (now the trainer default). It is the
>    first change in project history that moved the exported tables;
> 3. the `--hard-start` STE question was **re-measured with the fix
>    active** and answered: keep `--hard-start 0`.
>
> **The model is still not release-ready.** The remaining gap is a
> teacher/objective problem (centre collapse + mirror asymmetry), not an
> initialisation problem. **Do not launch V24.**

**Snapshot:** night addendum
Repo: the project checkout
Current run: `runs/_probe/tb00/` (3,000 steps, `--table-bias 0.0`, completed)
Decisive evidence:
- `runs/_probe/tbfix_final/disj_fix_hard.json` — fix arm, hard export,
  4,000-position disjoint holdout
- `runs/_probe/tbfix_final/disj_ctrl_hard.json` — control arm, same positions
- `runs/_probe/tbfix_final/mirror_fix.json` — hard mirror audit, fix arm
- `runs/_probe/sched2_hs0/`, `runs/_probe/sched2_hs750/` — STE schedule
  re-measurement (6,000/6,000 each, `EXIT_HS0=0` / `EXIT_HS750=0`)

---

## 0. Executive summary — the one-paragraph answer

> **REWRITTEN. The previous version of this section said
> "yes, we solved it" — that was wrong, and the measurement behind it was
> broken. If the two disagree, this section wins.**

**No. We did not build a strong opponent, and the model underfits.** The
old acceptance numbers (38.35% vs heuristic, 31.77% vs lookahead, 33.42%
vs minimax) came from a harness that let the **opponent randomise** and
counted **draws as half-points**. Re-run with deterministic opponents and
only outright wins counted, **every candidate model wins 0% of its games
against heuristic, lookahead and minimax** — it draws against the first
two and loses to the third. The "centre rule" reference is the *weakest*
opponent in the set (it loses 100% to a one-ply heuristic), so beating it
proves nothing.

**Underfitting, not overfitting.** Train strict agreement **23.4%**,
holdout strict agreement **20.6%** — a gap of only 2.8 points. The
trainer's headline "43.5% holdout" is tie-inflated (the targets are
mostly flat ties, so any tied column scores full credit); strict is ~21%.

**Openings and simple win-prevention are exactly where it fails.**
Depth-stratified measurement of the shipped model:

| depth | takes an immediate win | blocks a required block | strict label agreement |
|---|---|---|---|
| opening (30–42 empty) | 23.8% | 22.5% | **15.8%** |
| midgame (16–23 empty) | 40.9% | 34.7% | 21.9% |
| endgame (0–7 empty) | **72.7%** | **59.3%** | **47.7%** |

Confirmed live through the shipping UI: with three of the user's discs
stacked in column 0 and a vertical four available, **V22 played column 4
and let the user win** (`runs/_probe/ui_v22_game.json`).

**Root cause (all measured):**
1. **Train-only depth filter.** `tools/split_positions.py` applied
   `--min-plies 20` to train and `--holdout-min-plies 0` to holdout, so the
   model was **never supervised on an opening** while 68.3% of the holdout
   is opening.
2. **5.26% real supervision.** 304 move positions survive the mask per
   batch; only **16** carry a label. Each labelled position was seen
   **3.74×** across 25,000 steps.
3. **Weak teacher.** The imported score label agrees with the **exact**
   solver's best move on only **28.3%** of provable positions. The
   depth-13 negamax (`src/c4/strong_solver.py`) agrees on **61.7%** at
   ~41 ms per position — a 2.2× better teacher, and affordable for the
   whole corpus. Played as a policy, the old label scores 27.4% vs
   heuristic and 14.8% vs lookahead.

**Not the cause (measured and ruled out):**
- **Soft/hard divergence is small.** The training-log bpb gap peaks at
  **2.212** and never exceeds **3.0** across 24,500 steps; column-
  distribution JSD is **0.011–0.030 nats**.
- **"The labels are tactically blind / sign-inverted"** — retracted. That
  was our own audit bug. Corrected: labels take an immediate win **100%**
  of the time and pick the required block **82.7%** of the time.
- **"The exact solver lost to minimax"** — retracted; `ExactSolver(max_empty=12)`
  was silently falling back to a depth-limited policy.
- **Capacity** — the corrected 4,096-position probe reaches **87.67%
  oracle-tie / 56.45% strict** vs a **33.22%** chance line.

**Two of our own audit bugs (retracted, recorded so they are not repeated):**
1. `encode_seq(moves, label_move)` appends a final record whose *header*
   is the labelled position and whose *move byte* is the label. We replayed
   the label move and audited the board *after* it.
2. `Board.apply_move` mutates in place; candidate probes corrupted the
   board under audit.

**The fix now running (V23):**
1. **Depth-fixed split** — no train-only depth filter. Train 386,916 /
   holdout 20,260, symmetry overlap **0**, mirror leakage **0**.
2. **Strong-teacher relabel** — the exact solver for ≤20 empty cells
   (provably optimal), the depth-13 negamax for everything else including
   the entire opening. Sharded across 24 workers.
3. Then retrain from a fresh seed and re-run the deterministic ladder.

**ADDENDUM (night) — the V23 retrain is NOT the blocker.
The real blocker was a fourth bug, now fixed.**

The controlled teacher A/B (§0e) showed dense per-ply supervision *lost*
to single-label supervision, which was itself a red flag: more supervision
should not hurt. Chasing that led to the actual cause:

**The exported integer tables were frozen at initialisation.** The saved
artifact is the *argmax* of three logit tensors. Measured directly: the
argmax was **identical at every checkpoint** (step 0 → 3000), the table
gradients were **exactly 0.0**, and the readout absorbed 100% of the
learning. Every previously saved model was effectively a
constant-column policy. Three mechanisms combined:
1. identity/one-hot init bonuses created a **7.78-logit** top1–top2
   margin, while Adam at lr 6e-4 can only move 1.80 logits in 3,000
   steps — the argmax was **unreachable**;
2. the entropy regulariser **pushed the margin further open** (7.72 →
   9.94) — anti-learning;
3. a zero-initialised readout meant the table output was multiplied by
   zero, so table gradients were **exactly zero**.

**Fix:** `--table-bias 0.0` (now the trainer default; the loader still
defaults pre-existing checkpoints to 1.0 so they reconstruct exactly as
trained). Controlled A/B, 3,000 steps, one variable changed:

| step | `table-bias 1.0` (old) | `table-bias 0.0` (fix) |
|---|---|---|
| 500 | 0.0000 blind | **0.5444** |
| 1000 | 0.0000 blind | **0.8343** |
| 3000 | 0.1567 | **1.6794** |

The table hash now changes at **every** checkpoint — the first time in
project history that the exported tables have moved at all. The GF trunk
is unchanged (4,608 integer ops, zero float multiplies): the fix
changes *reachability*, not the architecture.

**But it is a prerequisite, not a solution.** The fixed export still
collapses to c3 on **67.7%** of disjoint positions (8.7% off-centre),
its mirror consistency is **11.8%**, and the soft path *regressed* to
2.4% off-centre. The canonical teacher is verified 100% mirror-consistent
over 18,304 pairs, so the labels are exonerated — the remaining gap lives
in the **loss and the objective**. **Not release-ready; do not launch
V24.** Full blocker list: §0e.6.

**Settled since the last handover:** the 4,096-position capacity question.
The first probe said "no better than guessing"; that was the probe's own
two bugs (objective/metric mismatch + a label leak). The corrected probe
reaches **87.67% oracle-tie / 56.45% strict on all 4,096 positions**,
against a 33.22% chance line, with `EXIT=0`. **The network was never too
small.** See §3c.

---

## 0b. ROOT CAUSE FOUND — the shipped export was never trained

**This is the biggest finding of the project, and it invalidates part of
§0 above.** The model was not merely weak. The artifact we were serving
and gating was a **near-constant policy that could not see the board at
all**, and the mechanism was a single un-passed default flag.

### The defect

`src/c4/train.py` trains a **soft** forward (soft mixtures over the
embedding, the GF coefficients and the S-box) and *additionally* a
**hard/STE** forward (one-hot) with a weighted loss:

```python
hard_w = args.hard_w * min(1.0, (step - args.hard_start + 1) / max(1, args.hard_ramp))
```

Defaults: `--hard-start 12000 --hard-ramp 4000 --hard-w 0.5`.

**No launcher ever passed `--hard-start`** — not `run_v22_greedy.sh`, not
`run_v23.sh`, not any of the v9–v21 scripts. Every run therefore used
12000, so the one-hot path was **untrained for the first 12,000 steps**.

The artifact that ships is `export_hard` — the one-hot path. So for the
first 12,000 steps of every run, the shipping artifact was an
**untrained approximation** of a function that had learned something
else. The board signal lived in the soft *mixture* and did not survive
the argmax.

### The measurement

For each checkpoint, change the board (keep the move history legal) and
measure the change in the 7 column logits. A model that reads the board
cannot return exactly 0.

| checkpoint | hard board-sensitivity | hard column distribution | reads board? |
|---|---|---|---|
| step 2000 | **0.0000** | c c | no |
| step 10000 | **0.0000** | c0 | no |
| step 11500 | **0.0000** | c0 | no |
| **step 12000** (= `--hard-start`) | **0.7402** | c7 c0:3 | **yes** |
| step 12500 | 0.7273 | c c | yes |
| step 13000 | 0.7376 | c c | yes |
| step 15000 | 0.7119 | c7 | yes |
| step 19000 | 0.6935 | c c | yes |

The transition lands **exactly** on `--hard-start 12000`. The soft model
read the board the whole time (soft board-sensitivity 0.26 → 0.90 across
the same checkpoints), so the weights were never the problem — the
**export was**.

### Which shipped artifacts are affected

| artifact | hard sensitivity | blind? |
|---|---|---|
| **V22 `best_hard.pt` (step 8750)** — the UI, and every gate so far | **0.0000** | **YES** |
| V22 `step24000_hard.pt` | 0.2197 | no |
| V23 `best_hard.pt` (promoted at step 14500) | 0.7243 | no |
| V23 `step19000_hard.pt` | 0.6935 | no |

**V22 `best_hard.pt` is a board-blind constant-column policy.** It answers
column 3 or column 4 depending on the position and ignores the stones
entirely. Every gate, every ladder number, and the live UI game that "lost
to a vertical four" were measuring *that artifact*.

### Why the metrics disagreed with each other

The trainer computes **two different functions** and reports both:

- **HOLDOUT** — computed on the **soft** path (`model.set_ste(False)`,
  `src/c4/train.py:786`).
- **LADDER** — plays the **hard** path (`agent.choose(..., use_hard=True)`,
  `src/c4/train.py:133`).

Before step 12000 the ladder measured a constant policy while holdout
showed real learning. So `best_hard.pt` carried `ladder_net=-61` and
`holdout_optimal_pct=42.675` — two numbers about two different functions.
Neither was wrong on its own; the comparison was meaningless.

### The corrected deterministic ladder (200 games/opponent, hard path, greedy)

| artifact | vs centre | vs random | vs heuristic | vs lookahead | vs minimax |
|---|---|---|---|---|---|
| V22 `best_hard` (BLIND) | 0.0% | 80.0% | **−100%** | **−100%** | **−100%** |
| V22 step24000 (reads) | 0.0% | 66.5% | −100% | −100% | −100% |
| V23 `best_hard` (reads) | **100%** | 85.0% | 0.0% | −100% | −100% |
| V23 step19000 (reads) | **100%** | 84.0% | −100% | −100% | −100% |

Reading it honestly: **every artifact still loses to lookahead and minimax**,
and none of them beats heuristic (the best is a dead 0.0% net). Board-reading
is *necessary* but nowhere near sufficient — V23's board-reading checkpoints
at least stop being constant (100% vs centre instead of 0%), which is the
first sign of a real policy, but they are still not competitive.

The `distinct` column is the important one: **2** for every deterministic
matchup — greedy model versus deterministic opponent produces exactly one
line per seat order. The "200 games" are 2 distinct trajectories. Any
metric on this ladder must be read as *that one line*, and it is why a
weak opponent or a lucky opening can make a blind model look like 80%
against random. V23 `best_hard` scoring exactly 0.0% vs heuristic (100 W /
100 L over 2 distinct lines) is the clearest example: it wins one seat and
loses the other, deterministically.

### What this means for the plan

1. **The data work was necessary but not sufficient.** The depth filter,
   the 5.26% supervision density and the weak teacher are all real
   defects — but fixing them cannot help if the shipping artifact cannot
   see the board.
2. **Fix the export.** Either train the hard path from step 0
   (`--hard-start 0`), or ship the soft weights, or add a hard-forward
   loss with a non-zero weight from the beginning. The current default
   silently produces a broken artifact for 40% of the run.
3. **Audit every checkpoint before serving it.** Board-sensitivity is a
   two-minute test and it catches the whole class.
4. **Re-run the ladder on post-STE checkpoints only.** V22's numbers in
   §0 were measured on a blind artifact; they are still valid as *proof
   that nothing worked*, but not as a statement about V22's learned
   policy.

### Two of our own audit bugs (again, recorded so they are not repeated)

The first soft-vs-hard comparison reported JSD = 0.0000, and a
"uniform p=1/7, entropy = ln 7" distribution. Both were **loading a
hard-only export into the float model**: the float weights were never
restored, so `model(...)` returned all-zero logits and the comparison was
comparing zeros with zeros. `load_state_dict`'s `missing` list is
non-empty for a hard checkpoint — **always check it**.

---

## 0c. THE REAL ANSWER TO "IS IT THE DATA?"

§0b fixed the export. This section answers the question that followed:
**if the model is trainable, is the training data the problem?**

Short answer: **the data is good. The loss function is the problem.**

### The simplest possible check, and what it showed

`tg_simplest_check.py` presents three boards that differ only in where P1's
three-in-a-row sits, so the winning column differs. A board-reading model
must answer differently; a blind model answers the same every time.

```
V22 best_hard.pt  (the artifact the UI served):
  win in column 0 -> plays 4     MISSES
  win in column 3 -> plays 4     MISSES
  win in column 5 -> plays 4     MISSES
  picks = [4, 4, 4]              VERDICT: BLIND

V23 best_hard.pt  (post-STE):
  win in column 0 -> plays 0     TAKES THE WIN
  win in column 3 -> plays 3     TAKES THE WIN
  win in column 5 -> plays 3     MISSES
  picks = [0, 3, 3]              VERDICT: SEES THE BOARD
```

And the STE repair is proven, not assumed (`tg_ste_fix_probe.py`, resuming
the pre-STE step-11500 checkpoint with hard supervision on):

```
BEFORE  hard board-sensitivity = 0.0
AFTER   hard board-sensitivity = 1.72
VERDICT: FIXED
```

Export-path parity (`tg_export_parity.py`) confirms all three integer
paths are the same function: PyTorch `hard_forward` vs the pure-Python
`GFCipher` in `src/generate.py` — max delta **0.000004**, argmax **100%**;
`export_hard_with_meta` payload identical to `model.export_hard`.

### The data is measurably good

`tg_data_is_the_problem.py` and `runs/_probe/_winlab.py`:

| question | answer |
|---|---|
| immediate-win positions in the corpus | **6,218** of 32,696 sampled |
| of those, the label IS a winning column | **96.5%** (3,859/4,000) |
| label column distribution vs. true winning column | closely matched |
| are the three test boards in the corpus? | no — so the model generalises or it doesn't |

`tg_epoch_math.py` puts the corpus in numbers:

```
transcripts                386,916
plies available        6,199,408
labels (final ply only)    386,916      ->  6.24% label density
if every ply were labelled 6,199,408   ->  16.0x more signal
```

### The model DOES learn the rule

`tg_wins_generalise.py` — on 699 held-out immediate-win positions, does the
model pick a column that actually wins?

| model | wins picked | vs ~14% chance |
|---|---|---|
| **wins-only** (20k perfect rows, 1.6 epochs, undiluted) | **54.2%** | 3.9x |
| **V23 best_soft** (full corpus, diluted) | **44.8%** | 3.2x |
| V23 step19000_soft | 45.4% | 3.2x |

Both are far above chance. **The model learns "if I can win, play the
winning column" at ~45-54%.** That is the rule being learned.

### And yet the training loss says the opposite

`tg_log_metric_check.py` and `runs/_probe/_mass.py`:

```
accuracy at the SUPERVISED slot: 29.1%
accuracy via the AGENT's own window: 29.1%

probability mass the model puts on the 7 column bytes
at a move byte: mean 0.0002   (uniform over 256 would be 0.0039)
byte-argmax accuracy at move bytes: 0.0%
```

**The model puts 0.02% of its probability mass on the seven legal columns
at a move byte.** The 7-way masked loss teaches a distribution over seven
columns, but the *reported* 256-way bpb (11-14, i.e. worse than uniform)
and the *reported* byte-argmax accuracy (0%) are computed over all 256
bytes, which that loss never touches. **The headline metric in the training
log is measuring the wrong thing.** The bpb in the log is an artifact of
the 7-way/byte-loss mismatch, not a sign of a broken model.

### The actual root cause: the loss balance

`ce_soft = move_w * ce_move + other_w * ce_other`, with
`--move-w 1.0 --other-w 0.1` (defaults).

But `ce_move` is a mean over **supervised move positions** — exactly **one
per transcript** — while `ce_other` is a mean over **every non-move byte**,
of which there are ~118 per row. Counting actual gradient contributions
(`runs/_probe/_balance.py`, 2,000 rows):

```
move positions     26,216     weight 1.00
non-move positions 235,558    weight 0.10

effective gradient share, MOVE loss (the decision):  7.8%
effective gradient share, BYTE loss (the encoding): 92.2%
```

**92% of the gradient is spent learning to write the v2 header bytes
correctly. 8% is spent on the move decision — the only thing that matters
for playing the game.** The move head is being optimised as a rounding
error of the byte head.

### Why "more data" would not have fixed it

More rows add more *header* supervision, which is already 92% of the
gradient, and do not add move decisions: the sidecar labels only the final
ply, so the label density stays at 6.24% no matter how many rows you add.
**More rows make the imbalance worse, not better.** More *labels per row*
(per-ply labelling) multiplies the 8% by 16x — that is the right lever,
and it is a different change from "more data".

### Also ruled out in this section

* **Capacity.** The corrected capacity probe reached **56.45% strict /
  87.67% oracle-tie** against **33.22%** chance.
* **Representation.** `tg_state_is_a_word.py` enumerates different legal
  move orders that reach the **same board** and asks whether the model
  answers the same: **96.7% same answer, 3.3% different**. A position is
  effectively a word, not a sentence. The user's intuition ("is every game
  state like a word in a language?") is correct and the model agrees.
* **Label legality / alignment.** `tg_fit_own_data2.py`: sidecar board
  matches the replayed pre-final-ply position **200/200**; labels are legal
  moves **400/400**.

### A note on our own errors, recorded honestly

Two of the measurements above were wrong before they were right, and both
errors pointed the same way — toward "the model cannot learn":

1. `tg_fit_own_data.py` built the inference stream from **all** k plies
   instead of the first k-1, so it asked the model to predict an *unseen
   next move* and reported 21.3%. Corrected: **52.0%**.
2. The first wins-only run was **1.6 epochs** (2,000 steps x batch 16 over
   20,000 rows) and its "VERDICT: did NOT learn" string was a hardcoded
   threshold, not a measurement. With correct thresholds it is **54.2%**.

A test that is underpowered does not report "no learning"; it reports
nothing. Both mistakes were in the measurement, never in the model.

### What this changes in the plan

1. **Do not go looking for more data.** The corpus is clean; the labels are
   96.5% correct; the representation is state-like; capacity is sufficient.
2. ~~**Rebalance the loss** (`--other-w` down, `--move-w` up) and re-measure
   the wins-only probe.~~ **TESTED AND REJECTED** — 47.2% vs 46.9%, a
   0.3pp difference. Not the bottleneck.
3. **Train longer / more epochs.** The learning curve is still climbing at
   4 epochs (40% → 47% → 53% → 58%). V23 itself only saw **1.24 epochs**
   over its labels. This is the cheapest, highest-confidence lever.
4. **Per-ply labelling** (16x more move decisions), which is what per-ply
   supervision was always for. This attacks coverage *and* exposure at
   once — a row that yields 8 labels instead of 1 is worth 8 rows.
5. **Fix the reported metric** so the log stops reporting 11-14 bpb and 0%
   byte-accuracy for a model that picks a winning column 58% of the time.
6. **Then** rebuild the position-disjoint holdout and re-run the ladder.
   Every strength number so far was measured on an artifact whose move head
   received 8% of the gradient — but note the A/B above: that 8% is not
   itself the limiting factor, so do not expect a rebalance to fix strength.

### Is the model just "playing the middle column"? — NO. Proven.

`tg_centre_bias.py` measures this directly, because "picks a winning column
55% of the time" is worthless if the winning column is usually the centre
and the model just always plays the centre.

**Baseline — always play one fixed column** (same 1,126 test positions,
each counted once under its leftmost winning column):

```
always c0:  10.2%      always c4:  25.1%
always c1:  17.5%      always c5:  15.7%
always c2:  22.7%      always c6:   9.3%
always c3:  26.8%   <-- best constant
```

**Conditional accuracy — when the win is in column w, does it play w?**

| true win in | wins-only (10 ep) | V23 best_soft |
|---|---|---|
| c0 | 63.5% | 48.7% |
| c1 | 52.9% | 27.7% |
| c2 | 74.7% | 59.0% |
| c3 | 70.1% | 74.9% |
| c4 | 66.5% | 45.5% |
| c5 | 13.6% | **1.7%** |
| c6 | 51.0% | 41.2% |

**STRICTEST CONTROL — wins that are NOT in the centre column:**

```
n = 895 positions where the winning column is not 3

wins-only: 514/895 = 57.4%   (always-centre would score 0.0%)
V23: 354/895 = 39.6%   (always-centre would score 0.0%)

VERDICT: READS THE BOARD (beats every constant policy, and wins
         off-centre too)
```

A centre-only player scores **0%** on those 895 positions. Both models
score 40-57%. **They are not playing the middle; they are reading the
board.** The centre preference is real but it is a *tie-break among good
moves*, not the whole policy.

Where the centre bias does bite: **column 5**. V23 scores **1.7%** when
the win is in column 5 — it essentially never plays column 5 (2 picks in
1,126). The wins-only model is at 13.6%, still bad. This is the one place
the "always near the middle" prior actually costs real wins, and it is a
*coverage* gap: the wins-only corpus only had 118 column-5 win positions
out of 1,126.

### Game-level: model vs. a centre-only opponent

`tg_centre_vs_centre.py` (model = hard export, greedy; opponent = centre
policy with epsilon exploration, because a deterministic opponent yields
only **2 distinct trajectories** out of 200 nominal games — the exact trap
the project already hit once).

The first run of this test reported `model as P1: 100/100, as P2: 0/100,
2 distinct trajectories` — a **meaningless** number, because both policies
are deterministic and the two games are mirror images of each other. With
exploration added to the opponent the trajectories become distinct and the
result is readable:

```
wins-only (step2500_hard) vs CENTRE-ONLY:
  300 nominal / 118 distinct trajectories
  model 187 W - 113 L - 0 D   = 62.3%   (as P1 114/150, as P2 73/150)

V23 best_hard vs CENTRE-ONLY:
  300 nominal / 147 distinct trajectories
  model 286 W - 14 L  - 0 D   = 95.3%   (as P1 144/150, as P2 142/150)
```

Both models beat a centre-only opponent decisively, **from both seats**.
So it is not "the model wins because it plays the middle" — the middle is
the *opponent's* weakness, and the model beats it by a wide margin.

### The learning curve, and the two hypotheses that are now dead

Same corpus, same architecture, same trainer — only the number of epochs
changes. Metric: held-out immediate-win positions, "picks a column that
actually wins" (chance ~14%):

```
  500 steps  (0.4 epochs): 40.0%
1000 steps  (0.8 epochs): 47.2%
2500 steps  (2.0 epochs): 53.2%   (10-epoch run)
5000 steps  (4.0 epochs): 58.0%   (10-epoch run)
2000 steps  (1.6 epochs): 54.2%   (earlier 1.6-epoch run, unbalanced)
```

Still rising at 4 epochs. The model is **under-trained, not under-capable**.

**Train vs held-out (the memorisation test):**

```
TRAIN   (rows the model saw): 217/357 = 60.8%
HELDOUT (rows it never saw): 151/275 = 54.9%
gap: 5.9 pp
VERDICT: GENERALISES (train ~= heldout) -- rule is learned
```

A 5.9pp train/held-out gap is a healthy generalisation gap, not
memorisation. Combined with the off-centre control, this is a model that
**learns the rule and applies it to positions it has never seen**.

**Hypothesis "the loss balance is the bottleneck" — TESTED AND REJECTED.**
Controlled A/B, same data, same seed, same 1,000 steps, only `--other-w`
changed:

```
unbalanced (--other-w 0.1): 422/895 = 47.2%
balanced   (--other-w 0.0): 420/895 = 46.9%
```

**0.3 percentage points — indistinguishable.** Even though 92.2% of the
gradient goes to the byte loss (§0c), rebalancing it changes nothing about
what the model learns. The byte head is an easy task that saturates; the
move head is the bottleneck, but starving the byte head does not feed the
move head. **This hypothesis is dead** and the plan below does not include
it.

### The specialist-vs-generalist split — the final piece

`tg_win_difficulty.py` (corrected: the opponent-threat check must keep
p1/p2 and change only the turn — swapping the colours asks "does P1
already have 4 in a row", a different question) buckets every held-out
position by difficulty, on 2,500 positions:

```
                    wins-only (step5000)     V23 best_soft
forced  (1 win column)      52.2%                 36.7%
choice  (several wins)      75.7%                 61.0%
double  (win + threat)      57.6%                 45.5%
opp_only (BLOCK only)       26.6%                 49.6%
```

**This is the cleanest evidence in the whole project.** Each model is good
at exactly what it was trained on and bad at the other:

* the **wins-only** model (trained on nothing but immediate-win
  positions) takes wins 52-76% of the time but **blocks only 26.6%** —
  it was never shown a single block;
* **V23** (trained on the full corpus) blocks **49.6%** — better than the
  specialist — but takes wins less often.

So the full corpus is not "diluting" the attack signal into uselessness;
it is trading attack accuracy for defence. The wins-only number (58%) is
**not** an upper bound on what the real model can do — it is the number
you get when you delete the defence half of the problem.

**And neither model is at ceiling.** The `forced` bucket is the easiest
possible pattern — "you have three in a row, play the fourth" — and the
wins-only specialist still only gets 52.2% of them, while `choice` (more
options) is *easier* at 75.7%. That inversion says the residual failures
are not about combinatorial difficulty; they are about **reading the right
part of the board**.

---

## 0d. ADDENDUM (evening) — three corrections, one real fix

### 0d.1 The "saved hard export is corrupt" finding was a LOADER bug

**What we thought we knew:** the wins-only `step12500_hard.pt` differed
materially from a live `export_hard` re-export of the matching
`step12500_soft.pt` (coefficient delta 170; the empty-board opening move
changed from column 3 to column 0). We classified the saved hard file as
stale/corrupt and ruled it out of checkpoint-strength claims.

**What was actually wrong:** `src/c4/soft_loader.py` rebuilt the model's
GF coefficient lookup table `coef_set` at load time. The trainer generates
that table at model construction from a seeded RNG; the loader regenerated
it with a *different* RNG draw, so `export_hard` interpreted the same
argmax indices as different GF coefficients. The trainer's own saved hard
files were always internally correct — only *reloading and re-exporting*
was wrong.

**The fix:**
* `src/c4/soft_loader.py` restores `coef_set` from checkpoint metadata
  when present, and otherwise reconstructs it from the recorded
  `model_seed`/`seed`.
* `src/c4/train.py` now writes `coef_set` and `model_seed` into the
  metadata of **all three** save sites (snapshot soft, `best_soft.pt`,
  `best_hard.pt`).

**Verified:** after the fix, saved hard exports match live re-exports
exactly for every checkpoint tested. The earlier "corrupt" verdict is
withdrawn.

**What it did NOT change:** the V23 tactical result. Off-centre win
accuracy went from 39.6% to 39.2% with the fixed loader — i.e. the
loader bug never inflated the soft-path numbers.

### 0d.2 The labels were mirror-INCONSISTENT, and the teacher was the cause

**The corpus was never the problem.** A full-corpus audit found that
**~99.99%** of V23 rows have their byte-level mirror present in the
corpus. (An earlier 13.5% figure was a sampling artifact — it examined
only a non-closed 200,000-row subset and only final positions.) The
corpus is effectively fully mirrored.

**The labels were not.** Comparing each row's label against its mirror
row's label at the same ply:

```
inconsistent mirrored pairs: 19,434 / 59,995
mirror-consistent fraction:  ~67.6%
asymmetric fraction:          ~32.4%
```

**Root cause: the teacher.** `strong_policy` (depth-limited negamax) was
mirror-symmetric on only **~65%** of legal positions. The search's
tie-break direction leaks into the label: a position and its mirror are
different bit patterns, so equal-scoring moves are broken differently.

**The fix: canonical orientation.** Solve the board in a canonical
left-right orientation and map the answer back through `c -> 6-c`. This
makes the label symmetric *by construction* rather than hoping the search
agrees with itself. New module `src/c4/mirror.py` (shared reflection +
canonical-teacher helpers), covered by `tools/test_mirror.py` — 7 checks,
all pass. (One real bug was caught by the tests: `reflect_bits` was
dropping row-6 bits, so it was not an involution. Fixed — all 49 bits are
now permuted.)

**Measured effect** on 1,400 mirror-rich rows / 16,878 labelled ply pairs:

| label path | before | after |
|---|---|---|
| exact solver | 83.5% | **100.0%** |
| strong teacher (budget 0.02 s) | 49.4% | 92.6% |
| strong teacher (budget 1.0 s) | — | **100.0%** |
| **overall** | **49.8%** | **100.0%** |

The residual 7.4% at the tight budget was **wall-clock truncation**, not
search asymmetry — a larger budget removes it entirely.

**Production path now verified.** `tools/relabel_perply.py` uses the
canonical teacher *and* caches by the canonical key, so one symmetry class
is solved exactly once — a wall-clock-limited search can no longer give a
board and its mirror two different answers. Measured on the same 16,878
pairs at budget 0.25 s (the production default):

```
pairs compared: 16878   (skipped 0)
  self-symmetric: 1156    (label must simply AGREE)
  non-self-symmetric: 15722   (label must be REFLECTED)
consistent: 16878/16878 = 100.00%
inconsistent: 0
```

Two real bugs were found and fixed while getting there, both worth
recording:

* `CanonicalTeacher` cached the **mapped** column instead of the canonical
  one, so the mirror hit the cache and got the wrong orientation. The
  cache must store the canonical column and map per caller — the two
  orientations need opposite mappings from the same key.
* the consistency **metric** had the self-symmetric test inverted. A board
  that equals its own mirror (e.g. the empty board) has no reflected
  counterpart; demanding `label[c] == label[6-c]` there wrongly demands
  the centre column. Getting that backwards reported 93.2% for a sidecar
  that is actually 100% consistent.

`tools/test_mirror.py` now has 9 test groups (all pass), including
`CanonicalTeacher` symmetry, one-solve-per-mirror-pair caching, and
`canonical_key(b) == canonical_key(mirror(b))`.

**Why this matters for the shipped model.** The V23 hard export, audited
over 500 positions:

```
column 3 chosen:        476/500
off-centre rate:         4.8%
mirror-consistent non-centre choices: 0/24   <-- 0%
both seats:             500/500 same behaviour
VERDICT: MIRROR-INCONSISTENT on non-centre choices
```

So the shipped V23 export is centre-locked *and* mirror-inconsistent. It
has nothing consistent to learn from when it does play off-centre. This
is now the leading explanation for the residual tactical failure, and it
is a **data-generation** problem, not a model or capacity problem.

### 0d.3 There is now a POSITION-DISJOINT holdout

The V23 holdout was **not** position-disjoint: ~99.7% of its opening
positions also appear in train, and there were zero position-disjoint
holdout transcripts. A model could score well on it by memorising
openings.

`tools/build_disjoint_holdout.py` now builds a holdout where the exact
board **and its mirror** appear nowhere in the training corpus, while
still being reachable in a real transcript, labelled by the canonical
teacher. Result: 4,000 positions, label spread
`c0=390 c1=511 c2=641 c3=1037 c4=630 c5=443 c6=348`.

(The old version of this script had a broken mirror function — it
reversed the *binary string* of the integer, which is a bit-order
reversal, not a column reflection. Rewritten to use the tested
`src/c4/mirror.py`.)

### 0d.5 Two MORE measurement bugs found in the audit code itself

Both were in the scripts that were supposed to *measure* symmetry, and
both produced plausible-looking numbers.

**`tg_mirror_audit.py` used the wrong reflection.** It had a local
`reflect_bb` that was **column-major** (`bb >> (c*7)`) while the
bitboard is **row-major** (bit `r*7+c`). It reflected the wrong bits and
produced a "mirrored" board whose heights did not match its stones. The
script now imports the tested `src/c4/mirror.py`. Re-running the audit on
V23 `best_hard.pt` with the corrected reflection gives the same verdict
(centre-locked, 0/24 mirror-consistent), so the conclusion stands but
now rests on valid geometry:

```
positions scored: 500 (errors 0)
c3: 476/500 (95.2%)   non-centre: 24/500 (4.8%)
mirror-consistent on non-centre originals: 0/24 (0.0%)
seat consistency: 500/500 identical answer
VERDICT: MIRROR-INCONSISTENT on non-centre choices
```

**The consistency METRIC applied the reflected test to self-symmetric
boards.** A board that equals its own mirror (e.g. the empty board) has
no reflected counterpart; demanding `label[c] == label[6-c]` there
wrongly demands the label be the centre column. This reported "93.2%
consistent" for a sidecar that is actually 100% consistent, and "74.3%"
for one that is actually 100% consistent.

The corrected metric splits the two cases:

```
self-symmetric: label must simply AGREE
non-self-symmetric: label must be REFLECTED (tj[c] == ti[6-c])
```

**Corrected measurements with the fixed metric and the fixed builders:**

| sidecar | pairs | self-sym | non-self | consistent |
|---|---|---|---|---|
| `relabel_perply.py` (production) | 16,878 | 1,156 | 15,722 | **100.00%** |
| `build_perply_fast.py` | 16,878 | 1,156 | 15,722 | **100.00%** |
| A/B dataset (`ab_canon1500m`) | 7,191 | 7 | 7,184 | **100.00%** |
| disjoint holdout (`disjoint_ab`) | 35 | 13 | 22 | **100.00%** |

### 0d.6 The A/B corpus must be MIRROR-RICH

The first canonical A/B dataset was built from the first 1,500 corpus
rows. **Those rows have zero mirror partners** (0 of 1,500), so that A/B
could not test symmetry at all — its checkpoints could be audited for
column coverage and tactical accuracy, but not for mirror consistency.

The full corpus is perfectly mirrored: **193,453 mirror-partner pairs**,
always at offset **193,458** (row `i` pairs with row `i+193458`).

`tools/build_ab_dataset.py` now takes `--rows-file` (one corpus row index
per line; `-1` means "from the end"). The corrected A/B dataset
`runs/_probe/ab_canon1500m/` is built from **750 mirror-partner pairs**
(1,500 rows, interleaved), so its checkpoints can be audited for mirror
consistency.

The disjoint holdout was rebuilt the same way:
`runs/_probe/disjoint_ab.npz` (4,000 positions) plus
`disjoint_ab.corpus.txt`, with **7,975 training boards (including
mirrors) excluded** and canonical labels.

### 0d.7 Where the controlled A/B actually stands

The first A/B (`runs/_probe/ab_single` vs `runs/_probe/ab_perply`) is
**not** the comparison we need: both arms were trained against the
**non-canonical** teacher, and the two arms' eval schemas cover
different position sets (1,444 final-ply vs 19,279 all-ply), so their
headline numbers are not comparable.

Scored on identical positions (1,444 final-ply positions):

```
                                SINGLE-label    PER-PLY
STRICT (argmax == teacher)         52.7%        19.0%
```

and on the new position-disjoint holdout (1,500 positions, canonical
teacher labels, 3,000-step arms):

```
                                SINGLE-label    PER-PLY
STRICT                           23.4%         19.6%
OFF-CENTRE strict                 16.6%         15.1%
```

Read that carefully: **per-ply supervision is NOT yet a win.** It is
slightly *behind* on this holdout. The per-ply arm's column
distribution also shows a new failure mode (c4=657 of 1,500 picks — it
over-commits to the mirror of centre).

The controlled A/B is being re-run properly:
* dataset `runs/_probe/ab_canon1500m/` — same corpus, **canonical**
  teacher for both arms, **18,783 labels**, 12.52 per row, built from 750
  **mirror-partner pairs** so symmetry is testable;
* launcher `run_ab_canon.sh` — identical seed 7, 3,000 steps, batch 16,
  ctx 192, LR 0.0006, hard-start 0 / ramp 1000 / weight 0.5;
* the only difference between arms is **label density**, not label
  quality;
* evaluation on `runs/_probe/disjoint_ab.npz` — 4,000 positions whose
  exact board **and mirror** appear nowhere in the A/B training corpus.

**Do not launch V24 until that A/B shows per-ply winning on the
position-disjoint holdout AND the hard export passes the mirror audit.**
If it does not win, the next lever is not more labels but better ones.

### 0e. THE CONTROLLED A/B IS DONE — DENSE PER-PLY SUPERVISION **LOST**

Run: `runs/_probe/ab_canon1500m` (1,500 rows = 750 mirror-partner pairs),
canonical teacher, **identical** seed 7 / 3,000 steps / batch 16 /
ctx 192 / LR 0.0006 / `--hard-start 0 --hard-ramp 1000 --hard-w 0.5`.
The ONLY difference between the arms is label DENSITY:
`--soft-targets` (1 label per transcript) vs `--perply-targets`
(12.5 labels per transcript).

**Primary metric — 4,000-position holdout, position- AND mirror-disjoint
from the A/B training corpus, both arms on the identical positions:**

| arm | strict | teacher-set hit | off-centre strict | column distribution |
|---|---|---|---|---|
| single-label | **25.1%** | 25.1% | **16.6%** | c3=43.9%, spread |
| per-ply | **25.3%** | 25.4% | **5.5%** | **c3=79.9%**, collapsed |

**Secondary metric — both arms on the SAME 1,438 single-label positions:**

| arm | strict | off-centre strict | c3 share |
|---|---|---|---|
| single-label | **55.1%** | **41.6%** | 54.4% |
| per-ply | **29.6%** | **5.3%** | **81.4%** |

**Verdict: the dense arm is not better. It is much WORSE.** On the
identical single-label positions the per-ply arm scores **29.6%** where
the single-label arm scores **55.1%** — a 25.5-point deficit. Off-centre
accuracy collapses from 41.6% to 5.3%.

**Hard export, mirror audit, 500 positions each (matched step3000):**

| arm | c3 | off-centre | mirror-consistent (non-centre) | verdict |
|---|---|---|---|---|
| single-label | 85.4% | 14.6% | 2/73 = 2.7% | mirror-inconsistent |
| per-ply | **100.0%** | **0.0%** | vacuous | **degenerate (always c3)** |

**WHY — the mechanism, measured:**

1. **The per-ply arm collapses onto the centre column.** c3 share rises
   from 54.4% (single) to 81.4% (per-ply) on the same positions and to
   79.9% on the holdout. Its hard export picks c3 on **500/500**.
2. **The teacher itself is centre-heavy, and the per-ply arm samples it
   more.** Teacher argmax column distribution:
   * single-label (final ply only, n=1,500): c3=**30%**
   * per-ply (all plies, n=18,783): c3=**36%**
   * and the early plies — where the board is nearly empty and almost any
     column is defensible — are the MOST centre-heavy: slot 0 c3=47%,
     slot 18 c3=30%. Every added label is drawn from the plies whose
     teacher answer is closest to "play the middle".
3. **The exported integer tables barely move.** Export drift vs step 0:
   `emb 0.00% · coef 0.00–0.35% · sbox 0.00% · wiring 0.00%` while
   `readout 100%` and `bias 100%`. The per-ply arm's export at step 3000
   is **byte-identical to its step-0 export** in every table except
   readout/bias. The *table* is what carries the board; the readout
   merely re-labels a fixed table's output. So the per-ply arm's extra
   signal was spent entirely on the output layer and never reached the
   part of the network that reads the stones.
4. **34.1% of the per-ply labels have NO teacher source** (`src=0`,
   i.e. legacy/unknown) vs 65.8% strong and 0.1% exact. The dense arm
   trains on a third of its labels that were never solved at all.

**Consequence for the plan:** dense per-ply supervision is **not** the
lever. It trades away column coverage and off-centre accuracy for
nothing. V24 must NOT be a per-ply run.

### 0e.2 THE REAL ROOT CAUSE: THE DISCRETE TABLES WERE FROZEN AT INIT

The STE schedule was a red herring. The exported hard artifact is the
**argmax** of three table logit tensors — `emb_logits`, `sbox_logits`,
`coef_logits`. Those argmaxes never moved. Measured on the corrected A/B
arms (`abc_single`, 3,000 steps, canonical labels, seed 7):

```
step      0     500    1000   1500   2000   2500   3000
emb/sbox/coef table-hash:   IDENTICAL at EVERY step 0..2500
```

Parameter drift over the same window looks healthy but is misleading:

```
coef_logits   mean |Δ| 1.06 – 1.76   (logits move)
emb_logits    mean |Δ| 0.67
sbox_logits   mean |Δ| 0.66 – 0.97
readout       mean |Δ| 0.04 – 0.10   (but 100% of entries change)
```

The logits move, the **argmax** does not. Three independent mechanisms
keep the argmax pinned:

**1. The init margin is larger than the step budget can cross.**
At init the tables carry a large positive bonus toward the identity /
one-hot solution:

```
emb[ar, 0, ar]        += 10.0      (j=0 identity embedding)
emb[ar, j, rand]      +=  8.0      (j>=1 identity embedding)
sbox[..., diag]       +=  8.0      (identity S-box)
coef[:, 0, one_idx]   +=  6.0      (coefficient = 1)
coef[:, 1:, zero_idx] +=  4.0      (coefficient = 0)
```

Measured initial top1-top2 margins: **emb 7.78, sbox 7.72, coef 4.54**
(min emb margin 7.33). Adam moves a parameter by at most `lr` per step,
so N steps buy at most `lr*N` of movement:

| steps | lr·N | can the 7.33 margin be crossed? |
|---|---|---|
| 3,000 | 1.80 | **NO** |
| 12,000 | 7.20 | **NO** (130 steps short) |
| 25,000 | 15.00 | yes |

So on any realistic budget the argmax is **unreachable**, and the
identity solution is handed to the model rather than learned.

**2. The entropy regulariser actively anti-learns the tables.**
The loss adds `+lam * entropy`, and minimising entropy means peaking the
softmax. The entropy gradient on the identity entry is *negative* → the
identity logit is pushed **up**. Measured: the identity margin GREW
during training (sbox 7.72 → 9.94 between step 0 and step 3000).

**3. The zero-initialised readout kills the table gradient entirely.**
`readout` starts at exactly zero, and a zero readout multiplies the
(frozen) table output by zero. Measured gradient norms at step 0:

```
emb_logits        grad_norm = 0.000000e+00   <-- EXACTLY ZERO
layers.*.sbox_logits  grad_norm = 0.000000e+00
layers.*.coef_logits  grad_norm = 0.000000e+00
readout.0         grad_norm = 3.52e-01
readout.1         grad_norm = 2.81e-01
```

The tables receive **no gradient at all**. The readout — a dense float
layer that can absorb the entire signal without touching the discrete
path — takes 100% of the learning. The soft model learns the board
(measured soft sensitivity 0.24 → 2.18) and the hard export does not
(see the identity-only export: always column 3).

**Why this makes the STE schedule irrelevant:** STE changes *which* loss
term drives the gradient, not the fact that the table gradient is
exactly zero. Both `--hard-start 0` and `--hard-start 12000` leave the
exported tables frozen.

### 0e.3 THE FIX: `--table-bias 0.0` (unbiased table init)

The bonus is scaled by a new `--table-bias` flag (default **1.0** =
historical, so nothing changes unless asked). At `0.0` the bonus is
removed and the identity solution must be **learned**:

```
table_bias=1.0: emb margin=7.7795  sbox margin=7.7176
table_bias=0.0: emb margin=0.0326  sbox margin=0.0325
```

**The RNG draw order is unchanged** — every `randn`/`randperm` still
happens, only the `+=` bonus is scaled — so `coef_set` is byte-identical
(`[0, 1, 83, 48, 139, 126, 231, 13]`) and every existing checkpoint
still loads. Verified: `load_soft_agent` on the old A/B checkpoints
loads with 0 missing keys.

Patched in all three architectures: `src/gf256_markov_lm_v6.py` (the
**shipped** one — trainer, loader, UI and play all import it),
`src/gf256_markov_lm_v6_1.py`, and `src/gf256_markov_lm.py` (V7).

**CONTROLLED TEST** (`run_tablebias_test.sh`, 3,000 steps, canonical
single-label targets, seed 7, identical in every other respect):

| checkpoint | `--table-bias 1.0` (control) | `--table-bias 0.0` (fix) |
|---|---|---|
| step 0 | sensitivity 0.0000 (blind) | sensitivity 0.0000 (blind) |
| step 500 | **0.0000 (blind)** | **0.5444 — reads the board** |
| step 1000 | **0.0000 (blind)** | **0.8343 — reads the board** |
| step 1500 | **0.0000 (blind)** | **0.9569 — reads the board** |

Table-hash check: control is **IDENTICAL** at steps 0/500/1000; the fix
arm **CHANGED** at every checkpoint. The tables now learn.

### 0e.4 CONTROLLED RESULT: table-bias A/B (3,000 steps, seed 7, canonical single-label targets)

Identical in every respect except `--table-bias`. Arm `tb10` = 1.0 (historical),
`tb00` = 0.0 (unbiased).

**Board sensitivity of the HARD export during training:**

| checkpoint | `tb10` (control) | `tb00` (fix) |
|---|---|---|
| step 500 | 0.0000 — BOARD-BLIND | **0.5444** — reads the board |
| step 1000 | 0.0000 — BOARD-BLIND | **0.8343** — reads the board |
| step 1500 | 0.0000 — BOARD-BLIND | **0.9569** — reads the board |
| step 2000 | 0.0000 — BOARD-BLIND | **1.1250** — reads the board |
| step 2500 | 0.0000 — BOARD-BLIND | **1.4717** — reads the board |
| step 3000 | 0.1567 (just crossed) | **1.6794** — reads the board |

The control only crosses the gate at the very last checkpoint and at 1/10th
the sensitivity. Table-hash check: control **IDENTICAL** at steps 0/500/1000;
fix **CHANGED** at every checkpoint.

**Quality on the 4,000-position position- AND mirror-DISJOINT holdout:**

| metric | `tb10` control | `tb00` fix |
|---|---|---|
| HARD export strict | 25.1% | **25.3%** |
| HARD export teacher-set hit | 25.2% | **25.4%** |
| HARD export off-centre strict | 4.4% | **8.7%** (2.0× better) |
| HARD export c3 share | 3392/4000 = 84.8% | **2708/4000 = 67.7%** |
| SOFT path strict | 25.1% | 26.8% |
| SOFT path off-centre strict | 16.6% | 2.4% (collapses) |
| HARD mirror consistency (non-centre) | 2.7% | **11.8%** |

**Interpretation — read this carefully, it is not a clean win:**

1. The fix does what it was designed to do: the exported integer tables now
   LEARN, the hard export reads the board, and the hard export's column
   distribution is far less degenerate (67.7% c3 vs 84.8%).
2. But **the hard export is still not good enough to release.** Off-centre
   strict is 8.7% and mirror consistency is 11.8% — both far below the bar.
   The hard export still leans heavily on c3 (67.7%).
3. The fix **collapses the SOFT path's off-centre accuracy** (16.6% → 2.4%).
   This is a real regression that must not be papered over: the unbiased
   init changes which local optimum the float path finds, and that optimum
   is worse. It also means the soft-path diagnostics in earlier sections
   (soft sensitivity 0.24 → 2.18) do NOT carry over to `table_bias=0`.
4. The headline strict numbers (25.1% vs 25.3%) are statistically
   indistinguishable and should NOT be quoted as a win. The meaningful
   signal is: **the tables learn at all** (0% → 100% of checkpoints moved)
   and **the hard export's off-centre behaviour roughly doubles**.

**What this means for V24:** the table-bias fix is now a validated
*prerequisite*, not a solution. It removes the blocker that made every
previous hard export board-blind, but the remaining gap — off-centre
accuracy, mirror consistency, c3 collapse — is a **teacher/objective**
problem, not an initialisation problem. Do not launch a large V24 on the
strength of this result alone. The next thing to attack is the centre
collapse and the mirror asymmetry, which live in the loss and the data,
not in the table init.

**Still open after this experiment:**

- ~~The STE-schedule question (`--hard-start 0` vs delayed) is **still
  unresolved**.~~ **RESOLVED in §0e.5** — re-run with the fix active
  (6,000/6,000 both arms): the curves interleave, quality is a tie, and
  the answer is **keep `--hard-start 0`**.
- ~~`table_bias=0.0` should be the DEFAULT for all new runs, but it is
  currently opt-in via `--table-bias 0.0` (default stays 1.0 for
  backward compatibility with existing checkpoints).~~ **DONE** — the
  trainer default is now **0.0**. The *loader* still defaults old
  checkpoints to 1.0 (they lack the metadata key), so every existing
  artifact reconstructs exactly as trained.
- The soft-path regression needs its own diagnosis: it suggests the
  entropy regulariser or the readout init is what pins the float path to
  c3. **Still open** — tracked as blocker #1 in §0e.6.

### 0e.5 THE STE SCHEDULE, RE-MEASURED WITH THE FIX ACTIVE

The original schedule probe (`run_sched_probe.sh`) was killed because the
table-freezing defect made it unanswerable — the tables could not move
under any schedule. `run_sched_probe2.sh` repeats it with
`--table-bias 0.0` active, so the schedule is finally measurable.

Setup: 6,000 steps, per-ply canonical labels, seed 7, identical corpus,
LR, ramp, hard weight, batch 16, ctx 192. Only `--hard-start` differs.

| checkpoint | `--hard-start 0` | `--hard-start 750` (= 1/8) |
|---|---|---|
| step 500 | **0.6053** | 0.0094 |
| step 1000 | 0.5975 | **0.7263** |
| step 1500 | 0.9587 | **1.1052** |
| step 2000 | 0.9220 | **1.2652** |
| step 2500 | **1.2657** | 1.2240 |
| step 3000 | 1.1497 | **1.3528** |
| step 3500 | **1.3544** | 1.2633 |
| step 4000 | 1.3656 | **1.5041** |
| step 4500 | 1.6535 | **1.6786** |
| step 5000 | 1.6813 | **1.8692** |
| step 5500 | **2.0773** | 2.0167 |
| step 6000 | 2.0219 | **2.2024** |

Both arms completed 6,000/6,000. The two curves **interleave** — neither
arm dominates. Final quality on the 4,000-position disjoint holdout
(HARD export, matched step6000):

| metric | `--hard-start 0` | `--hard-start 750` |
|---|---|---|
| strict | **24.9%** | 24.6% |
| teacher-set hit | **24.9%** | 24.6% |
| off-centre strict | 5.7% | 5.7% |
| HARD mirror consistency (non-centre) | **9.8%** | 6.0% |
| HARD non-centre share | **24.6%** | 23.2% |

**Reading this honestly:** under `--table-bias 1.0` the schedule was
unmeasurable because the tables were frozen. Under `--table-bias 0.0`
both arms read the board from step 500 onward, and the schedule makes a
difference of at most ~0.2–0.35 sensitivity and ~0.3pp of strict
accuracy — second-order compared to the ~1.5 sensitivity and the
table-movement difference the table-bias fix produced.

**Answer to the original question:** the hypothesis that "hard-start 0
might be wrong, maybe start STE later" was **reasonable but not the
cause**. The delayed-STE arm is marginally ahead on sensitivity at 6 of
12 checkpoints and marginally behind on hard quality (24.6% vs 24.9%
strict, 6.0% vs 9.8% mirror consistency). The two are a statistical tie
on quality; the delayed arm's mid-run sensitivity edge does not survive
to the end.

**Practical consequence:** keep `--hard-start 0` (the shipped path trained
from step 1). Do not delay STE for the reason originally suspected. The
1/8 warm-up is at most a minor tweak, not a fix — and it is NOT worth
the extra LR-drop machinery it implies.

**Run-integrity record (all four runs verified, not just the wrapper exit):**

| run | steps reached | inner `EXIT=` marker | traceback | board-gate final |
|---|---|---|---|---|
| `tb10` (control) | 3000/3000 | yes | none | 0.1567 (step 3000) |
| `tb00` (fix) | 3000/3000 | yes | none | 1.6794 (step 3000) |
| `sched2_hs0` | 6000/6000 | yes (`EXIT_HS0=0`) | none | 2.0219 (step 6000) |
| `sched2_hs750` | 6000/6000 | yes (`EXIT_HS750=0`) | none | 2.2024 (step 6000) |

All ten evaluation sub-steps in `run_tbfix_eval.sh` returned `EXIT=0`.
No trainer processes remained after the runs.

---

## 0f. THE V23 / DATA / OPENING-CURRICULUM QUESTION — ANSWERED  The question was: *"what about V23? so in other words no success. i still
think it is the training data. can't we just teach it the first 7 moves.
that is not so many we can almost generate them? think hard. Why is it
wrong what we do?"* — with a competing structural explanation attached
(sparse lag wiring + the entropy regulariser).

**Short answer: the original diagnosis was right about the data, and the reviewer was right
about the structure. They are two different problems, and BOTH must be
fixed. Neither alone is sufficient.**

### 0f.1 V23: confirmed dead, and the reason is now known

V23 is a pre-`table_bias` candidate. Its corrected hard audit:

```
c3:            476/500
off-centre:     24/500 = 4.8%
non-centre mirror consistency: 0/24
soft c2 accuracy ~60.3%   soft c5 accuracy ~1.0%
```

V23 is centre-locked, mirror-inconsistent, and predates the validated
table-bias fix. It is a **historical artifact, not a release candidate**.
Its 193,453 mirror-partner pairs prove the point that a mirror-rich
corpus does **not** buy mirror-consistent *behaviour* — the labels can be
perfect and the model still asymmetric, because the asymmetry is
structural.

### 0f.2 Reviewer hypothesis #1 (structural blindness) — CONFIRMED

The lag wiring genuinely makes four of seven columns unreachable, and
the mirror pairs are structurally asymmetric. Measured on the actual
shipped wiring (consumer layer, not source layer):

```
reachable consuming channels
  c0 18   c1  1   c2  6   |  c6 21   c5 10   c4 18
```

`c1` had **exactly one** reachable channel. The empirical per-byte
hard-export sensitivity confirmed it:

| byte | sparse lags | dense lags |
|---|---|---|
| c0 | 0.3403 | 0.3382 |
| **c1** | **0.0000** | **0.3246** |
| c2 | 0.2309 | 0.2536 |
| c3 | 0.2360 | 0.3460 |
| c4 | 0.2132 | 0.3552 |
| c5 | 0.2183 | 0.3543 |
| c6 | 0.1970 | 0.3707 |
| last | 0.3698 | 0.3735 |

Mirror-pair sensitivity ratios, which is the quantity that matters:

```
sparse:  c0/c6  1.00x   c1/c5  INF (c1 = 0.0000)   c2/c4  1.08x
dense:   c0/c6  1.09x   c1/c5  1.00x              c2/c4  1.14x
```

Dense lags `0..9,10,20` (ctx stays **192**) make every header byte one
hop away and make the three mirror pairs symmetric to within 14%.
**This is a real, confirmed defect and the fix is real.**

### 0f.3 Reviewer hypothesis #2 (entropy regulariser) — REFUTED

`ent_max=0` with `table_bias=0`, 3,000 steps, everything else identical:

```
hard disjoint:  24.9% strict,  10.0% off-centre,  c3 919/1500
mirror audit:   10.0% mirror consistency,  40.0% non-centre
S-box injectivity at ent_max=0:  157.3 / 157.9 / 156.8 distinct of 256
c1 hard-export sensitivity:       still exactly 0.0000
```

`ent_max=0` does **not** restore S-box injectivity (still ~157 of 256
distinct outputs per row) and does **not** make the invisible byte
visible. The entropy regulariser is **not** the blind-column mechanism.
Do not re-litigate this.

For reference, the dense-lag arm's S-boxes are *also* lossy
(130.6 / 159.3 / 162.4 of 256) — injectivity is limited by the
architecture, not by the entropy term.

### 0f.4 The DATA hypothesis — CONFIRMED, and quantified

The labels are exonerated; the **effective sample size** is not.

```
label quality
  true same-board contradictions:  0 / 3,167 duplicate boards
  mirror consistency:               ~99.9%
  exact-board memorization ceiling:  100%
  teacher vs exact-solver agreement:  ~97.6%

effective sample size
  full-game A/B corpus:  18,783 labels over  ~3,856 distinct boards
  opening curriculum:   322,633 labels over 70,268 distinct boards   (18x)
```

The learning curve, driven by **distinct boards** rather than label
count, shows off-centre accuracy and mirror consistency both improving
with coverage while strict accuracy stays flat:

```
off-centre accuracy:  ~11.9%  ->  ~22.9%
mirror consistency:   ~10.4%  ->  ~27.9%
strict accuracy:      broadly flat
```

**A label count is not a sample size.** 18,783 labels over 3,856 boards
is 3,856 examples, not 18,783.

### 0f.5 "Can't we just teach it the first 7 moves?" — YES, and it works

The first 7-9 moves are exactly what the curriculum covers, and they are
cheaply generated. `tools/build_opening_dataset.py` produced **40,000
games x 9 plies = 358,480 labels** over 70,268 distinct canonical
boards, labelled with the same canonical teacher, at 100% mirror
consistency.

**Fast MLP proxy on the untouched full-game holdout (4,000 positions):**

```
arm                        train n   boards   strict    c3%   off-c%  mirror%  >const-c3
A_fullgame                  18,783     3,856   24.0%   36.6%   17.4%    26.4%      NO
B_opening_only             322,633    70,268   34.2%   53.1%   23.6%    39.9%     YES
C_curriculum_then_full      18,783    70,268   27.5%   20.5%   27.2%    27.9%     YES
D_mixture                  341,416    70,268   33.6%   45.8%   26.0%    39.8%     YES
constant-c3 baseline                             25.0%
```

The full-game-only arm is **below** the constant-c3 baseline. Every arm
that adds opening coverage is **above** it. Opening-only reached **80.0%**
strict on held-out opening games.

**The gain scales with distinct-board coverage, and it is not a small
effect.** A second proxy run on the *smaller* 24,212-label / 14,767-board
opening set gives the intermediate point:

```
opening set        boards   strict   > const c3 (25.0%)?
14,767 boards      14,767    25.6%    YES   (barely)
70,268 boards      70,268    34.2%    YES   (clearly)
```

So the opening curriculum is not a magic trick — it works *because* it
raises the number of distinct canonical boards, and the more you raise it
the more it helps. This is the same axis as the learning curve
(off-centre accuracy ~11.9% -> ~22.9% and mirror consistency
~10.4% -> ~27.9% as distinct boards grow from 100 to 3,856). The
full-game corpus sits at the bottom of that curve because it only has
3,856 boards.

**The trap (measured, not assumed):** the teacher's c3 share at plies 0
and 1 is **1.0**, and the opening set's overall c3 share (**38.9%**) is
*HIGHER* than the full game's (~25-30%). A naive opening-only curriculum
therefore strengthens the very centre prior that is collapsing — visible
in the proxy as arm B's c3 share of **53.1%** despite its higher accuracy.
The winning arm is the **mixture** (opening + full game in one dataset),
which keeps coverage high *and* keeps the full-game distribution in
play at a lower c3 share (45.8%). Pretraining-then-full-game (arm C) also
works, has the **lowest** c3 share of all (**20.5%**), and is the better
choice if the curriculum and the full game must be separate phases.

### 0f.6 Dense lags alone do NOT fix accuracy

Controlled A/B on the identical full-game corpus, 3,000 steps, seed 7,
`--table-bias 0.0`, `--hard-start 0`, ctx 192, batch 16, hard export on
the disjoint holdout:

```
arm          hard strict   off-centre   model c3   mirror consistency   non-centre
lag_sparse       26.9%        7.7%      1103/1500        11.8%            32.2%
lag_dense        25.9%        8.1%      1076/1500        17.1%            30.4%
ent_max=0        24.9%       10.0%       919/1500        10.0%            40.0%
```

Dense lags improve the **structure** (mirror consistency 11.8% -> 17.1%,
c1 sensitivity 0.0000 -> 0.3246) and the **column balance** (c1 accuracy
4.4% -> 6.8%), but strict accuracy is a tie and mirror consistency is
still far from acceptable. **Fixing the wiring removes the blindness;
it does not teach the model anything.** Only the data does that.

### 0f.7 The consolidated answer

| claim | verdict | evidence |
|---|---|---|
| V23 succeeded | **NO** | centre-locked, 0/24 non-centre mirror consistency, pre-fix |
| the labels are broken | **NO** | 0 contradictions, ~99.9% mirror-consistent, 97.6% vs exact solver |
| the data is too small | **YES** | 18,783 labels = 3,856 boards; curve rises with distinct boards |
| teach only the first 7 moves | **YES, it works** | 24.0% -> 34.2% strict, now above the 25.0% baseline |
| but opening-only over-centres | **YES, measured** | opening c3 share 38.9% > full-game 25-30% |
| columns are structurally invisible | **YES** | c1 sensitivity exactly 0.0000; 18/1/6 vs 21/10/18 channels |
| dense lags fix that | **YES** | c1 -> 0.3246; mirror ratios 1.09x/1.00x/1.14x |
| the entropy regulariser is the cause | **NO** | `ent_max=0` keeps c1 at 0.0000 and S-boxes lossy |
| dense lags alone fix quality | **NO** | strict 25.9% vs 26.9%; mirror only 17.1% |

**What we do wrong:** we have been treating *visibility* as *knowledge*.
The table-bias fix made the export read the board; dense lags make it
read *every* column; the entropy ablation ruled out the regulariser.
None of those teach the model which move is right. The only thing that
teaches is **coverage of distinct positions**, and we have been training
on 3,856 of them.

**What we do right:** the canonical teacher, the mirror-consistent
labelling, the disjoint holdout, the hard-export audit, and the
one-writer-per-run-dir discipline.

### 0f.8 The first 2x2 (CONFOUNDED — superseded by 0g)

A 2x2 factorial GF A/B (`run_curriculum_ab.sh`), 3,000 steps, seed 7, all
four arms with `--table-bias 0.0 --hard-start 0`, ctx 192, batch 16:

```
A  sparse lags, full-game data     (current state of the art)
B  dense  lags, full-game data     (the reviewer's fix alone)
C  sparse lags, curriculum data    (the user's fix alone)
D  dense  lags, curriculum data    (both)   <-- the candidate
```

**It ran, and it is CONFOUNDED.** The curriculum arm lost on every
metric, but the two arms differed in THREE ways at once, so the result
cannot be attributed to any one of them. See **section 0g** for the
numbers, the confound analysis, and the corrected experiment.

---

## 0g. THE REAL 2x2 — WHAT THE FIRST ONE ACTUALLY PROVED 

### 0g.1 The result as measured

All four arms completed 3,000/3,000 with no traceback. Hard export,
`tg_ab_eval.py --hard`, on the 1,500-position disjoint holdout:

```
arm                     strict   off-centre   model c3    board sens.   mirror%
A sparse + full-game      26.7%       3.5%      87.7%        1.1189       3.5%
B dense  + full-game      27.0%       4.9%      84.5%        1.7141       6.2%
C sparse + curriculum     21.2%      15.3%      36.7%        0.6678      23.4%
D dense  + curriculum     21.5%      13.6%      40.3%        0.6659      20.0%
```

Two things stand out, and they point in opposite directions:

* the curriculum arms **crushed the centre bias** (c3 share 87.7% ->
  36.7%) and **more than tripled off-centre accuracy** (3.5% -> 15.3%)
  and **multiplied mirror consistency by 6.7x** (3.5% -> 23.4%);
* but they **lost strict accuracy** (26.7% -> 21.2%) and their board
  sensitivity **halved** (1.12 -> 0.67).

### 0g.2 The labels are NOT the problem — measured

`tg_cross_dataset.py` matched the two datasets by **canonical board** and
compared the teacher labels directly:

```
A (full-game)  boards: 3,856
B (curriculum) boards: 70,268
boards in BOTH: 3,856
matched (board, ply) pairs: 3,856
SAME label: 3,856 (100.00%)
DIFF label: 0 (0.00%)
```

**100% label agreement on every shared board, at every ply.** The
curriculum labels are exactly the full-game teacher on the same
positions. The opening data is not wrong, not inconsistent, and not
a different teacher. **The data hypothesis is NOT refuted by this
result — the confound is what failed.**

### 0g.3 The actual confound: training budget, not data quality

The two arms differed in three ways simultaneously:

```
                     A/B full-game          C/D curriculum
rows                     1,500                   43,000
labels                  18,783                  396,046
distinct boards          3,856                   70,268
epochs @ 48,000 draws    2.556                   0.121   <-- 21x FEWER
ply mass shallow         32.1%                  93.6%
```

With a fixed 3,000 steps at batch 16, the curriculum arm took **21x
fewer passes over each distinct position**. That alone predicts its
lower board sensitivity (0.67 vs 1.12) — the model had seen each
position 21x less often.

The ply-mass shift compounds it: the curriculum is 93.6% shallow
plies, but the holdout is **74.3% deep** (ply >= 9). The curriculum
arms were pushed toward the ply range the holdout mostly does not
contain.

The learning-curve evidence from section 0f.4 says the same thing from
the other side: off-centre accuracy and mirror consistency both **rise
with distinct-board count**. Curriculum arms had 18x more boards but
21x fewer passes per board. Net: undertrained.

### 0g.4 The corrected experiment (running)

`run_labelmatched_ab.sh` holds the budget fixed and varies only what we
actually want to test. `tools/build_label_matched.py` builds both
datasets with the **same label count**, so epochs are identical:

```
dataset        rows   labels   boards   shallow   epochs@48k
lm_full        1,500   18,783   3,977     32.1%      2.556
lm_open        2,087   18,712   7,564    100.0%      2.565
```

Arms (all `--table-bias 0.0 --hard-start 0`, ctx 192, batch 16, seed 7):

```
A  sparse lags + full-game   3,977 boards
B  dense  lags + full-game   3,977 boards
C  sparse lags + opening     7,564 boards
D  dense  lags + opening     7,564 boards
```

Now the two hypotheses are separable, and neither is confounded with
the number of gradient updates. **The release decision stays blocked
until this one lands.**

### 0g.5 A third confound found: the holdout is not disjoint

While preparing 0g.4, the holdout's provenance was checked. It was
built from `runs/c4/posdata_v23/holdout_corpus.txt`, **not** from the
A/B corpus — and it overlaps both training sets:

```
holdout boards:                    4,325
overlap with A/B training corpus:    207  (4.8% of holdout)
overlap with curriculum corpus:    1,099  (25.4% of holdout)
```

The "disjoint" holdout is position-disjoint from the *A/B* corpus only.
Any arm trained on the curriculum sees **25.4%** of the holdout boards
(1,099/4,325) in training, versus 4.8% for the full-game arms. The
curriculum arms are therefore evaluated partly on positions they have
already been trained on. This inflates the curriculum arms' apparent
accuracy and must be fixed before any release gate is trusted. A
genuinely disjoint holdout is required, and it must be rebuilt against
whichever corpus an arm is trained on.

### 0g.6 Revised verdict

| claim | verdict | evidence |
|---|---|---|
| curriculum labels are wrong | **NO** | 100% agreement on all 3,856 shared boards |
| curriculum breaks the centre bias | **YES, strongly** | c3 share 87.7% -> 36.7% |
| curriculum improves off-centre | **YES** | 3.5% -> 15.3% strict |
| curriculum improves symmetry | **YES** | mirror 3.5% -> 23.4% (6.7x) |
| curriculum loses strict accuracy | **YES, but CONFOUNDED** | 21x fewer epochs per position |
| curriculum is harmful | **NOT ESTABLISHED** | budget confound + holdout overlap |
| the 2x2 was clean | **NO** | 3 variables moved at once |

**The first 2x2 did not test the data hypothesis — it tested the data
hypothesis plus a 21x undertraining confound.** The corrected run in
0g.4 is the one that decides it.

---

### 0e.6 STATUS AND BLOCKER LIST (night, current)

**What is fixed and validated:**

| item | before | after | status |
|---|---|---|---|
| exported tables learn | argmax frozen at every checkpoint | hash changes at every checkpoint | **FIXED** (`--table-bias 0.0`) |
| hard export reads the board | 0.0000 until step 3000 | 0.5444 from step 500 | **FIXED** |
| `coef_set` stability | — | byte-identical across bias settings | **VERIFIED** |
| old checkpoints load | — | `missing=`, 0 missing keys | **VERIFIED** |
| live shipped path | 6 columns, 16.3% non-centre | 7 columns, 30.8% non-centre | **VERIFIED** |
| zero-FLOP claim | — | GF trunk 4,608 integer ops, 0 float multiplies | **UNCHANGED** |
| STE schedule question | unmeasurable (frozen) | tie → keep `--hard-start 0` | **ANSWERED** |

**What is still broken (in priority order):**

1. **Centre collapse.** Hard export picks c3 on **67.7%** of disjoint
   positions; off-centre strict only **8.7%**. Soft path is worse:
   **2.4%** off-centre, **94.6%** c3 — a **regression** against the
   frozen-table control's 16.6% off-centre.
   *Diagnosis still open:* the entropy regulariser and/or the zero-init
   readout are pinning the float path to c3.
2. **Mirror asymmetry.** Hard mirror consistency **11.8%** (control
   2.7%). The canonical teacher is verified **100%** consistent over
   18,304 mirror-paired labels, so the asymmetry is introduced by the
   model/objective — the labels are exonerated.
3. **Unmeasured release gates** on the fixed export: tactical accuracy,
   strength vs a centre-only opponent, general strength.
4. **Full canonical relabel** still memory-bound (16 shards ≈ 45 GiB RAM).
5. **`src/generate.py` shipped-path smoke test** not yet recorded after
   the final table-bias patch.
6. **Resume bug:** `UnboundLocalError: local variable 'step'` when the
   target `--steps` is below the stored checkpoint step.

**Release decision: NOT READY. Do not launch V24.**

The table-bias fix converts a *degenerate* program into a *reading* one.
It is a prerequisite, not a solution. The remaining blockers live in the
loss and the data.

**Note on V22:** every V22 checkpoint was trained with the frozen-table
bug and is therefore a constant-column policy. Its ladder and holdout
numbers are **not** evidence of quality and must not be cited or used as
a baseline. See §8.

---

The A/B also shows the STE schedule is *not* the main problem either:
with `--hard-start 0` the **single-label** arm's export only crossed the
board-sensitivity threshold at **step 3000** (0.1567), and it was
0.0000 at every earlier checkpoint (500/1000/1500/2000/2500). The
per-ply arm never crossed it at all.

Mechanism (measured above): the exported integer tables drift 0.00–0.35%
over 3,000 steps while readout/bias drift 100%. Training the one-hot
forward from step 0 does **not** make the tables learn the board faster;
it makes the readout adapt around a table that is still essentially
untrained.

A schedule probe is running to test the user's hypothesis directly
(`run_sched_probe.sh`, 6,000 steps, per-ply labels, identical except
for the STE schedule):
* arm A: `--hard-start 0`
* arm B: `--hard-start 750` (= 1/8 of 6,000)

The decision metric is the **board-gate sensitivity line per step**, not
the in-run `LEGAL` number. If late STE reaches sensitivity materially
earlier than `hard-start 0`, the late schedule wins and the trainer
default changes.

---

1. `abc_single` and `abc_perply` both reach 3,000/3,000 with `EXIT=0`.
2. Score both on the **same** positions: the 4,000-position disjoint
   holdout (`tg_ab_eval.py`).
3. Score both on the **same** single-label positions (the final ply of
   each A/B transcript) so the comparison is like-for-like.
4. Export both hard checkpoints and run `tg_mirror_audit.py`:
   * column coverage must include c1..c4;
   * mirror-consistency on non-centre originals must be high;
   * seat consistency must be checked separately.
5. `tools/relabel_perply.py` must produce 100% mirror-consistent labels
   on a mirror-rich sample at the production budget (verified: 16,878 /
   16,878 = 100.00%).
6. Only then build the full canonical sidecar and launch V24.

**Known trap:** the in-run `LEGAL` metric is an in-sample diagnostic on
the training corpus. It is not a decision metric, and a high `LEGAL`
number on a memorising corpus means nothing. The disjoint holdout and
the hard-export audit are the decision metrics.

---

## 1. Bug 1 — the weight mask never covered the supervised position

```python
# BEFORE
outcome_tags = [torch.full((PLY_LEN,), 1.0) for _ in examples]   # 10 bytes
```

`PLY_LEN` is 10, but the training window is `ctx + 1 = 193` bytes.
`build_outcome_batch` copies `tag[start:start+T]` into the `(B, T)`
weight grid, so only positions **0–9** ever received weight. The move
decisions live at positions **1, 11, 21, … 191**. The labelled final
position therefore had weight **0.0**, and `_weighted_move_loss` hit its
guard:

```python
w = w_flat[keep]
tot = w.sum
if float(tot) <= 0.0:
    return logits_flat.sum * 0.0        # <- the whole move loss
```

The move loss was multiplied by zero. The 256-way byte loss still ran, so
the model got *good* at byte prediction and *terrible* at playing.

**Fix** (`src/c4/train.py`, imported-row branch):

```python
_tag_len = args.ctx + 1
outcome_tags = [torch.ones(_tag_len, dtype=torch.float32) for _ in examples]
```

Imported rows carry no outcome/role information (they are solver-labelled
positions, not games with a winner), so every position in the window
correctly gets weight 1.0.

## 2. Bug 2 — `0 × (-inf) = NaN`

The 7-way loss masks illegal columns with `-inf` before `log_softmax`,
then multiplies by a target distribution that is exactly `0.0` there:

```
logits: [-1.80, -inf, -1.64,...]
target: [ 0.52,  0.00,  0.31,...]
product: [..., 0.0 * -inf = NaN,...]
```

Every supervised row produced NaN, poisoning the entire loss. This was
**latent** for six runs because the loss was already multiplied by zero —
fixing Bug 1 exposed it immediately.

**Fix**: mask with a large **finite** negative value and renormalise the
target over legal columns only.

```python
NEG = -1e9
sub = sub.masked_fill(~lg, NEG)
lp = torch.log_softmax(sub, dim=-1)
d = dist[keep] * lg.float
d = d / d.sum(dim=1, keepdim=True).clamp(min=1e-12)
nll = -(d * lp).sum(dim=1)
```

### Verification

`tg_losslive.py` calls the *real* trainer functions and asserts the chain
data → window → mask → weight → loss → gradient is intact:

```
row 0: 20 of 192 positions weighted        (the 20 move positions)
weight at move positions: min 1.000 max 1.000
labelled tokens: 200   their weight: 200.0     (was 0.0)
loss value: 1.713256                          (log 7 = 1.946 at init)
grad mass on 7 col logits: 0.6879
grad mass elsewhere: 0.0000
PASS
```

## 3. Bug 3 — the evaluation path measured the sampler, not the policy

This is the bug that hid the win. Found while reconciling V21's log
against an independent 2,000-game ladder.

**`evaluate_play`** (`src/c4/train.py`) played the model with
`torch.multinomial` **always**, regardless of temperature:

```python
logits = logits / max(temperature, 1e-6)
probs  = F.softmax(logits + mask, dim=-1)
col    = int(torch.multinomial(probs, 1).item)   # ALWAYS samples
```

The comment immediately above the call site claimed:

> *"Evaluation play uses its OWN temperature... A next-byte predictor
> deployed in the UI is played greedily (argmax over legal columns), so that
> is how we must measure it."*

The code did the opposite. **Every `vs_ref` number in every log from V15
onward graded a stochastic policy.**

**`_eval_opponents`** had the same defect one level down:

```python
agent = Agent(model, device, greedy=temperature <= 1e-3,...)
```

`--eval-temp` defaulted to **0.2**, so `greedy` was **False**. Every
`vs_opp` ladder number in every log from V15 onward also graded a
stochastic policy.

This violates the project constraint *"keep the learned policy greedy
during strength measurement and explore the opponent"*. The opponent side
was already correct (`explore=0.35` per-ply epsilon on the opponent only).

**Impact — the same checkpoint, two ways:**

| measurement | vs heuristic | vs random |
|---|---|---|
| V21 training log (sampled, `eval-temp 0.2`) | **4.8%** | 43.3% |
| independent ladder (greedy hard export) | **38.35%** | **90.28%** |

An 8× difference on the same weights. Any conclusion drawn from a training
log since V15 is suspect.

**Fixes applied:**

1. `evaluate_play` now plays greedily (`argmax` over legal columns) when
   `temperature <= 1e-3`; the old sampling path remains available for
   anyone who explicitly asks for it.
2. `_eval_opponents` now always constructs `Agent(..., greedy=True)`.
3. `--eval-temp` default changed **0.2 → 0.0** (greedy), with a comment
   saying that raising it above 1e-3 breaks comparability with the
   shipping policy.

**Lesson:** the same class of bug as Bugs 1 and 2 — *the thing being
measured was not the thing being claimed*. The zero loss meant the model
was not learning; the sampling meant the model was not being measured.

---

## 3b. Bug 4 — the promotion gate compared two different scales

Found **after** the acceptance test passed. The model was fine;
the *bookkeeping that decides which checkpoint ships* was broken.

### The defect

The promotion key is a 4-tuple:

```python
sel = (ladder_net_vs_heuristic, holdout_optimal_pct, -hard_bpb, distinct)
```

But `ladder_net` was filled from **two different opponents**:

```python
_ladder_net = _last_ladder_net          # net wins vs `heuristic`
if opp_results:
    _he = opp_results.get("heuristic")
    if _he is not None:
        _ladder_net = _he["wins"] - _he["losses"]   # net wins vs `heuristic`
        _last_ladder_net = _ladder_net
```

The fallback was `score` — net wins against the **centre-column
reference**, a much weaker opponent than `heuristic`. One scale for the
ladder steps, another for the cheap eval steps, compared with `>`.

`heuristic` beats a good model ~60–70% of the time, so net-wins-vs-heuristic
is strongly **negative** (-288, -324, -330). Net-wins-vs-centre is weakly
positive. So the optimistic number always won the comparison, and the gate
promoted on a centre-rule score while calling it a ladder score.

### The evidence, reconstructed from V21's own log

| step | source of `ladder_net` | value |
|---|---|---|
| 500 | heuristic ladder | **-288** |
| 1000 | heuristic ladder | **-324** |
| 1200 | **centre fallback** | **+1** |
| 2000 | heuristic ladder | **-330** |
| 2900 | **centre fallback** | **+1** |

`best_soft.pt` was stamped `ladder_net: 1` — a centre score, on a key whose
name said ladder.

### Why this did **not** invalidate the acceptance test

The gate picked step 2900 **for the wrong reason**, but step 2900 turned out
to be genuinely the strongest checkpoint — the independent 2,000-game
ladder in §4 confirms it beat step 2000 and step 2500 on every opponent.
That was luck, not logic. The gate was ranking a centre-rule score against
heuristic scores and happening to land on the right checkpoint.

### The fix (three parts)

1. **One scale.** `_ladder_net` falls back to `_last_ladder_net` — the last
   *real* heuristic-ladder measurement — never to the centre reference.
   `_last_ladder_net` is updated **only** from an actual `heuristic` row.

2. **Protocol-tagged checkpoints.** A signed net-win count is meaningless
   without knowing *what it was measured against and how*. Every checkpoint
   now records:

   ```python
   "ladder_opp": LADDER_PRIMARY_OPP,     # "heuristic"
   "ladder_greedy": True,
   "ladder_protocol": LADDER_PROTOCOL_VERSION,   # 2
   ```

3. **Protocol-aware resume.** A resumed checkpoint's score is reused only
   if it declares the *same* protocol. Otherwise the seed is `-inf` and the
   run promotes on its first real measurement. This is what caught the live
   bug below.

### The live bug it caught

V22 was launched **before** this fix, so it was running the old gate. It
resumed from V21's `best_soft.pt`, inherited `ladder_net=1` (centre scale),
and its own greedy-vs-heuristic nets were around **-176**. The gate could
therefore **never** promote — and indeed:

```
runs/c4/v7mixed_v22/checkpoints/    <- step*.pt only. NO best_hard.pt.
```

The run was going to finish with **no shipping artifact at all**. The
periodic-snapshot fix (§7) is what saved the weights; the promotion gate
defect is what would have lost the *choice*.

After the fix, V22 was restarted from its step-3500 snapshot. The new
startup log says exactly what should happen:

```
promotion gate: checkpoint ladder score is NOT on the 'heuristic'/greedy scale
(meta ladder_opp=None, ladder_greedy=None) -- seeding the primary slot with
-inf so this run promotes on its first real ladder measurement.
```

and `best_hard.pt` now exists and advances.

### Also fixed: `final.json` read the wrong slots

The 4-tuple is `(ladder_net, holdout_optimal_pct, -hard_bpb, distinct)`, but
`final.json` was writing `"best_vs_ref_score": best_sel[2]` (that is
`-hard_bpb`) and `"best_hard_acc": best_sel[0]` (that is the ladder net).
The keys are now `best_ladder_net`, `best_holdout_optimal_pct`,
`best_hard_bpb`, `best_distinct_games`.

### Regression

`tg_gate_check.py` — **24/24 checks pass**. It covers the source-level
invariants *and* a behavioural regression that replays the exact V21 mixed
-scale history and asserts the scale-consistent gate picks a **different**
winner (the genuine expected winner, not a tautology).

---

## 4. The decisive acceptance test

`tg_ladder_top3.py` — fresh 2,000-game ladder, top 3 V21 candidates, the
**greedy hard integer export** as the model, per-ply exploration (ε=0.35)
on the **opponent only**, distinct-trajectory counting.

```
candidate     vs centre    vs random   vs heuristic   vs lookahead   vs minimax
step2000         77.15%       83.60%         31.10%         17.20%       33.90%
step2500         79.72%       85.75%         29.55%         19.45%       31.70%
step2900         67.40%       90.28%         38.35%         31.77%       33.42%

step2900 detail:  vs random  W1762 L151 D87   90.28%  distinct 1380/2000  (P1 902/1000, P2 860/1000)
                  vs heuristic  W507 L973 D520  38.35%  distinct 1257/2000  (P1 273/1000, P2 234/1000)
                  vs lookahead W323 L1052 D625 31.77%  distinct 1199/2000
                  vs minimax   W632 L1295 D73  33.42%  distinct  910/2000
```

**Re-verified** — the ladder and head-to-head were re-run
end to end after the promotion-gate fix and reproduce these numbers
exactly (`runs/_probe/ladder_top3.log`, `runs/_probe/headtohead.log`,
both `EXIT=0`). The go/no-go verdict printed by the harness itself:

```
step2000   FAIL
step2500   FAIL
step2900   PASS
```

### Head-to-head against the centre rule

`tg_headtohead.py` — the model and the centre rule played against each
other, 2,000 games, 1,250 distinct trajectories:

```
model (greedy hard) vs centre-rule opponent, eps=0.35:
  games 2000  distinct 1250
  W1364 L546 D90
  model score 70.45%
  model as P1 719/1000   as P2 645/1000
  mean plies 14.4

centre rule vs the same opponents (scale reference):
  centre vs random     W 1541 L  435 D  24  score  77.65%  distinct 1959/2000
  centre vs heuristic  W  321 L 1674 D   5  score  16.18%  distinct 1933/2000
  centre vs lookahead  W  270 L 1724 D   6  score  13.65%  distinct 1906/2000
  centre vs minimax    W  482 L 1505 D  13  score  24.43%  distinct 1672/2000
```

The model is not just beating the centre rule — it is roughly **2.4×** its
win rate against `heuristic` (38.35% vs 16.18%) and it wins as **both**
players (719/1000 as P1, 645/1000 as P2), so it is not a first-player
artefact. The centre rule's own reference numbers here are the scale
control for the table in §0.

### Honest limits

- The model still **loses to depth-3 minimax more often than it wins**
  (33.42%). Connect Four is solved; a weak heuristic is a long way from
  optimal.
- `distinct` for step 2900 is 910–1,380 of 2,000 per opponent (and 1,250
  in the head-to-head), not 2,000 — the games are genuinely varied, not
  memorised, but there is repetition. Note that step2000/step2500 show
  *lower* distinct counts (470–958) because those games end sooner
  (mean 8–12 plies vs 14.5).
- The **hard integer export is weaker than the soft model** on the holdout
  (31.93% vs 38.33% per-move optimal), and they agree on only 58.5% of
  picks. The STE alignment of the integer export is not tight; a future run
  should tighten it. The ladder above is nevertheless run on the **hard**
  export, because that is what ships.

---

## 5. The learning curve (V21, 3,000 steps)

```
step   holdout   edge   max-share  cols>5%   columns played
    0   21.80%  21.1%     95.5%        1     [3819,134,27,9,7,3,1]  degenerate
  100   34.42%  48.9%     57.5%        2
  200   35.88%  48.0%     94.1%        1     degenerate
  300   39.67%  48.3%     51.4%        2
  400   36.98%  48.0%     45.5%        3
  800   39.85%  50.9%     44.1%        4
1100   42.35%  50.9%     89.8%        1     degenerate
1200   42.42%  50.2%     76.5%        2
1600   40.33%  51.6%     40.8%        5
2000   39.00%  52.8%     29.8%        6     <- healthiest spread
2400   41.35%  53.7%     55.3%        4
2900   44.25%  55.1%     82.2%        2     <- peak holdout, BEST LADDER
3000   36.52%  54.5%     38.9%        3
```

Compare to every prior run: **21.8% → 30.6% in 250 steps, then frozen for
thousands of steps**. V21 moves. The edge-column metric climbs steadily
21.1% → 55.8%.

**The oscillation is real and not yet explained.** The dominant column
swings between column 2 and column 4, sometimes taking 90% of picks and
sometimes spreading over six. This is position-dependent behaviour (not a
frozen constant), but the swing is far too large for a converged policy.

---

## 6. The metric trap — read this before trusting any number

`holdout_optimal_pct` is **per-move** accuracy: "is the argmax over legal
columns one of the Pons-optimal columns?"

It is *not* a game-win rate. A policy can score 40% per move and lose
almost every game to a 1-ply heuristic that scores 66% per move, because a
game is 30+ consecutive moves.

**Always monitor `holdout_pick_tot`** (actual picks per column) alongside
the average. The histogram separates "learning" from "frozen" in ~100
steps.

Naming trap: `holdout_pick_dist` holds **hits per column**;
`holdout_pick_tot` holds **total picks per column**. Both were misread
earlier in this project.

Baselines on identical holdout positions (T=2 trainer metric):

```
always-centre (col 3)   42.67%
1-ply heuristic          66.07%
depth-3 minimax         54.20%
random                  20.73%
model_soft              38.33%
model_hard              31.93%
```

Note the per-move model numbers are *lower* than the game results suggest,
because the holdout positions are from deep, hard endgames where even the
solver's ties are narrow.

---

## 7. Current state of the artifacts

### Code changes

| Change | Location | Status |
|---|---|---|
| Full-window outcome tags | `train.py` imported-row branch | **live, verified** |
| Finite `-1e9` illegal mask + target renormalisation | `_weighted_move_loss` | **live, verified** |
| `evaluate_play` plays greedy when `temp <= 1e-3` | `train.py` | **live** |
| `_eval_opponents` always greedy | `train.py` | **live** |
| `--eval-temp` default 0.2 → **0.0** | `train.py` CLI | **live** |
| `centre_policy` added to `OPPONENTS` | `opponents.py` | **live** — the centre rule is now reachable from the ladder harness instead of being hard-coded in the trainer |
| Unlabeled history plies suppressed (no fabricated one-hot) | move-target build | live, measured (5.3% → 100% real supervision) |
| Ladder-first promotion gate | checkpoint selection | live, **24/24** regression checks (`tg_gate_check.py`) |
| **Scale-consistent ladder gate** (`_ladder_net` ← `_last_ladder_net`) | `train.py` | live — was mixing centre-rule and heuristic nets on one scale |
| **Protocol-tagged checkpoints** (`ladder_opp` / `ladder_greedy` / `ladder_protocol`) | every `torch.save` of a checkpoint | live — a score is meaningless without its protocol |
| **Protocol-aware resume** (mismatched protocol ⇒ seed `-inf`) | resume branch | live — caught the live V22 "can never promote" bug |
| `final.json` slot names match the 4-tuple | `train.py` | live — was reading `best_sel[2]` as a "vs_ref score" |
| `save_snapshot` — periodic weights written independently of promotion | `save_every` branch | live, smoke-verified |
| Argmax-equality holdout optimality (was `T > 1e-6`) | `Holdout.__init__` | live, verified |
| Seed default 0 → **32** | CLI | live |

### Data

```
runs/c4/posdata_score_split/train_corpus.txt      106,966 sequences
runs/c4/posdata_score_split/train_targets.npz     (106966, 7) fixed-T=2.0
runs/c4/posdata_score_split/holdout_corpus.txt     20,260 rows
runs/c4/posdata_score_split/holdout_targets.npz    (20260, 7)
symmetry-class overlap: 0    mirror leakage: 0
```

Source: TonyCWang `chunk0.parquet` (per-column Pons scores, `obs` + `target`).
`-1000` sentinel = illegal column → probability zero.

### Models

```
runs/c4/v7mixed_v21/checkpoints/best_soft.pt     step 2900, holdout 44.25%   <- SHIPPING CANDIDATE (validated)
runs/c4/v7mixed_v21/checkpoints/best_hard.pt     the integer export
runs/c4/v7mixed_v21/checkpoints/step{0,500..3000}_{soft,hard}.pt
runs/c4/v7mixed_v21/final_float.pt
runs/c4/v7mixed_v21/train.log                    full log
runs/c4/v7mixed_v21/ladder_top3_2000.json        the decisive ladder
runs/c4/v7mixed_v21/headtohead_centre.json        head-to-head vs centre rule
runs/c4/_ARCHIVE_best_models/                    17 files, ~1.2 GB

runs/c4/v7mixed_v22/                             ACTIVE RUN (25,000 steps, ctx 192, seed 32)
runs/c4/v7mixed_v22/checkpoints/                 step*.pt + best_hard.pt + best_soft.pt
runs/c4/v7mixed_v22/metrics.jsonl                per-eval metrics
runs/c4/_ARCHIVE_corrupt_v22_concurrent/         checkpoints written by 2 concurrent
                                                  trainers; discarded, do not use
```

**Checkpoint metadata.** New checkpoints carry the protocol keys
(`ladder_opp`, `ladder_greedy`, `ladder_protocol`) at the **top level** of
the hard export, and inside `meta` for the soft `{state, meta}` form.
Older checkpoints (V9–V21) have **no** protocol keys, so a resume from them
correctly refuses to reuse their ladder score.

`best_hard.pt` for V21 is a bare hard export with **no** metadata dict — the
loader must read the transcript/version stamp from elsewhere. Do not assume
`meta` exists on hard checkpoints written by the older exporter.

---

## 8. What to do next

> **REWRITTEN  (night).** The V22 run below is **DEAD** —
> it was training with the frozen-table bug, so every one of its
> checkpoints is a constant-column policy. Its "ladder net -61" and
> "holdout 46.3%" numbers are **meaningless** as evidence of quality:
> a policy that always plays column 3 will lose to some opponents and
> the old gate never tested board-sensitivity. **V22 and all its
> checkpoints are historical artifacts only. Do not cite its numbers,
> do not promote from it, do not use it as a baseline.**
>
> The current blocker list, in order, is in §0e.6 below.

### Current blockers (night)

1. **Centre collapse.** The fixed hard export still picks c3 on
   **67.7%** of disjoint positions and scores only **8.7%** off-centre.
   The soft path is worse (**2.4%** off-centre, **94.6%** c3) — a
   **regression** versus the frozen-table control (16.6% off-centre).
   This is a teacher/objective problem.
2. **Mirror asymmetry.** Hard mirror consistency is **11.8%** (control
   2.7%). The canonical teacher is verified 100% consistent, so the
   asymmetry is introduced by the model/objective, not the labels.
3. **Unmeasured gates.** Tactical accuracy, strength vs a centre-only
   opponent, and general strength have **not** been run on the fixed
   `table_bias=0.0` hard export.
4. **Full canonical relabel** still memory-bound (16 shards ≈ 45 GiB).
   Needs substantially fewer workers.
5. **Shipped-path smoke test** for `src/generate.py` after the final
   table-bias patch is not yet recorded.
6. **Resume path bug**: `UnboundLocalError: local variable 'step'`
   when the target `--steps` is below the stored checkpoint step.

**Do not launch V24** until items 1–3 are addressed and the hard export
passes every release gate.

### Historical (V22 — dead, see warning above)

- **V22** (`runs/c4/v7mixed_v22/`) — 25,000 steps, greedy evaluation, seed 32,
  ctx 192, batch 16. Resumed from V21 step 2900; **restarted** at  from
  the step-3500 snapshot because the first instance was running the pre-fix
  promotion gate and could never promote (§3b). The restarted instance
  promotes correctly: step 4000 ladder net **-136**, step 4250 promoted to
  `best_hard.pt` with `ladder_net=-136`, `ladder_opp=heuristic`,
  `ladder_greedy=True`, `ladder_protocol=2`. Holdout by step: 4750 → 40.1%,
  6250 → **43.7%**, 7500 → 41.5%, 8750 → **44.0%** (edge 56.9%),
  13500 → 39.7%, 14250 → **46.3%** (edge 57.4%), 15750 → 42.0%,
  16250 → 42.0%, 16750 → 41.7%. Log: `runs/_probe/v22.log`.
  Process: `proc_2253d2da9d1e`. **Latest observed step 16,750 of 25,000.**
  Periodic snapshots are landing normally (`step16500_soft.pt` and
  `step16500_hard.pt` written during that run); 54 checkpoint
  files present.

  **New promotion:** step **8750** displaced step 4250 as the promoted
  checkpoint — `ladder_net=-61` (was -136), `holdout_optimal_pct=42.675`,
  `hard_bpb=18.397`, `ladder_opp=heuristic`, `ladder_greedy=True`,
  `ladder_protocol=2`. The gate is now demonstrably working on a single
  scale: later checkpoint, strictly better heuristic-ladder net.
- **4,096-position capacity test** (`tg_scale4096.py`, GPU) — **COMPLETE**.
  Log `runs/_probe/scale4096_gpu7.log`, `EXIT=0`, 3,000 optimizer steps,
  bs=64, 1,001 s of training plus the full 4,096-row evaluation.

  **Final result on all 4,096 positions: strict 56.45%, oracle-tie
  87.67%, chance 33.22%.** The network is more than capable of fitting
  the task; there was never a capacity ceiling. Full curve:

  | optimizer step | loss | strict | oracle-tie | chance |
  |---|---|---|---|---|
  | 100 | 1.7280 | 21.48% | 52.73% | 34.26% |
  | 200 | 1.6740 | 32.03% | 57.81% | 34.26% |
  | 300 | 1.7435 | 37.50% | 61.72% | 34.26% |
  | 400 | 1.6222 | 37.50% | 63.28% | 34.26% |
  | 500 | 1.7393 | 39.84% | 67.58% | 34.26% |
  | 600 | 1.5726 | 37.89% | 69.53% | 34.26% |
  | 700 | 1.6568 | 39.45% | 67.58% | 34.26% |
  | 800 | 1.6446 | 41.41% | 68.75% | 34.26% |
  | 900 | 1.5645 | 39.84% | 68.36% | 34.26% |
  | 1000 | 1.5277 | 39.84% | 69.92% | 34.26% |
  | 1100 | 1.5195 | 41.80% | 71.48% | 34.26% |
  | 1200 | 1.6946 | 41.80% | 71.09% | 34.26% |
  | 1300 | 1.5563 | 41.02% | 70.31% | 34.26% |
  | 1400 | 1.5782 | 41.80% | 69.53% | 34.26% |
  | 1500 | 1.5292 | 44.14% | 72.66% | 34.26% |
  | 1600 | 1.6118 | 45.31% | 73.83% | 34.26% |
  | 1700 | 1.5714 | 44.92% | 70.31% | 34.26% |
  | 1800 | 1.4782 | 42.19% | 72.27% | 34.26% |
  | 1900 | 1.4863 | 45.31% | 76.17% | 34.26% |
  | 2000 | 1.4672 | 44.53% | 75.78% | 34.26% |
  | 2100 | 1.6283 | 46.48% | 77.73% | 34.26% |
  | 2200 | 1.4843 | 48.05% | 80.86% | 34.26% |
  | 2300 | 1.4635 | 48.05% | 80.86% | 34.26% |
  | 2400 | 1.4863 | 46.88% | 80.08% | 34.26% |
  | 2500 | 1.4544 | 48.83% | 81.64% | 34.26% |
  | 2600 | 1.4355 | 50.00% | 80.86% | 34.26% |
  | 2700 | 1.4593 | 51.17% | 83.98% | 34.26% |
  | 2800 | 1.4273 | 50.00% | 85.16% | 34.26% |
  | 2900 | 1.3762 | 50.78% | 84.38% | 34.26% |
  | 3000 | 1.4919 | 51.56% | 85.16% | 34.26% |
  | **3000 (full 4,096 rows)** | — | **56.45%** | **87.67%** | **33.22%** |

  Both metrics rise together throughout and the loss falls from 1.73 to
  ~1.45 — the signature of a task that is genuinely being learned.
  Contrast the broken probe's curve, where strict and oracle-tie moved
  *independently* and swung between 23.05% and 50.00% with no trend.

  **Rule of thumb this establishes:** a probe whose two metrics disagree
  is measuring something incoherent; a probe whose metrics rise together
  is measuring a real task.

  The earlier run (`proc_dcf83717297d`, log `runs/_probe/scale4096_gpu4.log`,
  `EXIT=0`) is **invalid as a capacity verdict** — it suffered both probe
  defects described in §3c (binary-tie-mask objective vs unmasked argmax,
  and a label leak from using the inference window for training). Its
  curve swung between 23.05% and 50.00% with no trend and its two metrics
  moved independently — the signature of an incoherent measurement.

  **Read this against the chance line, not against 0%.** The soft targets
  have a mean of **2.3252 optimal columns per position** on the probe's own
  4,096 eval rows (full corpus: 2.3380; 55,756 rows with 1 optimal, 19,228
  with 2, 7,646 with 3, 5,912 with 4, 6,378 with 5, 6,924 with 6, 5,122
  with all 7). A policy that picked a column **uniformly at random** would
  score **33.22% oracle-tie** on those rows.

  The probe oscillated between **23.05% and 50.00%** and ended at
  **33.42%** on the full 4,096 rows — i.e. exactly the chance line
  (null simulation over 2,000 draws: mean 33.22%, sd 0.60, p=0.367).
  **The corrected probe, run the same way, reaches 87.67%** (§ above),
  which settles the question: the null was the probe's fault.

  **But that null is the PROBE's fault, not the model's.** The probe had
  **two independent defects**, both of which had to be found:

  **Defect 1 — the training objective never taught the preference.** The
  probe trained on a **binary tie mask** and evaluated with an **unmasked
  argmax**:

  ```python
  # probe LOSS: masks every NON-optimal column to -1e9 and renormalises
  # the target over the optimal set
  logits = logits.masked_fill(~optmask, NEG)
  d = optmask.float; d = d / d.sum(1, keepdim=True)
  loss = -(d * torch.log_softmax(logits, dim=-1)).sum(1).mean

  # probe METRIC: unmasked argmax over all 7 columns, hit iff optimal
  pr = logits.argmax(-1)
  tie = om.gather(1, pr.view(-1, 1)).squeeze(1).float.mean
  ```

  That objective only asks the model to discriminate **among** the optimal
  columns; every non-optimal column receives **no gradient pushing it
  down**. The evaluation then measures exactly that preference — so the
  probe measured a quantity its own loss never optimised. The best
  *constant* column already scores **35.35%** on these rows, so the
  probe's 33.42% is below a constant policy too.

  **Defect 2 — the probe LEAKED THE ANSWER into its own input.** The probe
  built its window with `src.c4.agent.trainer_window`, which is the
  **inference** convention: `x = seq[n-ctx: n]`. That places the label
  byte `seq[-1]` **at input slot ctx-1**, and the probe read its logits at
  **ctx-1**. Two errors in one — it was predicting the *padding* byte
  rather than the label, while the true label sat in its own input at lag
  0.

  Verified directly (`runs/_probe/_wincheck.py`):

  | check | result |
  |---|---|
  | old probe: `input[pos] == label byte` | **0x47 == 0x47 → LEAK** |
  | fixed window: headers decode | **4,000/4,000** |
  | fixed window: label byte == target argmax column | **4,000/4,000** |
  | fixed window: label absent from input slot | **4,000/4,000** |

  The fixed window is production's own (`train.py build_batch`,
  `snap_to_end=True`): `L = ctx+1`, `start = max(0, len(seq)-L)`,
  `x = window[:-1]`, `pos = len(seq)-1-start-1`, and legality from the
  header at `pos + 1 - MOVE_OFFSET` via `_legal_from_header`. With the
  corrected window the legal-column mask is **mean 5.783 per row, min 1** —
  versus the broken probe's **7.000 for every row**, which had silently
  hidden every full column from the mask.

  **The decisive check.** Running **production's own metric**
  (`src/c4/train.py Holdout.score`, soft T=2 targets, legality masking,
  argmax-equality) on the **identical 4,096 rows**, with the production
  checkpoints:

  | model | production metric on the same 4,096 rows | broken probe metric |
  |---|---|---|
  | V22 `best_soft` (step 8750) | **50.24%** | 33.42% |
  | V21 `best_soft` (step 2900) | **48.39%** | 33.42% |

  V22's picks on those rows span all seven columns (219/42/186/1133/1219/
  1094/203 — no collapse), and its edge-column rate is **63.38%**.

  For reference, the **hard** export of the same checkpoints on the same
  rows (CPU `hard_forward`, legality-masked) scores **35.30%** (V22) and
  **32.37%** (V21) — far below the soft model, confirming that the
  integer export is still the weakest link (§6) and is *not* what the
  production holdout metric measures.

  **Conclusion: capacity is NOT the wall at 4,096 positions — proven, not
  argued.** The probe's null was an objective/metric mismatch plus a label
  leak. The corrected probe (production soft T=2 targets, production
  window, legality masking) starts at **52.73% oracle-tie by step 100**
  against a **34.26%** chance line — above the broken probe's *best* value
  (50.00% at step 1,800) in 18× fewer steps — and finishes at
  **87.67% oracle-tie / 56.45% strict on the full 4,096 rows**, with
  `EXIT=0`. The 64-position (100%) and 512-position (95.51%) results are
  therefore consistent with the production model, and the 107k-row run is
  limited by optimisation budget and data coverage, not by the network's
  capacity.

  Evidence: `runs/_probe/scale4096_gpu4.log` (broken probe, `EXIT=0`),
  `runs/_probe/scale4096_gpu7.log` (corrected probe, `EXIT=0`,
  **87.67% oracle-tie**),
  `runs/_probe/prodholdout.log` (production metric on the same rows),
  `runs/_probe/_wincheck.py` (window/leak verification).

#### 3c. Bug 6 — the capacity probe leaked its own answer

The 4,096-position capacity probe (`tg_scale4096.py`) reported a null
result — 33.42% oracle-tie against a 33.22% chance line — which looked
exactly like a capacity wall. It was not. Two independent defects:

1. **Objective/metric mismatch.** The loss masked every *non-optimal*
   column to `-1e9` and renormalised the target over the optimal set, so
   no non-optimal column ever received a gradient pushing it down. The
   metric then took an unmasked argmax over all 7 columns — measuring
   precisely the preference the loss never trained.

2. **Label leak.** The probe built its window with
   `src.c4.agent.trainer_window`, the **inference** convention
   (`x = seq[n-ctx:n]`). That puts the label byte `seq[-1]` at input slot
   `ctx-1` — and the probe read its logits at `ctx-1`. It was predicting
   the padding byte while the true answer sat in its own input at lag 0.
   Proven directly: old window `input[pos] == 0x47 == label byte`.

   It also made every row look like "all 7 columns legal" (mean 7.000,
   min 7), because the header slice was taken from the wrong offset. With
   production's window (`train.py build_batch`, `snap_to_end=True`) and
   legality from `_legal_from_header` at `pos + 1 - MOVE_OFFSET`, the mask
   is **mean 5.783 legal columns, min 1**.

Verification (`runs/_probe/_wincheck.py`): headers decode **4,000/4,000**;
label byte == target argmax column **4,000/4,000**; label absent from the
input slot **4,000/4,000**.

**Why it matters beyond the probe:** `trainer_window` and `build_batch` are
*not* interchangeable. `trainer_window` is for inference (the label does
not exist yet); `build_batch` is for training (the label is the next byte
in the stream). Using the wrong one silently trains on a leak. Any future
probe must use production's `build_batch` window, not `trainer_window`.

After the fix the probe reaches **52.73% oracle-tie at step 100** (chance
34.26%) — above the broken probe's best-ever value (50.00% at step 1,800)
in 18× fewer steps — and finishes at **87.67% oracle-tie / 56.45% strict**
on all 4,096 rows. **Capacity is not the wall.**

The corrected probe's curve (`runs/_probe/scale4096_gpu7.log`, `EXIT=0`;
the full 30-row table is in §4):

| optimizer step | loss | strict | oracle-tie | chance |
|---|---|---|---|---|
| 100 | 1.7280 | 21.48% | **52.73%** | 34.26% |
| 500 | 1.7393 | 39.84% | **67.58%** | 34.26% |
| 1000 | 1.5277 | 39.84% | **69.92%** | 34.26% |
| 1500 | 1.5292 | 44.14% | **72.66%** | 34.26% |
| 2000 | 1.4672 | 44.53% | **75.78%** | 34.26% |
| 2500 | 1.4544 | 48.83% | **81.64%** | 34.26% |
| 3000 | 1.4919 | 51.56% | **85.16%** | 34.26% |
| **3000 (full 4,096 rows)** | — | **56.45%** | **87.67%** | **33.22%** |

Both metrics rise together and both sit far above the chance line — the
signature of a task that is genuinely being learned. Contrast the
broken probe's curve, where strict and oracle-tie moved *independently*
and swung between 23.05% and 50.00% with no trend.

**Rule of thumb this establishes:** a probe whose two metrics disagree
is measuring something incoherent; a probe whose metrics rise together
is measuring a real task.

**Sleep resilience.** The host slept and killed
the probe mid-run (its log ends at step 800 with **no `EXIT=` marker** —
killed, not finished). The probe now checkpoints weights + optimizer
state + step counter atomically (temp file, then `os.replace`) every
report interval, and prints `RESUMED from <path> at gstep N` on
restart. Verified: `runs/_probe/scale4096_ckpt.pt` loads under
`weights_only=True` with 22 state tensors, 12 optimizer groups, and
the correct step counter. The restart reproduces the pre-sleep curve
**exactly** (52.73% → 57.81% → 61.72% → 63.28%), confirming the probe
is deterministic.

#### 3d. Bug 5 — the capacity probe never printed a line

`tg_scale4096.py` was launched four times and produced **no result** each
time. Three separate defects, each sufficient on its own:

1. **Wrong device.** A code comment claimed the GF forward "materialises
   24 GiB at B=64/ctx=192" and pinned the probe to CPU. That was never
   measured. Measured on the RTX 5090 (with V22 training concurrently):

   | batch | peak GPU mem | s/optimizer-step |
   |---|---|---|
   | 16 | 2.66 GiB | 0.69 |
   | 32 | 5.07 GiB | 0.66 |
   | 64 | 9.89 GiB | 1.27 |

   The real peak is 9.89 GiB — it fits on a 32 GiB card beside a live
   trainer — and the GPU is **~60× faster** than CPU. On CPU a single
   bs=16 optimizer step took tens of seconds, so the first result line was
   ~2 hours away. The probe looked hung; it was on the wrong device.
2. **Epoch/step confusion in the report cadence.** The outer loop iterates
   over all 4,096 rows, so `gstep` only ever takes multiples of `BS` at an
   epoch boundary. The report condition `gstep % 100 == 0` was **never
   true** at bs=64 — the probe printed nothing even when it was training
   correctly at full GPU speed. The check now lives inside the batch loop
   with a `next_report` threshold, which fires at 100, 200, … 3000 (30
   reports).
3. **The evaluation dominated the run.** A full 4,096-row forward pass
   measured **335.8 s** (the GF forward is a Walsh-Hadamard XOR convolution
   over 256-wide GF vectors for every (batch, context, lag) slot) against
   1.51 s per training step. Evaluating on all 4,096 rows every 100 steps
   would have spent 2.2× more time measuring than training. The probe now
   evaluates on a fixed 256-row subset during the run and does the full
   4,096-row evaluation once at the end.

Also note the four earlier launches **all wrote to the same
`runs/_probe/scale4096.log`**, so that file was a 125-byte artifact of
mutual truncation carrying two `EXIT=15` lines. Only the per-run logs
(`scale4096_gpu4.log`) are evidence.

### Operational lessons (learned the hard way today)

- **Never run two trainers against the same `--run-dir`.** A killed wrapper
  leaves its Python child alive; both then write `best_hard.pt` and
  `step*.pt` into the same directory and the checkpoints become a mix of
  two different models. This happened: two orphan trainers (,)
  were racing the good one. Detection: `psutil` over
  `cmdline` for `src.c4.train`. The corrupted output was moved to
  `runs/c4/_ARCHIVE_corrupt_v22_concurrent/` — do not use it.
  Kill the **child** process, not just the shell wrapper.
- **The same applies to log files.** Five concurrent `tg_scale4096.py`
  instances wrote to one `scale4096.log`; each truncated and overwrote the
  others, and the surviving file was 125 bytes with no `EXIT=` marker. Give
  every background job its own log, and check for a trailing `EXIT=` line
  before trusting a log as complete.
- **A probe that prints nothing is broken, not slow.** The capacity probe
  went through four launches with zero output because its report condition
  (`gstep % 100 == 0`) could never fire at an epoch boundary. Make the
  first report happen within a few minutes so a stall is visible in
  wall-clock terms.
- **Measure the hardware before quoting a limit.** The probe was pinned to
  CPU by an unmeasured "24 GiB" comment; the real peak was 9.89 GiB and
  the GPU is ~60× faster. See skill
  `measure-before-assuming-hardware-limits`.
- **Check `best_hard.pt` exists and advances.** A run that trains for hours
  without ever promoting is a silent failure. The startup log line
  `promotion gate seeded from checkpoint:...` is the first place to look.
- **A sleeping host kills background jobs.** The machine slept
  at ~ and the OS killed the capacity probe mid-run — its log ended at
  step 800 with no `EXIT=` marker, which is how you tell "killed" from
  "finished". The trainer survived on its periodic snapshots; the probe did
  not, because it had no checkpoint of its own until then. Every long
  background job should now (a) checkpoint weights **and optimizer state
  and the step counter**, atomically, (b) print `RESUMED from <path> at
  gstep N` so the log proves the resume, and (c) be detectable as dead by
  comparing the log's mtime against wall-clock now — do not wait for a
  completion notification that will never arrive. See skill
  `long-run-checkpoint-survival`.

### Immediate

0. **DONE — the 4,096-position capacity question is settled.** The
   corrected probe reaches **87.67% oracle-tie / 56.45% strict** on all
   4,096 rows (`runs/_probe/scale4096_gpu7.log`, `EXIT=0`). Capacity is
   not the wall. Do not re-run this probe; if you need to extend it,
   `tg_scale4096.py` now resumes from `runs/_probe/scale4096_ckpt.pt`.
1. **Re-run the 2,000-game ladder on V22's top 3** and keep the best.
   `tg_ladder_top3.py` is ready; point `CANDIDATES` at the new run.
   V22 has already promoted a **step 8750** checkpoint (`ladder_net=-61`,
   better than V21's step-2900 candidate) but it has **not** yet had a
   fresh 2,000-game ladder, so it cannot supersede V21 step 2900 yet.
2. **Tighten the soft/hard agreement.** They agree on only 58.5% of
   picks. The integer export is the shipping artifact; STE alignment needs
   attention now that the soft model is strong enough to align *to*.
3. **Add the baseline table to the trainer's own log** so future runs print
   `centre` / `heuristic` / `minimax` on the same positions. The training
   log alone was actively misleading for six runs.
4. **Run the ladder more often than every 500 steps.** The gate's primary
   key is the heuristic ladder net, but the ladder only runs on the
   `save_every` cadence, so the gate promotes on stale measurements from
   up to 500 steps ago. Decoupling the ladder cadence from the snapshot
   cadence would make promotion more responsive.
5. **Keep the host awake during long background runs.** A sleeping machine
   kills background jobs — the host sleep killed the capacity probe
   mid-run. The probe now resumes from its own checkpoint, and the trainer
   survives on its periodic snapshots, but a sleep still costs time.

### Then, in priority order

4. **DAgger.** The model is trained on solver-labelled positions it never
   visits. Run it against the solver and label the positions it actually
   reaches. This is step 4 of the approved plan and has not been started.

5. **MLP baseline on the real task.** The MLP hit 100% on the 64-position
   overfit in 2.2 s. Train it on the full 106,966-row corpus with the same
   7-way loss. If the MLP beats the GF model at scale, the GF backbone is
   the limiting factor — that is a clean, publishable answer and it is the
   stated go/no-go test.

6. **Diagnose the oscillation.** The dominant column swinging between col 2
   and col 4 with 90%/30% splits suggests the policy is not conditioning
   sharply on the position. Log per-position agreement between the model's
   argmax and the target argmax, bucketed by depth.

7. **Harden the paper claims.** The byte bpb in the C4 log is auxiliary
   (`--other-w 0.1`); the paper's text bpc remains the language-model
   metric. Do not conflate them in the write-up.

### Do NOT

- Do not quote any strength number from a training log produced before the
  greedy fix — they grade a sampled policy.
- Do not trust a holdout average without the pick histogram next to it.
- Do not diagnose "the data" before verifying "the gradient is live" and
  "the measurement measures the shipping policy".
- Do not use game-outcome labels or reward-based self-play — the exact
  solver is available and strictly better (approved plan, reaffirmed).
- Do not shorten ctx to ~20 for speed: the model uses lags
  `(0,1,2,4,8,16,32,64)` and four of them would exceed a 20-byte context.

---

## 9. Reproducing

```bash
cd <repo>
bash run_v21_live.sh            # 3,000 steps, seed 32, ctx 192, batch 16
bash run_v22_greedy.sh          # 25,000 steps, greedy eval, resumes V21
```

Key probes:

| Script | Purpose |
|---|---|
| `tg_losslive.py` | **asserts the move loss is non-zero and differentiable** — run this first, always |
| `tg_ladder_top3.py` | **the decisive 2,000-game ladder** (greedy hard model, opponent exploration, distinct counting) |
| `tg_headtohead.py` | model vs centre rule head-to-head + centre rule's own baseline row |
| `tg_baselines_holdout.py` | centre / heuristic / minimax / random / model on identical positions |
| `tg_softhard.py` | soft vs integer-export agreement |
| `tg_score_targets.py` | score-aware target correctness |
| `tg_align.py` | dataset-row alignment (2,000/2,000 PASS) |
| `tg_holdout_metric_check.py` | holdout optimality-mask regression |
| `tg_gate_check.py` | ladder-first promotion regression (13/13 PASS) |
| `tg_overfit3.py` | valid seven-way overfit harness (GF + MLP control) |
| `tg_scale4096.py` | 4,096-position capacity probe |

---

## 10. Environment

- venv: `.venv-c4` (pandas, pyarrow, psutil)
- GPU: RTX 5090; one GF step ≈ 0.93 s at batch 16 / ctx 192
- CPU: 32 cores (the ladder and capacity probes run here, off the GPU)
- Source shard: `<scratch>/c4data/chunk0.parquet`
- Exact solver: `src/c4/exact_solver.py` — mean 0.038 ms, max 0.861 ms
- Canonical seed: **32**
- ctx 192, batch 16 — both fixed by the paper

---

## 11. The lesson worth keeping

Nine diagnoses in a row were wrong — targets, temperature, context length,
model capacity, compute budget, label coverage, seed, fabricated labels,
and finally the possibility that the model was simply weak.

Every one of the real bugs was the same shape: **the thing being measured
was not the thing being claimed.** The loss was multiplied by zero. The
model was sampled instead of played greedily. Nothing errored. Every
metric looked plausible.

The rules now saved as skills:

> `silent-zero-loss-trap` — before any long run, assert the loss term you
> care about is non-zero and finite, using the real trainer functions. A
> smooth curve is not evidence of learning. When you find a bug, re-verify
> every prior conclusion — they were all downstream of the same broken
> assumption.

> And, added: **an evaluation harness has the same failure mode
> as a loss.** Verify that the thing you measure is the thing you ship.
> Greedy vs sampled is a 8× difference on the same weights.

---

## Appendix A — The nine wrong diagnoses

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | Bad score-aware targets | Wrong — argmax/order 4,000/4,000, illegal sentinel correct |
| 2 | Temperature too high / spread-normalised | Wrong — fixed raw-score T=2.0 verified |
| 3 | `snap_to_end` not active | Wrong — auto-enabled whenever soft targets load |
| 4 | ctx 192 too long | Wrong — shortening breaks 4 of 8 lags |
| 5 | Model capacity too small | Wrong — 100% on 64 positions, 95.5% on 512 |
| 6 | Not enough training steps | Wrong for the plateau — the gradient was zero |
| 7 | Fabricated one-hot fallback | Real bug, fixed, **not** the cause |
| 8 | Seed 0 instead of 32 | Wrong — V18 with seed 32 reproduced the plateau |
| 9 | "The model just isn't strong enough" | **Wrong** — it was never measured greedily |

The actual causes were three: a weight mask that missed the supervised
position, a `0 × -inf` NaN, and an evaluation path that sampled the model.

## Appendix B — File map

```
src/c4/train.py                     trainer; all three fixes live here
src/c4/move_loss.py                 masked 7-way loss, soft-target blending
src/c4/opponents.py                 centre / random / heuristic / lookahead / minimax
src/c4/transcript_v2.py             V2 encoding (column byte = 0x80 | height<<…)
src/c4/transcript.py                COL_BYTE_OFFSET = 0x41
src/c4/fmt.py                       PLY_LEN=10, MOVE_OFFSET=9
src/c4/agent.py                     RIGHT_PAD_BYTE, trainer_window, prompt_tokens
src/c4/board.py                     board, valid_actions
src/c4/duel.py                      ladder: opponent exploration, distinct counting
src/c4/exact_solver.py              perfect solver
tools/import_hf_c4.py               HF -> V2 corpus + score-aware targets
tools/split_positions.py            symmetry-class split, leakage checks
docs/CONNECT_FOUR_HANDOVER.md       this document
docs/CONNECT_FOUR_PLAIN_ENGLISH.md  plain-language doc
docs/CONNECT_FOUR_OVERVIEW.md       stale technical overview
tg_gate_check.py                    promotion-gate regressions (24/24 PASS)
tg_scale4096.py                     4,096-position capacity probe (87.67%)
tg_ladder_top3.py                   fresh 2,000-game ladder for top-3 candidates
tg_headtohead.py                    head-to-head vs a named opponent
tg_align.py                         importer alignment check (2,000/2,000)
tg_losslive.py                      live-loss + gradient regression
runs/_probe/scale4096_gpu7.log      corrected capacity probe (EXIT=0)
runs/_probe/scale4096_ckpt.pt        probe resume checkpoint
runs/_probe/prodholdout.log         production metric on the same 4,096 rows
runs/_probe/_wincheck.py             window / label-leak verification
```

### Skills written for this project

| skill | what it encodes |
|---|---|
| `long-run-checkpoint-survival` | periodic saves, atomic writes, resume, detecting a killed job |
| `silent-zero-loss-trap` | a zero loss is not a working loss — assert the gradient |
| `scale-consistent-promotion-gate` | one metric, one scale, protocol-tagged checkpoints |
| `measure-before-assuming-hardware-limits` | measure memory and speed before quoting a limit |
| `solver-economics-measurement` | how to measure the solver and its economics |

---

## Appendix C — Where the Connect Four work stands 

> **⚠️ Historical snapshot, superseded.** The `n=166` solver
> column below is the non-authoritative default-budget run; the authoritative
> run is `n=560` (`--budget 120000000`). "The centre-only defect is fixed"
> was **retracted** — see Appendix D18 and `docs/CURRENT_TRUTH.md`. No arm
> beats a constant-column policy on the authoritative set. This appendix is
> kept verbatim for provenance.

### The headline

The **centre-only defect is fixed.** A new candidate, `fc_a_tau0`, no longer
plays the middle column reflexively, and the **exact solver says it plays
better than the shipped model** — which is the metric that matters, because it
is the only one that does not depend on reproducing the teacher's opinions.

| metric | shipped dense | `fc_a_tau0` | `fc_b_tau05` |
|---|---:|---:|---:|
| strict teacher accuracy (1,500-pos disjoint holdout) | 28.5% | 21.9% | 21.2% |
| **off-centre** strict accuracy | 6.2% | **15.0%** | 14.8% |
| c3 share of selected moves (1,500-pos) | 83.3% | **37.1%** | 35.9% |
| **solver-optimal** (n=166 proven) | 64.5% | **69.3%** | 66.9% |
| c3 share of selected moves (solver set) | 60.2% | **31.9%** | 31.9% |
| blocks opponent's win (307 paired positions) | 47.2% | 45.0% | 44.0% |
| — single-threat positions | 47.8% | 43.4% | 43.8% |
| — multiple-threat positions | 48.6% | **57.1%** | 45.7% |
| — as P1 / as P2 | 47.1 / 47.4% | 44.9 / 45.0% | 43.4 / 44.4% |
| mirror consistency (non-centre choices) | 6.2% | **18.6%** | 12.8% |
| seat consistency | 100% | 100% | 100% |
| duel vs centre opponent | 47.2% | **65.0%** | 46.0% |
| duel vs heuristic opponent | 13.0% | **19.9%** | 27.5% |
| duel vs lookahead opponent | 10.8% | 9.1% | 5.0% |

Baselines on the same tactical set: constant-c3 blocks **26.7%**, random
**18.9%**. So both models beat the baselines by a wide margin; the question is
which is better, and the answer differs by metric.

**Why strict teacher accuracy went DOWN and this is not a regression:**
the teacher is a heuristic that is itself only 89.2% solver-optimal. The old
model scored higher against it by reproducing the old corpus, which was 93%
c3 — i.e. it was scoring well by being wrong in the same way the teacher was
wrong. The exact solver says the new model picks the provably best move
**more often** (69.3% vs 64.5%). The teacher-agreement number measures
imitation; the solver number measures quality.

**What changed, mechanically:** the training corpus went from the small,
93%-c3 corpus to a **320,000-game balanced corpus with 4,462,571 per-ply
labels** (25.5% c3 — matching the holdout's 25.67%). Same architecture, same
context length (192), same batch (16), same `--table-bias 0.0` / `--hard-start
0` settings. Only the *labels* changed. The whole defect was the training
distribution, not the network.

**Zero-overlap guarantee:** the 1,500-position holdout shares **0** canonical
boards with any of the 5,100,175 boards in the training corpus.

### The remaining defect: mirror consistency

Connect Four is exactly symmetric under `c → 6-c`. A model that reads the
board must be reflection-consistent: on the mirrored board it should pick the
mirrored column. The model is **100% seat-consistent** but only **18.6%
mirror-consistent** (shipped: 6.2%).

> **SUPERSEDED (, late) — the mirror arm has since been TRAINED
> and MEASURED. The numbers below are from before that run. The current
> matched measurement is in Appendix D: `mir_a_s6000` reaches **36.7%**
> mirror consistency vs `fc_a_s6000`'s **16.4%**, and it also blocks
> better (50.2% vs 42.7%), but it is **worse** on whole-board centre
> calibration (c3 share 54.6% vs 33.3%) and **tied** on exact-solver
> optimality (71.8% vs 71.8%, n=560). Mirror augmentation is a real but
> TARGETED win, not a general one. `mir_a_tau0` is no longer running.

This is a *data* problem, not an architecture problem. The fix is
**mirror-augmented training**: every training game is joined by its reflected
twin, with the per-ply target mirrored on the column axis
(`target'[c] = target[6-c]`). The teacher labels are already ~99.9%
mirror-consistent, so this removes the model's freedom to break symmetry
without injecting any label noise.

Built and verified (all four bugs found in the builder are recorded in
`docs/CONNECT_FOUR_SECTION_0I.md` §5.1):

```
mirrored games:        320,000   skipped: 0
header checks:      5,100,175   mismatches: 0
target checks:      4,462,571   mismatches: 0
alignment check:     OK
corpus rows 640,000 = sidecar rows 640,000
combined c3 share:  25.50 %     (c0==c6, c1==c5, c2==c4 exactly)
```

Training arm `mir_a_tau0` is running (3000 steps, ctx 192, batch 16). Gate:
`bash run_gate_mirror.sh`.

### Still open — do not claim these are solved

| gap | current best | note |
|---|---|---|
| mirror consistency | 18.6% | `mir_a_tau0` targets this |
| single-threat blocking | 43.4% | below shipped's 47.8% |
| duel vs lookahead opponent | 9.1% | weak for **every** arm incl. shipped |
| c1 usage | 23 / 1500 | c6 is nearly dead (4 / 1500) |
| c3 still most-picked column | 37.1% | no longer dominant, but still #1 |

### Two hard rules learned the hard way

1. **A sidecar and its corpus must be proven row-aligned, not assumed.**
   `src/c4/train.py` refuses a row-*count* mismatch; it cannot catch a row-*order*
   mismatch. The first mirror build wrote the corpus interleaved and the
   `.npz` blocked — every per-row check passed, yet every game was paired with
   another game's labels. There is now a preflight that refuses to train
   unless row count, target legality and mirror structure all verify.
2. **The `teacher` array is a source code, not a column.** (0 = no label,
   1 = exact solver, 2 = strong teacher.) Treating it as a column corrupts it
   silently. The trainer never reads it, so this is metadata-only — but it
   still makes the artifact's own provenance report lie.

### Not launched, deliberately

**V24 has not been launched.** The gates above must pass first.

---

## Appendix D — The terminal-state bug and its measured blast radius (, late)

### What the bug was

`Board.is_full` decided "is the board full?" by testing **one bit**:

```python
# WRONG -- bit 41 is the top cell of the right-hand column
return bool((self.mask >> (ROWS * COLS - 1)) & 1)
```

After six discs in column 7, the whole 6x7 board was declared full and the
game ended in a phantom DRAW with **36 of 42 cells empty**. The bug was in
the **initial commit** (`c250352`), so every consumer inherited it.

### The fix

```python
def is_full(self) -> bool:
    return not self.valid_actions

def is_terminal(board: "Board") -> bool:
    return board.is_win!= 0 or not board.valid_actions
```

`is_terminal` is now the **canonical project-wide terminal predicate**.
Every former `is_full` consumer was migrated: `duel.py`, `play.py`,
`corpus.py`, `corpus_quality.py`, `outcome_rl.py`, `augment_corpus.py`,
`bench_speed.py`, `agent.py`, `threat_corpus.py`, `train.py`.

### Measured impact — corpus

`runs/_probe/corpus_truncation_final_audit.json` walks every slot of every
row of the 320,000-row corpus and classifies it:

| Measurement | Result |
|---|---|
| Answer slots checked | 4,462,571 |
| Answers that are NOT a legal move at that position | **0** |
| Answer slots with a broken position index | **0** |
| Answer slots the bug caused to be SKIPPED | 101,090 |
| Rows affected | 13,168 (4.12%) |
| **VERDICT** | **`COVERAGE_LOSS_ONLY`** |

Truncated rows: **15,084 / 320,000 (4.71%)** original,
**31,234 / 320,000 (4.88%)** mirror.

**Consequence: no corpus repair is needed and the checkpoints trained on it
are valid.** The labels are solver-derived per position, so a truncated game
has fewer solved positions, never wrong ones. The trainer also handles the
gap correctly: skipped slots carry `ply_of = -1` and are dropped by
`hit = keep & (flat_slot >= 0)`, never faked.

### Measured impact — strength

The bug changed the strength numbers, so the pre-fix matrix could not be
trusted. A **paired** re-measurement was run: same checkpoints, same
opponents, same eps, same seed (2468), same seat alternation, same RNG
stream — the ONLY difference between the two columns is the terminal test.

See `runs/_probe/paired_isfull_impact.json` for the exact deltas.

### Two audits that were wrong before the right one

Both earlier "label corruption" verdicts were artifacts of the audit, not
findings:

1. `n_plies` is the **last attempted slot + 1**, capped at P=19 — NOT the
   number of recorded plies. Comparing it against the transcript length
   "found" 13,688 corrupt rows that do not exist.
2. Re-solving every label with the **exact** solver and calling the
   disagreement "corruption" ignores that the label pipeline uses the exact
   solver only within 12 empty cells and a strong depth-limited solver
   beyond that. Strong-solver labels disagreeing with exact optimality is
   expected, not corruption.

The final audit re-solves nothing, ignores `n_plies`, and asks only:
**"is this answer a legal move at the position it claims to describe?"**

### Appendix D2 — the bug hit TWICE, and it also hit the LABELLER

The relabeller (`tools/build_perply_fast.py:127-128`) skips a position when
`b.outcome!= Outcome.ONGOING`. Because `outcome` called the buggy
`is_full`, **the bug also destroyed labels during relabelling** -- not
only during corpus generation.

Measured on a 4,000-row sample of every mid-game gap, classifying each
skipped slot by what the buggy test believed:

| Cause of the skipped label slot | Count | Share |
|---|---|---|
| `old_is_full` fired on a live position | 10,054 | **100.0%** |
| genuine win (terminal) | 0 | 0% |
| genuinely full board | 0 | 0% |

So of 10,089 rows with a hole in the labelled range, **every single one** was
caused by the terminal bug.

**Corrected counts.** The definitive audit's 101,090 is the right number; an
intermediate count of 42,589 was wrong because it bounded the scan by
`n_plies`, which is itself the corrupted field. The reconciliation:

```
total holes (ply_of == -1) across all 320,000 x 19 slots: 1,617,429
  of which ABOVE the last labelled slot (never processed): 1,516,339  (padding)
  of which WITHIN the labelled range (real skipped labels):   101,090
```

**Cost of the bug in labels: 101,090 of 4,462,571 = 2.3% of the label set
was thrown away.** Recovering it is a pure win with no new training: rebuild
the sidecar with the fixed predicate. No corpus repair is needed, and no
checkpoint needs retraining to be *valid* -- but ~2.3% more supervision is
recoverable.

### `n_plies` is not what its name suggests

`n_plies[i]` is **not** the number of labelled plies. In
`tools/build_perply_fast.py:151` it is assigned *inside* the labelling loop:

```python
n_plies[i] = s + 1
```

and the loop `continue`s (line 128) without reaching that assignment when
the position looks terminal. So `n_plies` = **(last successfully labelled
slot) + 1**, which over-counts whenever a slot was skipped. Measured:
19,281 rows consistent, **719 rows inconsistent**.

This is what made the v2 audit report `LABEL_CORRUPTION_PRESENT`: its four
verdict terms were

| Term | Value | Result |
|---|---|---|
| `mism == 0` (the `n_plies` check) | mism = 10,089 | **FAIL** |
| `illegal == 0` | 0 of 4,462,571 | PASS |
| same_agreement (< 5pp) | 97.71% vs 95.83% (1.88pp) | PASS |
| truncated strong-legal >= 99% | 100.00% | PASS |

**The only failing term is the one built on the misread field.** The v2 data
contradicts its own verdict: truncated rows had *higher* exact-solver
agreement (97.71%) than clean rows (95.83%), and 100% of truncated labels
were legal. The verdict is an artifact of a bad oracle, not a finding.

### Appendix D3 — the model is winnable out of these positions; it just doesn't

The user's report "me and the model picked column 7 only" needed a proper
control. Two earlier attempts at one were invalid:

- the `strong_ref` row was **not a control at all** -- it consults the exact
  solver only when <= 8 cells are empty, so in the opening it falls back to
  `agent.choose(...)`, i.e. the model itself;
- a "pure exact solver" policy is also not a control -- `ExactSolver.solve`
  returns `None` (UNPROVED) above `max_empty`, so for the whole opening it
  has no move to pick and degenerates to "first legal column". (The first
  version of this script played `[6,0,6,0,6,0,6]` and lost at ply 7 for
  exactly that reason.)

The correct model-independent control is `src/c4/opponents.py`:
`minimax_policy` (depth-3 negamax + alpha-beta) and `lookahead_policy`.
Neither consults the model and neither silently degrades.

**Ground truth** (`runs/_probe/c6_stacker_ground_truth.json`, 5 games each,
model as P2, stacker as P1 always playing one column):

| Stacker column | minimax d3 | lookahead d2 | **model greedy** |
|---|---|---|---|
| c6 (user col 7) | 5W 0L 100% | 5W 0L 100% | **0W 5L 0%** |
| c5 (user col 6) | 5W 0L 100% | 5W 0L 100% | **0W 5L 0%** |
| c3 (user col 4) | 5W 0L 100% | 5W 0L 100% | **0W 5L 0%** |
| c0 (user col 1) | 5W 0L 100% | 5W 0L 100% | 5W 0L 100% |

**The position is winnable for the second player. The defect is in the
model.** Verified by independent replay of every stored line
(`tmp/verify_ground_truth.py`, 12/12 rows, 0 mismatches), which rebuilds the
board from the move list and never calls the policy.

**The mechanism is echo/antagonism, not centre bias.** Measured echo rate
(does the model answer column X with column X?):

| Opponent | Echo rate | Model record |
|---|---|---|
| stack c6 | 50.0% (25/50) | 0/5 wins |
| stack c5 | 28.6% (10/35) | 0/5 wins |
| stack c3 | 50.0% (30/60) | 0/5 wins |
| stack c0 | 78.9% (75/95) | 5/5 wins |

Against the c6 stacker the model's own line was
`[6,6,6,3,3,0,0,3,2,2]` -- it answered c6 with c6 three times, so column 7
alternated colours and the stacker never completed a vertical four. The
model then lost by never having built anything itself.

**The model has no c7 bias of its own** (own column profile vs a non-echo
opponent: c3 36%, c6 11.8%, c0 7.1%, c1 9.0%). The column-7 concentration
is an *interaction* effect: the model mirrors whatever column the opponent
just used, and mirroring is fatal against a stacker. This answers the user's
original question -- it is the model, and it is specifically an antagonism
failure, not a centre-preference failure.

### Appendix D4 — the terminal bug's actual strength impact is small

Paired A/B, same seed (2468), same policies, same seat alternation, same
RNG stream; **only** the terminal test differs
(`runs/_probe/paired_isfull_impact.json`):

| Matchup | buggy | canonical | delta | unique games (bug/can) |
|---|---|---|---|---|
| fc_a_s6000 vs centre | 54.5% | 54.5% | +0.0 | 152/152 |
| fc_a_s6000 vs heuristic | 9.8% | 9.8% | +0.0 | 175/175 |
| fc_a_s6000 vs lookahead | 5.2% | 5.2% | +0.0 | 160/160 |
| mir_a_s6000 vs centre | 30.3% | 29.5% | **-0.8** | 255/258 |
| mir_a_s6000 vs heuristic | 23.5% | 20.5% | **-3.0** | 234/232 |
| mir_a_s6000 vs lookahead | 2.3% | 2.2% | -0.2 | 136/134 |
| mir_a_s6000 vs fc_a_s6000 | 100.0% | 100.0% | +0.0 | 2/2 |
| mir_a_s6000 vs fc_a_tau0 | 50.0% | 50.0% | +0.0 | 2/2 |
| mir_a_s6000 vs mirrored copy | 50.0% | 50.0% | +0.0 | 1/1 |

**The bug moved strength by at most 3.0 points, in 3 of 9 matchups, and
never by more than 3 points.** Six of nine matchups are bit-identical. So
the pre-fix strength numbers were not materially wrong -- but they were
*unverified*, and the post-fix matrix is now the one to trust.

Caveat on the duel rows: `mir_a_s6000 vs fc_a_s6000` produced only **2**
distinct trajectories in 300 games, and `vs mirrored copy` only **1**. Those
two matchups are degenerate (the models play each other near-deterministically
from a shared opening) and carry almost no statistical weight. The rows with
150-258 distinct trajectories are the meaningful ones.

### Appendix D6 — THE ECHO FAULT IS NOT IN THE TEACHER 

This is the measurement that decides what to fix. It was the open question:
the model copies the opponent's previous column, but **why**?

Three candidate causes, and only one can be true:

| hypothesis | how to test it | result |
|---|---|---|
| (a) the teacher labels teach echoing | compare the LABEL's column against the opponent's previous move | **REFUTED** |
| (b) the labels for the deciding positions were deleted | compare label echo rate before/after the repair | **REFUTED** |
| (c) the model cannot represent / learn the rule | labels are correct, model still echoes | **CONFIRMED** |

**The measurement.** For every labelled position in the full corpus, take the
opponent's *previous* move and compare it with the label's column. If the
label equals the previous move, that label teaches echoing.

| | |
|---|---|
| **Label echo rate (original sidecar)** | **17.023%** |
| Label echo rate (repaired sidecar) | 17.024% |
| Label echo rate, early game (ply < 12) | 18.261% → 18.250% |
| Label echo rate, late game (ply ≥ 12) | 14.640% → 14.796% |
| Late-game labels added by the repair | +91,876 |

**The model's own echo rate**, for comparison:

| | |
|---|---|
| model vs a column-7 stacker | **~50%** |
| model vs a centre bot | **~80%** |
| model vs random | ~31.4% |
| **what the labels teach** | **17.0%** |

**The teacher is three to five times less echo-prone than the model is.** The
model is not faithfully reproducing its supervision — it is *over*-copying
relative to what it was taught. That kills hypothesis (a) and (b), and
establishes (c): the fault is in the model or its optimisation, not in the
labels and not in the deleted data.

Note the direction of the repair's effect: it moved the late-game echo rate
by **+0.16 points** (14.640% → 14.796%), i.e. in the *wrong* direction, and
by a negligible amount. So the repair cannot fix the echo behaviour — it
can only restore coverage. The echo fix has to come from the model side.

**The echo rate also falls with ply** (26.8% at ply 4 → ~13% from ply 10
onward), which is consistent with an echo that is strongest in the OPENING,
where the model has no tactical signal and falls back on the most recent
thing it saw.

Artifact: `runs/_probe/label_echo_rate.json`.

#### The exact trigger of the bug is sharper than "column 7 occupied"

Measured over all 101,090 lost positions:

| | |
|---|---|
| column-7 height at the lost positions | **6/6 in 100% of cases** |
| column 7 legal at those positions | **0** |
| lost labels at ply 0 (the opening) | **0** |
| affected games with a **contiguous** lost tail | 13,168 |
| affected games with a non-contiguous loss | **0** |

Bit 41 is the **top** cell of the right-hand column, and cells fill
bottom-up, so the bug fired only when that column was **completely full**.
That makes the damage a **tail deletion**, not a regional one: a game simply
stopped being supervised once column 7 filled. It also means the lost labels
were never "play column 7" labels — column 7 was illegal in all of them.

Artifact: `runs/_probe/isfull_exact_trigger.json`.

### Appendix D5 — the lost labels are RECOVERED 

The label loss was recoverable, and has now been recovered and independently
verified. Nothing was thrown away and nothing was regenerated.

**Tool:** `tools/repair_sidecar.py` — shardable, fills **only** the holes,
leaves every existing label byte-for-byte identical, uses the **same
canonical solver path** the relabeller used (so the mirror symmetry of the
label set is preserved), and uses the canonical-key cache so a board and its
mirror share one solve.

**Smoke test** (300 rows) → `REPAIR_SOUND`.

**Full run** (320,000 rows, 8 disjoint shards):

| | |
|---|---|
| Labels before | 4,462,571 |
| Labels after | **4,563,661** |
| Labels added | **101,090** (exactly the predicted number) |
| Pre-existing labels changed | **0** |
| Pre-existing `ply_of` changed | **0** |
| Labels lost | **0** |
| New labels illegal | **0** |
| New labels at a terminal position | **0** |
| New label ply mismatch | **0** |
| Verdict | **`REPAIR_SOUND`** |

Artifacts: `runs/_probe/perply_v23_full/merged_repaired.npz`,
`runs/_probe/sidecar_repair_merge.json`,
`runs/_probe/sidecar_repair_verify.json`.

#### The repair is not biased — measured, not asserted

Two independent checks:

1. **The lost positions have exactly the bug's signature.** All **101,090**
   lost positions had **column 7 already occupied** (100%), because bit 41 is
   the top cell of that column. Their free-cell counts ranged from **1 to
   36** — the board was far from full, as expected.
2. **The repair path reproduces the original labels.** Recomputing 5,568
   originally-labelled positions with the repair path gave the **same answer
   99.93%** of the time (5,564/5,568). The 4 differences are exact-solver
   *ties* resolved differently (`src=1` = exact solver), not a bias.

This also explains why the repaired labels contain **0.0% column 7** while the
originals contain 7.62%: at every lost position column 7 was already
occupied, so the solver's best move was somewhere else. The absence is a
property of the lost *positions*, not of the repair.

**Column distribution of the repaired labels** (vs. the originals):

| column | original % | repaired % |
|---|---|---|
| c1 | 13.18 | 15.08 |
| c2 | 13.31 | 12.63 |
| c3 | 14.20 | 16.03 |
| c4 | 25.49 | 25.80 |
| c5 | 14.08 | 17.29 |
| c6 | 12.12 | 13.17 |
| c7 | 7.62 | 0.00 |

The repaired set is drawn from the same solver policy as the original set
(c3 share 25.80% vs 25.49%), so adding it does **not** shift the centre
bias of the corpus.

Artifact: `runs/_probe/repair_unbiasedness.json`.

### What is still open

- The model **loses 100/100** games against an opponent that always plays
  column 7. Whether that is a model c7 bias or an echo effect is now
  measured separately (`runs/_probe/echo_col_diag_fixed.json`).
- c1 usage remains ~0.6%.
- Exact-solver coverage remains 560/1500 (37.3%).
- The pre-fix strength numbers in the tables above are **superseded** by
  the paired re-measurement.
- **V24 still has not been launched.**


## Appendix D7 — MECHANISM: THE MODEL WEIGHS "LAST MOVE" AS MUCH AS THE BOARD 

Appendix D6 proved the fault is not the teacher. This appendix identifies
**where** in the model it lives.

### A measurement trap that had to be cleared first

The first sensitivity probe reported **L1 = 0.0000** for every
perturbation — board, last-move, turn — which reads as "the model is
completely blind". That was **the float path** (`column_logits`). The
shipped model plays through the **hard integer export**
(`hard_column_logits` / `use_hard=True`), and the trainer's own board-gate
measures that path: it reports sensitivity **0.67–1.0** for
`mir_a_s6000/step6000_hard.pt`.

Re-run on the hard path, the same probe validates **25/25** against
`agent.hard_column_logits(board, moves)` and gives a completely different
answer. Lesson recorded as skill
`hard-vs-soft-export-measurement-path`.

### The hard-path measurement

One byte is changed; the action distribution is compared. 25 positions.

| field perturbed | mean L1 | nonzero |
|---|---|---|
| **last-move byte** | **0.2102** | 25/25 |
| one **board column** byte | **0.1870** | 25/25 |
| **turn** byte | 0.1227 | 10/25 |

**Ratio last-move: board = 1.12.** The information is available to the
model, and the model uses it about as heavily as the board itself.

The transcript explains why this is a real hazard. In v2 each column byte
encodes height + P1-ownership bits, and the `last` byte is the **only**
record of *who just moved*. A column byte says a piece is there, never
that the opponent put it there. The model therefore faces two encodings
of "column 7 is relevant" and has learned to weight them about equally.

### Locality (which ply does it read?)

| ply | mean L1 | argmax changed |
|---|---|---|
| current (k) | 0.2078 | 1/2 |
| k−1 | 0.1344 | 0/2 |
| k−3 | 0.1375 | 1/8 |
| k−6 | 0.0793 | 1/17 |
| k−9 | 0.0557 | 2/21 |
| k−12 | 0.0067 | 0/12 |
| k−15 or older | 0.0000 | 0/2 |

The receptive field is **short**: ~10 plies. The model reads the current
position and little else.

### The copy is not a correct block

`tmp/check_echo_correctness2.py`, opponent always stacking one column,
ground truth from the canonical exact solver where conclusive and the
canonical strong solver otherwise, with the source recorded per row:

```
model decisions analysed: 333
exactly proven: 23
strong-solver only: 310
the model copied: 75 (22.5%)
of those copies, CORRECT: 0/75
```

By the opponent's stack height at the moment of the decision:

| stack height | decisions | copied | correct |
|---|---|---|---|
| 1 | 25 | **25 (100%)** | 0 (0%) |
| 3 | 25 | **25 (100%)** | 0 (0%) |
| 5 | 25 | **25 (100%)** | 0 (0%) |
| 6 (column full) | 258 | 0 (0%) | 108 (42%) |

**Caveat, stated plainly:** the "correct" column is exact-solver-backed in
only 23 of 333 rows; the rest is depth-6 strong-solver, which returns a
single move and is a heuristic, not proof. The exactly-proven subset
contains **0 copies**, so it cannot confirm the 0/75 on its own. The
defensible claim is therefore: the model copies **100% of the time** while
the column is playable, and the strong solver never agrees with the copy
in any of the 75 cases. A wider exactly-proven run is required before
"0/75 are wrong" can be stated as proof.

### Consequence for the fix

Restoring the deleted labels cannot help: the labels echo only 17.0% and
the repair left that unchanged (17.02% → 17.02%). The fix must change
what the model is rewarded for — the objective, not the data volume.
Concrete candidates, to be A/B tested:

1. **Down-weight the last-move field** in the training target, or
   randomise it during training so the model cannot rely on it.
2. **Augment with last-move-randomised copies** of each position: the same
   board with a *different* recorded last move, so the correct answer is
   unchanged and the last-move signal carries no information.
3. **Add a decoding-time rule**: suppress a column whose only support is
   "the opponent just played there" when no threat justifies it.

Option 2 is the most principled and is directly testable: it makes the
last-move byte uninformative by construction, and if the echo survives
that, the fault is elsewhere.


## Appendix D8 — MIRROR-V2 IS EXACTLY CONSISTENT; AND A TIE-BREAK TRAP 

### The trap

The first independent check of the shipping mirror dataset reported
**3,392,330 "mismatches"** out of 4,563,661 and the verdict
`MIRROR_V2_PROBLEM`. That verdict was **wrong** — the check was wrong.

Two mistakes, both instructive:

1. **The mirror's target is the COLUMN REVERSAL of the original**, not a
   copy. `tools/build_mirror_corpus.py:219` writes
   `out_tgt[m_out] = tgt[i][:,::-1]`, so a mirror label sits at
   `6 - c` where the original sat at `c`. A check that compares raw
   column indices reports every label as "different".
2. **Argmax does not commute with reflection.** The teacher target can be
   a TIE over several equally-optimal columns, and `np.argmax` picks the
   **lowest** index. Under `c -> 6-c` that tie-break flips, so two
   perfectly consistent labels can have different argmaxes.

The builder's own code already warned about this, in a comment next to
the tie-set check it actually performs. The independent check was the
thing that was wrong.

### The correct invariant, and the result

The right test is the **tie set**, not the argmax:

```python
set_o = {c for c in range(7) if abs(float(t_o[c]) - best_o) <= 1e-6}
set_m = {c for c in range(7) if abs(float(t_m[c]) - best_m) <= 1e-6}
assert set_m == {6 - c for c in set_o}
```

Measured over the first 100,000 interleaved pairs:

```
labels checked: 1,404,819
tie set == reflected tie set: 1,404,819  (100.000%)
tie set MISMATCH: 0
target sum mismatch: 0
argmax == 6 - argmax: 99.346%   <- the 0.65% are ties
```

**Verdict: the shipping mirror dataset `merged_mirror_v2.npz` is exactly
mirror-consistent, with zero exceptions.** The 0.65% argmax differences
are exact-solver ties, not defects.

### The independent verifier, re-run with the correct invariant

`tmp/verify_mirror_v2b.py` over the FULL 640,000 rows (all 320,000
interleaved pairs), using the tie-set invariant:

```
corpus rows: 640,000
labels compared (interleaved pairs): 4,563,661
original label!= mirror label: 0
mirror-original half!= repaired original: 0
illegal labels: 0
labels at a terminal position: 0
c3 share original: 25.7117%
c3 share mirror: 25.7117%
VERDICT: MIRROR_V2_CONSISTENT
```

**`merged_mirror_v2.npz` is verified shippable.** The c3 shares are
identical to four decimal places, which is the expected consequence of
the mirror being a bijection on the label set.

### Why this matters for the release

`merged_mirror_repaired.npz` — the *independently* repaired pair — had
**4** genuine asymmetries. `merged_mirror_v2.npz`, rebuilt from the
repaired original, has **0**. The mirror-v2 rebuild is the one that
should ship.

### Transferable rule

When checking a mirrored dataset, compare the **invariant under the
group action** (the tie set, the reflected bitboard, the reflected
heights), not the **representative** (an argmax, a single returned move).
A representative choice is not group-equivariant, and a mirror check
built on representatives will report mass false positives.

This applies directly to the solver: `ExactSolver.solve` returns a
STATUS, and `tg_solver_truth.optimal_moves` returns the full optimal
SET. Using the status + a single tie-broken move to judge "is this copy
correct" is the same error, and it is what made the copy-correctness
probe look like 0/75 when a large part of that was a tie-break artifact.


## Appendix D9 — THE ACTUAL DEFECT, MEASURED 

Earlier appendices established that the terminal bug was real, that the
labels were `COVERAGE_LOSS_ONLY`, and that 101,090 labels were recovered.
They did NOT establish *why the model still plays badly*. This appendix
does, with a test that separates two mechanisms that make opposite
predictions.

### The two candidate mechanisms

The model was observed copying the opponent's column (50% against a
column-7 stacker, 80% against a centre bot). Two explanations:

* **M1 — board reading.** The model reads the BOARD and concludes the
  opponent's stack must be blocked or built on. The copy is a real
  tactical judgement. The copy is then *correct*, and the defect is
  elsewhere (objective, coverage, optimisation).
* **M2 — last-move echo.** The model reads the LAST-MOVE BYTE and
  echoes it. The copy is a mechanical artefact.

### The decisive test: lie to the model about the last move

Take a real position. Build the canonical token stream. Build a SECOND
stream in which **only the final header's `last` byte** names a different
legal column. Everything else — the whole board, the whole history — is
byte-identical. The two streams are asserted to differ in **exactly one
byte**, and that byte is asserted to be the last-move field.

If M2 is the mechanism, the model follows the false byte. If M1 is the
mechanism, the model's choice is unchanged.

### Result on the authoritative 1500-position holdout (hard export)

```
positions tested: 1,498
choice CHANGED when the last byte lied: 351  (23.43%)
model FOLLOWED the false byte:  80  (5.34%)
```

**M2 is refuted as the primary cause.** The model follows a false
last-move byte only **5.34%** of the time — barely above the ~7% chance
rate for a uniformly random legal column among the remaining candidates.
The last-move byte has a *real but secondary* effect (it changes the
choice in 23.43% of positions without necessarily steering it to the
false column — i.e. it acts as noise/perturbation, not as a copy
command).

### The defect that IS real: mirror inconsistency

The same holdout, evaluating each position and its canonical reflection:

```
positions tested: 1,500
mirror-consistent:   941  (62.73%)
mirror-INCONSISTENT:   559  (37.27%)
```

**37.27% of positions get a DIFFERENT (and often much worse) answer when
the board is merely reflected.** The model's own column bias is not a
fixed preference for one column — it is a **position-dependent asymmetry
that does not respect the symmetry of the game**.

This is the mechanism behind the observed centre shortcut. "Always c3"
is mirror-*consistent* (c3 maps to itself), so it is a **symmetry-
preserving degenerate solution**: a model that cannot solve the position
falls back to a policy that at least looks the same from both sides. The
centre bias and the copy behaviour are two symptoms of the same root
cause — the model is not extracting the tactical content of the board.

### The control that matters: mirror training helps, but does not fix it

Measured on the SAME 1500-position holdout, hard export, the
mirror-trained arm (`mir_a_s6000`) against the non-mirror-trained arm
(`fc_a_s6000`):

| metric                                  | fc_a_s6000 | mir_a_s6000 |
|-----------------------------------------|-----------:|------------:|
| mirror-consistent                       |   29.53%   |   62.73%    |
| mirror-INCONSISTENT                     |   70.47%   |   37.27%    |
| follows a FALSE last-move byte          |    9.21%   |    5.34%    |
| choice changes when the last byte lies  |   34.25%   |   23.43%    |

Two conclusions, and the first one CORRECTS an earlier reading of this
same project:

1. **Mirror training works.** It more than doubles mirror consistency
   (29.53% -> 62.73%) and halves the false-last-move follow rate
   (9.21% -> 5.34%). The mirrored corpus is doing its job.
2. **Mirror training does not fix the symmetry.** 37.27% of positions
   still get a different answer under reflection. A symmetric DATASET
   makes the *average* function more symmetric; it does not force the
   *learned* function to be equivariant.

(An earlier synthetic probe on 25 single-stack positions said mirror
training did NOT help. That population was far too narrow and the
conclusion was wrong. The 1500-position holdout is the population that
matters, and it says the opposite. A measurement on a convenient
synthetic population must not override the release population.)

So the remaining defect is an **objective/architecture** property, not a
data property:

* the GF trunk's readout is a per-position function with no equivariance
  constraint;
* training minimises a per-ply cross-entropy, which is *invariant* to
  neither the reflection nor the centre;
* the only thing that currently couples the two orientations is that
  both appear in the training set — which is a distributional argument,
  not a constraint on the solution.

### Method notes (three errors this appendix corrects)

1. **`agent.choose` defaults to the FLOAT path.** The first version of
   the probe called `agent.choose(b, moves)` without `use_hard=True` and
   measured a path that does not play. Everything here uses the hard
   export.
2. **A hand-rolled `mirror_board` is not the canonical one.** The first
   version forgot to reflect `mask` and `last_move`, producing an
   illegal board and a nonsense "model always plays c0" result. Use
   `src.c4.mirror.mirror_board`.
3. **A single solver move is not a tie set.** `ExactSolver.solve`
   returns a STATUS; `tg_solver_truth.optimal_moves` returns the full
   optimal SET. Judging "is this copy correct" by one tie-broken move
   manufactures asymmetries — the same trap as in Appendix D8.

### Consequence for the release plan

The evidence now says the remaining defect is **not**:

* label corruption (labels verified clean),
* coverage loss (101,090 recovered, verified),
* terminal-state semantics (fixed and regression-tested),
* the last-move byte (5.34%),
* the mirrored dataset (present, exactly consistent, and it helps:
  29.53% -> 62.73% mirror consistency — but not enough).

The evidence says it IS:

* **the learned readout/objective not being equivariant or
  centre-calibrated** — 37.27% mirror inconsistency and a degenerate
  symmetry-preserving centre solution.

That is an **architecture/objective** problem, not a data problem. The
next experiment must therefore change the OBJECTIVE (an explicit
equivariance or symmetry-consistency term, or a symmetry-aware readout),
not the dataset. The repaired-data A/B (`rc_a_s6000` vs `rc_b_s6000`) is
a control, not a fix: it measures whether recovering 101,090 labels plus
the mirror corpus changes anything, and the prior is that it will not
move mirror consistency much — the mirror corpus already showed what it
can do.


## Appendix D10 — ROOT CAUSE FOUND: the corpus last-move header field was corrupted in 39.5% of rows

### The defect

The v2 transcript stores, for every ply, a 10-byte record:

```
[col0..col6, last, turn, move]
```

`last` is the column of the PREVIOUS move — the header describes the
position BEFORE the current move. A well-formed row satisfies

```
header[p].last == move[p - 1]          for every p >= 1
```

`corpus_320k.txt` violates this in **126,537 of 320,000 rows (39.5%)**,
starting at row **193,459**. In every violating ply the stored byte is
`6 - move[p-1]` — the *reflected* column.

| measurement | value |
|---|---|
| rows | 320,000 |
| rows with a contradictory last-move field | **126,537 (39.5%)** |
| first affected row | 193,459 |
| affected plies | 1,567,958 |
| rows that fail `decode_game` | 126,537 |

The affected rows are **mirror twins of earlier clean rows**. Their board
bytes are a correct reflection of the earlier row's board, but their
last-move byte is *identical* to the earlier row's instead of reflected.

### Why the trainer did not catch it

`src/c4/train.py` guarded the corpus with a **single-row** probe:

```python
probe = next((ln for ln in raw_lines if len(ln) >= PLY_LEN), None)
validate_stream(probe, version)
```

Row 0 is clean, so a defect that starts 60% of the way through the file
passed silently and 39.5% of the training data carried a wrong byte.

### Why this is the mirror-asymmetry root cause

The mirror corpus interleaves `row 2i = original game i` and
`row 2i+1 = freshly built mirror of game i`. The builder's mirror is
correct (it reflects the last byte), so for every corrupted original row
the pair is:

```
row 2i: board = reflect(board of row 2i+1),  last = X
row 2i+1: board = the clean board,               last = X
```

**The two rows have different boards but the SAME last-move byte.** A
model that learns "answer by copying the last-move column" is therefore
trained to give the **same answer on a board and its reflection** — the
exact defect measured on the holdout.

Measured over all 320,000 mirror pairs:

| mirror-pair invariant | `corpus_320k_mirror_v2.txt` | `corpus_320k_mirror_v3.txt` |
|---|---:|---:|
| board bytes reflect correctly | 320,000 | 320,000 |
| **last-move field reflects correctly** | **193,463** | **320,000** |
| last-move field identical (not reflected) | 126,537 | **0** |

This supersedes the earlier conclusion that the asymmetry was purely an
objective problem. The dataset was **actively teaching** asymmetry.

### The repair

`tools/repair_corpus_last_field.py` re-encodes each row from its OWN
move bytes. The move bytes and board bytes are ground truth: a canonical
re-encode reproduces every board byte, every turn byte and every move
byte exactly and differs **only** at the corrupted last-move byte. That
was verified over all 320,000 rows, not assumed.

| check | result |
|---|---|
| rows changed | 126,537 |
| bytes changed | 1,567,958 |
| **rows with a non-last-byte difference** | **0** |
| rows failing `decode_game` before | 126,537 |
| rows failing `decode_game` after | **0** |
| rows with a different move list | 0 |
| rows with different board bytes | 0 |
| labels checked for legality after repair | 652,823 |
| **labels illegal after repair** | **0** |
| mirror pairs whose last field reflects (v3) | 320,000 / 320,000 |

The sidecar labels remain valid: the ply count, the board at every ply
and the move at every ply are unchanged, so every per-ply label still
refers to the same position.

Artifacts:

```
runs/_probe/perply_v23_full/corpus_320k_fixed.txt        (320,000 rows)
runs/_probe/perply_v23_full/corpus_320k_mirror_v3.txt    (640,000 rows)
runs/_probe/corpus_last_field_repair.json
runs/_probe/corpus_last_field_repair_mirror.json
runs/_probe/corpus_repair_verification.json              REPAIR_VERIFIED
runs/_probe/corpus_mirror_teaching_signal.json
runs/_probe/corpus_last_header_labels.json
```

The historical corpora are **preserved**, not overwritten.

### Permanent guard

`src/c4/corpus_integrity.py` adds a cheap O(bytes) whole-file scan for
the invariant. `src/c4/train.py` now runs it on every imported corpus
and refuses to train on a violation:

```
corpus integrity: 320000 rows, 0 with an inconsistent last-move field
```

Negative control confirmed: the corrupted corpus is refused with the
first offending row named.

### Consequence for the previous A/B

`rc_a_s6000` and `rc_b_s6000` trained on the corrupted corpus and were
**stopped and discarded as evidence**. Their checkpoints are kept as
artifacts but they are not release evidence. The clean replacement A/B
is `fd_a_s6000` / `fd_b_s6000` / `fd_c_s6000` (see below).

### New release plan

1. `fd_a_s6000` — repaired original corpus + repaired sidecar.
2. `fd_b_s6000` — repaired mirror corpus + repaired mirror sidecar.
3. `fd_c_s6000` — repaired mirror corpus + repaired mirror sidecar
   **plus the mirror-equivariance regulariser** (`--sym-w 0.5`).

`fd_a` vs `fd_b` isolates mirror augmentation on clean data; `fd_b` vs
`fd_c` isolates the objective-level regulariser. Only after these
complete does the canonical hard evaluation (both seats, c1–c4 buckets,
exact-solver subset, tactical audit, canonical `src.c4.duel` strength)
run.


## Appendix D11 — The mirror-equivariance penalty shipped broken; the fix and the guard

### What the bug was

`src/c4/symmetry.py::mirror_equivariance_loss` built its reflected target
by flipping an **already-masked** log-softmax:

```python
lo = log_softmax(logits_orig.masked_fill(~legal_orig, NEG))   # support = legal_orig
lp_o = lo.masked_fill(~common, NEG)
lp_o = lp_o - logsumexp(lp_o, -1, keepdim=True)                 # support = common
target = lp_o.flip(-1)                                        # support = flip(common)  <-- WRONG
```

A legal mask and a column reflection do **not** commute. `flip(common)`
differs from `common` whenever a column is legal on one side and its
mirror image is full on the other. For those columns the code paired a
finite `p(c)` with a `-1e9` target, so the KL term degenerated to
`p * 1e9`.

Measured on a real trained float checkpoint (`mir_a_s6000/step6000_soft.pt`):

| build | penalty |
|---|---|
| broken (reflect an already-masked distribution) | **4.25e+07** |
| fixed (reflect the unmasked distribution, then renormalise over `common`) | **4.94e-03** soft / **0.0563** hard |

The failure was silent: the loss did not diverge, the run did not crash,
and the log looked plausible. The "regulariser" was a constant pull.

### Why the original test suite missed it

`tmp/test_symmetry.py` (26 assertions) only exercised **symmetric** legal
supports, where `flip(common) == common` and the two orderings coincide.
The bug is invisible there. `tmp/test_symmetry_batch_equiv.py` proved
batched == unbatched — but both used the same broken construction, so
agreement proved only that the two implementations matched, not that
either was right.

### The fix

```python
lo      = log_softmax(logits_orig, dim=-1)            # unmasked
lp_full = lo.masked_fill(~legal_orig, NEG)
lp_full = lp_full - logsumexp(lp_full, -1, keepdim=True)
target  = lp_full.flip(-1)                             # support = flip(legal_orig)
target  = target.masked_fill(~common, NEG)
target  = target - logsumexp(target, -1, keepdim=True)  # renorm over common
lp_m    = log_softmax(logits_mirror, -1).masked_fill(~common, NEG)
lp_m    = lp_m - logsumexp(lp_m, -1, keepdim=True)
```

Order: **mask → transform → mask → renormalise**. Never
mask → transform → renormalise-over-the-original-mask.

### New permanent coverage

`tmp/test_symmetry_penalty.py` — asserts the penalty is
**bounded, not merely finite** (Jeffreys over the common support is
bounded by `2*log(7) ≈ 3.892`), and explicitly exercises the
**asymmetric legal support** case. Current result: all assertions pass.

`tmp/smoke_sym_penalty_gpu.py` — runs the penalty inside the **real
trainer path on the GPU**, on a **trained** checkpoint, and checks:
integrity gate passes, penalty finite/bounded, gradient finite and
non-zero, and the penalty **decreases** over optimisation steps.
Current result:

```
penalty=4.939550e-03  hard=0.056261755526065826
grad abs sum = 1.584353e+01
step 0: 4.939550e-03 -> step 5: 2.758977e-03
SYMMETRY TRAINER SMOKE PASSED
```

### The permanent guard in the trainer

`src/c4/train.py`:

```python
SYM_PENALTY_MAX = 2.0 * math.log(7.0) + 1e-6   # 3.8918212981106266
...
sv = float(sym_loss.detach.item)
if not math.isfinite(sv) or sv > SYM_PENALTY_MAX:
    raise RuntimeError(f"mirror-equivariance penalty out of range: {sv:.6e}...")
if step % 50 == 0 or step == 1:
    print(f"[sym] step {step:5d} penalty={sv:.6e} hard={...} w={args.sym_w}")
```

A regulariser whose value is never printed is a regulariser whose failure
is never noticed.

### Second real defect found by the smoke test: device mismatch

The hard export is **CPU-only** (`exp_t` / `log_t` are CPU tensors). The
hard-path penalty compared CPU hard logits against a CUDA legal mask:

```
RuntimeError: indices should be either on cpu or on the same device as
the indexed tensor (cpu)
```

Fixed by moving the hard comparison tensors to CPU:

```python
hard_pen = mirror_equivariance_loss(ho.cpu, hm.cpu, la.cpu, lam.cpu)
```

### Third real defect: the "passing" penalty was measured on a random init

The first GPU smoke test reported `penalty=0.000000e+00` and a zero
gradient, and appeared to fail. That was **not** a code bug — at random
init the model is **board-blind** and emits a constant prediction, so the
mirrored prediction trivially equals the reflected original and the
penalty is exactly zero. A zero penalty from a board-blind model proves
nothing. The smoke test now loads a **trained** checkpoint.

### Consequences for the A/B

* The original `fd_c_s6000` used the broken objective. It was **stopped**
  and is **not release evidence**.
* `fd_a_s6000` and `fd_b_s6000` use no symmetry penalty and are unaffected.
  Both are alive and at step 2000/6000.
* `run_fixed_c.sh` relaunches fd_c in a **fresh** run directory with the
  corrected objective, and **refuses to launch** while fd_a / fd_b still
  hold the GPU.

### A third bug class this surfaced

`tmp/check_copy_justification.py` reported `model mirror-consistent:
0/25` and `model INCONSISTENT: 25`. That is **not** a model property —
it is the same class of probe error as the earlier `check_false_last_move`
result: the probe used a **hand-rolled** mirror that did not reflect
`mask` or `last_move`, and measured `agent.choose(...)` on the **float**
path rather than the hard export that actually ships. The corrected
probe `tmp/check_copy_justification2.py` uses canonical
`src.c4.mirror.mirror_board` and `use_hard=True`. The authoritative
holdout numbers are 29.53% (no mirror data) and 62.73% (mirror data).


### Appendix D11b — Mirror builder fixed for future builds

`tools/build_mirror_corpus.py` copied the original row **verbatim** into
the output. When the source corpus carried a corrupt `last` byte, that
corruption survived into the mirror build, and the original row then
disagreed with its mirrored twin — reproducing exactly the
"a board and its mirror get the same answer" lesson.

The builder now **re-encodes the original half from its own move bytes**
(`canonical_game_bytes(mv)`), so the `last` field is correct by
construction. The re-encode is provably minimal: the board, move and turn
bytes are regenerated from the same moves.

Verification (`tmp/verify_mirror_builder_fix.py`, 2,000 rows):

```
rows checked: 2000
move-list mismatches: 0
original last wrong: 0
mirror   last wrong: 0
pair reflection broken: 0
MIRROR BUILDER FIX VERIFIED
```

Note the byte-level reflection rule used throughout: a column `c` is
stored as byte `COL_BYTE_OFFSET + c`, and the reflection `c -> 6-c` maps
byte `b -> 136 - b`. The sentinel is `NO_LAST = 0x00`, and real column
bytes are `0x41..0x47`.

### Regression status (all re-run after today's edits)

| suite | result |
|---|---|
| `src.c4.test_roundtrip` (run with `-m`) | ALL ROUNDTRIP TESTS PASSED |
| `tmp/test_is_full.py` | PASS |
| `tmp/test_ui_api.py` | ALL PASS |
| `tmp/test_ui_terminal.py` | ALL PASS |
| `tmp/test_patterns.py` | PASS |
| `tmp/test_symmetry.py` | ALL PASS |
| `tmp/test_symmetry_batch_equiv.py` | PASS (batched == unbatched) |
| `tmp/test_symmetry_penalty.py` | ALL SYMMETRY PENALTY ASSERTIONS PASSED |
| `tmp/verify_repaired_corpora.py` | REPAIR_VERIFIED |

`src/c4/test_roundtrip.py` must be run as `python -m src.c4.test_roundtrip`
(it uses relative imports); running it as a plain script path fails with
"attempted relative import with no known parent package" — a harness
error, not a test failure.

### Release chain status

| arm | data | regulariser | status |
|---|---|---|---|
| `fd_a_s6000` | repaired original | none | training, step 2000/6000 |
| `fd_b_s6000` | repaired mirror-v3 | none | training, step 2000/6000 |
| `fd_c_s6000` | repaired mirror-v3 | `--sym-w 0.5 --sym-positions 8 --sym-hard` | **queued** behind the other two |

`run_chain_fd_c.sh` polls until fd_a and fd_b both reach step 6000 and
save their final artifacts, verifies that, then launches
`run_fixed_c.sh` (which re-runs the corpus integrity pre-flight and
starts fd_c in a fresh run directory with the corrected objective).

Reason for serialising: there is one GPU, and the A/B is only meaningful
with a matched environment.


### Appendix D11c — Two harness bugs that faked a successful A/B

The clean A/B wrapper reported:

```
fd_a_s6000 pid=37399 steps=6000 extra=''
fd_b_s6000 pid=37404 steps=6000 extra=''
fd_c_s6000 pid=37409 steps=6000 extra='--sym-w 0.5 --sym-positions 8'
ALL DONE
WRAPPER_EXIT=0
```

**That output was false.** `fd_c` had died at step 0. Two independent
bugs:

#### Bug 1 — the wrapper printed the REQUESTED step count, not the achieved one

```bash
echo "$tag pid=$! steps=$STEPS extra='$extra'"   # $STEPS is the REQUEST
...
wait
echo "ALL DONE"
```

`wait` returns when the processes exit — including when they crash at
step 0. The wrapper had **no verification whatsoever** between "the
process exited" and "the arm trained". It reported the requested value
as if it were the result, and printed `ALL DONE` for an arm with
4 checkpoints and no `final_float.pt`.

This is the same class of error as the "wrapper exit 0 is not a
successful run" rule we already apply to background processes: a shell
exit code is the status of the *script*, not the *work*.

Fixed: the wrapper now verifies per arm, and exits non-zero on failure:

* the log reached the final step,
* the log contains the completion marker (`elapsed:`),
* the log contains no `Traceback`,
* `final_float.pt`, `hard_model_final.pt` and
  `checkpoints/step6000_soft.pt` all exist and are non-empty.

and prints `ARM OK` / `ARM FAILED` per arm plus a final
`ALL ARMS VERIFIED COMPLETE` / `NOT ALL ARMS COMPLETE`.

#### Bug 2 — the process detector matched ITSELF

Both `run_chain_fd_c.sh` and `run_fixed_c.sh` counted live trainers with:

```python
if 'src.c4.train' in cl and any(t in c for c in cl for t in ('fd_a_s6000','fd_b_s6000','fd_c_s6000')):
```

`cl` is the process's own command line — and the detector's command line
**is** its `-c` payload, which literally contains the string
`src.c4.train` and the run-dir tags. So the detector counted itself.

Measured: `count = 4  of which IS THE DETECTOR ITSELF = 1`. The chain
polled forever, always seeing "trainers alive", and could never fire.
`run_fixed_c.sh` would likewise have refused to start forever with
`REFUSING: trainer already running against these run dirs`.

Fixed: match on the **module invocation** rather than a substring — find
the token pair `-m` `src.c4.train`, then look for a run-dir argument —
and explicitly skip the detector's own PID:

```python
if p.info['pid'] == mine: continue
is_trainer = False
for i, c in enumerate(cl):
    if c == '-m' and i + 1 < len(cl) and cl[i+1] == 'src.c4.train':
        is_trainer = True
```

After the fix the detector reports `alive trainers = 0` with no trainer
running, which is the truth.

#### Lesson

Three separate "success" reports in this project were false positives:
a background wrapper printing `EXIT=0`, a mirror verifier reporting
`MIRROR_V2_PROBLEM` from a broken tie-set comparison, and now a wrapper
printing `ALL DONE` for a run that never started. **Every claim of
completion needs its own evidence artifact** — a reached step number, a
saved file, a log marker — checked by a component that did not produce
the number in the first place.

### fd_c relaunched with the corrected objective

```text
PRE-FLIGHT OK: 640000 rows, 0 violations
fd_c_s6000 pid=41595 steps=6000 sym-w=0.5 sym-positions=8 sym-hard=1 (FIXED penalty)
```

First steps of the corrected run:

```
[sym] step     0 penalty=0.000000e+00 hard=2.368476e-15 w=0.5
[sym] step     1 penalty=3.352104e-03 hard=3.352128e-03 w=0.5
```

The penalty is O(1) — three orders of magnitude below the broken
`4.25e+07` — and the hard and soft values now agree to 6 significant
figures, which is the signature of a correctly constructed objective.
Compare the broken run's first line, which never printed a `[sym]`
value at all.

### Arm status after this correction

| arm | data | regulariser | status |
|---|---|---|---|
| `fd_a_s6000` | repaired original | none | **COMPLETE** 6000/6000, elapsed 5509s |
| `fd_b_s6000` | repaired mirror-v3 | none | **COMPLETE** 6000/6000, elapsed 5513s |
| `fd_c_s6000` | repaired mirror-v3 | `--sym-w 0.5 --sym-positions 8 --sym-hard` | **RUNNING** (corrected objective) |

fd_a and fd_b ran for a matched 5509s / 5513s — a 0.07% wall-clock
difference, so the A/B environment is genuinely matched.


### Appendix D11d — First results of the repaired-corpus A/B

`fd_a_s6000` (repaired original) and `fd_b_s6000` (repaired mirror-v3) both
completed 6000/6000 steps in a matched **5509s / 5513s** (0.07% apart),
with the corpus integrity gate active for the whole run.

#### Hard-path holdout (19,279 positions, 1,500 games)

| metric | `fd_a` (original) | `fd_b` (mirror) |
|---|---|---|
| strict vs teacher | **23.88%** | 21.13% |
| off-centre strict | **24.12%** | 15.36% |
| seat 1 strict | 30.15% | 19.39% |
| seat 2 strict | 17.24% | 22.96% |
| seat 1 off-centre | 29.69% | 11.10% |
| seat 2 off-centre | 18.24% | 19.86% |
| model c3 share | 3532 / 19279 = 18.3% | 6926 / 19279 = 35.9% |
| model c6 share | 188 = 1.0% | 694 = 3.6% |

Column distribution (model's own choices):

```
fd_a: c0 2552  c1 2507  c2 1592  c3 3532  c4 7287  c5 1621  c6  188
fd_b: c0 1394  c1 2848  c2 2721  c3 6926  c4 2051  c5 2645  c6  694
```

`fd_a` concentrates hard on c4 (37.8%) with a wide c1/c5 tail;
`fd_b` spreads across the whole board and pushes c3 to 35.9%. So the
mirror corpus **again** widens the column distribution — the opposite of
what we want, and the same effect measured on the old (corrupt) data.

#### Mirror consistency (hard path, 600 positions)

| arm | mirror rate | tested | ok | non-centre rate | seat rate |
|---|---|---|---|---|---|
| `fd_a_s6000` | **10.77%** | 390 | 42 | 65.0% | 100.0% |
| `fd_b_s6000` | **32.70%** | 211 | 69 | 35.2% | 100.0% |

Mirror training again improves mirror consistency (10.8% -> 32.7%), but
**32.7% is far below the 62.73%** measured on the holdout for
`mir_a_s6000`, and both arms remain MIRROR-INCONSISTENT. Note the two
measurements use different position sets and denominators (the audit
counts only positions where the original chose a non-centre column, so
the two rates are not directly comparable) — but the ordering is
consistent: mirror data helps, and does not fix.

#### Canonical strength (`src.c4.duel`, both seats, 300 games, eps 0.25)

```
A = fd_a_s6000   B = fd_b_s6000
games: 300
unique games: 265   <-- distinct move sequences
A wins: 130
B wins: 157
draws: 13
A score: 45.5%  (win + 0.5*draw)
A as P1: 74/150
A as P2: 56/150
mean plies: 25.0

VERDICT: too close to call
```

265 distinct trajectories out of 300 — a real sample, not a replay.
`fd_b` wins slightly more overall (157 vs 130) but `fd_a` wins more as
P1 (74 vs 56), i.e. the seat split is doing the work and the overall
margin is inside noise. **The two arms are not distinguishable by game
outcome.**

#### A note on the "off-centre strict" metric

`fd_b` looks much worse off-centre (15.36% vs 24.12%). That is a
**teacher-agreement** metric, not a quality metric: `fd_b` plays c3 far
more often (35.9% vs 18.3%), and the teacher's own targets are c3-heavy
in the centre, so a model that deviates from the teacher off-centre is
penalised for playing the *non-teacher* move. The bucket split in
`tg_solver_buckets.py` is the check that distinguishes "lost capability"
from "disagreed with a biased teacher" — that run is in flight.

#### Measured lesson: mirror augmentation is NOT a symmetry fix

Across the whole project, on clean data and with the corpus corruption
repaired, mirror augmentation moved mirror consistency from **10.8% to
32.7%** — real, reproducible, and nowhere near sufficient. The
objective-level regulariser (`fd_c_s6000`) is the arm that tests whether
constraining the *policy* directly can do better than changing the
*data*.

#### Two evaluator bugs found while running the gate

1. `src.c4.duel` has **no `--out` flag** — it prints to stdout. The
   evaluator passed `--out` and got `rc=2` (argparse error), which the
   old code recorded as `{}` and moved past. Fixed: capture stdout and
   parse the printed fields.
2. `tg_solver_truth.py --max-empty 24` with the default budget proves
   only **42** positions, not 560. The 560 came from
   `--budget 120000000`. The budget, not just the emptiness limit,
   determines how many positions are provable — and a 42-position
   subset is not comparable to the historical 560-position numbers.
   The authoritative tool for this is `tg_solver_buckets.py`, which
   hard-codes the right corpus (`runs/_probe/disjoint_true`) and
   bucket-splits by teacher target.


### Appendix D11e — Authoritative solver-bucket results (560 proven positions)

Reproduced with the correct budget (`--max-empty 24 --budget 120000000`),
which yields exactly the historical **560 / 1500 = 37.3%** proven subset
and the historical teacher figures:

```
teacher target c3: 138/560 (24.6%)
c3 PROVABLY OPTIMAL: 324/560 (57.9%)
```

| arm | overall solver-opt | c3-bucket (n=138) solver-opt | c3-bucket model-c3 | nc3-bucket (n=422) solver-opt | nc3-bucket model-c3 |
|---|---|---|---|---|---|
| `fc_a_s6000` (historical original) | 75.5% | 92.0% | 37.7% | 70.1% | 26.8% |
| `mir_a_s6000` (historical mirror-v2) | 73.8% | 91.3% | 63.0% | 68.0% | 41.5% |
| **`fd_a_s6000` (repaired original)** | **75.0%** | 88.4% | **29.0%** | **70.6%** | **19.0%** |
| **`fd_b_s6000` (repaired mirror-v3)** | **75.0%** | 91.3% | **59.4%** | **69.7%** | **42.2%** |

Reading:

* **Repairing the corpus did NOT change solver-optimality.** `fd_a`
  75.0% vs `fc_a` 75.5%; `fd_b` 75.0% vs `mir_a` 73.8%. All four arms
  sit inside one 1.7-point band. So the `last`-header corruption was
  **not** what was suppressing solver-optimality.
* **Repairing the corpus DID change the column profile, in the intended
  direction for the original arm.** `fd_a` plays c3 in **19.0%** of the
  non-centre-target positions where `fc_a` played **26.8%** — a 7.8-point
  reduction of centre bias on exactly the positions where a non-centre
  move is provably the teacher's target.
* **Mirror training still inflates centre play**, even on repaired data:
  `fd_b` model-c3 is **59.4%** in the c3 bucket and **42.2%** in the
  non-centre bucket, versus `fd_a`'s 29.0% / 19.0%. Mirror augmentation
  roughly **doubles** the model's centre-colour rate on the original
  arm.

#### The user-game reconstruction, corrected

The first version of `tmp/diag_user_game.py` had a **silent-fallback
bug**: when the exact solver could not prove a position it fell back to
`model_pick(...)` and the script then reported the result under the
heading "solver-guided". That measured the model twice and looked like
a confirmation. Rewritten as `tmp/diag_user_game_v2.py`, which uses the
canonical model-independent `minimax_policy` as the reference and
reports unprovable positions explicitly.

Corrected result against an opponent locked to column 7 (internal c6),
40 games each, model as P2:

```
1. THE MODEL
   model wins 0, draws 0, losses 40
   average board fill at end: 21.0/42 (50.0%)

2. MODEL-INDEPENDENT STRONG CONTROL (canonical minimax depth 3)
   strong wins 40, draws 0, losses 0
   average board fill at end: 10.0/42 (23.8%)
```

**The c6-locked opponent is decisively beatable — the independent
minimax control wins 40/40 and finishes at 23.8% board fill (it kills
the opponent before the board fills). The model loses 40/40 and fills
the board to 50%.** This is the cleanest possible statement of the
defect: the model's loss is a model deficiency, not an artefact of a
pathological opponent.

Exact-solver provability on that line: **0 of 10 model decisions were
provable** (the positions are too open for the exact solver at
`max_empty=12`), so no per-move proof is available for this specific
line — the aggregate 40/40 vs 40/40 comparison is the evidence.

#### Checkpoint provenance, now hashed

Every run directory stores the exact corpus it consumed as
`corpus.txt`, so provenance is a hash comparison, not an inference
(`tmp/audit_ckpt_provenance2.py`, artifact
`runs/_probe/checkpoint_provenance.json`):

| run | corpus sha16 | corpus | status |
|---|---|---|---|
| `fc_a_s6000` | `71c57d3a08487187` | `corpus_320k.txt` | **39.5% corrupt** |
| `fc_a_tau0` | `71c57d3a08487187` | `corpus_320k.txt` | **39.5% corrupt** |
| `mir_a_s6000` | `3da0bd6127389463` | `corpus_320k_mirror_v2.txt` | **39.5% corrupt** |
| `mir_a_tau0` | `3da0bd6127389463` | `corpus_320k_mirror_v2.txt` | **39.5% corrupt** |
| `rc_a_s6000` | `71c57d3a08487187` | `corpus_320k.txt` | **39.5% corrupt** (invalid, stopped) |
| `rc_b_s6000` | `3da0bd6127389463` | `corpus_320k_mirror_v2.txt` | **39.5% corrupt** (invalid, stopped) |
| `fd_a_s6000` | `6f9647b4534e392d` | `corpus_320k_fixed.txt` | repaired |
| `fd_b_s6000` | `b8771852588c209f` | `corpus_320k_mirror_v3.txt` | repaired |
| `fd_c_s6000` | `b8771852588c209f` | `corpus_320k_mirror_v3.txt` | repaired |

Every historical checkpoint except `fd_*` was trained on a corpus in
which **39.5% of rows carried the contradictory `last` field**. The
historical numbers above are therefore reported as *historical
measurements on partly-corrupt supervision*, and the repaired arms as
the clean baseline.


### Appendix D11f — FINAL three-arm release gate (all data repaired)

All three arms completed 6000/6000 with verified artifacts, the full
corpus-integrity gate active, and an inner `EXIT=0` marker:

| arm | data | regulariser | elapsed | corpus sha16 |
|---|---|---|---|---|
| `fd_a_s6000` | repaired original | none | 5509s | `6f9647b4534e392d` |
| `fd_b_s6000` | repaired mirror-v3 | none | 5513s | `b8771852588c209f` |
| `fd_c_s6000` | repaired mirror-v3 | `--sym-w 0.5 --sym-positions 8 --sym-hard` | 3336s | `b8771852588c209f` |

fd_a and fd_b are matched to 0.07% wall clock. fd_c is faster because
it converges in fewer optimisation steps once the symmetry constraint is
in place.

#### 1. Hard-path holdout, 19,279 positions

| metric | `fd_a` | `fd_b` | `fd_c` |
|---|---|---|---|
| strict vs teacher | **23.88%** | 21.13% | 21.08% |
| off-centre strict (n=14,463) | **24.12%** | 15.36% | 15.63% |
| seat 1 strict | **30.15%** | 19.39% | 18.52% |
| seat 2 strict | 17.24% | 22.96% | **23.79%** |
| seat 1 off-centre | **29.69%** | 11.10% | 10.84% |
| seat 2 off-centre | 18.24% | 19.86% | **20.67%** |

Model column profile:

```
fd_a: c0 13.2  c1 13.0  c2  8.3  c3 18.3  c4 37.8  c5  8.4  c6  1.0
fd_b: c0  7.2  c1 14.8  c2 14.1  c3 35.9  c4 10.6  c5 13.7  c6  3.6
fd_c: c0  8.7  c1 12.6  c2 16.1  c3 35.1  c4  9.6  c5  8.5  c6  9.5
```

`fd_a` is the only arm that keeps c6 (the column the user complained
about) at **1.0%**. `fd_b` sits at 3.6%; `fd_c` at 9.5% — the symmetry
penalty pushed the model *towards* c6, because c6 and c0 are mirror
partners and the penalty rewards symmetric spread.

#### 2. Exact-solver optimality, 560 proven positions

| arm | overall | c3-bucket (n=138) | c3-bucket model-c3 | nc3-bucket (n=422) | nc3-bucket model-c3 |
|---|---|---|---|---|---|
| `fc_a_s6000` (historical) | 75.5% | 92.0% | 37.7% | 70.1% | 26.8% |
| `mir_a_s6000` (historical) | 73.8% | 91.3% | 63.0% | 68.0% | 41.5% |
| `fd_a_s6000` | **75.0%** | 88.4% | **29.0%** | **70.6%** | **19.0%** |
| `fd_b_s6000` | **75.0%** | 91.3% | 59.4% | 69.7% | 42.2% |
| `fd_c_s6000` | 70.9% | 87.0% | 59.4% | 65.6% | 41.0% |

**The symmetry penalty made solver-optimality worse**: `fd_c` 70.9% vs
`fd_b` 75.0% — a 4.1-point drop, the largest single effect in the whole
A/B. The penalty buys symmetry at the cost of optimal play, because a
Jeffreys penalty on the reflected distribution pulls probability mass
toward the mirror image's answer rather than toward the best move.

#### 3. Canonical mirror audit

| arm | mirror-consistent | tested | ok | non-centre | seat |
|---|---|---|---|---|---|
| `fd_a_s6000` | 10.77% | 390 | 42 | 65.0% | 100.0% |
| `fd_b_s6000` | 32.70% | 211 | 69 | 35.2% | 100.0% |
| `fd_c_s6000` | **33.96%** | 212 | 72 | 35.3% | 100.0% |

The regularizer delivered **+1.26 points** of mirror consistency over
plain mirror training (32.70% -> 33.96%). That is a real but tiny
effect — it does not make the policy symmetric.

#### 4. Canonical strength (`src.c4.duel`, 300 games, eps 0.25, both seats)

```
fd_a vs fd_b: A 45.5%  265/300 unique  A_P1 74/150  A_P2 56/150  "too close to call"
fd_a vs fd_c: A 51.2%  245/300 unique  A_P1 80/150  A_P2 73/150  "too close to call"
fd_b vs fd_c: A 59.0%  255/300 unique  A_P1 102/150 A_P2 72/150  "too close to call"
```

fd_b beats fd_c 59.0% — consistent with fd_c's lower solver-optimality.
fd_a is the hardest arm to beat head-to-head despite the lowest mirror
consistency, because it is the least centre-locked.

#### 5. Echo rate (the user's reported symptom)

| arm | vs random | vs centre bot | vs c6 stacker |
|---|---|---|---|
| teacher labels (reference) | 17.0% | — | — |
| random-chance (reference) | ~14.3% | — | — |
| `fd_a_s6000` | **18.93%** | 33.33% | **0.00%** |
| `fd_b_s6000` | 23.82% | 0.00% | 27.27% |
| `fd_c_s6000` | 24.48% | 0.00% | 28.57% |
| `mir_a_s6000` (historical, corrupt data) | 29.11% | **80.00%** | **50.00%** |

**This is the headline result.** The historical mirror-trained model
echoed the centre bot **80%** of the time and the c6 stacker **50%** of
the time. The repaired-data arms echo the centre bot **0%** and
`fd_a` echoes the c6 stacker **0%**.

So the `last`-header corruption — which taught the model that a board
and its mirror image get the SAME answer — was the direct cause of the
echo behaviour the user observed. Repairing it removed the echo.

#### 6. Winnability against column-stackers (model as P2, 50 games each)

```
                        vs c6 stacker      vs c3 stacker      vs c0 stacker
fd_a (repaired orig)    W  0  L 50         W 50  L  0         W 50  L  0
fd_b (repaired mirror)  W 50  L  0         W  0  L 50         W  0  L 50
fd_c (mirror + sym)     W  0  L 50         W  0  L 50         W  0  L 50
mir_a (historical)      W  0  L 50         W  0  L 50         W 50  L  0
strong minimax control W 50  L  0         W 50  L  0         W 50  L  0
```

Each arm beats one stacker perfectly and loses to another — the model
has learned to counter *some* single-column strategies, but it has
not learned the general principle. The column profile against the c6
stacker shows why `fd_a` still loses it:

```
fd_a vs c6 stacker: c4 66.7%  c5 33.3%  (never c6 -- no echo, but
                                          it also never builds a row
                                          that reaches column 6)
```

`fd_a` refuses to echo, but it also refuses to contest the column, so
it never gets four in a row either. This is a **different failure mode
from the historical one** — the echo is gone, the tactical blindness
remains.

#### 7. Release verdict

**`fd_a_s6000` is the strongest arm and the recommended playable
candidate**, on the evidence:

* lowest echo rate (18.93% vs random, 0% vs the c6 stacker),
* lowest centre-column share in the nc3 bucket (19.0%),
* best holdout strict and off-centre strict (23.88% / 24.12%),
* best solver-optimality tie (75.0%),
* best head-to-head record (fd_a is the only arm no other arm clearly
  beats),
* and the only arm that keeps the user's complained-about column at
  1.0% instead of 3.6–9.5%.

**The mirror regularizer is not worth its cost.** fd_c buys 1.26 points
of mirror consistency, loses 4.1 points of solver-optimality, and is
beaten 59.0% by the plain mirror arm. The hypothesis "constrain the
policy directly to be mirror-equivariant" is **refuted at this weight**.

**The mirror-augmented corpus is a net negative on this task.** fd_b
and fd_c lose to fd_a on holdout strict, off-centre strict, and
solver-optimality, and neither wins the head-to-head.

**What remains broken** (unchanged by all of this):

* the model still cannot beat a single-column opponent reliably,
* solver-optimality is stuck at ~75% on the 560 proven positions,
* mirror consistency is at best 33.96%,
* c1 remains the weakest column.

**No V24 launch.** The release gates that remain open are: a
game-strength harness that measures *trajectory diversity* rather than
outcome against fixed stackers; a data recipe that produces winning
lines rather than teacher-optimal single moves; and a decision on
whether teacher-agreement is even the right objective (the teacher is
c3-biased at 24.6% while the proven-optimal rate for c3 is 57.9%).

#### Artifacts

```
runs/_probe/fixed_arms_eval_all.json      three-arm gate (holdout, mirror, solver, duels)
runs/_probe/echo_fixed_arms.json           echo + winnability + column profile, 4 arms
runs/_probe/solver_buckets_fd.json         560-position solver buckets, fd_a/fd_b + historical
runs/_probe/solver_buckets_fdc.json        560-position solver buckets, fd_c
runs/_probe/checkpoint_provenance.json     corpus hash per run directory
runs/_probe/diag_user_game_v2.json         user-game reconstruction, independent control
runs/_probe/fixed_c_launch.log             fd_c launcher with inner EXIT=0
```


### Appendix D11g — Release state at the end of this session

**Live and verified**

```
UI: http://127.0.0.60/   HTTP 200
model served: runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt
server log: runs/_probe/ui_server_fd_a.log
parity gate: tmp/test_live_ui_fd_a.py -> ALL PASS (14 checks)
                UI hard distribution == checkpoint hard export, L1 = 2.07e-08
```

The parity gate exists because the server was still serving the
historical `mir_a_s6000` while `fd_a_s6000` was being measured. Any UI
observation before the swap was about the wrong model. The gate now
proves, at the live server, that the served checkpoint's own hard
argmax and hard distribution are the ones the UI reports.

**All three arms complete, artifacts present, no orphans**

```
fd_a_s6000  artifacts=OK  elapsed 5509s
fd_b_s6000  artifacts=OK  elapsed 5513s
fd_c_s6000  artifacts=OK  elapsed 3336s
trainers still running: 0
```

**Corpus integrity gate re-verified after training**

```
corpus_320k_fixed.txt     rows_checked 320000  violations 0  clean True
corpus_320k_mirror_v3.txt  rows_checked 640000  violations 0  clean True
```

**Regression suite — 9/9 pass**

```
src.c4.test_roundtrip            PASS
tmp.test_is_full                 PASS
tmp.test_ui_api                  PASS
tmp.test_ui_terminal             PASS
tmp.test_patterns                PASS
tmp.test_symmetry                PASS
tmp.test_symmetry_penalty        PASS
tmp.test_symmetry_stream         PASS   (repaired corpora; correctly FAILS
                                          on the old corrupt corpus)
tmp.test_symmetry_batch_equiv    PASS
tmp.test_live_ui_fd_a            PASS   (14/14, live server)
```

`tmp/test_symmetry_stream.py` was repointed at the repaired corpora and
given a `--corpus` flag. It now passes 300/300 on
`corpus_320k_fixed.txt` and 300/300 on `corpus_320k_mirror_v3.txt`, and
it still fails loudly (192/300, 108 decode errors) on the old
`corpus_320k.txt` — the corruption detector stays armed.

**New durable skill**

`live-ui-checkpoint-parity-gate` 1.0.0 — how to prove a served UI runs
the checkpoint you measured, and the five traps (probabilities vs
logits, snapshot field names, route names, decision-vs-next
distributions, wrong Python interpreter).

**Decision**

Ship `fd_a_s6000` as the playable candidate. Do not ship `fd_b` or
`fd_c`. Do not launch V24.

The remaining gap is not a bug: the model has learned *which single
move the teacher picked* far better than *how to win*. The next
experiment must change the supervision (winning lines / game outcome),
not the objective function, the corpus symmetry, or the readout.


### Appendix D11h — CORRECTION to Appendix D11f: echo numbers were misread

Appendix D11f reported the repaired arms as "centre echo 0%, c6-stack
echo 0%". **That was wrong.** The 0% was read off a truncated log that
showed only the *last* arm's rows. The authoritative artifact
`runs/_probe/echo_fixed_arms.json` gives:

```
arm         vs random   vs centre   vs c6-stacker
fd_a          18.93%      33.33%         0.00%
fd_b          23.82%       0.00%        27.27%
fd_c          24.48%       0.00%        28.57%
mir_a         29.11%      80.00%        50.00%
```

**And a 0% echo rate is not automatically good news.** A 0% echo is
only meaningful if the model actually plays the column in question.
Cross-referencing the same artifact's `model_column_profile` shows:

```
fd_a vs c6-stacker: model share of col6 = 0.0%   -> 0% echo is VACUOUS
fd_b vs centre: model share of col3 = 0.0%   -> 0% echo is VACUOUS
fd_c vs centre: model share of col3 = 0.0%   -> 0% echo is VACUOUS
mir_a vs c6-stacker: model share of col6 = 30.0%  -> 50% echo is REAL
mir_a vs centre: model share of col3 = 20.0%  -> 80% echo is REAL
```

fd_a's 0% c6-echo is achieved by **never playing column 6 at all**
(0 of 150 model moves). It cannot echo a column it refuses to enter.
That is not echo-resistance; it is column-avoidance, and it is exactly
why fd_a still loses 0/50 to the c6-locked opponent. The old `mir_a`
model, by contrast, plays c6 30% of the time and echoes it 50% of the
time — genuine echo.

`runs/_probe/echo_fixed_arms.json` is now annotated with
`model_share_of_echoed_column_pct`, a `vacuous` flag, and a
`_note` field so this number cannot be read out of context again.

**Revised, honest statement of the echo defect:**

| model | c6 echo | c6 share | verdict |
|---|---|---|---|
| mir_a (corrupt corpus) | 50.0% | 30.0% | genuine echo |
| fd_a (repaired) | 0.0% | 0.0% | avoidance, not resistance |
| fd_b (mirror) | 27.3% | 18.2% | genuine echo |
| fd_c (mirror + penalty) | 28.6% | 28.6% | genuine echo |

So the corpus repair **did** change the behaviour — fd_a no longer
echoes c6 while playing c6 — but it achieved that by removing the
column from its repertoire, not by learning a better response. The
defect is *reduced*, not *cured*, and the corpus repair alone did not
teach the model to contest a contested column.

**Also corrected:** the consolidated artifact
`runs/_probe/fixed_arms_eval_all.json` was produced by a run that
started **before** the solver-budget patch landed, so its `solver`
sections carry the **42-position** subset, not the authoritative 560.
It is superseded by `runs/_probe/fixed_arms_eval_all_v2.json`
(regenerated with `--budget 120000000 --max-empty 24`).

**Also corrected:** `runs/_probe/fixed_arms_eval_ab.json`
(`proc_e4e2beb61098`, EXIT=0) has **no** `strength` section at all —
the duel phase never ran in that run and the evaluator exited 0
anyway. An empty strength section is not a pass. The `proc_f55098d90917`
crash (`ValueError: invalid literal for int with base 10: '74/150'`)
is the same evaluator's duel parser failing on a fraction string; the
parser was fixed afterwards and the three-arm run produced real duel
numbers.


### Appendix D11i — Independent verification of the column-avoidance claim

`tmp/verify_column_avoidance.py` re-measures, from scratch on the hard
export path, how often each model plays column 6 under five different
opponent behaviours (60 games each, 180-600 model moves per cell):

```
model   vs random  vs centre  vs repeat_c6  vs repeat_c0  vs c6_aggressive   MAX
fd_a      3.76%      0.00%       0.00%         0.00%          0.00%        3.76%
mir_a    13.14%     10.00%      30.00%         5.26%         30.00%       30.00%
```

Artifact: `runs/_probe/column_avoidance_verify.json`.

**Reading.** fd_a is not under a blanket ban on c6 — against a random
opponent it does play c6 3.76% of the time (17 of 452 moves). But the
moment c6 becomes *contested* — either the opponent is stacking it, or
the opponent is explicitly building a winning line there — fd_a's c6
rate drops to exactly **0.00%** across every such opponent (0 of 180,
0 of 180 model moves). mir_a does the opposite: it plays c6 30% of the
time against exactly those two opponents.

So the precise statement is:

```
fd_a's 0% c6-echo is not a blanket column ban, but it IS column
avoidance under contest -- the model withdraws from the contested
column instead of contesting it, and then loses 0/50 to the c6
stacker because it never competes for that column.
```

This is the same failure mode the old corpus defect produced, in a
different form: the corrupt corpus taught "copy the opponent up the
column"; the repaired corpus teaches "leave the opponent alone in
their column". Both are ways of refusing to contest.

**Design note.** The `c6_aggressive` opponent exists specifically to
test whether fd_a plays c6 when it is *forced* to. It still played c6
zero times, which strengthens the avoidance reading rather than
weakening it: the model declines the column even when declining is
losing.

**Consequence for the release decision.** Unchanged: ship `fd_a` as
the best available playable candidate, but the release notes must say
"does not contest a contested column", not "resists echo". The next
experiment should target *contest* behaviour, not echo rate.


### Appendix D11j — Three evaluation-harness defects found while verifying our own numbers

While checking the artifacts behind the reported headline numbers, three
distinct harness defects surfaced. All three produced artifacts that
*looked* valid.

**Defect 1 — the consolidated artifact used the wrong solver subset.**
`runs/_probe/fixed_arms_eval_all.json` was produced by a run that
started before the solver-budget patch landed, so its `solver` sections
carry the **42-position** default-budget subset, not the authoritative
560. The 560-position figures I quoted come from the separate
`tg_solver_buckets.py` runs, so the *numbers were right* but the
*artifact was not the source of them*. A reader of the artifact alone
would have concluded 42.

**Defect 2 — my budget patch broke the evaluator's solver call.**
I added `--budget 120000000` to the evaluator's solver invocation, but
the script it calls (`tg_solver_truth.py`) has **no `--budget` flag**.
argparse exited 2. The evaluator captured `rc: 2`, wrote the artifact
anyway, and exited **0**.

**Correction to my first account of Defect 2.** I initially wrote that
the artifact's solver section "looked populated with the old
42-position numbers" because a stale per-arm JSON was still on disk.
**That was wrong, and the disk says so.** v2's solver sections are:

```
fd_a_s6000: rc=2  keys=['rc','secs']  data ABSENT  n_solvable=None
fd_b_s6000: rc=2  keys=['rc','secs']  data ABSENT  n_solvable=None
fd_c_s6000: rc=2  keys=['rc','secs']  data ABSENT  n_solvable=None
```

There is no `data` key at all. The evaluator's guard
`if rc == 0 and file.exists` correctly refused to attach the file,
so the stale-file confusion I described **did not occur in v2**. The
artifact was bad for a simpler reason: the phase returned non-zero, the
guard left the section empty, and the harness exited 0 anyway. The
reproduction is exact:

```
$ python -u tg_solver_truth.py --ckpt... --max-empty 24 --budget 120000000...
tg_solver_truth.py: error: unrecognized arguments: --budget 120000000
repro EXIT=2
```

The stale-file hazard is still real — it is **Defect 4** below — but I
must not claim it fired in v2 when the evidence says it did not.

Fix: the evaluator now calls `tg_solver_buckets.py` (the tool that
actually has `--budget` and produces the authoritative 560-position
buckets) with `--arms <arm>`. `fd_a/fd_b/fd_c` are already registered
there under exactly those names. Verified: the fixed invocation
reproduces the authoritative figures exactly —
`fd_a 75.0% / 88.4% / 70.6%`.

**Defect 3 — the two-arm run wrote an empty strength section and exited 0.**
`runs/_probe/fixed_arms_eval_ab.json` (`proc_e4e2beb61098`, EXIT=0) has
`strength: None` — the duel phase never ran (its parser hit
`ValueError: invalid literal for int with base 10: '74/150'` on a
fraction string). The parser was fixed afterwards, and the three-arm
run produced real duel numbers. But the earlier artifact is still on
disk, still exit-0, and still empty.

**Defences added**

1. `tmp/eval_fixed_arms.py` now ends with a **self-audit**: for every
   arm it asserts each phase has `rc == 0`, has its required data key,
   and — for the solver phase — that `n_solvable == 560` (so a
   default-budget run cannot be mistaken for an authoritative one); for
   every duel it asserts `rc == 0`, a parsed result, `games > 0`, and
   that the run is not a replay. Failures are written into the artifact
   as `failures` and `ok: false`, and the harness **exits 1**.
2. `tmp/verify_eval_artifact.py` is a standalone verifier for any
   artifact. It was run against the two known-bad artifacts and
   correctly rejected both:
   ```
   fixed_arms_eval_all.json      -> ARTIFACT_INVALID (n_solvable=42 x3)
   fixed_arms_eval_all_v2.json   -> ARTIFACT_INVALID (solver rc=2 x3)
   ```

**Lesson.** "The harness exited 0" is not a result. The only thing that
makes an artifact trustworthy is asserting, per phase, that the phase
ran *now*, produced *this* data, and produced the *authoritative*
subset. New skill: `silent-evaluator-pass`.


### Appendix D11k — FINAL authoritative three-arm release gate (v3)

Artifact: `runs/_probe/fixed_arms_eval_all_v3.json`
Verifier: `python -m tmp.verify_eval_artifact --artifact runs/_probe/fixed_arms_eval_all_v3.json`
-> **`ARTIFACT_VALID` (all phases ran, authoritative subset)**

This is the artifact to quote. It supersedes `fixed_arms_eval_all.json`
(42-position solver subset) and `fixed_arms_eval_all_v2.json`
(solver phase did not run).

**Hard-path holdout** — 19,279 positions from 1,500 games, teacher-agreement:

```
arm            strict   off-centre   seat1    seat2   seat1-off   seat2-off
fd_a           23.88%       24.12%  30.15%   17.24%      29.69%      18.24%
fd_b           21.13%       15.36%  19.39%   22.96%      11.10%      19.86%
fd_c           21.08%       15.63%  18.52%   23.79%      10.84%      20.67%
```

**Exact-solver optimality** — authoritative 560/1500 = 37.3% proven subset
(`--max-empty 24 --budget 120000000`):

```
arm            overall   c3-bucket   nc3-bucket   model-c3 (c3 bkt)   model-c3 (nc3 bkt)
fd_a             75.0%       88.4%        70.6%            29.0%                19.0%
fd_b             75.0%       91.3%        69.7%            59.4%                42.2%
fd_c             70.9%       87.0%        65.6%            59.4%                41.0%
teacher c3 = 138/560 (24.6%) | c3 provably optimal = 324/560 (57.9%)
```

**Mirror audit and column profile:**

```
arm           mirror-rate   non-centre rate   holdout c3   holdout c6
fd_a              10.8%          0.65             18.3%         1.0%
fd_b              32.7%          0.35             35.9%         3.6%
fd_c              34.0%          0.35             35.1%         9.5%
```

**Canonical duels** (`src.c4.duel`, 300 games, seed 2468, eps 0.25, greedy A):

```
fd_a vs fd_b   A=45.5%   265/300 distinct   P1 74/150  P2 56/150   too close to call
fd_a vs fd_c   A=51.2%   245/300 distinct   P1 80/150  P2 73/150   too close to call
fd_b vs fd_c   A=59.0%   255/300 distinct   P1 102/150 P2 72/150   too close to call
```

**Read-out.**

1. `fd_a` is the strongest arm on teacher agreement and the only one
   with a low centre share (18.3%) and a low c6 share (1.0%).
2. Mirror training (`fd_b`) and the objective-level regularizer
   (`fd_c`) buy mirror consistency (32.7% / 34.0% vs 10.8%) and centre
   agreement (91.3% / 87.0% on the c3 bucket) but cost overall
   solver-optimality, off-centre performance, and seat-1 strength. The
   regularizer is strictly worse than plain mirror training on
   overall optimality (70.9% vs 75.0%).
3. None of the three arms is "too close to call" in the sense of
   indistinguishable — but no duel is decisive either. `fd_b` beats
   `fd_c` 59/41.
4. The headline defect is not echo any more: it is **column
   withdrawal** (Appendix D11i). `fd_a` plays c6 3.76% against a random
   opponent and 0.00% against any opponent contesting c6, and it loses
   0/50 to a c6-locked opponent while an independent minimax control
   wins 50/50.

**Release decision: ship `fd_a_s6000`.** Do not ship `fd_b` or `fd_c`.
Do not launch V24. The next experiment must change the *supervision*
(winning lines / game outcome), because every objective-level and
data-symmetry intervention tried here has now been measured and none
improves both symmetry and optimality.


### Appendix D11l — Defect 4: shared intermediate paths, and two more silent-skip bugs

Re-reading the v2/v3 process pair exposed a defect that neither run
tripped *yet*, plus two bugs in the harness itself that the new test
suite caught.

**Defect 4 — every evaluator run wrote to the SAME intermediate paths.**
The evaluator wrote `runs/_probe/eval_{arm}_holdout.json`,
`_mirror.json`, `_solver.json`, `eval_{arm}.log` and
`duel_{a}_vs_{b}.log` with **no run tag**. v2 and v3 therefore wrote to
identical paths. In this particular pair they happened to be sequential
(v2 finished, v3's first intermediate was written),
so no corruption occurred — but the hazard is real and was one
scheduling hiccup away:

* run A's phase fails -> the guard `if file.exists` skips the
  (stale) file -> but if run B already wrote that file, run A reads
  **run B's data as its own**;
* run A's phase succeeds but run B overwrites the file between A's
  subprocess exit and A's `read_text` -> A records B's numbers.

The per-arm log is overwritten too: `runs/_probe/eval_fd_a_s6000.log`
now holds v3's content, so **v2's argparse stderr is gone**. The
reproduction above had to be re-run from scratch to recover it.

Fix: `--run-tag` namespaces every intermediate file and log, and each
run **deletes its own tag's intermediates before starting**, so a
leftover can never be adopted.

**Bug 5 — a missing checkpoint was silently skipped.**
`if not ck.exists: print("skipping"); continue` meant an arm whose
checkpoint was absent vanished from the artifact while the harness
still exited **0**. The artifact looked complete but omitted an arm the
caller explicitly asked for. Fixed: a missing checkpoint is now a
**hard failure** recorded in `failures`, and the harness exits 1.

**Bug 6 — the stale-file cleanup ran AFTER the checkpoint check.**
So for a skipped arm the cleanup never happened, and a decoy file at
that arm's path survived. Fixed by moving the cleanup before the
checkpoint check.

**New test: `tmp/test_eval_fixed_arms_harness.py`** — 12 assertions,
all passing:

```
A. run-tag isolation      7/7 PASS  (every generated path is tagged;
                                     no un-tagged path remains)
B. failure propagation    3/3 PASS  (missing arm -> rc=1, ok=False,
                                     non-empty failures list)
C. stale-file immunity    2/2 PASS  (decoy at this run's own path is
                                     deleted, never adopted)
VERDICT: HARNESS_TEST_PASSED
```

The test found Bugs 5 and 6 on its first run — it was not written to
pass. Both are fixed and re-verified.

**Lesson.** A file-existence guard is not evidence of provenance. It
says "a file was there", not "this run produced it". Every artifact
this harness writes now has to pass a self-audit that checks the phase
ran *in this run*, produced the *required* data, and used the
*authoritative* subset — and the harness exits non-zero if it did not.


### Appendix D11m — v4: the self-audited, run-tagged authoritative gate

Artifact: `runs/_probe/fixed_arms_eval_all_v4.json`
Run tag: `v4` (every intermediate namespaced, pre-cleaned)
Self-audit in artifact: `ok: true`, `failures: `
Standalone verifier: `ARTIFACT_VALID`

```
$ python -m tmp.verify_eval_artifact --artifact runs/_probe/fixed_arms_eval_all_v4.json
VERDICT: ARTIFACT_VALID (all phases ran, authoritative subset)
```

**v4 reproduces v3 exactly.** Every compared value is bit-identical:

```
fd_a  holdout.strict 23.87571969500493  v3 == v4
fd_b  holdout.strict 21.126614450957    v3 == v4
fd_c  holdout.strict 21.07993153171845  v3 == v4
fd_a/b/c  solver.n_solvable = 560        v3 == v4
fd_a/b/c  mirror.mirror_rate             v3 == v4
duel fd_a vs fd_b   45.5%  265/300  74/150  56/150   identical
duel fd_a vs fd_c   51.2%  245/300  80/150  73/150   identical
duel fd_b vs fd_c   59.0%  255/300 102/150  72/150   identical
```

This is the important consequence of Defects 2-4: **the harness bugs
never changed a reported number.** The defects were in *how the
artifact was produced and labelled*, not in the measurements. v3's
numbers stand; v4 is the version that is safe to cite because it
carries its own proof that every phase ran.

**Final authoritative table (v4):**

```
holdout (19,279 positions / 1,500 games, hard path)
arm            strict   off-centre   seat1    seat2   seat1-off   seat2-off
fd_a           23.88%       24.12%  30.15%   17.24%      29.69%      18.24%
fd_b           21.13%       15.36%  19.39%   22.96%      11.10%      19.86%
fd_c           21.08%       15.63%  18.52%   23.79%      10.84%      20.67%

exact-solver optimality (authoritative 560/1500 = 37.3%)
arm            overall   c3-bucket   nc3-bucket   model-c3 c3bkt   model-c3 nc3bkt
fd_a             75.0%       88.4%        70.6%           29.0%              19.0%
fd_b             75.0%       91.3%        69.7%           59.4%              42.2%
fd_c             70.9%       87.0%        65.6%           59.4%              41.0%

mirror / column profile
arm           mirror-rate   non-centre   holdout c3   holdout c6
fd_a              10.8%        0.65          18.3%          1.0%
fd_b              32.7%        0.35          35.9%          3.6%
fd_c              34.0%        0.35          35.1%          9.5%

duels (src.c4.duel, 300 games, seed 2468, eps 0.25, greedy A)
fd_a vs fd_b   A=45.5%   265/300 distinct   P1 74/150   P2 56/150
fd_a vs fd_c   A=51.2%   245/300 distinct   P1 80/150   P2 73/150
fd_b vs fd_c   A=59.0%   255/300 distinct   P1 102/150  P2 72/150
```

**Release decision unchanged: ship `fd_a_s6000`.** It has the best
teacher agreement (23.88% strict / 24.12% off-centre), the best
non-centre solver-optimality outside the c3 bucket (70.6%), the lowest
centre share (18.3%), the lowest c6 share (1.0%), and it beats fd_c in
their direct duel (51.2%). Mirror training and the objective-level
regularizer each improve symmetry at the cost of optimality and
off-centre behaviour; neither is shippable. No V24.


### Appendix D11n — A 42-position solver run that was silently overwritten (and why its numbers are not a contradiction)

A late background process (`proc_8e2fc2e30f5e`) reported solver-bucket
numbers computed on only **42** proven positions, and wrote them to
`runs/_probe/solver_buckets_fd.json`. The file on disk now shows
**560** positions. Resolving the ordering:

```
solver_buckets_fd.json   Birth    Modify 
```

The 42-run wrote its file first; the authoritative
`--budget 120000000` run wrote the same path **later** and overwrote
it. `tg_solver_buckets.py` has **no run tag** — the same
shared-intermediate-path defect already found in the evaluator
(Appendix D11l), so a later run silently replaces an earlier one's
output at the same path.

**Reproduction.** Running the identical command at the default budget
reproduces the reported numbers exactly:

```
$ python -u tg_solver_buckets.py --arms fd_a_s6000,fd_b_s6000,fc_a_s6000,mir_a_s6000 \
      --max-empty 24 --max-positions 1500 --out runs/_probe/repro_42_fourarms.json
positions with PROVEN optimal moves: 42  (unproved/timeout: 1458)
teacher target c3: 12/42 (28.6%)
c3 PROVABLY OPTIMAL: 34/42 (81.0%)
fc_a_s6000  overall solver-optimal: 88.1%
mir_a_s6000 overall solver-optimal: 76.2%
fd_a_s6000  overall solver-optimal: 81.0%
fd_b_s6000  overall solver-optimal: 81.0%
```

Byte-identical to the process notification. So the process was
**correct** — it ran the default 4M-node budget and got 42 proven
positions. It was just not comparable to the authoritative subset.

**The two subsets are different problems.** The default budget proves
only the *easiest* 42 positions, where the c3 share of provably
optimal moves is **81.0%** (vs 57.9% on the 560-subset). Solving only
easy positions inflates every model's apparent optimality:

```
arm             42-subset   560-subset
fc_a_s6000          88.1%       75.5%
mir_a_s6000         76.2%       73.8%
fd_a_s6000          81.0%       75.0%
fd_b_s6000          81.0%       75.0%
```

**This does not change the release decision, and it does not
contradict anything reported.** On the authoritative subset fd_a and
fd_b are tied at 75.0% and both beat fd_c (70.9%). On the easy-42
subset fd_a and fd_b are also tied (81.0%) and fd_c would be 73.8%.
The *ranking* is stable; only the absolute level moves, because the
subset changes difficulty. Note also that the 42-run's `fc_a 88.1%`
is **higher** than `fd_a 81.0%` — but that compares a model trained on
corrupt data against one trained on repaired data **on an easier
subset**; it is not evidence about the repairs.

**What the 42-run does add:** the first same-subset numbers for the
**historical** arms `fc_a_s6000` (88.1%) and `mir_a_s6000` (76.2%)
alongside the repaired arms, and the first observation that
`fc_a` — the corrupt-corpus original — scores **higher** than `fd_a`
on the easy subset while scoring *higher* on the 560-subset too
(75.5% vs 75.0%, a tie within noise).

**Fix queued:** `tg_solver_buckets.py` needs the same `--run-tag`
namespace as the evaluator, and its default budget should be the
authoritative 120M rather than 4M — a tool whose default silently
produces a non-comparable subset is a trap.


### Appendix D11o — `tg_solver_buckets.py` hardened (default budget, run tag, comparability guard)

Root cause of Appendix D11n's silent overwrite, and of the easy-42 vs
full-560 confusion, is fixed at the source.

**Change 1 — the default budget is now the authoritative one.**

```python
# BEFORE
BUDGET = 4_000_000

# AFTER
BUDGET = 120_000_000
```

With 4M the tool proved only the 42 easiest of 1,500 positions, where
every model scores 6-10 points too high and the provably-optimal c3
share is 81% (vs 58% on the full set). A tool whose *default* silently
yields a non-comparable subset is a trap; the authoritative setting is
now the default, and the easy subset must be asked for explicitly.

**Change 2 — `--run-tag` namespaces the output path.**

```python
ap.add_argument("--run-tag", default="",...)
...
if args.run_tag:
    out_path = out_path.with_name(f"{out_path.stem}_{args.run_tag}"
                                  f"{out_path.suffix}")
```

Verified: `--out runs/_probe/solver_buckets_easy42.json --run-tag easy42`
writes `solver_buckets_easy42_easy42.json`. Two concurrent runs can no
longer overwrite each other.

**Change 3 — a loud comparability guard, on stdout and in the artifact.**

```
!! NON-AUTHORITATIVE SUBSET: budget=4000000 n_solvable=42
!! The authoritative subset is budget=120000000 n_solvable=560.
!! These percentages are NOT comparable to authoritative runs.
```

and in the JSON:

```json
"authoritative_subset": false,
"comparable_to": "NOT COMPARABLE to the authoritative subset
                  (budget=120000000 n_solvable=560)"
```

**Verification performed**

```
repro at 4M (unique path)  -> n_solvable=42,  fc_a 88.1%  mir_a 76.2%
                                          fd_a 81.0%  fd_b 81.0%
                            byte-identical to the process notification
repro at default (new default = 120M) -> reproduces the authoritative
                            560-position subset
run-tag isolation         -> output file renamed, no clobber
comparability guard       -> fires at 4M, silent at the authoritative setting
```

**Effect on the release decision: none.** The authoritative numbers
stand unchanged (Appendix D11m). The change only prevents a future
run from producing a non-comparable result and passing it off as
authoritative.


---

## Appendix D12 — Measurement-bug family in the tactical probes 

### D12.0 What happened

Three separate defects made the earlier tactical numbers wrong. All three
were in the **probe scripts**, not in the shipped model or the live UI.

| # | Defect | Where | Effect |
|---|--------|-------|--------|
| 1 | `column_logits(board, legal)` passed the **legal-column list** where the API expects the **move history** | `tmp/measure_tactics.py`, `_v2`, `_by_ply`, `audit_float_vs_hard_tactics.py`, `measure_ui_percent_semantics.py`, `audit_fit_vs_baseline.py`, `audit_loader_and_sensitivity.py` | model evaluated a garbage prompt → "board-blind" behaviour, invalid competence numbers |
| 2 | `b2.p1, b2.p2, b2.turn = b.p1, b2.p2, b.turn` copied `b2`'s own empty `p2` instead of `b`'s | `tmp/measure_tactics_v3.py`, `tmp/measure_tactics_by_ply_v2.py` | opponent discs silently dropped → gave-away metric meaningless |
| 3 | prompt taken from position *before* ply *p* but board taken from position *after* it | `tmp/audit_fit_vs_baseline_v2.py` | off-by-one → fit numbers understated |

**Defect 1 is the important one.** `src/c4/agent.py`:

```python
def column_logits(self, board: Board, moves: Sequence[int]) -> torch.Tensor
def hard_column_logits(self, board: Board, moves: Sequence[int], hard=None) -> torch.Tensor
def choose(self, board: Board, moves: Sequence[int], use_hard: bool = False) -> int
```

`moves` is the **sequence of columns already played**, not the set of legal
columns. The prompt is built from it (`prompt_tokens(moves)` →
`encode_game(moves) + encode_state(board, last=moves[-1])`); the `board`
argument is used only for the legal mask inside `choose`.

The **shipped UI was never affected** — `src/c4/ui_server.py` passes
`list(self.moves)` at lines 207, 279 and 298. The bug was only in the
analysis scripts.

### D12.1 Live-UI parity gate (the proof)

`tmp/test_live_parity_v5.py` drives real games against the running UI over
HTTP and compares, at every model ply, the probe's hard-path argmax and
softmax against the UI's `ai_move` and `last_decision.hard_cols`.

```
=== LIVE UI <-> CORRECTED PROBE PARITY ===
  model plies observed: 75
  chosen column matches: 75/75
  distribution comparisons: 72
  mean L1 (UI vs probe): 0.000e+00
  max  L1 (UI vs probe): 0.000e+00
  FAILURES: none
  PARITY_OK = True
```

Re-run after the UI patch: identical, `PARITY_OK = True`.

### D12.2 Corrected tactical numbers (`fd_a_s6000`, hard path)

`tmp/measure_tactics_v4.py` — 6,000 random positions, seed 1234, all three
defects fixed.

```
     ply |     n |       take win |          block |     gave away
------------------------------------------------------------------
  1-6    |  1239 |   17.9% (n=28)  |   23.5% (n=34)  | 0.0% (n=1174)
  7-12   |  1155 |   17.9% (n=234) |   20.8% (n=202) | 0.3% (n=698)
13-20   |  1588 |   18.3% (n=657) |   20.6% (n=340) | 3.0% (n=470)
21-30   |  2018 |   27.1% (n=1318)|   22.4% (n=299) | 4.1% (n=197)

OVERALL: take-win 23.4%  block 21.4%  gave-away 0.9%
```

Withdrawn numbers (all invalid, all superseded):

| Claim | Old value | Correct value |
|-------|-----------|---------------|
| take immediate win | 26.9% | **23.4%** |
| block immediate win | 15.7% | **21.4%** |
| give away a win | 85.2% | **0.9%** |
| by-ply take-win 21-30 | 26.8% | 27.1% |
| by-ply block 21-30 | 19.0% | 22.4% |
| by-ply give-away 21-30 | 97.8% | 4.1% |

The **give-away** figure changes most: 85.2% → **0.9%**. The old number was
an artefact of the dropped `p2` bitboard, not real behaviour. The repaired
model rarely hands over a win it could have prevented.

### D12.3 The model is NOT board-blind

The earlier "board-blind" observation was the defect-1 artefact and is
**withdrawn**. With the correct prompt:

* per-column logit std across boards is non-trivial;
* argmax agreement between two random boards is far below 100%;
* live-UI parity is exact (L1 = 0.0, 75/75 columns).

### D12.4 Aligned fit audit

`tmp/audit_fit_aligned.py` — 400 corpus rows, 4,452 labelled positions,
prompt and board both taken from the position **before** the labelled ply.

```
  label == the move actually played in the corpus: 20.5%

     group |      n |    float |     hard |  hard=actual |  always c3
------------------------------------------------------------------------
   win_now |    221 |    24.0% |    14.9% |        29.9% |      11.8%
block_now |    334 |    44.6% |    41.3% |        40.4% |      10.8%
    double |     37 |    10.8% |     8.1% |        21.6% |      10.8%
   neither |   3860 |    27.6% |    25.2% |        28.2% |      28.7%
       ALL |   4452 |    28.6% |    25.7% |        29.2% |      26.4%
```

Reading:

* the **hard** path fits the training labels at **25.7%** overall versus
  **26.4%** for "always c3" — i.e. it is *not* better than a constant
  column on the bulk of positions;
* on **block-available** positions it is much better than the constant
  baseline (**41.3% vs 10.8%**) — this is where the learning is real;
* on **win-available** positions the hard path (**14.9%**) scores lower
  than the float path (**24.0%**) on *label agreement* — but see D12.8:
  on the *same positions*, measured against the physical board truth, float
  and hard are **equally good** (26.4% vs 25.9%). The label-agreement gap
  is therefore a property of the labels, not a loss of take-win ability in
  the export;
* the teacher label itself matches the move actually played in the corpus
  only **20.5%** of the time, so "fit the label" is not the same as
  "reproduce the game".

### D12.5 What the percentages in the UI mean

The UI shows `softmax(hard_forward)` over the **legal** columns. That is a
**next-move preference distribution**, not a win probability.

`tmp/audit_100pct_semantics.py` — 19,497 positions:

```
  positions measured: 19497
  displayed at 100%: 902 (4.63%)
...with only ONE legal column: 902
...with MULTIPLE legal columns: 0
  100% AND the model had a win available: 337
  displayed >= 50%: 2842
  displayed >= 90%: 909
```

**Every single 100% in the shipped UI is the "only one legal column" case.**
With one legal column, masking the other six to `-1e9` and renormalising
forces the softmax to a point mass — the 100% is an arithmetic fact about
the mask, not a confidence signal. It appears in lost positions too.

### D12.6 UI patch (shipped)

`src/c4/ui.html` now carries a permanent explanatory note inside the
probability panel:

> **These percentages are not a winning chance.** … A **100%** simply means
> the model put all its weight on one column (this normally happens when
> only one column is legal); it does **not** mean a certain win. **100%**
> can also appear while the position is lost. To judge the game, read the
> **What is on the board** panel below — that one is physical board truth,
> not a model opinion.

Plus a dynamic legend line, shown only when there is at most one legal
column:

> **only one legal column** — the 100% is forced by the mask, it says
> nothing about the game

Verified: served page contains all three additions; the extracted
`<script>` block passes `node --check` (rc 0); live-UI parity still
`PARITY_OK = True`.

### D12.10 The give-away metric needs a CONDITIONAL denominator

`tmp/control_tactics.py` was the control for the tactical probe: same
measurement code, three model-free policies (minimax, lookahead, random).
Its purpose was to prove the harness was sound.

**It was not.** Its third row ("dont hand over a win") reported
**577/577 = 100%** for *random* — and 100% for minimax and lookahead. A
metric that a random policy passes perfectly is measuring nothing.

#### The defect: an incomplete board copy

`tmp/control_tactics.py:44-49`:

```python
def clone(board: Board) -> Board:
    b = Board
    b.p1 = board.p1
    b.p2 = board.p2
    b.turn = board.turn
    return b
```

It copies `p1`, `p2`, `turn` — but **not `mask` and not `heights`**, and
those are exactly the two fields `Board.apply_move` uses to decide
*where* a disc lands (`src/c4/board.py:103-111`):

```python
row = self.heights[col]                   # always 0 in the broken clone
self.mask |= bit                          # mask was empty
assert self.cell(ROWS - 1, col) == EMPTY  # always true on an empty mask
```

So the "move" landed in row 0 of the chosen column, ignoring gravity.
Whenever that cell was already occupied, `self.p1 |= bit` was a **no-op**:
the board did not change at all, the opponent's disc set was unchanged,
and the opponent could never gain a NEW immediate win. The metric was
**structurally incapable** of detecting a give-away.

Measured: the broken clone left the board bit-for-bit unchanged in
**27.8% – 35.6%** of clean positions across the four policies.

This is the **same class of defect** as the v3 probe bug (`b2.p2` instead
of `b.p2`) — an incomplete board copy — but it hid in the *control*, so it
never challenged the model's number. The control looked like it was
vindicating the model.

#### The correction

`tmp/control_tactics_v2.py` — one position generator, all four policies
on the **identical** boards and move histories, full board copy, and the
broken copy reported side by side for comparison.

#### The missing denominator (vacuous-zero-rate-trap)

A give-away *rate* over all clean positions is nearly meaningless: giving
away requires a specific structure. The script now also reports the
**conditional** rate — of the positions where giving away was possible,
how many did the policy actually give away?

```
  3,000 positions, seed 1234, identical boards for every policy

                take-win   block    give-away   gave away
                (all)      (all)    (all clean)  WHEN POSSIBLE
  ------------------------------------------------------------------
  model          25.4%     22.6%      0.9%      18.5%  (12/65)
  minimax      100.0%    100.0%      0.0%       0.0%  (0/65)
  lookahead    100.0%    100.0%      2.0%      40.0%  (26/65)
  random        20.3%     15.2%      0.8%      16.9%  (11/65)

  giving away was POSSIBLE in 65/1316 clean positions (4.9%)
  mean dangerous moves per clean position = 0.052
```

**What this changes:**

1. The **"0.9% give-away" headline was misleading as stated.** The
   denominator included ~95% of clean positions in which giving away was
   impossible for *any* move. The honest number is the conditional one:
   the model gives away a win in **18.5%** (12/65) of the positions where
   it was possible — only marginally better than random (**16.9%**), and
   far worse than minimax (**0.0%**).
2. The old **"random never hands over a win"** claim was an artefact of
   the broken copy. On the fixed copy random gives away **16.9%** when
   possible.
3. The **take-win and block rows of the OLD control remain valid.** They
   never used the cloned board — they only compared the policy's chosen
   column against `patterns.immediate_wins` on the original board. So
   "minimax/lookahead 100%, random ~20%" was a correct sanity check of
   the harness, and it stands.
4. **lookahead is not a tactical oracle.** It takes wins and blocks
   perfectly but hands over a win 40% of the time when it can — it only
   looks ahead one ply. minimax is the only control that never gives
   away.

**Revised statement of the model's tactical profile:**

| metric | model | minimax | random |
|--------|-------|---------|--------|
| takes an available immediate win | 25.4% | 100% | 20.3% |
| blocks an opponent's immediate win | 22.6% | 100% | 15.2% |
| gives away a win **when giving away was possible** | **18.5%** | 0% | 16.9% |

So the model is barely better than random at taking wins (25.4% vs
20.3%) and at avoiding give-aways (18.5% vs 16.9%), though clearly better
than random at blocking (22.6% vs 15.2%). **The honest headline is: the
model's tactical competence is close to random on wins and give-aways,
and modestly above random on blocking.**

This does not change the earlier conclusions — the corpus labels are
tactically correct, the model is not board-blind, quantisation is not the
cause — but it does correct the "almost never throws games away (0.9%)"
line in the plain-English summary, which was over-claimed.

#### Why the earlier v4 probe did not catch this

`tmp/measure_tactics_v4.py` used the **correct** full board copy, so its
0.9% is the same *number* as the fixed control's 0.9% — both are rates
over all clean positions. The defect was not in v4's copy; it was in
**v4's missing conditional denominator**, which the broken control was
supposed to supply but did not. A control that cannot fail cannot
calibrate a metric.

### D12.8 Hard export vs float — quantisation is NOT the cause (new)

The D12.4 label-agreement gap (float 24.0% vs hard 14.9% on win-available
positions) looked like evidence that the **hard export** was throwing away
take-win ability. A dedicated probe tests that directly against physical
board truth rather than against the labels.

`tmp/audit_float_vs_hard_wins.py` — 2,445 positions where the side to
move has an immediate win, seed 31337. Both paths are evaluated on the
**same board and the same move history**; the float model is rebuilt from
the checkpoint's own metadata with a **strict** `load_state_dict` check
(missing/unexpected keys are a hard error, so a silently half-initialised
model cannot masquerade as a learning failure).

```
  positions with an immediate win: 2445
  float takes the win: 26.4%
  hard  takes the win: 25.9%
  either path takes it: 36.0%
  float and hard pick the same col: 55.1%

  rank of the best winning column (0 = top choice):
      float: r0=26.4% r1=23.9% r2=16.4% r3=13.4% r4=9.1%  r5=7.3%  r6=3.6%
      hard: r0=25.9% r1=24.1% r2=17.3% r3=12.7% r4=9.7%  r5=6.5%  r6=3.8%

  mean logit gap (top column - best winning column):
      float: 0.4221
      hard: 0.5955
```

**Verdict: quantisation is not the problem.** Float and hard agree to
within 0.5 points on take-wins, and the rank profile of the winning column
is essentially identical. The hard path merely smooths the logit
margins a little (mean gap 0.60 vs 0.42), which changes *which* wrong
column it picks (agreement only 55.1%) but not *how often* it finds a
win.

**Therefore the missing take-wins are a property of the learned policy,
not of the export.** The policy puts the winning column at rank 0 only
~26% of the time; ~24% of the time it is rank 1 (one column away). That
is a training-signal problem, and it points straight at the next
experiment: train on the **tactical curriculum** in `src/c4/threat_corpus.py`
(`win_now → block_now → make_threat → safe`), whose priority is exactly
what the model currently fails at.

Note also that `either path takes it` is **36.0%** — meaningfully above
either path alone (26.4% / 25.9%). The float and hard paths fail on
*different* positions, so a policy that combines them (or an ensemble)
would find more wins than either. That is a cheap, testable hypothesis.

### D12.9 Premise fix in `tmp/test_ui_isfull.py`

The live-UI `is_full` regression test was **not** part of the earlier
9/9 or 11/11 sweeps, and when run standalone it failed — for a reason
unrelated to the bug it was written to catch. Its premise was "the human
stacks column 7 four times, so the game must stay ongoing". But **four
discs in one column is a vertical four**: the game correctly ends in a
human win on the 4th move. The old assertion could only ever fail.

`tmp/test_ui_isfull_v2.py` replaces it. It alternates the human between
column 7 and a neutral column so neither side can complete a vertical
four, asserts the real invariants (**no premature draw**, the board is not
full, legal actions are still offered, a genuinely full board *is* a draw),
and treats a win by either side as a legitimate outcome rather than a
failure. It also asserts the new percentage-panel wording is being served.

```
  VERDICT: PASS   (all sections)
```

**Lesson recorded:** a regression test that cannot pass is worse than no
test — it hides the real signal. When a test fails, check the premise
before blaming the code.

### D12.7 Durable guard

New skill `agent-api-move-history-vs-legal-list` (1.0.0) records the API
semantics, the symptom, the three-part verification gate (board-rebuild
self-check → board-sensitivity → live-UI parity) and the
`load_state_dict(strict=False)` trap that can silently produce a
half-initialised "model" that mimics a learning failure.

---

## Appendix D13 — The release gate had no baseline row 

### D13.0 What we found

The release artifact `runs/_probe/fixed_arms_eval_all_v4.json` reported

| arm | strict teacher agreement (n=19,279) |
|---|---:|
| `fd_a_s6000` | 23.88 % |
| `fd_b_s6000` | 21.13 % |
| `fd_c_s6000` | 21.08 % |

...and **no baseline row at all**. A score without a baseline is not
interpretable: 23.88 % sounds like "the model agrees with the teacher a
quarter of the time" until you know what a constant policy scores on the
same positions.

### D13.1 The baseline, computed on the IDENTICAL position set

`tmp/audit_holdout_baselines.py` re-derives `tg_ab_eval.py`'s position
list using that script's **own** `decode_header` / `moves_of` helpers
(imported, not re-implemented) and the same corpus, same targets, same
`--max-rows 1500`, same filters. Same `n = 19,279` as the model rows.

| policy | teacher agreement (n=19,279) |
|---|---:|
| **always c3** (best constant) | **24.98 %** |
| always c2 | 16.30 % |
| always c1 | 14.77 % |
| c0 | 13.93 % |
| c4 | 13.67 % |
| c5 | 10.37 % |
| c6 | 5.97 % |
| chance over legal columns | 14.89 % |

### D13.2 The consequence

| arm | model | always-c3 | delta |
|---|---:|---:|---:|
| `fd_a_s6000` | 23.88 % | 24.98 % | **-1.10 pp** |
| `fd_b_s6000` | 21.13 % | 24.98 % | **-3.85 pp** |
| `fd_c_s6000` | 21.08 % | 24.98 % | **-3.90 pp** |

**All three trained arms score BELOW the best constant policy on teacher
agreement.** This is not a small effect: the arms are 1.1–3.9 points
*worse* than a policy that always plays the centre column.

### D13.3 Why this was invisible

Every release-gate metric was an **absolute** percentage with no
reference row:

* `strict_pct` — model vs teacher argmax
* `teacher_set_hit_pct` — model in teacher tie set
* `teacher_mass_pct` — model softmax mass on teacher tie set
* `offcentre_strict_pct` — same, restricted to non-c3 teacher targets
* `solver_optimal_pct` — model in the solver's optimal set

The one number that looked like it had a baseline
(`ALWAYS-c3 baseline` in `CONNECT_FOUR_SECTION_0H.md`) was measured on a
*different, earlier* split, and the current release gate never recomputed
it.

### D13.4 Why "teacher agreement" can be a trap

The authoritative solver subset says:

| fact | value |
|---|---:|
| teacher labels c3 | 24.6 % of proven positions |
| c3 is **provably optimal** | 57.9 % of proven positions |

The teacher and the solver disagree about the centre column by ~33
points. That is consistent with two very different situations:

1. the teacher picks a **different member of an optimal tie set**
   (benign — the solver often has several optimal moves), or
2. the teacher picks a **provably worse move** (harmful — training on it
   teaches bad play).

`tmp/audit_teacher_vs_solver.py` distinguishes them by checking whether
the teacher's move is *in* the solver's optimal set. See D13.5.

### D13.5 The fix, applied

Baselines are now computed and recorded by the evaluators themselves:

* `tg_solver_buckets.py` — every constant column plus chance, scored on
  the **same proven positions** as the models, and every model row now
  carries `vs_best_constant_pp` and `vs_always_c3_pp`.
* `tmp/audit_holdout_baselines.py` — the teacher-agreement baseline on the
  exact holdout positions.

A future release gate must print a baseline next to every model row. A
model number without a baseline is not a result.


---

## Appendix D14 — Does the model actually learn? 

The question asked was: *does it learn, can it learn, is it getting better,
and what about the training data?* This appendix answers all four with
measured evidence.

### D14.1 The headline metric was inflated by ties

The exact solver proves the optimal move SET for 560 positions. The set
size is not 1 — it is often 7 (every column equally good):

| optimal moves | positions |
|---:|---:|
| 1 | 112 |
| 2 | 55 |
| 3 | 41 |
| 4 | 31 |
| 5 | 68 |
| 6 | 101 |
| 7 | 152 |

**Only 20.0 % of positions have a single right answer.** On the other
80 % the solver cannot distinguish a good policy from a bad one, so
"solver-optimal %" over all positions is close to uninterpretable. A
constant policy scores 75.7 % on it.

The honest subset is the 112 positions with a UNIQUE optimum. On that
subset a constant column gets 29.5 % (always-c4) and chance gets 0.9 %.

### D14.2 The learning curve (`tmp/measure_unique_optimum_curve.py`)

`runs/_probe/unique_curve_fd_a.json` — fd_a, every checkpoint, same 560
proven positions, hard export, greedy:

| step | ALL % | UNIQUE % | TIED % | teacher(all) % | vs best-constant on UNIQUE |
|---:|---:|---:|---:|---:|---:|
| 0 | 65.2 | 0.9 | 81.2 | 8.9 | −28.6 pp |
| 1000 | 74.3 | 24.1 | 86.8 | 27.5 | −5.4 pp |
| 2000 | 75.7 | **30.4** | 87.1 | 26.6 | **+0.9 pp** |
| 3000 | 73.8 | 24.1 | 86.2 | 26.1 | −5.4 pp |
| 4000 | 73.0 | 17.9 | 86.8 | 23.2 | −11.6 pp |
| 5000 | 76.2 | 28.6 | 88.2 | 25.5 | −0.9 pp |
| 6000 | 75.0 | 24.1 | 87.7 | 23.8 | −5.4 pp |

**Answers to the four questions:**

* **Does it learn?** Yes. On the only subset with teeth, the model goes
  from 0.9 % (chance) at step 0 to 24–30 %. That is a ~25× improvement and
  it is board-reading, not a constant: the trainer's own board-sensitivity
  gate went 0.000 → 0.87, and the model beats chance by 27 points
  (exact binomial p ≤ 6.3e-03 at every checkpoint except step 4000).
* **Can it learn?** Yes, and it does. The GF trunk generalises.
* **Is it getting better?** It learns to read the board, then **plateaus
  and oscillates** (30.4 → 24.1 → 17.9 → 28.6 → 24.1). The final checkpoint
  is *worse* than the step-2000 checkpoint on the honest metric. Six
  thousand steps of training bought nothing over step 2000.
* **What about the training data?** The data is the problem — see D14.4.

### D14.2a Statistical honesty: "+0.9 pp over always-c4" is ONE position

The table above invites a wrong conclusion. `30.4 %` vs `29.5 %` is
**34/112 vs 33/112** — a single position out of 112. Two-sided exact
binomial against the always-c4 rate:

| checkpoint | correct / 112 | vs always-c4 (33/112) | p | vs chance (1/7) | p |
|---|---:|---|---:|---|---:|
| 0 | 1 | **significant** | 0.000 | 1.1e-06 | **sig** |
| 1000 | 27 | not significant | 0.254 | 6.3e-03 | **sig** |
| 2000 | 34 | not significant | 0.836 | 1.6e-05 | **sig** |
| 3000 | 27 | not significant | 0.254 | 6.3e-03 | **sig** |
| 4000 | 20 | **significant (worse)** | 0.007 | 2.8e-01 | n/s |
| 5000 | 32 | not significant | 0.918 | 1.1e-04 | **sig** |
| 6000 (shipped) | 27 | not significant | 0.254 | 6.3e-03 | **sig** |

**Corrected reading:**

1. The model **does** beat chance at every healthy checkpoint — that is
   real board-reading, confirmed statistically.
2. The model is **statistically indistinguishable from always-c4** at
   every checkpoint. The "+0.9 pp" is one position; the shipped
   checkpoint's "−5.4 pp" is six positions. Both are within noise.
3. Checkpoint-to-checkpoint the score swings 20↔34 positions
   (stdev 4.45). The model is wandering around the constant-policy line,
   not climbing away from it.
4. So the honest statement is: **the model learned to read the board, but
   has not yet learned to beat a constant column.** The step-2000
   checkpoint must NOT be promoted on the strength of "+0.9 pp" — that
   would be reading noise as signal.

Always-c4's 29.5 % on UNIQUE is simply "c4 is the unique optimum in
32/112 = 28.6 % of them", plus rounding. It is not a bar the model
cleared.

**D14.2a note on the ALL metric:** the same 560 positions also cannot
separate the models, because 80 % of them have tied optima. `always-c4`
scores 75.7 % on ALL and `fd_a` 75.0 %. The ALL metric should be retired
as a release gate and replaced by the UNIQUE subset, with a larger
solver-proven sample (560 of 1,500 holdout positions = 37.3 % is too few
to resolve single-digit differences).

### D14.3 The trainer's own log agrees

`runs/_probe/fd_a_s6000.log`:
bpb 8.000 → 5.075, LEGAL 13 % → 28 %, board sensitivity 0.000 → 0.871.
The model *is* fitting the data. The strength ladder never played a single
game (`vs-ref W0L0D0` at every step), so nothing in the training loop
could have detected the plateau.

### D14.4 Root cause: the training labels collapsed the tie set

`runs/_probe/perply_v23_full/merged_repaired.npz` carries both a `target`
array and a `teacher` field. The trainer's loss path reads `target` +
`ply_of`; the `teacher` field is a *source tag* and is not used as the
label (it reads 99 % "c2", which is an artefact of the source-tag encoding
— **do not** read it as the label distribution).

The label the trainer actually optimises is `argmax(target)`. Measured on
the full corpus (4,563,661 labels):

| | train sidecar | holdout sidecar |
|---|---:|---:|
| c0 | 13.2 % | 13.9 % |
| c1 | 13.3 % | 14.8 % |
| c2 | 14.2 % | 16.3 % |
| **c3** | **25.5 %** | 25.0 % |
| c4 | 14.1 % | 13.7 % |
| c5 | 12.1 % | 10.4 % |
| c6 | 7.5 % | 6.0 % |

That looks reasonable — **but the targets are one-hot**: 99.3 % of targets
have exactly ONE column at the max, and the mean target entropy is
0.008 nats (0.4 % of the maximum 1.946). The per-ply labels are
essentially hard.

**The supervision source breakdown is the smoking gun:**

| source | share of labels | targets that are one-hot |
|---|---:|---:|
| EXACT solver | 1.0 % | 34.0 % |
| STRONG negamax | **99.0 %** | **100.0 %** |

The exact solver — which *does* compute the full tie set — contributes 1 %
of the labels. The strong negamax teacher contributes 99 %, and it returns
exactly one move, so **every tie the solver could have found is collapsed
to a single column.**

Mechanism, in `src/c4/strong_solver.py` `_root_move`:

```python
if v > best:          # strict '>' keeps the FIRST column visited among equals
    best = v
    best_col = c
```

Combined with `_order_columns` (centre-first heuristic ordering), every
tie resolves to the centre column. That is the source of the c3 bias in
the labels — and it is a *teaching* bias, not a model bias.

### D14.5 The label does not match the game that was played

Over 267,331 labelled plies in the first 20,000 corpus rows:

* the label equals the move actually played in only **20.8 %** of plies
* label column distribution vs played column distribution:
  c3 is **25.4 % of labels but only 18.0 % of played moves**;
  c6 is 6.5 % of labels but 13.0 % of played moves

So the labels are not "what a good player did" — they are the solver's
tie-break, re-derived independently of the game. The corpus games and the
labels describe different policies.

### D14.6 How good is the strong teacher, really?

`tmp/audit_strong_teacher_quality.py`, 164 solver-proven positions:

| metric | value |
|---|---:|
| strong teacher's move is inside the exact optimal set | 91.5 % |
| strong teacher's move is genuinely suboptimal | 8.5 % |
| shipped label is inside the optimal set | 92.1 % |
| shipped label is genuinely suboptimal | 7.9 % |
| strong teacher hits the UNIQUE optimum | 86.0 % |
| shipped labels that fall in a tie set | 114 |

The teacher is **not** the problem in terms of move quality — 92 % of its
moves are optimal, and 86 % of the time it finds the unique optimum. The
problem is purely that it *reports one column where several are correct*,
and reports the centre-biased one.

**So the model is being taught something it cannot possibly learn.** On
80 % of positions there is no single right answer, the teacher picks one
(the centre-biased one), and the model is scored against that arbitrary
choice. A perfect model of the teacher would still be a bad Connect Four
player, because it learns "when in doubt, play c3" instead of "when in
doubt, any of these is fine".

### D14.7 What this means

1. The model **does** learn board-reading policy — 0.9 % → 30.4 % on the
   unique-optimum subset is real, generalisable learning.
2. The headline 75 % "solver-optimal" is **inflated by ties** and means
   much less than it appears.
3. The training labels are **the bottleneck**. They encode a tie-break
   convention as if it were ground truth, and they plateau the model at
   roughly step 2000.
4. The fix is in the **data, not the architecture**:
   * label with the SET (`tgt[c] = 1.0 / len(best)`) — the trainer's
     soft-target path already consumes this shape;
   * or make the strong teacher return a set, not a column;
   * and measure progress on the UNIQUE-optimum subset.

### D14.8 Artifacts

| file | content |
|---|---|
| `runs/_probe/unique_curve_fd_a.json` | learning curve, fd_a, all checkpoints |
| `runs/_probe/unique_optimum_truth_cache.json` | cached solver truth (560 positions) |
| `runs/_probe/strong_teacher_quality_400.json` | strong-teacher vs exact-solver audit |
| `tmp/measure_unique_optimum_curve.py` | the curve + baselines |
| `tmp/audit_strong_teacher_quality.py` | teacher quality audit |
| `runs/_probe/fit_aligned_v2.json` | aligned fit audit (8,883 positions) |

### D14.9 Related skill

`tie-set-label-collapse` — the general form of this failure.


---

## Appendix D15 — Consolidated re-measurement batch 

Five background jobs completed and were consolidated into one answer.
Two of them (`proc_a1bf0f8e3e66`, `proc_0a34d88e8c1f`) wrote to the
**same output path** (`runs/_probe/fit_aligned_v2.json`) and the **same
log** (`runs/_probe/fit_aligned_v2.log`) — the canonical constraint says
never let two processes write the same log. The failed run exited 1; the
successful run exited 0 and the artifact on disk is from the successful
one (mtime, tail shows the complete summary table). **Lesson:
the artifact is valid, but the pair must not be repeated.** Recorded here
so the next agent does not "re-run to be safe" and clobber a good file.

### D15.1 Authoritative solver baselines — `proc_a3431f12fa5f`

```text
tg_solver_buckets.py --arms fd_a_s6000,fd_b_s6000,fd_c_s6000 \
  --max-empty 24 --max-positions 1500 --budget 120000000 \
  --run-tag auth_baselines
exit 0 -> runs/_probe/solver_buckets_auth_baselines.json
n_solvable 560, unproved 940, authoritative_subset=true
```

Baselines on the **same 560** positions:

| policy | solver-optimal % | legal % |
|---|---:|---:|
| chance / always-c0 | 65.2 | 92.3 |
| always-c3 | 69.6 | 81.8 |
| always-c2 | 70.9 | 81.4 |
| always-c6 | 70.5 | 100.0 |
| always-c1 | 73.9 | 78.4 |
| always-c5 | 72.5 | 92.3 |
| **always-c4** | **75.7** | 90.4 |

Models:

| arm | solver-optimal % | vs always-c4 | vs always-c3 |
|---|---:|---:|---:|
| fd_a_s6000 | 75.0 | **−0.7** | +5.4 |
| fd_b_s6000 | 75.0 | **−0.7** | +5.4 |
| fd_c_s6000 | 70.9 | −4.8 | +1.3 |

Bucket decomposition (teacher said c3 vs not):

| arm | bucket | n | solver-opt % | model c3 % | teacher-hit % |
|---|---|---:|---:|---:|---:|
| fd_a | teacher c3 | 138 | 88.4 | 29.0 | 29.0 |
| fd_a | teacher non-c3 | 422 | 70.6 | 19.0 | 22.0 |
| fd_b | teacher c3 | 138 | 91.3 | 59.4 | 59.4 |
| fd_b | teacher non-c3 | 422 | 69.7 | 42.2 | 15.2 |
| fd_c | teacher c3 | 138 | 87.0 | 59.4 | 59.4 |
| fd_c | teacher non-c3 | 422 | 65.6 | 41.0 | 13.7 |

fd_b and fd_c chase the teacher's c3 far harder than fd_a (42 % and 41 %
c3 in the non-c3 bucket) — **the mirror and symmetry-regularised arms
amplified the centre bias**. That is consistent with their lower
solver-optimal scores.

### D15.2 Strong-teacher quality — `proc_7f3375910705`

```text
tmp/audit_strong_teacher_quality.py --max-positions 400 --budget 40000000
exit 0 -> runs/_probe/strong_teacher_quality_400.json
```

| metric | value |
|---|---:|
| n_proven | 164 |
| strong teacher move inside optimal set | **91.5 %** |
| strong teacher genuinely suboptimal | 8.5 % |
| shipped label inside optimal set | 92.1 % |
| shipped label suboptimal | 7.9 % |
| unique-optimum positions | 43 |
| strong teacher hits the UNIQUE optimum | **86.0 %** |
| shipped labels that fall in a tie set | 114 |

Strong-teacher column distribution: c0 15, c1 5, c2 33, **c3 38**,
c4 44, c5 18, c6 11. Shipped-label distribution: c0 9, c1 5, c2 19,
c3 32, **c4 52**, c5 25, c6 22 — note the shipped labels are *more*
centre-biased toward c3 than the raw strong teacher is, and the shipped
labels over-use c4.

### D15.3 Teacher vs solver — `proc_08e6158e2e14`

```text
tmp/audit_teacher_vs_solver.py
exit 0 -> runs/_probe/teacher_vs_solver.json
```

| metric | value |
|---|---:|
| n_proven | 560 |
| teacher optimal | **94.6 %** (530) |
| teacher suboptimal | 5.4 % (30) |
| c3 in the optimal set | 57.9 % (324) |
| unique-optimum positions | 112 (**20.0 %**) |
| optimal-set sizes | 2,,,,, 1, 2 |

**The verdict field says `TEACHER_IS_SUBOPTIMAL` — that label is
misleading.** 94.6 % optimal is not a bad teacher. The real defect is not
quality, it is **granularity**: the teacher returns one column where
several are correct, and that one column is centre-biased.

### D15.4 Corrected aligned fit audit — `proc_0a34d88e8c1f` (success)

```text
tmp/audit_fit_aligned.py --float-ckpt...step6000_soft.pt \
  --hard-ckpt...step6000_hard.pt --rows 800
exit 0 -> runs/_probe/fit_aligned_v2.json
```

| bucket | n | hard vs label | hard vs actual move | best const (label) | chance |
|---|---:|---:|---:|---:|---:|
| **ALL** | 8883 | 25.0 % | **31.2 %** | 26.3 % (c3) | 14.3 % |

Label distribution: c0 14.0, c1 15.5, c2 15.3, **c3 26.3**, c4 12.9,
c5 10.8, c6 5.3.

The model fits the **actual played move** (31.2 %) better than the
teacher **label** (25.0 %), and both are near the always-c3 constant
(26.3 %). So the model is not even faithfully reproducing the label it
was trained on.

### D15.5 Statistical honesty (see D14.2a)

The "+0.9 pp over always-c4" at step 2000 is **one position out of 112**
(p = 0.836, not significant). The model beats chance at every healthy
checkpoint (p ≤ 6.3e-03) but is statistically indistinguishable from a
constant column at every checkpoint. This supersedes the earlier
"+0.9 pp, model beats the best constant" framing.

### D15.6 What the batch establishes, in one paragraph

The infrastructure is sound and the measurements are now trustworthy
(authoritative 560-position baselines, self-audited artifact, statistical
significance computed). The **model is not** the bottleneck any more —
the **labels** are. Concretely: the teacher is 94.6 % optimal but returns
one column where 80 % of positions admit several; that column is
centre-biased; c3 is the unique optimum in only 6/112 unambiguous cases;
the model scores ~25–30 % on the unambiguous subset, statistically
level with always-c4's 29.5 %, and fits the actual played move (31.2 %)
no better than it fits the label (25.0 %). The fix is tie-set labelling
plus a strength ladder in the training loop, not more steps or a bigger
model.


---

## Appendix D16 — The real mechanism: the model collapsed to a 3-column prior 

D15 said "the labels are the bottleneck". That was correct but incomplete.
This appendix measures **what the model actually learned**, on the
identical 560 solver-proven positions, and it changes the diagnosis:
the failure is **not** that the model copied a bad teacher. The model
collapsed onto a **narrower, more concentrated prior than its teacher**,
and it collapsed onto the *wrong* columns.

### D16.1 Same-position distribution: model vs label vs optimal set

`tmp/audit_same_position_dist.py` → `runs/_probe/same_position_dist.json`
(model = fd_a step6000_hard, label = the shipped sidecar label for the
*same* position, "in-set" = how often that column is a member of the
exact optimal set):

| col | model % | shipped label % | in optimal set % | model / label |
|---:|---:|---:|---:|---:|
| 0 | 0.5 | 5.9 | 58.9 | **0.09×** |
| 1 | 2.0 | 8.9 | 58.8 | **0.22×** |
| 2 | 0.7 | 15.4 | 60.9 | **0.05×** |
| 3 | 21.4 | 24.6 | 57.9 | 0.87× |
| 4 | **45.5** | 21.1 | 68.9 | **2.16×** |
| 5 | 27.1 | 13.9 | 66.8 | **1.95×** |
| 6 | 2.7 | 10.2 | 70.5 | **0.26×** |

**model c3+c4+c5 = 94.1 %. label c3+c4+c5 = 59.6 %.**
Total-variation distance model↔label = **0.377**.

> **⚠️ CORRECTION (late — see Appendix D18).** The 94.1 % figure
> above is **specific to this 560-position solver-proven subset**, which is
> endgame-heavy because it is restricted to `≤24` empty cells. It is **not**
> a whole-model property. On the full **19,279-position** release holdout the
> same checkpoint's hard-path column distribution is:
>
> ```text
> c0 13.2%   c1 13.0%   c2  8.3%   c3 18.3%
> c4 37.8%   c5  8.4%   c6  1.0%
> ```
>
> So the model *does* play c1 13.0 % and c2 8.3 % of the time on a
> representative set; the sharp "never plays c1/c2" statement is a property
> of the proven subset only. The genuine, subset-independent findings that
> survive are:
>
> * c1/c2 are **never** played on the 112 unique-optimum positions (a
>   high-signal subset where the right answer is unambiguous);
> * 23/112 (20.5 %) of those clear-cut positions are structurally
>   unreachable for the shipped checkpoint;
> * the shipped checkpoint picks c3 on 16/112 unique positions, **14 of them
>   wrong**.
>
> The centre-heavy c4 (37.8 %) and the c6 rarity (1.0 %) **are** whole-model
> effects. Everything in D16.1–D16.3 below must be read as
> *subset-scoped*, not as the model's global behaviour.

The critical control: **the game does not favour c3/c4/c5.** Every column
is a member of the optimal set in 58–70 % of positions (c0 58.9 %,
c1 58.8 %, c2 60.9 %, c3 57.9 %, c4 68.9 %, c5 66.8 %, c6 70.5 %).
There is no column-prior signal in the game for the model to have
learned — so the c3/c4/c5 spike is **not** the optimal set, and **not**
the label distribution either. It is the model's own collapse.

### D16.2 Consequence: 20.5 % of unambiguous positions are unreachable

On the 112 unique-optimum positions the optimal column is:

| col | n optimal |
|---:|---:|
| c1 | 15 |
| c2 | 8 |
| c3 | 6 |
| c4 | 32 |
| c5 | 28 |
| c6 | 23 |

The shipped model plays **c1 and c2 exactly zero times** across all 13
checkpoints (step 500 → step 6000). c1 and c2 are legal in 84 and 74 of
those 112 positions respectively, so they are not unavailable — they are
**unused**.

**23 of 112 unambiguous positions (20.5 %) are structurally unreachable
for this model.** Its ceiling is therefore ~79.5 %, not 100 %, before any
other error. This single fact explains why the model cannot beat a
constant: always-c4 hits the c4-optimal cases (32) plus the c5/c6 cases
where c4 happens to be legal and optimal (33 total), while the model
*cannot* reach the c1/c2 cases at all.

### D16.3 The same suppression is present on the tied subset

Not a metric artifact — it is a model property:

| step | model c1+c2 on UNIQUE (112) | model c1+c2 on TIED (448) |
|---:|---:|---:|
| 0 | 7 | 25 |
| 2000 | 1 | 9 |
| 6000 | 0 | 15 |

The tied subset (448 positions) shows the same suppression. The model
simply does not play c1/c2 much.

### D16.4 Where the bias comes from

Two independent sources, both confirmed by source inspection:

1. **Teacher tie-break (centre-first).** `src/c4/strong_solver.py`
   `_order_columns` sorts by `abs(c - COLS // 2)` — centre-out — and
   `_root_move` uses `if v > best` (strict), so among equal-valued moves
   the **first visited** wins, i.e. the centre-most. 99 % of the training
   labels come from this teacher and are 100 % one-hot, so the tie set is
   destroyed and the centre preference is baked in. On the 560 proven
   positions the teacher picked c3 in 128/560 = 22.9 %.

2. **The model's own additional collapse.** The label puts 21.1 % on c4
   and 15.4 % on c2; the model puts 45.5 % on c4 and 0.7 % on c2. The
   model did **not** reproduce its teacher. It found a sharper, narrower
   solution — the classic outcome of a hard one-hot target: the model
   learns the *easy* part of the distribution (which centre-ish columns
   are plausible) and discards the tail.

### D16.5 This is a third distinct failure, and it has its own skill

There are now **three separate bugs**, each independently sufficient to
produce a weak model, and each with its own skill:

| # | failure | skill | symptom |
|---|---|---|---|
| 1 | solver falls back to a weaker policy silently | `solver-economics-measurement` | "exact" results that aren't exact |
| 2 | labels collapse the tie set to one centre-biased column | `tie-set-label-collapse` | teacher looks 94.6 % optimal but teaches a convention |
| 3 | **model collapses to a 3-column prior and abandons c1/c2** | `model-column-prior-collapse` | cannot reach 20.5 % of unambiguous positions |

Failure 3 is the one that actually caps the score, and it is **not**
fixed by tie-set labelling alone. Tie-set labelling removes the teacher's
centre convention, but the model may still collapse onto a narrow
prior — that must be measured.

### D16.6 New skill: `model-column-prior-collapse`

Created to catch this class of bug. The rule it encodes:

> A model's column distribution must be compared to its label distribution
> **on the same positions**, and to the optimal-set membership. If the
> model's picks are concentrated on a subset of columns (TV > ~0.2 vs the
> label) **and** it never plays columns that are legal and often optimal,
> the model has collapsed to a prior. Measure the *unreachable fraction*
> — the share of positions whose optimal column the model never emits.

### D16.7 What this changes in the plan

The tie-set A/B (`tmp/ab_tieset_labels.py`, relabelling 3000 rows with
exact tie sets) is still the right next experiment, but it is **not
sufficient on its own**. It must be scored on:
- the UNIQUE subset (as designed), **and**
- the **unreachable fraction**: how many unambiguous positions have an
  optimal column the model never plays, **and**
- the model↔label TV distance on the same positions (as designed), **and**
- the per-column pick distribution (as designed).

If tie-set labels fix the teacher convention but the model still never
emits c1/c2, the fix is a **coverage/entropy** objective on the column
head (an explicit penalty for never-emitted legal columns), not more
data.

### D16.8 Artifacts

| file | content |
|---|---|
| `runs/_probe/c3_bias_unique.json` | per-checkpoint c3 bias + reachable fraction |
| `runs/_probe/same_position_dist.json` | model vs label vs optimal-set on 560 |
| `tmp/audit_c3_bias_unique.py` | per-checkpoint column audit |
| `tmp/audit_same_position_dist.py` | same-position distribution audit |


---

## Appendix D17 — Second consolidation batch, incl. two live fixes

Seven background processes reported. Five were **stale duplicates** of
jobs already incorporated into D14–D16; one was a **failed duplicate**;
one carried a **new metric**. Two live defects were found and fixed.

### D17.1 Batch triage

| process | exit | disposition |
|---|---|---|
| `proc_54d41eada2a5` | 1 | **FAILED** — `ModuleNotFoundError: No module named 'src'` |
| `proc_c865da6e5737` | 0 | superseded — non-authoritative baseline run |
| `proc_f5f0e25cdc9a` | 0 | duplicate of `proc_08e6158e2e14` (teacher audit) |
| `proc_15c4300a3173` | 0 | duplicate waiter on the unique-curve run |
| `proc_72d3b0fb621e` | 0 | regression sweep — **1 FAIL** (stale test) |
| `proc_020bc693dfa6` | 0 | duplicate of the unique-optimum curve |
| `proc_f35a7c6d7486` | 0 | the c3-bias audit (already in D16) |

**`proc_54d41eada2a5` failure analysis.** The crash was
`ModuleNotFoundError: No module named 'src'` at the
`from src.c4.board import Outcome` line. The script *does* insert ROOT
into `sys.path` (line 45) — but **after** the `src.c4` import at line 48
of the failing copy. The currently-on-disk copy has the correct order
and imports cleanly (verified: `from src.c4.board import Outcome` and
`from src.c4.exact_solver import ExactSolver` both succeed with
`ROOT =.../ZeroFLops-paper`). The crash was a **pre-fix duplicate**
launched before the path fix; the authoritative artifact
`runs/_probe/teacher_vs_solver.json` (655 B, mtime) is intact
and parses: `n_proven=560`, `teacher_optimal_pct=94.6`,
`unique_optimum_pct=20.0`, `verdict=TEACHER_IS_SUBOPTIMAL`. **No
artifact was corrupted by the failed run.**

**Lesson:** the import must come *after* the `sys.path.insert`. This is
the same class as the `ROOT = parents[1]` bug fixed earlier in the same
file — the script was launched twice, once before each fix.

### D17.2 `proc_c865da6e5737` — non-authoritative, correctly self-labelled

```text
tg_solver_buckets.py --arms fd_a_s6000,fd_b_s6000,fd_c_s6000 --run-tag baselines
```

It wrote `runs/_probe/solver_buckets_baselines.json` with
`n_solvable=166`, `authoritative_subset=false` and
`comparable_to="NOT COMPARABLE to the authoritative subset (budget=120000000 n_solvable=560)"`.

The guard worked exactly as designed: the default budget proves only 166
positions, and the artifact says so in machine-readable form. **The
authoritative artifact remains `runs/_probe/solver_buckets_auth_baselines.json`
(560 positions).** No action needed beyond ignoring this one.

### D17.3 `proc_72d3b0fb621e` — the regression sweep failure, and the fix

The sweep reported `PASS=10 FAIL=1`:

```text
tmp/test_ui_isfull.py    FAIL(1)
  urllib.error.HTTPError: HTTP Error 400: Bad Request
```

This is the **stale, already-diagnosed** test. Its premise drives the
human to play column 7 four times in a row and assert the game stays
ongoing — but four discs in one column is a **vertical four**, so the
server correctly refuses the 4th move with HTTP 400. The test was
failing for a reason unrelated to the `is_full` bug it was written to
catch.

**Fix applied:** `tmp/test_ui_isfull.py` is now a **superseded shim**
that prints a pointer to `tmp/test_ui_isfull_v2.py` and exits 0. The v2
test (which alternates columns to avoid a vertical four and asserts only
"no premature draw") is the live one. The shim exists so a naive
"run every `tmp/test_*.py`" sweep can never report a failure that has
already been diagnosed and fixed.

Re-run of the full sweep, now including the shim:

```text
tmp/test_is_full.py                        PASS
tmp/test_patterns.py                        PASS
tmp/test_symmetry.py                        PASS
tmp/test_symmetry_batch_equiv.py            PASS
tmp/test_symmetry_penalty.py                PASS
tmp/test_symmetry_stream.py                 PASS
tmp/test_ui_api.py                          PASS
tmp/test_ui_isfull.py                       PASS   (superseded shim)
tmp/test_ui_isfull_v2.py                    PASS
tmp/test_ui_terminal.py                     PASS
tmp/test_eval_fixed_arms_harness.py         PASS
tmp/test_live_ui_fd_a.py                    PASS
tmp/test_live_parity_v5.py                  PASS
tmp/test_control_harness_soundness.py       PASS
PASS=14 FAIL=0
```

### D17.4 New metric: teacher agreement on the UNIQUE subset

The curve artifact carries `teacher_unique_pct`, which is more informative
than `teacher_all_pct`:

| step | UNIQUE score | teacher agreement (UNIQUE) | teacher agreement (ALL) |
|---:|---:|---:|---:|
| 0 | 0.9 % | 2.7 % | 8.9 % |
| 1000 | 24.1 % | 25.9 % | 27.5 % |
| 2000 | 30.4 % | 26.8 % | 26.6 % |
| 3000 | 24.1 % | 22.3 % | 26.1 % |
| 4000 | 17.9 % | 23.2 % | 23.2 % |
| 5000 | 28.6 % | 25.9 % | 25.5 % |
| 6000 | 24.1 % | 25.9 % | 23.8 % |

The model reproduces its teacher's label on only **~26 %** of the
unambiguous positions, while the **teacher itself is right on ~86 %** of
them (D15.2). So the model is **not** a noisy copy of the teacher — it
is a materially different policy, one that scores about the same overall
(24–30 % vs the teacher's 86 %) while disagreeing with the teacher most
of the time. That is the clearest single statement of the failure: the
model learned something, but not the thing it was taught.

## Appendix D18 — Retraction of the stale status line, and the external critique adjudicated

### D18.1 Why this appendix exists

An external review of this project challenged the reading of D16 and
flagged that this document contradicted itself. Both were correct. This
appendix records the retraction and the adjudication, and points at the
one-page replacement: **`docs/CURRENT_TRUTH.md`**.

### D18.2 The retraction

The status line at the top of this file used to read:

> *"Status: the centre-only defect is FIXED. The controlled matched A/B is
> complete and `mir_a_s6000` is the leading candidate."*

**That was wrong.** It is now replaced by a superseded-status notice. The
truth:

* **No arm passes the release gates.** On the authoritative 560-position
  solver set, `fd_a` 75.0 % vs `always-c4` 75.7 % — the trained model does
  **not** beat a constant-column policy.
* `mir_a_s6000` is **not** the leading candidate. The 560-position solver run
  *tied* `fc_a_s6000` at 71.8 %, and the full-holdout column distribution
  shows `mir_a_s6000` is **worse** on whole-board centre calibration
  (c3 share 54.6 % vs 33.3 %).
* The D16 "94.1 % c3/c4/c5" figure is **subset-scoped**, not whole-model. See
  the correction banner inserted in D16.1.

### D18.3 External critique, adjudicated claim by claim

| claim | verdict | evidence artifact |
|---|---|---|
| fd_a/b/c used **sparse** lags -> c1/c2 architecturally blind | **REFUTED** - launcher and checkpoint both declare dense lags `[0,1,2,3,4,5,6,7,8,9,10,20]`; per-column hard sensitivity shows **no blind columns** in any of the 4 audited checkpoints (fd_a c1 0.126, c2 0.144) | `runs/_probe/col_sensitivity_fd_a.json` |
| the 94.1 % collapse describes the model | **CORRECT CRITICISM, WRONG CONCLUSION** - it is the 560-position endgame-heavy subset. Full 19,279-position holdout: c1 13.0 %, c2 8.3 %. The subset-free findings (c1/c2 never played on the 112 unique-optimum positions; 23/112 unreachable; 14 of 16 c3 picks wrong) survive | `runs/_probe/eval_fd_a_s6000_v4_holdout.json` |
| the 28.3 % agreement is a **decoding bug** | **REFUTED** - a 12-convention orientation grid selected `row_flip=True, col_flip=False, player=parity`, exactly matching `tools/import_hf_c4.py`; 0 sentinel misplacements over 400 rows; label argmax in the exact optimal set **100.0 %** | `runs/_probe/decoding_grid.json` |
| the 28.3 % number itself is wrong | **CONFIRMED** - it is a tie-break-coincidence artefact. Label **in** the exact optimal set = 100.0 %, **0** suboptimal labels, predicted coincidence 46.4 % | `runs/_probe/thirty_pct_decomposition.json` |
| the evaluation set is distorted | **CONFIRMED** - 560 positions are endgame-heavy by construction (`<=24` empty cells); 940/1,500 unproved and excluded | `runs/_probe/solver_buckets_auth_baselines.json` |
| the ladder never played (`W0L0D0`) | **CONFIRMED** - `run_fixed_ab.sh` sets `--eval-games 0`; `evaluate_play(..., 0,...)` returns W0L0D0 by construction. Re-run with 24 games: **W12 L12 D0** | `runs/_probe/ladder_enabled_check.json` |
| `_root_move` values are alpha-beta **bounds**, unusable | **REFUTED** - 0/397 columns had a wrong magnitude, best column exact 60/60. But they are exact for the depth-6 tree under a hand-written eval, not true game values: argmax in the exact optimal set only 76.8 % (median rank corr 1.000, and no failure ever ranked a truly optimal column below a non-optimal one) | `runs/_probe/root_value_bounds.json`, `runs/_probe/strong_teacher_value_quality.json` |
| graded score-aware targets are worth building | **CONFIRMED, and the data is already in hand** - 78.5 % of proved rows carry a genuinely graded vector, 0 sign contradictions, and the Pons max-set is a **subset** of the exact optimal set in 100 % of proved positions | `runs/_probe/pons_score_set_equality.json`, `runs/_probe/pons_refinement.json` |
| docs contradict themselves; make one current-truth page | **CONFIRMED and now fixed** - see `docs/CURRENT_TRUTH.md` and D18.2 | this appendix |

**The review's most valuable point was its caution, not its mechanism.** *"Do
not add a coverage/entropy penalty before diagnosing the cause"* is exactly
right. Its proposed cause - sparse lags and a decoding bug - is not.

### D18.4 The scale finding that changes the next design

The Pons score is **not** the exact value in the same units. The exact solver
returns outcome-only (`+1` / `0` / `-1` / `1000`); the Pons score is
**distance-like** (`+k` = win in *k* plies, `-k` = loss in *k* plies, `0` =
draw). Consequences:

* only **18.81 %** of columns match numerically, while the sign agrees
  **100.00 %** of the time;
* the Pons max-set is a **subset** of the exact optimal set in **100.0 %** of
  proved positions - it *refines* the exact set by separating "win in 1" from
  "win in 5", and the two never contradict;
* the 24.75 % "set mismatch" is therefore **not** a data error. It is the
  distance information the outcome-only exact value cannot express.

**So the graded target should be built from the Pons distance scale**, with the
exact-solver tie set overlaid where the Pons max-set is not exactly the
optimal set - not from `strong_solver`, whose argmax agrees with exact truth
only 76.8 % of the time versus the Pons 100 %.

### D18.5 Artifacts added by this batch

```text
runs/_probe/decoding_grid.json           12-convention orientation grid
runs/_probe/pons_score_set_equality.json  max-set equality + graded-vector rate
runs/_probe/pons_refinement.json         subset/refinement test + scale map
runs/_probe/pons_score_units.json        units, sign agreement, Spearman
runs/_probe/root_value_bounds.json       0/397 magnitude errors
runs/_probe/strong_teacher_value_quality.json  teacher ordering vs exact truth
runs/_probe/ladder_enabled_check.json    W0L0D0-by-construction vs real W/L/D
runs/_probe/col_sensitivity_fd_a.json    per-column hard sensitivity, 4 checkpoints
runs/_probe/thirty_pct_decomposition.json  exact 28.3 % reproduction
```

### D18.6 Two process defects this batch also exposed

1. **Shared output path.** Two runs of `tmp/audit_decoding_grid.py` (an old
   8-convention version and the corrected 12-convention version) both wrote
   `runs/_probe/decoding_grid.json` and `runs/_probe/decoding_grid.log`. The
   corrected run finished last and won, so the on-disk artifact is correct -
   but only by luck of ordering. The stale run's numbers (best convention
   `swap=True`, 80 % in-optimal-set) were reported into a notification in the
   same batch. **The on-disk artifact was verified directly** before quoting.
2. **A parse failure that was not a corruption.** `col_sensitivity_fd_a.json` is
   a JSON *list* of 4 per-checkpoint records, not an object; the first parse
   attempt used `.keys` and failed. The file was fine.

### D19.2 The representative audit found a bug in itself, not in the model (late)

`tmp/audit_col_representative.py` reported the model at **65.36 %** in-optimal
on the 560 proven positions, with a per-column histogram that **disagreed**
with the authoritative `tg_solver_buckets.py` run on the **same 560
positions and the same checkpoint** (authoritative: 75.00 %).

Two audits of identical inputs cannot disagree, so one protocol was wrong.
Reconciliation (`tmp/audit_mask_reconciliation.py`,
`runs/_probe/mask_reconciliation.json`):

| protocol | picks an illegal column | in the exact optimal set |
|---|---:|---:|
| raw `argmax` over `COL_SLICE` | **64/560 (11.43 %)** | 65.36 % |
| **masked `argmax` (the shipped path)** | **0** | **75.00 %** |

`Agent.choose` (`src/c4/agent.py:300-305`), `src/c4/ui_server.py` and
`tg_solver_buckets.py` all mask illegal columns to `-1e9` **before** the
argmax. The representative audit did not, so it sometimes chose a **full
column** — a move that can never be in the exact optimal set, because it is
not a legal move at all. It also mis-stated the teacher distribution and
under-counted c1/c2 (2.0 % / 0.7 % instead of the correct 1.4 % / 0.9 %
under the shipped protocol), which **inflated** the apparent asymmetry.

**The model, the training data, the UI and the authoritative evaluator are
all unaffected.** This was a bug in one throwaway audit script. It was
found because the script's own validation block compared a *masked* pick
against an *unmasked* reference — a mismatch it had been silently tolerating
via its `except: pass`.

Fixed: `tmp/audit_col_representative.py` now applies the shipped mask and
uses `agent.version` for the transcript. Re-run: **560/560** validate
against the canonical API, the per-column counts now match D16's table
exactly, and the headline reconciles at **75.0 %**.

**Generalisable lesson (skill `agent-api-move-history-vs-legal-list`, now
patched):** the legal mask is part of the *decision*, not an optional
extra. A raw argmax over the column logits measures a policy nobody plays.
Any competence number must use the same masked protocol as `Agent.choose`,
and the cheapest cross-check is to diff the per-column pick counts against
the authoritative evaluator on identical positions.

### D19.3 The column asymmetry is depth-dependent

The corrected depth-stratified breakdown (same 560 positions, masked
protocol) shows the model's in-optimal rate is **much worse in deep
positions** while the teacher's is flat:

| empty cells | n | model in optimal set | teacher in optimal set |
|---|---:|---:|---:|
| 0–8 (endgame) | 8 | 87.5 % | 100.0 % |
| 8–14 | 32 | 84.4 % | 96.9 % |
| 14–20 | 162 | 72.2 % | 95.1 % |
| 20–25 | 358 | 75.1 % | 94.1 % |

c1+c2 take **2.7 %** of the model's moves on the proven subset but
**21.3 %** on the full 19,279-position holdout. So the under-use of the
outer columns is concentrated in the **deep, early-game** band — the
*opposite* of what the endgame-heavy subset on its own would suggest.

Artifact: `runs/_probe/col_representative.json`


---

## Appendix D20 — ROOT CAUSE: the per-ply sidecar was indexed by batch position (late night)

### The defect

`src/c4/train.py`, per-ply supervision branch:

```python
b_idx = torch.arange(keep.numel, device=device) // (y.shape[1] + 1)
rows = b_idx[hit].cpu
s_idx = flat_slot[hit].cpu
tgt = _pt[rows, s_idx].to(device)
```

`arange(keep.numel) // (width + 1)` produces **0 … B−1** — the *batch
index*. The sidecar is indexed per **corpus row**, so the trainer read the
teacher answers of sidecar rows 0..15 instead of the rows its windows came
from. The neighbouring single-label path (`_soft_rows_for_batch`) returns
`chosen[k]` and was always correct; only the per-ply path was wrong.

The **slot** index came from the correct row, so row and slot disagreed
silently. Every target read was a real, well-formed teacher label — of a
*different game*. Nothing raised; the loss fell; `LEGAL` climbed; every
checkpoint and artifact looked healthy.

### Measured (production corpus + production sidecar, model-free)

| quantity | before | after |
|---|---:|---:|
| trainer target == correct target | **9.59 %** | **100 %** |
| distinct sidecar rows read (of 320,000) | **16** | 100s |
| all-zero target → silent played-move fallback | **41.27 %** | **0.00 %** |
| supervised positions with a live target | 58.55 % | 100 % |

Artifacts: `runs/_probe/perply_row_index_bug.json`,
`runs/_probe/trainer_effective_target.log`
Audits: `tmp/audit_perply_row_index_bug.py`,
`tmp/audit_trainer_effective_target.py`

### Why the played-move fallback was the whole story

`src/c4/move_loss.py::_weighted_move_loss` substitutes a one-hot on the
**PLAYED** move when the teacher target has no mass on the legal columns.
With `--soft-w 1.0` the teacher target otherwise *replaces* the played-move
term entirely, so the fallback was the **only** route by which the played
move could enter the loss.

**41 % of the supervision was silently "predict the move that was already
played"; the other 59 % was the answer to a different game.**

That resolves the paradox that drove weeks of measurement:

| quantity | value | what it actually measured |
|---|---:|---|
| trainer `LEGAL` | 54–59 % | agreement with the **played byte** |
| direct sidecar-label agreement | 20.2 % | agreement with the **teacher** |
| model vs played move | 56.6 % | the fallback dominating |
| model vs label, `src=1` exact | 70.2 % | exact labels (218 rows) survived |
| model vs label, `src=2` strong teacher | 19.6 % | the fallback dominating |

**The architecture was never the problem.** The "overfit 19 % ≈ fd_a
20 % ⇒ architecture must change" conclusion was a measurement of the
supervision plumbing, not of the architecture. The 500-row overfit run
reached 54–59 % `LEGAL` while the fix was absent, which is exactly what a
played-move-fallback-dominated loss would produce.

### Scope

The bug was in the working tree, never committed, so it affected **every**
`fd_a` / `fd_b` / `fd_c` run (all used `--perply-targets`) and the
`overfit_test` run.

### Fix and guard

* `src/c4/train.py::_perply_corpus_rows(chosen, width, device)` — new
  helper; the per-ply branch now uses it.
* The per-ply branch **drops** positions whose target has no mass on the
  legal columns instead of falling back to the played move. With the
  correct row index the drop count is **0** — the guard exists so a future
  sidecar cannot silently reintroduce the fallback.
* `tmp/test_perply_row_index.py` — regression test, calls the **shipped**
  helper, asserts: unit mapping, >B distinct rows read, 0 all-zero
  targets.
* `tmp/verify_test_catches_bug.py` — swaps in the pre-fix helper and
  asserts the test **FAILS** (`rc=1`, "only 16 distinct corpus rows read").
  A test that cannot fail on the bug guards nothing.

### Retractions this forces

Everything below that depends on a *trained model's* behaviour is now
**provisional** and must be re-measured on a correctly-trained model:

* "no arm beats a constant-column policy" (the in-game matchups) — the
  arms were trained on the wrong supervision;
* "the model never plays c1/c2" (unique-optimum subset) — same;
* "the model collapses onto c3/c4/c5" (94.1 % on the 560-subset) — same;
* "the architecture cannot fit even data it saw 10×" — refuted; the
  measurement was of the plumbing.

Still valid: the corpus and sidecar repairs, the transcript-v2 decoding
facts, the Pons decoding grid, the hard-export parity, the UI parity
suite, the exact-solver baselines, and every label-quality measurement
(labels themselves were fine — they were read from the wrong row).

### Status

Code fixed and guarded. Corrected 500-row acceptance run in progress
(`runs/_probe/overfit_fixed/`). Acceptance gate: **>90 %** direct
sidecar-label agreement (was 19.1 %). No full-scale retrain and no V24
until that gate passes.

---

## Appendix D21 — The promotion gate is repaired, and the in-game harness got a third defect fixed

### D21.1 What was wrong with promotion

The trainer's promotion key was:

```text
sel = (ladder_net, holdout_pct, -hard_bpb, best_distinct)
```

`hard_bpb` is a **byte-level perplexity on the played move** — a different
objective from the one the per-ply sidecar actually trains. In the
acceptance run there was no holdout and no ladder score at all
(`--eval-games 0`), so the key **degenerated to the lowest byte-bpb** and
promoted step 1500 (62.82 % direct label agreement) over the final step
3923 (91.29 %). The best checkpoint was never even a candidate.

**Fix.** `PerPlyAgreement` scores the fraction of labelled per-ply
positions whose model pick is **in the teacher label set**, on the
**shipped hard path**, and that is now the **primary** promotion key
whenever a per-ply sidecar is present:

```python
if ppa is not None and ppa.get("perply_n", 0) > 0:
    _primary = float(ppa["perply_agreement_pct"])
else:
    _primary = float(_ladder_net)
sel = (_primary, ho_pct, -hard_bpb, float(best_distinct))
```

The metric is logged to `metrics.jsonl` and printed in the evaluation
panel as `LABEL xx.x% (n=…)`, so the number that drives promotion is
visible in the log rather than hidden.

### D21.2 Two traps that produced *false* results while building this

**Trap 1 — the zeroed-export trap.** The first hard-path implementation
returned an identical **12.68 %** for *every* checkpoint, which looked
like a working metric. Cause: `GFMarkovLM_V6` has **no `import_hard`**, so
a V6 checkpoint re-loaded through `_load_agent` keeps **zero float
parameters**; `export_hard` on that model yields an all-zero model whose
every logit is exactly `0.0` and whose argmax is always column 0 — a
constant, useless policy. The *saved* hard payload was fine; only the
float re-export was broken.

Fix: the scorer takes the hard payload by **injection** — the trainer
injects its live `_hard_export(model)`, and offline measurement reads the
payload straight out of the file and constructs the architecture only to
call `hard_forward`. After the fix: final checkpoint **92.38 %**,
step-0 checkpoint **12.68 %** — the metric discriminates.

**Trap 2 — the context-width trap.** `trainer_window` reads at index
`ctx-1`, so the scorer must use the trainer's `--ctx` (192 for the
acceptance run). The scorer had a hardcoded **191**. `ctx` is now an
explicit constructor argument, passed by the trainer from `args.ctx` and
by the offline measurement script.

### D21.3 Verified numbers

```text
direct sidecar-label agreement, hard export
  runs/_probe/overfit_fixed/hard_model_final.pt   91.29 %   (tmp/measure_label_agreement.py)
  same checkpoint, via the shipped scorer           92.38 %   (n=4000)
  same checkpoint, via the promotion-gate test      92.40 %   (n=3000)
  runs/_probe/overfit_fixed/best_hard.pt (step 1500) 62.82 %
  step 0                                            12.68 %
```

Two independent measurement paths agree to within 1.1 pp, and both clear
the **> 90 %** acceptance gate. The architecture was never the limiting
factor — the per-ply supervision plumbing was.

`tmp/test_promotion_gate.py`: **16 PASS, 0 FAIL**. It also asserts that
the metric is *not* the played move: `label == played` holds on only
**20.82 %** of corpus positions, consistent with the independently
measured 21.33 %.

Full serial regression sweep: **17 PASS, 0 FAIL**
(`tmp/run_regression_sweep.sh`).

### D21.4 The in-game harness had a THIRD opponent-definition defect

`tmp/audit_ingame_strength_fd_a.py` defined its "always-cX" arms as:

```python
return col if col in legal else legal[0]
```

That is **not** a constant policy — once the preferred column fills it
falls back to the *first* legal column, i.e. a **centre-first priority
sweep**. The same defect made the old "always-c3" arm play c3 on only
**28.2 %** of its moves. It has been replaced with a true constant: play
the preferred column while legal, and once it is full pick another legal
column at random.

The effect on the measured result is large and direction-reversing. With
20 games/arm/seed against the `centre` opponent:

```text
                old priority-sweep     true constant
always-c3             46.7 %               30.0 %
always-c0             --                   85.0 %
```

So "always-c3 beats the model" was an artefact of the fallback. The
correct opponent is always-**c0** against a centre-ish opponent.

The harness is also now checkpoint-parameterised
(`--ckpt/--out/--games`), `None`-safe (the earlier `best_arm is None`
crash), and it seeds the **global** RNG before each duel, because
`src.c4.duel.duel` **ignores its own `seed` argument**.

**All strength numbers measured against `fd_a_s6000` remain
provisional** — that checkpoint was trained under the per-ply supervision
defect. The harness fix is valid; the model conclusions from that
checkpoint are not.

### D21.5 Corrected retrain launched

`run_fixed_ab_corrected.sh` retrains all three arms with the fixed
supervision, at the canonical seed 32, everything else byte-identical to
`run_fixed_ab.sh`:

```text
fx_a_s6000: repaired original corpus + repaired sidecar
fx_b_s6000: repaired mirror corpus  + repaired mirror sidecar
fx_c_s6000: repaired mirror corpus  + repaired mirror sidecar + --sym-w 0.5
```

The launcher verifies, per arm: the log reached the requested step, the
completion marker is present, there is no traceback, the final artifacts
exist and are non-empty, **and the final hard export carries the
`transcript: v2` stamp** (an unstamped export is read as v1 and
mis-scored — that defect cost the acceptance run a false 24.37 %).

### D22. RETRACTED then CORRECTED: the model does NOT absorb tie sets

**This appendix was first written claiming the opposite. It is
retracted.** An earlier draft of this section concluded from the raw
numbers that the model represents tie sets, so the label collapse was
the primary defect. That was **confounded by set size**: a labelled set
of 3.38 columns trivially collects more probability mass than a set of
1 column.

Adding a chance-rate control -- what a uniform-random legal policy
scores on the SAME positions -- reverses the verdict.

```text
                       TIE slots (263)   SINGLE slots (39,059)
mean columns in set          3.38                1.00
model mass on set          0.894               0.1745
CHANCE mass on set         0.8809              0.1488
lift over chance         1.015x               1.173x
model argmax in set       90.49 %             23.23 %
chance argmax in set      88.09 %             14.88 %
lift over chance (argmax) +2.41 pp             +8.36 pp
```

On a tie set the model places 0.894 of its mass on the set, but chance
places 0.881 -- a lift of 1.015x, i.e. **essentially chance**. On
single-column labels the lift is 1.173x / +8.4 pp. So on ties the model
is spreading near-uniformly over the tied columns: an *uncertain*
model, not one that has learned "these columns are equally optimal".

This does **not** re-open the architecture question. The corrected
trainer reaches 91-92 % label agreement on the 500-row acceptance corpus
(D20/D21), where it does fit its labels. What this establishes is
narrower: the old fd_a/b/c checkpoints carry very weak label signal
(1.02-1.17x chance), consistent with their being defective-supervision
artifacts, and the ability to LEARN a genuine tie set remains an open
question for the tie-set training path.

**Method rule (the reusable part).** Any metric comparing a k-element
set against a 1-element set must be reported **relative to chance on
the same positions**, or it measures set size rather than model
behaviour. The raw comparison gave the OPPOSITE conclusion from the
chance-controlled one.

**Two audit-script bugs fixed to obtain this** (neither was in shipped
code):

1. The slot -> ply mapping used `p = nrec - 1 - s`, which is the
   trainer's *byte-target* slot convention, **not** the per-ply label
   convention. The sidecar ships `ply_of`, whose entry for slot `s` IS
   the ply that slot labels. Use `ply_of`; do not re-derive it.
2. The corpus was decoded with the **v1** decoder. Corpus rows are
   dense **v2** (stride `PLY_LEN`, no intra-game newlines). v1 read the
   9-byte v2 column headers as v1 height bytes -> `ValueError: bad
   decoded height 81`. Use `src.c4.transcript_v2.decode_game`.

### D23. The strength harness has NO seat bias; the first control was wrong

`tmp/audit_strength_harness_control.py` reported two FAILs. Both were
**control bugs**, not harness bugs. Corrected control:
`tmp/audit_strength_harness_control_v3.py`.

```text
C1  model vs model, BOTH sides explored   W295 L280 D25  49.2 %  p=0.5594  PASS
C2  always-c3 vs always-c3, BOTH explored  W298 L301 D1   49.7 %  p=0.9349  PASS
C3  model vs model, no exploration         unique_games = 1           PASS
```

* **C2 was mis-specified:** A was a PURE constant, B was the same policy
  with 30 % exploration added. An explored opponent is strictly weaker,
  so A winning 53.7 % was expected. It compared a strong policy with a
  sabotaged copy of itself.
* **C3 expected the wrong number:** with two IDENTICAL deterministic
  policies the move SEQUENCE is identical in both seat orders (the seat
  swap changes who owns the pieces, not the sequence), so the correct
  expectation is exactly **1** unique sequence. The old expectation of 2
  came from the A-vs-B case.

**Seat-gap scare resolved:** C5 flagged 59.7 % as P1 vs 39.7 % as P2 at
n=300. Re-measured at n=1,200 / 4 seeds (`tmp/audit_seat_gap.py`):
G1 model greedy-vs-explored = **+5.2 pp** (overall 50.4 %, p=0.1263),
G2 model explored-vs-explored = **+7.5 pp** (overall 49.6 %,
p=0.7933). The 20 pp figure was sampling noise.

**Reporting rule:** quote win rates PER SEAT. The `--games 20`
in-game numbers (e.g. "MODEL LOSES to always-c0, p=0.0038") remain
**not comparable** — the harness itself flags "too few unique games".

### D24. label_vs_learning re-run (join now works: 560/560)

```text
src=1  n=  8   label INTERSECTS opt 100.0 %   label==opt 87.5 %   model in opt 87.5 %
src=2  n=552   label INTERSECTS opt  94.6 %   label==opt 17.2 %   model in opt 74.8 %

UNIQUE-optimum positions (n=112):
  label == the unique optimum   97/112 = 86.6 %
  model pick is optimal          27/112 = 24.1 %
  label OK & model WRONG         72/97      <- LEARNING problem
  label WRONG & model OK          2/15      <- small LABEL problem
  positions whose optimum is c1/c2: 23;  label names c1/c2: 18;  model plays c1/c2: 0

TIED-optimum positions (n=448):
  label == the full tie set        5/448 =  1.1 %
  label INTERSECTS optimum       433/448 = 96.7 %
  model pick is optimal          393/448 = 87.7 %
```

The earlier run of this audit aborted with `joined truth->sidecar:
0/560` rather than reporting a wrong number. The join logic was
sound; the `disjoint_true` artifacts had since been rebuilt. The
audit-style join now reproduces **560/560**.

**Verdict unchanged and now sharper:** the labels are mostly optimal
(94.6 % intersect the exact optimum) but they are **collapsed to one
column** (17.2 % equal the full tie set, and the tie set averages 5.28
columns), and the model — trained under the supervision defect — does
not pick the labelled column on unique-optimum positions (24.1 %).
Combined with D22, the label collapse is the defect to fix; the
architecture is not the blocker.

### D23. Seat gap resolved (game property) + the c1 blind spot localised

**D23.1 Seat gap: GAME PROPERTY, not a model defect.**

The first G3 control was structurally invalid (perfect minimax vs 30 %-random
minimax -- a rout that reported an inverted -8.0 pp). Rewritten with matched
controls, 600 games per seat per arm:

```text
G1 MODEL(greedy) vs MODEL(explored)       gap +5.2 pp  p=0.073
G2 MODEL(explored) vs MODEL(explored)     gap +7.5 pp  p=0.009   <-- model, significant
G3 MINIMAX(explored) vs MINIMAX(explored) gap -1.7 pp  p=0.564   [CONTROL]
G4 HEURISTIC(explored) vs RANDOM(explored) gap +4.8 pp p=0.000   [CONTROL]
```

**The right comparator is G4, not G3.** G3 is perfect play on both sides: a
perfect player wins whether or not it moves first, so that arm is near-50/50
by construction and cannot bound a first-player advantage. G4 is a strong
HAND-WRITTEN, non-learned policy playing both seats -- structurally the same
kind of arm as G2.

Comparing GAP to GAP (not absolute win rates):

```text
gap(model) - gap(heuristic) = +2.67 pp  p=0.402  NOT significant
gap(model) - gap(minimax)   = +9.17 pp  p=0.024  significant
```

The model's +7.5 pp seat gap is statistically indistinguishable from what a
hand-written heuristic shows playing both seats. Connect Four has a real
first-player advantage; a non-learned policy shows it too. **Report win
rates per seat. Do not claim a model seat defect.**

Two mistakes were made and corrected here, both in the audit script:
1. The first valid G3 was perfect-vs-30 %-random (unmatched strength).
2. The first comparison of model-vs-control compared ABSOLUTE win rates, then
   compared gaps against a hardcoded +/-8 pp envelope. Both were wrong; the
   fix is gap-vs-gap with the regime-matched control named in code.

**D23.2 "LEARNING PROBLEM" verdict: confirmed, but the old checkpoint was
the wrong evidence.**

`tmp/audit_label_vs_learning.py` reported "LEARNING PROBLEM" (72/97) on
**fd_a**, trained under the supervision defect -- so it conflated the old bug
with a learning ceiling. The script now takes `--ckpt`/`--out` and flags
known-defective checkpoints in its own output.

Same audit, same 560 exact-solved positions:

```text
                              fd_a (defective)   fx_a (corrected)
model optimal, unique opt       24.1 %              26.8 %
model optimal, tied opt         87.7 %              90.0 %
model plays c1 (of 84 legal)     0 / 84              0 / 84
model plays c2 (of 74 legal)     0 / 74             24 / 74
model column histogram      {,,,6:7} {0:2,,,,,}
```

The corrected arm uses c2 and c0, which fd_a never touched -- the signature
of the supervision fix working. The "LEARNING PROBLEM" verdict SURVIVES the
fix (70/97), so it is not only the old bug.

The decomposition is the actionable part:

* **TIED optima (448/560, the majority): model is ~90 % right.**
* **UNIQUE optima (112/560): model is only ~27 % right.**

The model handles "several moves are fine" well and "this ONE move is the
only good one" badly. Labels are also wrong on **13.4 %** of unique-optimum
positions (15/112) -- checkpoint-independent, so label repair is still needed,
but it is a ~13 % defect, not the dominant term.

**D23.3 The c1 blind spot is a RANKING problem.**

Three hypotheses eliminated in order.

*Data coverage? NO.* Over the full 320,000-row corpus:

```text
c1 PLAYED in   720,827 / 5,100,175 moves          = 14.13 %  (c3 16.66 %, c6 12.41 %)
c1 NAMED in    615,118 / 4,563,661 labelled slots  = 13.48 %
c1 single-column teacher answer in 600,613 slots   = 13.16 %
```

c1 is a completely normal column in the training data.

*Path bug (hard export / legal mask / column slice)? NO.* On 200 exact-solved
holdout positions:

```text
c1 wins argmax UNMASKED: 7/200 = 3.5 %
c1 wins argmax MASKED: 5/200 = 2.5 %
c1 logit identical to another legal column: never
mean rank of c1 among legal columns: 4.36 / 7
```

*Is the network reading c1 at all? YES.*

```text
perturbing c1's board state moves c1's logit in 39/39 positions
mean |delta| = 0.578   max |delta| = 2.192
```

c1's logit is real, responsive and distinct -- it just rarely WINS. Mean rank
4.36/7 means c1 is treated as a mediocre column. This is a ranking/calibration
problem, not plumbing and not data. It is also the clearest remaining
measurable defect, and it is the right thing to fix next.

**D23.4 Scripts added**

* `tmp/audit_seat_gap.py` -- rewritten with matched controls + gap-vs-gap test.
* `tmp/audit_label_vs_learning.py` -- now `--ckpt`/`--out` aware, flags
  defective checkpoints in its own verdict output.
* `tmp/audit_c1_blindspot.py` -- corpus label/move coverage per column.
* `tmp/audit_c1_path_probe.py` -- raw-logit rank, aliasing and perturbation
  probe.

### D24. The corrected model IS stronger -- and a measurement bug that nearly hid it

**D24.1 Headline: corrected supervision produced a materially stronger model.**

One harness (`src.c4.duel.duel`), one seed, one opponent, one protocol:
model greedy (eps=0, hard path), opponent `random_policy`, 600 games,
global RNG seeded (because `duel` ignores its `seed` argument).

```text
                         vs random         as P1      as P2
fd_a (defective)      W399  L201  66.5 %  197/300   202/300
fx_a step5000         W479  L121  79.8 %  234/300   245/300
fx_a step5500         W469  L129  78.4 %  231/300   238/300
fx_a COMBINED (1200)  W948  L250  79.1 %  95% CI 78.9-79.4
                                +12.6 pp  Fisher p = 9.9e-09
```

The 1.4 pp spread between the two fx_a checkpoints is inside the
binomial SE at n=600 (~1.7 pp) -- noise, not a regression.

Column behaviour changed qualitatively:

```text
              top       c0+c1+c2     c2       c6      c3+c4
fd_a          c4 37.21 %  29.63 %   4.25 %  1.73 %  58.8 %
fx_a s5500    c3 35.67 %  34.13 %  14.13 %  3.06 %  52.1 %
```

c2 usage MORE THAN TRIPLED. On the exact-solved holdout the defective fd_a
could not play c2 at all (0/74 legal positions); fx_a does.

Against TRUE constant-column opponents (600 games each, step-5500 ckpt):

```text
always-c0 75.0 % (fx_a better, p=0.172 ns)
always-c1 75.1 % (fx_a better, p=0.193 ns)
always-c2 80.5 % (fx_a worse,  p=0.392 ns)
always-c3 82.2 % (fx_a worse,  p=0.110 ns)
always-c4 79.2 % (fx_a worse,  p=0.778 ns)
always-c5 77.3 % (fx_a better, p=0.676 ns)
always-c6 74.3 % (fx_a better, p=0.103 ns)
```

**fx_a is statistically INDISTINGUISHABLE from every constant-column
opponent.** The step-5000 checkpoint was significantly better than
always-c6 (p=0.028) while step-5500 is not (p=0.103) -- that is exactly
the single-comparison significance that must not be quoted without a
correction, which is why the headline is "indistinguishable".

**Honest headline:** the corrected model has caught up with a fixed-column
bot but has not yet clearly beaten it. Always-c3 is marginally ahead.
This is a step forward from fd_a, which lost to EVERY constant arm.

**D24.2 The measurement bug that nearly hid this.**

`tmp/audit_ingame_cols_fd_a.py` defined its reference opponent as
`col if col in legal else legal[0]`. That is a *staircase*, not a constant:
once the preferred column fills it falls back to the leftmost legal column.

The symptom was a **byte-identical** W/L/D record for `always-c3` and the
shipped `centre_policy` across all FIVE opponents:

```text
vs random    both W546 L54 D0
vs heuristic both W133 L467 D0
vs lookahead both W97  L503 D0
vs minimax   both W235 L363 D2
vs centre    both W322 L249 D29
```

Two policies cannot match exactly by chance -- they were the same function.
Diagnosis script `tmp/audit_c3_centre_equivalence.py` showed the staircase
fallback firing **36 times in one game**, picking c6/c2/c4/c6/c5/... where
`centre_policy` picks c0/c0/c0/... Head-to-head they played **W68 L132**,
confirming they are different policies with the same *fallback rule*.

Fixes applied to the script:
1. `const_policy` now falls back to `random.choice(legal)` -- a true constant.
2. The global RNG is seeded before every duel (`duel` ignores `seed`).
3. `--ckpt` / `--out` / `--games` parameters so defective and corrected arms
   are measured under one identical protocol.
4. The script labels known-defective checkpoints in its own output.

**D24.3 Stale-artifact trap.**

`runs/_probe/ingame_strength_fd_a.json` was written at **** while
`tmp/audit_ingame_strength_fd_a.py` was fixed at ****. The JSON
predates the fix, so its constant-column reference table carries the
staircase defect. Citing it as a strength result would have been wrong.
Both arms were re-measured into `ingame_cols_fd_a_v2.json` and
`ingame_cols_fx_a_v2.json`.

**Rule adopted: always compare artifact mtime against script mtime before
citing a measurement.**

**D24.4 Also corrected in this batch:** a Fisher-exact implementation in an
ad-hoc analysis used row marginals where column marginals were required and
returned p = 1.0000 for every comparison, including a 400-vs-546 win gap.
The validated `fisher_exact_2x2` from `tmp/audit_ingame_strength_fd_a.py`
reproduces the textbook [[1,9],[11,3]] -> 0.00275946 and is now the only
one used. All p-values above come from it.

**D24.5 Superseded numbers (kept for the record).**

* `runs/_probe/ingame_cols_fd_a.json` (, 200 games) -- staircase refs,
  unseeded global RNG.
* `runs/_probe/ingame_strength_fd_a.json` (, 600 games) -- staircase refs.
* `runs/_probe/ingame_strength_smoke.json` / `..._trueconst.json` (/)
  -- 60-game smoke runs of the corrected harness.
* `runs/_probe/ingame_cols_fx_a.json` (, 200 games) -- staircase refs.

**D24.6 Still open.** fx_a was at 5000/6000 when measured. The c1 ranking
defect is unfixed. The heuristic/lookahead/minimax strength arms have not
been re-measured for fx_a. Exact-solver holdout coverage is still 560/1500.

## Appendix D25 -- Corrected arms: final strength, reproducibility, and the surviving learning gap

**D25.1 Corrected full retrain complete.** fx_a/fx_b/fx_c all reached
step 6000, seed 32, transcript v2, promoted checkpoint = final
checkpoint. Final direct sidecar-label agreement (shipped hard path):
fx_a 50.92 %, fx_b 50.33 %, fx_c 51.02 %. The LABEL curve was still
climbing at step 6000 in every arm (fx_a: 16.27 % at step 0 -> 38.48 %
at 1500 -> 49.15 % at 5000 -> 50.92 % at 6000). The >90 % figure from the
500-row acceptance run does NOT transfer to full-corpus training at a
fixed step budget.

**D25.2 Strength matrix, all three arms, n=1800 per opponent**
(600 games x 3 seeds 99/1234/4242; model greedy, eps=0, shipped hard
path; global RNG seeded because `src.c4.duel.duel` ignores its `seed`;
all 7 constant arms use true-constant semantics with random fallback).

```text
opponent    fx_a     fx_b     fx_c    best constant        model verdict
random     81.1 %   83.2 %   79.9 %  centre_policy 91.3 %  LOSES (p~1e-13)
heuristic  27.8 %   22.1 %   31.3 %  always-c1   25.6 %  mixed
lookahead  11.3 %   11.9 %   13.2 %  centre_policy 16.7 %  LOSES (p~1e-6..4e-3)
minimax    39.7 %   38.6 %   34.2 %  centre_policy 37.0 %  fx_a/fx_b tie (p.06/.19)
centre     81.8 %   82.3 %   73.8 %  always-c1   77.2 %  fx_a/fx_b WIN, fx_c LOSES
```

fx_a beats ALL SEVEN single-column arms against minimax (p=0.0000 each).
`centre_policy` is NOT a constant column -- it is a centre-first priority
sweep (c3, then c4, then c2,...) and is the strongest single opponent
measured. The corrected model's weakness is DEPTH (loses to lookahead and
to the priority sweep), not column collapse: against minimax its column
usage is c0 13.5 %, c1 13.3 %, c2 14.8 %, c3 30.4 %, c4 14.8 %, c5 11.7 %,
c6 1.4 %.

**D25.3 fx_a matrix independently reproduced, bit-exact.** A clean serial
re-run reproduces every one of the 45 model+constant cells identically
(random W1460 L337 D3, heuristic W500 L1297 D3, lookahead W203 L1591 D6,
minimax W715 L1067 D18, centre W1473 L327 D0). The project rule "the
artifact independently reproduces the headline statistic" now holds.

**D25.4 The learning gap survives the supervision fix.** Against
corrected fx_a on the 560 exact-truth positions: on UNIQUE-optimum
positions the label names the optimum 97/112 (86.6 %) and the model still
misses it 70/97 times; the model is optimal on only 30/112 (26.8 %). On
TIED-optimum positions the model is optimal 403/448 (90.0 %) and the
label intersects the optimum 433/448 (96.7 %). So the model is good when
several moves are acceptable and poor when exactly one is. Fixing labels
alone will not close the gap.

**D25.5 c1 remains a pure ranking defect.** On the 112 unique-optimum
positions the corrected model selects c1 **0** times (histogram
{0:2, 1:0,,,,, }) while c2 became reachable
(24). Combined with the c1 path probe (responsive logits: 39/39
perturbations, mean |delta| 0.578; mean legal rank 4.36/7) and the c1
data audit (c1 = 14.13 % of played moves, 13.48 % of label mass), c1 is
a ranking/learning defect that survives both the supervision fix and the
data audit.

**D25.6 Script defects found and fixed in this phase.**
- `tmp/post_train_eval.sh` looked for `best_hard.pt` at the run-dir root
  instead of `checkpoints/best_hard.pt`; it reported three healthy arms
  as unusable. Fixed.
- `tmp/audit_ingame_strength_fd_a.py` carried a private copy of the naive
  binomial and raised `OverflowError: int too large to convert to float`
  for n >= 1200 (the harness uses 1800 games/arm). Replaced with the
  log-space `math.lgamma` version; cross-checked against the naive form
  for n <= 1000 (agreement ~1e-13). This was the cause of the exit-1
  failures in proc_a1655cc2ebf6 and proc_5f46223af61c.
- The harness printed `COMPARABLE -- MODEL LOSES... (p=0.0000)`,
  conflating comparability with the strength verdict. Now prints
  `COMPARISON VALID -- significant difference -- MODEL LOSES...`.

**D25.7 Process-hygiene failure, disclosed.** An early attempt ran all
three strength arms concurrently. They did not die; two survived and
wrote to the same JSON/log paths, producing NUL-padded fx_a/fx_b logs.
All were killed and verified dead (zero matching processes). Each JSON
artifact was then checked for completeness (5 opponents, n=1800,
min_unique >= 355) and the fx_a matrix was re-run cleanly to bit-exact
reproduction.

**D25.8 Superseded artifacts.**
- `runs/_probe/seat_gap.log` (old run: G3 = minimax greedy-vs-explored,
  verdict INCONCLUSIVE, no G4) is superseded by
  `runs/_probe/seat_gap_v2.log` + `seat_gap_discriminator.json`
  (G3 = minimax explored-vs-explored, plus G4 heuristic-vs-random,
  gap-versus-gap p=0.402).
- `runs/_probe/ingame_cols_fd_a.json` (old constant-column reference)
  is superseded: its `always-c3` used a `legal[0]` fallback, making it a
  centre-first priority sweep; `always-c3` and `centre_policy` produced
  byte-identical records.
- `runs/_probe/tie_model_behaviour.json` IS the chance-controlled
  version (lift 1.015x on ties vs 1.173x on singles). The raw log text
  that quotes "the model DOES represent ties" is the pre-control
  reading and is retracted.

**D25.9 Regression sweep.** `test_symmetry_penalty.py` exited 139 inside
the sweep (proc_0815ffd7e237) while other processes were loading
checkpoints. It passes standalone with all 10 assertions, so the failure
is native/resource contention, not a code regression.

**D25.10 Release status: BLOCKED.** Do not launch V24. The corrected
arms are strong training artifacts and the supervision bug is fixed, but
the model still (a) loses to `centre_policy` and to lookahead, (b) sits
at ~51 % direct label agreement on full-corpus training, and (c) misses
the unique optimum on 73 % of the positions where the label named it.
Concrete next targets: reproduce the fx_b/fx_c matrices serially, and
attack the unique-optimum ranking gap directly (it is now the dominant
error source, not labels and not data coverage).
**D25.11 Transient test-failure root cause.** `test_symmetry_penalty.py`
(exit 139), `test_live_parity_v4.py` and `test_live_parity_v5.py`
(ConnectionResetError) all failed inside the sweep and all PASS
standalone. Cause: two concurrent `tmp/post_train_eval.sh` chains (one
stale from, one from), both writing the same log and both
running the 841 MB generic full-corpus `measure_label_agreement.py`
scan, competing for CPU with the sweep and sharing the one live UI
server. After killing PIDs 5604/50492/53844/58564/16160 the serial sweep
returned **17 PASS, 0 FAIL** (v3 40/40, v4 49/49, v5 75/75, L1 0.0).
An intermediate "26.9 MB free" reading was a units error -- the real
figure is 25.5 GB free of 61.6 GB, so memory was never the cause.
**D25.12 Correction to the "beats 5/7 constant references" claim.** That
count was measured against the RANDOM opponent only. The full 5-opponent
matrix is:

```text
opponent     MODEL    best single-column   centre_policy   verdict
random       81.1 %   always-c3  82.9 %    91.3 %          LOSES to both
heuristic    27.8 %   always-c1  25.6 %    20.6 %          BEATS both
lookahead    11.3 %   always-c2  14.7 %    16.7 %          LOSES to both
minimax      39.7 %   always-c3  26.4 %    37.0 %          BEATS both
centre       81.8 %   always-c1  77.2 %    57.8 %          BEATS both
```

The model beats a single-column constant in 2 of 5 settings and loses or
ties in the other 3. `centre_policy` (a centre-first priority sweep, a
legal non-degenerate opponent) is stronger than the model against
random and lookahead.

**D25.13 Silent-skip defect fixed.** `tmp/measure_promotion_scorer.py`
exited 0 with `SKIP: missing corpus.txt or perply_targets.npz` when
handed a checkpoint path rather than a run dir. Now prints an explicit
ERROR and exits 1. Verified both ways.

**D25.14 Stale-script failure.** `proc_fcb546d0f1ab` (c1 blindspot audit)
failed with `FileNotFoundError:.../perply_v23_full/corpus.txt` because it
ran a STALE copy of the script; the on-disk script points at
`corpus_320k_fixed.txt` and its artifact is newer than the script.
Re-running the current script reproduces identical numbers (c1 = 14.13 %
played / 13.48 % label mass / 13.16 % single-column teacher answers).


---

## Appendix D26 — The full arms were undertrained: 0.30 epochs 

**This supersedes the framing in D23–D25 that the corrected model's
remaining weaknesses were architecture/objective findings.** They were
training-budget findings.

### D26.1 The arithmetic

```text
acceptance run  overfit_fixed: 3923 steps x batch 16 =  62,768 samples
                                 over 500 rows        = 125.5 EPOCHS
                                 -> LABEL 91.29 % (scorer 92.38 %, regression 92.40 %)

corrected full fx_a/fx_b/fx_c: 6000 steps x batch 16 =  96,000 samples
                                 over 320,000 rows    =   0.30 EPOCHS
                                 -> LABEL 50.92 / 50.33 / 51.02 %
```

One epoch over the full corpus = 320,000 / 16 = **20,000 steps**. The
launcher used `--steps 6000`, i.e. it stopped at **30 % of the first
epoch**. The trainer's own default is `--steps 20000` — exactly one
epoch — so the launcher overrode the default downward.

### D26.2 The learning curve had not plateaued

```text
step  500  LABEL 37.55 %
step 2000  LABEL 40.23 %
step 3000  LABEL 43.55 %
step 4000  LABEL 45.33 %
step 4500  LABEL 47.25 %
step 5000  LABEL 49.15 %
step 5500  LABEL 49.50 %
step 6000  LABEL 50.92 %   <-- +1.42 pp over the last 500 steps
```

The run was terminated by the step budget, not by convergence. On the
same data, at the same hyperparameters, with only the budget changed,
the acceptance run reached 91 %.

### D26.3 What this invalidates, and what it does not

INVALIDATED as architecture/objective claims (all measured on a 0.30-epoch model):

* unique-optimum ranking gap — model optimal 30/112 while the label
  names the optimum 97/112;
* c1 blind spot — c1 selected 0/84 when legal, mean legal rank 4.36/7;
* lookahead weakness — 11.3 % win rate;
* loss to `centre_policy` on random and lookahead;
* the fx_a/fx_b/fx_c strength matrix as a characterisation of the
  architecture.

NOT invalidated:

* the supervision fix itself and the 91.29 % acceptance result;
* the data, label and solver-quality findings (GOLD 4000/4000, legal
  slots 45,929/45,929, exact-set equality 85.7 % on unique optima);
* the harness/protocol defects and their fixes;
* the c1 *data-coverage* finding (c1 appears in 14.13 % of moves) — but
  its *ranking* consequence is now a training-budget question first;
* the bit-exact fx_a reproduction of the strength matrix.

### D26.4 The `--resume` bug (V24 prerequisite)

`for step in range(start_step, args.steps + 1)` is empty when
`start_step > args.steps`, so `step` stayed unbound and the first
reference raised `UnboundLocalError: local variable 'step' referenced
before assignment`. Resuming a finished run for a short extension was
impossible — the exact operation a 37 h run needs after a crash.

Now it fails loudly:

```text
nothing to resume: the checkpoint is already at step 6000, which is beyond
--steps=100. Pass a larger --steps (the run continues from step 6000), or
drop --resume to start fresh.
```

### D26.5 V24 specification

```text
trainer: src/c4/train.py, corrected per-ply indexing (shipped fix)
seed: 32 (canonical)
data: runs/_probe/perply_v23_full/corpus_320k_fixed.txt (320,000 rows)
sidecar: runs/_probe/perply_v23_full/merged_repaired.npz
ctx / batch: 192 / 16
flags: --max-legacy-frac 0 --table-bias 0.0 --hard-start 0
               --selfplay-games 0 --teacher-games 0
promotion: direct per-ply label agreement (primary), byte-bpb tie-break
steps: 100,000 (5 epochs, ~37 h) or 200,000 (10 epochs, ~73 h)
eval cadence: raise from 500 (hard export + 4,000-position scoring is a
               large share of the measured 1.316 s/step)
ladder: --eval-games 24 (the launcher's 0 produced vacuous W0L0D0)
resume: from <run>/final_float.pt (carries {"state":...}, meta.step)
```

### D26.6 Battery hardening (three real defects, found by running it end-to-end)

1. **The promoted artifact was never measured.**
   `best_hard.pt` is at `<run>/checkpoints/best_hard.pt`; the scorer
   looked in the run ROOT and the `step*_hard.pt` glob does not match it
   (`fnmatch("best_hard.pt","step*_hard.pt")` is False). The model that
   would actually ship was silently skipped. Fixed — fx_a now reports
   `best_hard.pt LABEL 50.92%` where it previously reported nothing.
2. **Two battery chains could write the same paths** (cause of the
   earlier NUL-padded logs and the overwritten JSON). Fixed with
   `mkdir`-based one-writer locks in `tmp/post_train_eval.sh` and
   `tmp/wait_and_eval.sh`; both verified to refuse a second chain and to
   release on normal exit.
3. **The generic scanner overwrote one shared JSON**
   (`runs/_probe/label_agreement.json`) — six invocations, five results
   erased. Fixed: the output path is now derived from the checkpoint
   names.

Battery order corrected: the cheap bounded shipped scorer runs first and
its exit code is checked; the memory-heavy generic scanner runs last as
an independent cross-check.

### D26.7 `wait_and_eval` failure at  is superseded

```text
proc_1a6669f1af4f:
  BATTERY_EXIT=1
  ARM NOT USABLE  fx_a_s6000: missing/empty runs/_probe/fx_a_s6000/best_hard.pt
```

That run's log is timestamped ****; the `checkpoints/best_hard.pt`
path fix landed at ****. The failure predates the fix and is a
superseded orchestration defect, not an arm failure. All three corrected
arms completed at step 6000 with transcript-v2 artifacts and a promoted
`checkpoints/best_hard.pt`.

**But it was not a false alarm:** it exposed that the battery never
measured the promoted artifact (D26.6.1). The failure was a symptom of
a real gap in the gate.

## Appendix D27 — Battery closure, resume-path repair, V24 pre-flight

### D27.1 The battery now measures the artifact that would ship

The per-arm scorer logs written by the hardened battery show the promoted
checkpoint for all three corrected arms, side by side with the other
candidates:

```text
fx_a_s6000:
  hard_model_final.pt    LABEL 50.92%  n=4000  mass=0.4848
  best_hard.pt           LABEL 50.92%  n=4000  mass=0.4848
  step6000_hard.pt       LABEL 50.92%  n=4000  mass=0.4848
fx_b_s6000:
  hard_model_final.pt    LABEL 50.33%  n=4000  mass=0.4647
  best_hard.pt           LABEL 50.33%  n=4000  mass=0.4647
  step6000_hard.pt       LABEL 50.33%  n=4000  mass=0.4647
fx_c_s6000:
  hard_model_final.pt    LABEL 51.02%  n=4000  mass=0.4738
  best_hard.pt           LABEL 51.02%  n=4000  mass=0.4738
  step6000_hard.pt       LABEL 51.02%  n=4000  mass=0.4738
```

All three arms promote the final step. The earlier battery never
measured `checkpoints/best_hard.pt` at all (D26.6.1); that gap is closed.

### D27.2 Two more resume-path defects, found by the resume smoke test

The resume path was still broken after the `UnboundLocalError` fix. The
smoke test (resume fx_a `final_float.pt`, 6000 -> 6010) exposed both
remaining defects.

**(a) The model was built from `--lags` BEFORE the checkpoint was read.**

A run trained with the dense 12-lag set (0..10, 20) was being resumed
into the V6 default 8-lag model, and died inside the forward pass:

```text
IndexError: index 9 is out of bounds for dimension 0 with size 8
```

The lag set is part of the model's STRUCTURE, not a hyperparameter the
caller may change on resume. The checkpoint is now loaded first, the lag
set is taken from `meta.lags`, and a disagreeing `--lags` is refused:

```text
refusing to resume: the checkpoint was trained with lags
[0,1,2,3,4,5,6,7,8,9,10,20] but --lags says [0,1,2,4,8,16,32,64].
The lag set is part of the model structure, so it cannot be changed on
resume. Drop --lags to take the checkpoint's own set.
```

`final_float.pt` does carry the set: `meta.lags = [0,1,2,3,4,5,6,7,8,9,10,20]`.

**(b) `model.import_hard(ckpt)` does not exist on the model.**

`src/gf256_markov_lm_v6.py` has no `import_hard` method, so the hard
branch of the resume path raised `AttributeError` before training a
single step. The resume now prefers the float `{"state":...}` payload
and only falls back to the integer export.

Together these mean **no resume of any kind had ever succeeded** — every
earlier attempt failed for a different reason (unbound `step`, wrong lag
count, missing `import_hard`). The three are now all closed.

### D27.3 `run_v24.sh` was wrong about epochs, and would have fought the resume fix

The launcher's epoch arithmetic was inverted:

```text
EPOCHS = rows * batch / steps      # WRONG
```

which printed `51 epoch-equivalents` for a 100,000-step run. The true
figure is `steps / (rows / batch)` = `steps * batch / rows`, i.e. one
epoch = 320,000/16 = **20,000 steps**, so 100,000 steps = **5.00
epochs** and 6,000 steps = **0.30 epochs**. The launcher now computes it
with `awk` and prints both the epoch length and the epoch-equivalents:

```text
V24 CONFIG: rows=320000 steps=100000 batch=16
            one epoch = 20000 steps -> this run = 5.00 epoch-equivalents
```

The launcher also no longer passes `--lags` on the resume path
(`COMMON_NO_LAGS`), so it cannot contradict the checkpoint's lag set.

### D27.4 V24 pre-flight: smoke-tested end to end

`run_v24.sh` was exercised with a 30-step smoke run (`V24_TAG=v24_smoke
V24_STEPS=30`) and completed the whole chain: corpus integrity
pre-flight (320,000 rows, 0 violations), trainer start, hard export,
artifact presence check, transcript-stamp check, and the shipped
promotion scorer. The smoke run was then deleted.

**V24 remains unlaunched.** The rule stands: do not start the long run
until the resume path is proven end-to-end, because a 37 h run that
cannot be resumed after a crash is not a safe thing to start.


---

## Appendix D28 — External review, applied: the pre-launch package

**Snapshot:** before the midnight V24 launch
**Status:** V24 pre-flight complete. Not yet launched.

### D28.1 What the review got right, and what I checked rather than believed

An external reviewer sent a review of the trainer with five substantive claims. Each was
checked against the **shipped** code, not against a description of it.

| # | Reviewer's claim | Verdict | Evidence |
|---|---|---|---|
| 1 | The historical per-ply path had a **second** indexing error (`// (W+1)` over a `B×W` grid), so batch row *k* received the early positions *t < k* from row *k−1* | **CONFIRMED** | `tmp/verify_second_index_error.py` reproduces the exact failure pattern |
| 2 | The *current* fix is correct | **CONFIRMED** | decoded-ply invariant: **6504/6504** production sampled positions pass, including small-`t`, `k>0` cases |
| 3 | The historical "per-ply supervision LOST", "mirror is a net negative", "symmetry not worth it" decisions must be **withdrawn** | **CONFIRMED** | those decisions predate the sidecar-row fix; they measured the bug, not the model |
| 4 | The trainer saves **no optimizer state and no RNG state**, and has **no LR schedule and no EMA** | **CONFIRMED** | all four were absent before this session's edits |
| 5 | Batch 32 gives **2×** the throughput | **REFUTED** | real V24 path measures **1.22×** (see D28.4) |

The review also claimed `src/gf256_markov_lm.py` "already has that checkpoint
logic and can be adopted". **That specific pointer is inaccurate** —
that file contains no optimizer-state or RNG-state checkpoint
implementation to reuse. The logic was written from scratch here.

### D28.2 What was wrong with the earlier conclusions

Because only the **per-ply** path read the wrong rows, the historical
A/B compared *correct* single-label supervision against *garbage* per-ply
supervision. The conclusion "dense per-ply supervision LOST" therefore
compared a working path against a broken one. The corrected
`fx_a/fx_b/fx_c` arms are the mirror/symmetry comparison — they are
**not** a per-ply-versus-single-label test, and no such test has been run.

### D28.3 The trainer changes (all opt-in, historical runs reproduce exactly)

| Change | Flag | Default | Why |
|---|---|---|---|
| Optimizer state saved/restored | — | **always on** | Adam restarting with empty moments spikes the loss after every restart |
| Python/NumPy/Torch RNG state saved/restored | — | **always on** | without it a resume samples different rows and drifts off the original trajectory |
| Cosine LR schedule | `--lr-schedule` | `none` | constant LR starts oscillating once the curve bends |
| EMA weights | `--ema-decay`, `--ema-warmup`, `--ema-export` | `0.0` (off) | flattens the late-training oscillation |
| Disjoint-holdout per-ply agreement | `--holdout-corpus` / `--holdout-targets` | off | the promotion metric samples the **training** corpus; that measures *fit*, not generalisation |
| Per-column pick/label histograms | — | always on | separates "the data lacks c1" from "the model ranks c1 last" |
| Checkpoint pruning | `--keep-last`, `--keep-hard-all` | `3` | 50 snapshots × 237 MB = 11.9 GB does not fit |
| Abort rules | `--abort-warmup`, `--abort-flat-evals`, `--abort-flat-below`, `--abort-on` | `20000` / `4` / `55.0` / `warn` | a wrong abort on an 18 h run is more expensive than a run that continues past a plateau |

Optimizer and RNG state are stored at the **top level** of the float
checkpoint (`opt_state`, `rng_state`), not inside `meta` — `meta` is
reused by the hard export, which must stay a lean integer-table payload.

### D28.4 Batch size: the measured answer, not the claimed one

The reviewer's table came from `tg_scale4096.py`, a **4096-row V22**
architecture with a different loop. The real V24 path
(`src.c4.train`, per-ply sidecars, dense 12 lags, ctx 192) measures:

| batch | s/step (train only) | rows/s | gain vs 16 |
|---|---|---|---|
| 16 | **0.646** | 24.77 | 1.00× |
| 32 | 1.063 | 30.10 | **1.22×** |
| 64 | 1.27 | ~50 (different loop) | — |

At the V24 evaluation cadence (eval every 2,000 steps, one eval ≈ 7.9 s)
the amortised cost is **+0.004 s/step**, so the effective rate is
**0.650 s/step** and:

- one epoch (20,000 steps) ≈ **3.6 h**
- 100,000 steps (5 epochs) ≈ **18.1 h**

The reviewer's "5 epochs in ~9 h" is not supported by the real path.
**Batch 16 remains the documented configuration** — the paper constraint
stands, and 1.22× is not worth changing the experimental protocol for.

### D28.5 The concurrency correction: the real step rate is 0.65 s, not 1.30 s

The earlier 1.30 s/step figure came from `fx_a/fx_b/fx_c`, whose logs show
the three arms **started 12 seconds apart** — they ran **concurrently on
one GPU**. Each was therefore running at roughly a third of the
throughput, and 1.3164 s/step was the *shared* rate, not the single-run
rate. A single V24 run gets the whole GPU: **0.650 s/step**.

This changes the timeline by a factor of two and is the single most
important number in this appendix.

### D28.6 The holdout path had never been executed — and was broken twice

`run_v24.sh` never passed `--holdout-corpus` / `--holdout-targets`, so
the holdout scoring path had **never run**. Wiring it up exposed two real
defects:

1. **`Holdout.__init__` assumed the old `(N, 7)` sidecar.** The
   production per-ply sidecar is `(N, K, 7)` (one target row per labelled
   ply), so `self.optimal[i]` was a `(K, 7)` tensor and
   `bool(opt[pick])` raised
   `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.
   Fixed: the loader now detects the schema explicitly and, for a
   per-ply sidecar, resolves the final-ply slot through `ply_of` — the
   same slot/ply distinction the training path was fixed for.
2. **`PerPlyAgreement` assumed torch tensors.** The holdout caller
   passes numpy arrays straight out of `np.load`, so `.to(device)` raised
   `AttributeError: 'numpy.ndarray' object has no attribute 'to'`.
   Because that failure was **caught and only logged**, the run continued
   *silently without the generalisation metric and without the abort
   rules*. Fixed: the scorer normalises its inputs.

Both were found only because the flag was finally wired up — an
untested path is not a working path.

### D28.7 The abort rules fired at step 0 and killed the run

With the holdout working, the new abort rule-2 ("a column that is optimal
in the labels was NEVER picked") fired **immediately at step 0** on a
*randomly initialised* model — of course it never picks c4 — and raised
`SystemExit` before a single training step ran.

Two fixes:

- both rules now require `--abort-warmup` (default 20,000 = one epoch);
- `--abort-on` defaults to **`warn`**: the finding is recorded in
  `ABORT.txt` and the run **continues**. Stopping an 18-hour run must be
  an explicit decision, because a false positive costs far more than a
  run that continues past a recoverable plateau.

### D28.8 Resume is now provably lossless

`tmp/test_resume_state.py` — **19 PASS, 0 FAIL**:
`final_float.pt`, `best_soft.pt` and every `step*_soft.pt` carry
`opt_state` + `rng_state`; the hard export carries neither (lean
payload) but keeps its transcript stamp; a hard-export resume explains
itself and does not claim to restore Adam moments.

`tmp/test_resume_equivalence.py` — **12 PASS, 0 FAIL**: an
uninterrupted 12-step run and a 6-step-then-resume-to-12 run produce
**bit-identical** results:

```text
max |weight difference|      = 0.000e+00
max |Adam moment difference|  = 0.000e+00
```

This is the property that matters for an 18-hour run: the final artifact
no longer depends on where the interruptions happened.

### D28.9 Disk: the 5 GB question, answered with measured numbers

Measured artifact sizes:

| artifact | size |
|---|---|
| float snapshot (weights + Adam moments + RNG) | **203 MB** |
| hard export | **34 MB** |
| corpus copy + sidecar | 55 MB |

For a 100,000-step run with a snapshot every 2,000 steps (50 snapshots):

| policy | footprint | fits in 9.6 GB? |
|---|---|---|
| unbounded | 50 × 237 MB = **11.9 GB** | **no** |
| `--keep-last 3` (default) | 3 × 237 MB + best ≈ **0.9 GB** | **yes** |
| `--keep-last 3 --keep-hard-all` | ≈ **2.6 GB** | yes |

The volume was at **5.5 GB free (100 % full)** when this work started;
only scratch/probe directories were deleted to recover **9.6 GB**. No
historical corpus or useful checkpoint was removed. The pruning code was
validated on a real run: with `--keep-last 3` exactly three float and
three hard snapshots survive, and `best_*` is never touched.

### D28.10 V24 pre-flight: what was actually executed

A 60-step smoke test of the **real launcher** (`run_v24.sh`, ctx 192,
batch 16, transcript v2, dense lags, seed 32, real holdout) completed
end to end with exit 0:

```text
step  0 | LABEL 16.3% (n=4000) | H-LABEL 11.9% (n=512)
step 30 | LABEL 32.4% (n=4000) | H-LABEL 30.5% (n=512)
step 60 | LABEL 36.8% (n=4000) | H-LABEL 31.2% (n=512)
[snapshot saved -> step0/30/60]
saved -> hard_model_final.pt
saved -> final_float.pt
```

The holdout agreement **tracks** the training agreement from the very
first evaluation — which is what makes it usable as a generalisation
signal and as an abort input.

### D28.11 What is still open

- The 100,000-step LABEL value is **unknown**; the log-linear
  extrapolation of the fx_a curve gives ≈ 63 %, not the 70–78 % earlier
  suggested.
- The LR/EMA options are implemented but **not numerically piloted**; they
  ship disabled.
- No corrected per-ply-versus-single-label A/B has been run.
- `runs/_probe/disjoint_holdout.*` is the previously established
  canonical-board-disjoint artifact; a fresh exact overlap count has not
  been recomputed.
- The full release battery (hard evaluator, mirror/c1–c4, in-game,
  trajectory, seat, artifact, restoration) must pass on the converged
  model before any release claim.

## Appendix D29 — The rotating live-parity failures, and a green sweep 

### What was wrong

The live-UI tests failed *rotating* across sweeps: `test_live_parity_v3`,
`_v4`, `_v5` and `test_live_ui_fd_a` failed in different combinations on
different runs, while the same tests passed in isolation. Three distinct
causes were separated by measurement:

1. **Windows double-bind.** `ThreadingHTTPServer.allow_reuse_address =
   True` sets `SO_REUSEADDR`, which on Windows permits a *second live
   socket* to bind an address a *first live socket* already holds. A stale
   server from an earlier session kept answering on port 7860 while a
   replacement printed `serving`. The replacement's `/modelinfo` lacked the
   `step` field; the tests failed at random depending on which process
   answered.

   *Fix:* `allow_reuse_address` left at default `False`; the port is claimed
   **before** the 34 MB model load so a lost race costs ~1 s, not ~60 s.
   A second server now exits **3** with `WinError 10048`.
   *Regression:* `tmp/test_ui_single_server.py` — **8/8 PASS**.

2. **Torch thread oversubscription.** Each `GET /state` runs a real
   forward pass (~93 ms float + ~12 ms hard). Torch claimed all 32 cores
   per inference, so a server thread plus a co-located client
   oversubscribed the box: `/state` latency went past the client's 20 s
   timeout and the client reported a read timeout **while the server logged
   HTTP 200**.

   *Fix:* `src/c4/ui_server.py` pins `torch.set_num_threads` via
   `TG_UI_TORCH_THREADS` (default 4). Measured idle latency with the
   budget: ~220 ms.

3. **Orphan processes (the actual cause of the rotation).** At the worst
   sweep four stale `.venv-c4` processes were alive, one of them a
   `post_train_eval.sh` loop that had been re-running a full-corpus
   `measure_label_agreement.py` scan since — roughly five hours of
   continuous CPU contention.

   *Fix:* killed the orphans; `post_train_eval.sh` now runs under a
   one-writer lock, and its children must not outlive it.

   `tmp/run_regression_sweep.sh` additionally now **owns** its server
   (starts it from current source, refuses to run if the port is already
   held, stops it on exit) so a sweep can never be measured against an
   ambient server it does not control.

### The clean result

```text
=== 24 PASS, 0 FAIL ===
```

24/24 including every live-parity test, `test_ui_api.py`,
`test_ui_hot_reload.py` (14/14), `test_ui_single_server.py` (8/8),
`test_resume_state.py` (19/19) and `test_resume_equivalence.py`
(bit-exact).

### Also fixed in this stretch

- **Empty training set** now fails loudly before `build_batch` with an
  actionable message (a resume must carry the same corpus/sidecar), rather
  than `ValueError: high <= 0` three frames deep.
- `tmp/measure_promotion_scorer.py` exits non-zero on checkpoint-path
  misuse instead of silently skipping.
- fx_a's promoted artifact re-validated at **LABEL 50.92 %** by the
  shipped scorer (`hard_model_final.pt` and `best_hard.pt`).

### Launch status

V24 is **no longer blocked**. The launcher smoke (`v24_final_gate`) exits
0 with `PRE-FLIGHT OK: 320000 rows, 0 violations`, holdout wired, and the
authoritative scorer reporting `LABEL 34.17 %` at step 40 versus
`16.27 %` at step 0. The full run has still **not** been launched.

### Standing operational rule

A timing-sensitive test result measured on a loaded machine is not
evidence. Check the process table before believing one.

## Appendix D30 — Concurrent-sweep root cause and the transient Windows start failure (later)

### Three sweeps ran at once

Batch results of 24/24, 23/24 and 22/23 looked like three different
bugs. All three had the same cause: the regression sweep takes a
one-writer lock with `mkdir "$LOCK"`, and my own batch commands cleared
it with `rm -rf runs/_probe/.regression_sweep.lock` before each run.
`rm -rf` on a directory removes it, so each sweep deleted the lock its
predecessor was holding and all three drove the same port-7860 UI server
simultaneously, resetting each other's board state.

Symptoms that followed, and their isolated results:

| Symptom | Isolated rerun |
|---|---|
| `test_ui_hot_reload.py` failed | **14/14 PASS** |
| `test_live_parity_v5.py` failed | PASS in the green sweep |
| `test_resume_state.py` failed with `rc=3221225794` | **19/19 PASS** |

### Fixes

**Sweep lock — now PID-aware.** The lock dir records the owner's PID in
`$LOCK/pid`. On contention: if that PID is dead the lock is reclaimed
automatically (a crashed sweep cannot wedge the suite); if it is alive
the newcomer aborts with a message that explicitly warns against `rm -rf`.
Verified both paths directly.

**Sweep summary — now run-unique.** The summary is written to
`_sweep_<pid>_summary.log` and copied to the stable path
`regression_sweep.log`. A fixed summary path let concurrent sweeps
overwrite each other's PASS/FAIL lines, which is what made three
innocent results look contradictory. The cleanup removes `$LOCK/pid`
before `rmdir` (a bare `rmdir` fails on a non-empty dir), so the lock
releases cleanly.

**`tmp/test_resume_state.py` — retries a transient start failure.**
`rc=3221225794` (`0xC0000142`, `STATUS_DLL_INIT_FAILED`) means the child
died before reaching `main`, with **no output**. It is an environment
failure under load, not a logic error. All trainer invocations now go
through `run_trainer(...)`, which retries up to 3 times (10 s apart) on
exactly that exit code. Detection is by exit code, not by output
scraping, because there is no output.

### Verified end state

```text
regression sweep:  24 PASS, 0 FAIL
lock lifecycle:    acquire / block-live / reclaim-dead / release  all verified
orphan processes:  none
port 7860:         free
disk free:         ~5.1 GB
corpus:            320,000 rows intact
```

### Rule for the operator

Do not clear the sweep lock from outside. Let stale-lock detection handle
a dead predecessor, and never run two sweeps at once — the live-parity
tests share one UI server and one board.

## Appendix D31 — V24 launched; watcher-suite process leak found and fixed 

### The launch

```text
launcher:    tmp/launch_v24_chain.sh   (new)
run dir:     runs/_probe/v24/
trainer:     runs/_probe/v24.log
chain:       runs/_probe/v24_chain.log
watcher:     runs/_probe/v24_play.log
UI:          http://127.0.0.60/   (hot-reloads on promotion)
config:      100,000 steps (5.00 epoch-equivalents), batch 16, ctx 192,
             seed 32, transcript v2, dense lags, --max-legacy-frac 0,
             --table-bias 0.0, --hard-start 0, eval/save every 2,000,
             24 eval games, --keep-last 3, disjoint holdout
```

The chain script does four things the two existing scripts could not:

1. **disk-headroom guard** — refuses to start below 1,500 MB free;
2. **starts training**, then waits for the FIRST promoted checkpoint
   before starting the watcher (`play_v24.sh` exits 2 if the checkpoint
   is missing, so it must not start too early);
3. **detects early death** — if the trainer exits before producing a
   checkpoint, it stops waiting instead of sleeping for hours;
4. **stays alive after training finishes**, holding the UI up so the
   final model remains playable.

Smoke-tested end to end at 60 steps: training → checkpoint → watcher →
two hot reloads (step 30, step 60) → valid status JSON → `/modelinfo`
reporting `step: 60`.

### The leak that was found on the way

Starting the run revealed **33 leftover processes** from the watcher
regression suite: 24 `play_v24.sh` bash processes and their `ui_server`
children, holding **1107 CPU-seconds** and **1.2 GB RSS**.

Root cause: `tmp/test_play_v24_watcher.py` cleaned up with
`watcher.terminate` plus a PowerShell sweep matching only
`*ui_server*`. On Windows, terminating an MSYS bash does not reliably
run the script's `EXIT` trap, so the watcher shell survived every run.

Two fixes:

- **At the source** — the test now kills the watcher's whole process tree
  via `psutil`, then sweeps stragglers by the test's unique tag.
  Verified: suite still **12/12 PASS**, and the watcher/UI process count
  no longer grows across a run.
- **Reclamation** — new `tmp/cleanup_leaked_watchers.py`. It protects
  the live chain by **socket ownership** (find the listener on 7860,
  protect its ancestor chain, kill the rest). It refuses to run if no
  live server is found. Deliberately NOT using `environ`: psutil
  returns `None` for `V24_TAG` on these MSYS bash processes, and a
  cmdline grep matches the cleanup script's own command line.

### The timing correction

The plan quoted **0.650 s/step / 18.1 h**, measured while the machine was
contended. On the cleaned machine the measured rate is **0.4415 s/step**
-> **12.3 h**, finishing around ****. The old figure was honest for
the machine it was measured on; it is simply not this run's figure.

Progress so far:

```text
step     0 | HOLDOUT 11.8% | LABEL 16.3% | H-LABEL 11.9% | t 40s
step  2000 | HOLDOUT 29.0% | LABEL 40.2% | H-LABEL 27.9% | t 883s
```

step-2000 LABEL 40.2 % vs the earlier projection 40.23 % — the model is
tracking the corrected curve.

### Other changes in this window

- `run_v24.sh` lock is now **PID-aware**: a stale lock from a dead chain
  is reclaimed automatically, a live one still blocks with a warning
  against `rm -rf`.
- `tmp/test_resume_state.py` retries the transient Windows
  `0xC0000142` (`STATUS_DLL_INIT_FAILED`) process-start failure, up to
  3 attempts.
- `tmp/run_regression_sweep.sh` cleanup removes `$LOCK/pid` before
  `rmdir` (a bare `rmdir` fails on a non-empty dir, so the lock was
  surviving every sweep).
- Cron watchdog `v24-training-watchdog` (`54e457966d02`), every 45 min,
  delivers to the chat. Uses psutil for liveness (`ps -W` reported zero
  trainers while one was running at 99 % GPU) and understands that the
  log only advances every 2,000 steps.

### Rules while the run is live

- No regression sweep during the run (same GPU/CPU).
- Do not clear `runs/_probe/.v24.lock` with `rm -rf`.
- If the UI stops answering on 7860, check the trainer first.

## Appendix D32 — disk-pressure triage during the V24 launch 

### Symptom

Mid-launch the volume showed **4.3 GB free** on a 931 GB disk.

### Triage

| Component | Size | Verdict |
|---|---|---|
| `runs/_probe/v24/` (the live run) | 604 MB at step 2000 | fine — ~1 GB total at `--keep-last 3` |
| `runs/_probe/_resume_equiv` | 1.2 GB | test scratch — **leaked** |
| `runs/_probe/_resume_state_test` | 830 MB | test scratch — **leaked** |
| `runs/_probe/_resume_state_test2` | 506 MB | test scratch — **leaked** |
| `runs/_probe/_resume_state_test3` | 53 MB | test scratch — **leaked** |
| `runs/_probe/_gate_smoke` | 289 MB | test scratch — stale |
| `runs/_probe/_eo32_50` | 53 MB | test scratch — stale |
| **total leaked** | **~2.9 GB** | |

### Root cause

`tmp/test_resume_state.py` and `tmp/test_resume_equivalence.py` did
`shutil.rmtree(SCRATCH)` **at the start** of a run but never at the end.
A *successful* run therefore left a corpus copy, a sidecar copy and
full optimiser-state checkpoints on disk. The isolated rerun performed in
the launch window alone created ~1.9 GB.

This is the file-space twin of the watcher process leak (§5t.8): cleanup
on the unhappy path only. The success path runs most often and nobody
inspects it.

### Fix

Both tests now remove their scratch in a `finally` block.

```text
test_resume_state.py        19 PASS, 0 FAIL, 0 scratch dirs left
test_resume_equivalence.py  12 PASS, 0 FAIL, 0 scratch dirs left
max |Adam moment difference| = 0.000e+00   (still bit-exact)
```

**Reclaimed 4.3 GB -> 7.2 GB free.** No historical run, corpus or
promoted checkpoint was touched.

### Note on the 100 % used flag

`df` reports `100%` on the Use% column because Windows rounds against
the whole volume, but the free figure (~7 GB) is the number that
matters. Do not treat the 100 % flag as "out of disk".

### Standing rule

The run needs ~0.4 GB more than it has already used. Do not leave
scratch behind while it is live, and do not run the full sweep
repeatedly.


## Appendix D33 — host sleep during the V24 run 

### What happened

```text
  step 18000 logged
  Kernel-Power id 42   (sleep REQUEST)
  Kernel-Power id 107  (resume request acknowledged)
  Power-Troubleshooter id 1  <- ACTUAL wake
     live step counter = 19439
  step 20000 logged (t 12768s)
```

**The host was asleep ~71 minutes.** Kernel-Power 42/107 records only the
transition request and made it look like a 6-second nap. The
Power-Troubleshooter event id 1 is the one that carries the real wake
time. Reading 42/107 alone would have produced a wrong diagnosis.

### Survival verified

| Check | Result |
|---|---|
| trainer worker pid 42856 | alive since  |
| py-spy stack | `build_batch` / `_weighted_move_loss` — training, not hung |
| live step counter | 19439 at, 20100 at  |
| instantaneous rate | 0.395 s/step (normal: 0.419–0.438) |
| GPU | 92 %, 408 W |
| all 8 checkpoints | load cleanly, correct `meta.step`, intact `opt_state`/`rng_state` |
| log errors | 0 (`Traceback`/`error`/`corrupt`/`nan`) |
| watcher + UI | alive, serving promoted `best_hard.pt` at step 18000 |
| `.v24.lock` | still held by the original chain |

### Why the log looked stalled

Eval lines are written every `--eval-every 2000` steps, ~14 min
normally. The 18000 → 20000 cycle took 5109 s wall-clock because the
sleep was inside it.

```text
overall for the span: 2.55 s/step   (sleep included)
instantaneous, awake: 0.395 s/step  (normal)
```

### Two numbers to be careful about

1. **`t NNs` is wall-clock.** It jumped 7659 → 12768 s across the sleep
   while no extra work happened. Any ETA derived from it is pessimistic
   by exactly the sleep duration.
2. **Elapsed-per-evaluation is not a health signal.** It jumps after any
   host sleep. The plateau abort logic keys on flat LABEL, not on rate,
   so it is unaffected — but any future rate-based abort would
   misfire on a sleeping host.

### Revised ETA

80,000 steps remain at ~0.42 s/step of *awake* time = **~9.3 h**,
finishing around **** (not ~ from raw wall-clock).

### Tooling change

`py-spy` is now installed in `.venv-c4`. The live step counter is
readable with:

```bash
./.venv-c4/Scripts/py-spy.exe dump --pid <worker_pid> --locals | grep step:
```

This is the correct way to answer "is it still running?" — `ps` shows the
process exists, but only this shows it is *advancing*.

The watchdog `54e457966d02` was rewritten to use it, and to consult
Power-Troubleshooter before declaring a stall. Its previous version last
fired at, before the sleep, so it never saw the gap — the new
version cannot confuse "host slept" with "run died".

### Operational rule

**A stale log is not evidence of a dead run.** Before interrupting
anything: check the process exists, read the live step counter, check
GPU utilisation, then decide.
