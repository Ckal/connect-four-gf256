# Reading the Board: A GF(256) Integer Model for Connect Four

### And a measurement that shows teacher-agreement is not game strength

**Author:** Christof Ckal
**Project:** ZeroFLops
**Code:** https://github.com/Ckal/connect-four-gf256
**Artifact:** `best_hard.pt`, step 18000, 32.3 MB, 8,422,496 parameters

---

## Abstract

We apply a GF(2⁸)-based integer neural architecture, originally developed
for character-level language modelling, to Connect Four — a game whose state
is a 6×7 bit board with no hand-designed "how to play" rules. Training uses
dense per-ply supervision from a negamax alpha-beta teacher, with every ply of every game
labelled by the teacher's chosen move.

On a 320,000-game corpus the model reaches **59.6 %** agreement with the
solver's optimal-move labels on the training distribution and **41.0 %** on
a board-disjoint holdout, up from 16.3 % / 11.9 % at initialisation. It
beats a random opponent decisively (**85.2 %** win rate, n=600) and plays a
recognisable, non-degenerate game.

However, we report a **negative strength result**: across the whole of the
100,000-step training run (measured to step 46,000 at the time of
writing) the model remains **net-negative against a 1-ply
heuristic** (win > block opponent's immediate win > centre tiebreak), never
reaching parity. An independent in-game matrix over five opponent families
(600 games per arm, 3 seeds) confirms it: the model wins only **37.2 %**
against the 1-ply heuristic and **24.3 %** against 2-ply lookahead. The
model *does* beat a centre-column policy head-to-head (90.5 %), but that
policy scores higher than the model against a random opponent (91.0 % vs.
85.2 %, p = 0.00236), so the model's overall profile is the weaker one.

We argue this dissociation is the paper's principal contribution.
**Agreement with an expert teacher is not a proxy for game strength.** A
model can learn to reproduce a teacher's labels on the positions it has
seen while still failing to exploit structure the teacher relies on but the
supervision does not penalise. We release the full measurement apparatus
so the negative result is checkable rather than anecdotal.

---

## 1. Motivation

Connect Four is a small, fully-observed, deterministic, perfect-information
game. It is small enough to solve exactly. That makes it an unusually clean
testbed for a question that is hard to study in larger games:

> When you train a small model to imitate a strong search player, how much of
> the player's *strength* do you actually recover?

This question matters beyond Connect Four. Supervised imitation of a
teacher is the standard recipe behind a great deal of learned behaviour,
and "our loss went down" is routinely reported as if it implied "our
behaviour got better". Connect Four lets us measure both quantities
separately and exactly, on the same positions, with the same model.

The architectural substrate is the GF(256) Lagrange Field Transformer
(LFT), described in full in the companion language-modelling paper
*GF(256) Lagrange Field Transformer — A Discrete Algebraic Language Model
Compiled to Zero-FLOP Integer Programs*. We restate only what the game
setting requires.

---

## 2. Architecture

### 2.1 The GF(2⁸) trunk

Each layer operates over GF(2⁸) = GF(2)[x]/(x⁸+x⁴+x³+x+1). The forward
pass of the trunk uses:

- **XOR** for the linear mixing step,
- **table lookups** for the learned S-box non-linearity,
- **integer additions** for the read-out accumulation.

The S-box is learned, not fixed. Training proceeds in a continuous "soft
field" — probability distributions over field elements — and the hard
integer program is obtained by straight-through estimation.

### 2.2 What "zero FLOPs" means here, precisely

The **trunk** performs no floating-point multiply-accumulate. The
read-out layer is float32. So the honest statement is:

> The trunk is an integer program. The complete model is not.

A complete model is 8,422,496 parameters: embedding 8,192; coefficients
288; S-boxes 24,576; wiring 576; read-out 8,388,608; bias 256. The
read-out is **99.6 %** of the parameters.

### 2.3 Game-specific configuration

| Parameter | Value |
|---|---|
| Layers | 3 |
| `d` | 32 |
| `fan_in` | 32 |
| Lags | `[0,1,2,3,4,5,6,7,8,9,10,20]` |
| Context window | 192 |
| Batch size | 16 |
| Seed | 32 |
| Transcript version | v2 |
| Table bias | 0.0 (new runs) |

---

## 3. Data

### 3.1 Corpus

**320,000 games**, each a complete Connect Four transcript. Verified:
**5,100,175 boards** scanned, **4,462,571 per-ply labels** aligned.

Each ply record is `[9 header bytes][1 move byte]`. The 9 header bytes
describe the board state *before* that ply's move. Move bytes are the raw
column value `0x41 + column`, i.e. `0x41..0x47` for columns 0..6 — this
is **transcript v2**.

### 3.2 Dense per-ply supervision

Every ply is labelled with the **set of teacher moves** under a negamax
solver, not a single move and not an outcome. Where multiple moves are
equally optimal, the label is the full set and the loss treats them as a
tie.

This matters. The teacher is the search policy; it is never replaced by a
weaker policy, and no fallback silently substitutes a different teacher.

### 3.3 Board-disjoint holdout

A holdout corpus of **4,000 games** whose canonical initial board state
and board-set are disjoint from the training corpus. Scored separately so
that memorisation is visible rather than hidden.

### 3.4 Partial solver coverage — a stated limitation

Exact-solver coverage of the holdout is **560 / 1,500 = 37.3 %**. The
remaining **940** positions have no proved label and are **excluded** from
scoring rather than being guessed. All reported holdout numbers are
therefore computed over the proved subset and are labelled as such.

---

## 4. Training

### 4.1 Objective

Cross-entropy against the optimal-move set, with ties handled as a
distribution over the tied set rather than a single arbitrary member.

### 4.2 Defects found and repaired before this result was meaningful

Reporting these is part of the result. Each was a genuine bug that
produced a *plausible-looking* number:

1. **Terminal-state detection.** The board's terminal check returned a
   truthy value on terminal boards instead of "no legal action remains".
   Fixed to `return not self.valid_actions()`.

2. **Corrupt transcript field.** The final field of **126,537 / 320,000
   rows (39.5 %)** was corrupt, affecting **1,567,958** plies. Board, move
   and turn bytes were intact; only the trailing field was damaged. A
   dedicated repair tool fixed **only** the damaged bytes and verified the
   result (`REPAIR_VERIFIED`). The trainer now scans the whole corpus
   before training and **rejects corrupt input**.

3. **Sidecar row indexing — the most consequential bug.** The trainer
   looked up per-ply labels by **batch position** rather than by the
   **sampled corpus row**. The sidecar was therefore read almost entirely
   at the wrong rows: roughly **41.3 %** of supervised positions had
   **all-zero targets**, and the rest were misaligned. Fixed to derive
   rows from the sampled corpus indices. A regression test was written
   that **fails against the old behaviour** and passes against the fix.

4. **Promotion metric degeneracy.** The checkpoint-selection key was
   lowest-byte-bpb, which selected **step 1500** over the final step 3923.
   Replaced with **direct label agreement** as the primary key, with
   byte-bpb retained only as a tie-breaker. The new metric discriminates:
   **92.4 %** on the final checkpoint vs **12.7 %** at step 0.

5. **Checkpoint metadata.** Final exports omitted the transcript version,
   causing a checkpoint to be read as v1 when it was v2 — and to score
   **24.4 %** instead of its true **91.3 %**. Fixed; historical exports
   re-stamped.

6. **Per-ply index arithmetic (historical).** An earlier implementation
   contained a second indexing defect in which early positions `t < k` were
   assigned to row `k−1`. The current shipped mapping is verified correct
   on the production invariant: **6,504 / 6,504** positions decode
   consistently.

> **Consequence for the literature record:** historical conclusions in this
> project — "dense per-ply supervision is worse", "mirror consistency is
> net negative", "symmetry is not worth it" — were all drawn **before**
> the row-index repair. They are **invalid** as evidence about the
> corrected model and are not cited here.

### 4.3 Protocol

Seed **32**; `--hard-start 0`; `--table-bias 0.0`. Bit-exact resume is
supported and verified: resuming from a float checkpoint reproduces
subsequent steps with **maximum weight difference 0.000e+00** and
**identical Adam moments** across **12/12** regression checks.

---

## 5. Results

### 5.1 The convergence curve

Holdout of 4,000 games; holdout per-ply subset of 512 positions; 48-game
ladder against random and heuristic.

| Step | LABEL % (train) | H-LABEL % (holdout) | vs heuristic W-L | vs random W-L |
|---:|---:|---:|---:|---:|
| 0 | 16.3 | 11.9 | 9-39 | 36-12 |
| 2000 | 40.2 | 27.9 | 10-38 | 37-11 |
| 6000 | 50.9 | 36.7 | 15-33 | 38-10 |
| 10000 | 56.0 | 40.8 | 19-29 | 37-11 |
| 14000 | 57.5 | 38.1 | 23-25 | 44-4 |
| 18000 | **59.6** | 41.0 | 16-32 | 41-7 |
| 20000 | 59.0 | 38.9 | 15-32 | 37-11 |
| 26000 | 58.7 | 39.1 | 15-33 | 38-10 |
| 30000 | 59.4 | 42.0 | 18-30 | 38-10 |
| 34000 | 58.5 | 41.8 | 14-34 | 43-5 |

### 5.2 What improves, and what does not

**Improves:**

- Label agreement: **16.3 % → ~59 %** (train), **11.9 % → ~41 %**
  (holdout). A large, unambiguous gain.
- Win rate vs. random: **85.2 %** (511-88-1 of 600), stable.
- Win rate vs. the centre-column policy, head-to-head: **90.5 %**
  (543-57-0 of 600) — the model wins that match. However, that policy
  itself scores **91.0 %** against random (546-54), above the model's
  85.2 %, so the model's overall profile against random opposition is
  the weaker one (Fisher exact p = 0.00236).
- The model demonstrably **reads the board**. Its move distribution is
  spread across all seven columns (holdout picks
  `[39, 55, 94, 175, 78, 48, 23]`), it is not a degenerate constant
  policy. Per-column recall is non-zero for every column.

**Does not improve:**

- **Strength versus the 1-ply heuristic.** The model is net-negative at
  every measured step of the entire run. The best point (step 14000) is
  23-25 — close to parity, but never ahead.
- **Label agreement after ~step 14000.** The curve plateaus at 57–60 %
  train / 38–42 % holdout. Training length is no longer the bottleneck.
- **The train/holdout gap** (~59 % vs ~41 %) indicates a substantial
  memorisation component in the headline number.

### 5.3 The dissociation

The model at step 18000 agrees with the teacher's move on ~59 % of
training positions and 41.0 % of held-out positions, beats random 85.2 %
of the time, plays a genuine board-reading game — and **still loses to a
rule that only checks "can I win now" and "can I block now"**.

We read this as: the supervision signal rewards reproducing the teacher's
choice on *the labelled distribution*, but a large part of the teacher's
advantage comes from **structurally different** behaviour that dense
per-ply imitation does not express — for example, threat creation and
threat denial over horizons longer than one ply, and the timing of when
to abandon the centre.

### 5.4 Independent strength measurement

A separate in-game matrix over five opponent families (random, centre,
heuristic, 2-ply lookahead, minimax), using the canonical `src.c4.duel`
harness, 600 games per arm per seed across 3 seeds, model greedy
(ε = 0) while the opponent explores (ε = 0.3) so that trajectories are
genuinely distinct:

Measured results for the released checkpoint (step 18000), 200 games per
arm per seed × 3 seeds = **600 games per arm**:

| Opponent family | Model W-L-D | Win % | Best constant in the same block | Fisher p (model vs that constant) |
|---|---:|---:|---|---:|
| random | 511-88-1 | **85.2 %** | centre_policy 546-54 (**91.0 %**) | 0.00236 |
| heuristic (1-ply) | 223-376-1 | **37.2 %** | always-c0 159-441 | 7.2 × 10⁻⁵ |
| lookahead (2-ply) | 146-452-2 | **24.3 %** | centre_policy 97-503 | 4.2 × 10⁻⁴ |
| minimax | 251-343-6 | **41.8 %** | centre_policy 235-363 | 0.316 |
| centre (head-to-head vs centre_policy) | **543-57-0** | **90.5 %** | always-c0 470-130 | 6.9 × 10⁻⁹ |

Two findings, and they point in opposite directions, so both are reported:

1. **The model wins the direct match against the centre-column policy**
   (543-57-0, 90.5 % of 600 games). `centre_policy` is deterministic — it
   always plays column 3, then the leftmost legal column — so it is
   exploitable.
2. **But `centre_policy` is the stronger player overall.** Measured against
   a random opponent it scores **91.0 %** (546-54) versus the model's
   **85.2 %** (511-88), a significant difference (Fisher exact
   p = 0.00236). A deterministic policy converts more wins against a random
   opponent than a greedy model does, because a random opponent rarely
   occupies the centre.

The model is also **net negative against the 1-ply heuristic** (37.2 %) and
against 2-ply lookahead (24.3 %). Overall the model is a real but weak
player: it reads the board, it beats random decisively, and it loses to
three-line rules.

Seats were checked separately: model gap **+7.5 pp**, heuristic gap
**+4.8 pp**, gap-versus-gap **p = 0.402** — no compelling model-specific
seat asymmetry.

### 5.5 Head-to-head between promoted checkpoints

Promoted step 18000 vs. pinned step 20000, 400 games:
**W208 L192 (52.0 %)**, one-sided **p = 0.45**, **82.5 %** same-column
agreement, first divergence at ply 16. Statistically inconclusive;
step 18000 is released because its authoritative label agreement is
higher (59.55 % vs. 59.00 %).

---

## 6. Threats to validity

1. **The strength deficit is the headline negative result, not a
   measurement artefact.** It is reproduced by two independent methods
   (48-game ladder inside the trainer; 600-game-per-arm in-game matrix),
   and it holds across the whole run.
2. **Partial solver coverage (37.3 %)** limits holdout claims to the proved
   subset.
3. **Holdout size.** 512 positions for the holdout per-ply metric gives a
   standard error of roughly ±2.2 pp — enough to see the plateau, not
   enough to resolve 1-pp differences.
4. **Architecture-level conclusions are not established.** The controlled
   A/B arms comparing per-ply against single-label supervision reached only
   **0.30 epoch-equivalents** and are provisional. They are not used here
   to support any claim.
5. **The final 100,000-step result was not available when this was
   written.** The run is live; all numbers here are from a completed
   intermediate evaluation and are traceable in the release record.

---

## 7. Reproducibility

Training and evaluation are deterministic under seed 32, with bit-exact
resume. The full regression suite — **24 PASS, 0 FAIL** — covers live UI
parity, checkpoint atomicity, transcript stamping, resume equivalence, UI
hot reload, single-server binding, and the watcher.

The release record accompanying this paper lists, for every published
number, the exact command that produced it and the artifact it came from.

---

## 8. What we would do next

1. **Diagnose the strength gap directly** rather than through the label
   proxy: measure per-ply *decision regret* under the opponent's actual
   responses, not teacher agreement.
2. **Train against adversarial search** rather than static labels, so the
   supervision reflects positions the model actually reaches.
3. **Search-based distillation** from the negamax teacher, which converts a
   value function into a policy far more efficiently than per-ply
   imitation.
4. **Extend the architecture** so the read-out is also an integer program —
   the read-out is 99.6 % of the parameters and the only reason the
   complete model is not a zero-FLOP program.

---

## Appendix A — Artifact

| Field | Value |
|---|---|
| File | `best_hard.pt` |
| Size | 33,830,949 bytes |
| Step | 18000 |
| Transcript version | v2 |
| Lags | `[0,1,2,3,4,5,6,7,8,9,10,20]` |
| Parameters | 8,422,496 |
| Authoritative label agreement | 59.55 % |
| Corpus | 320,000 games / 4,462,571 per-ply labels |

## Appendix B — Provenance

Every measurement in this paper is traceable to `RESULTS.md`,
which records the command, artifact path, provenance anchor and scope limit
for each number.

## Appendix C — Relationship to the language-modelling paper

This is a **separate work**. It shares the GF(256) LFT architecture with
*GF(256) Lagrange Field Transformer — A Discrete Algebraic Language Model
Compiled to Zero-FLOP Integer Programs* (Hard BPC 3.071 on Tiny
Shakespeare), but its contribution, its data, its supervision, and its
central finding are different. Nothing in the language-modelling paper's
results should be read as evidence about Connect Four, and vice versa.
