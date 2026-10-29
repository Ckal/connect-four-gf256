# Follow-up Research Plan

**Project:** ZeroFLops — GF(256) Connect Four
**Author:** Christof Ckal
**Opened:** 2026-10-29

This plan starts from the **measured** state of the project, not from the
original ambition. The central measurement is:

> The model agrees with the exact solver on ~59 % of training positions
> and 41.0 % of held-out positions, beats random 85.2 %, plays a genuine
> board-reading game — and **still loses to a 1-ply heuristic** at every
> point in the run.

Everything below is aimed at explaining and fixing that gap.

---

## 0. The one-sentence thesis

> *Teacher-agreement is not game strength.* The project now has the
> instrumentation to measure both, on the same model, with an exact
> teacher. The next phase explains where the disagreement comes from and
> whether a different supervision target closes it.

This is a stronger contribution than a higher accuracy number, because it
is falsifiable, measurable, and reusable.

---

## Phase 1 — Diagnose the gap (highest value, no new training)

**Goal:** locate *which* decisions cost games.

### 1.1 Decision regret under real opponent responses

The current metric asks "does the model match the teacher's label on this
position?". The right question is "did the model's move cost us the win,
*given how the game actually continued*?".

Implement **per-ply decision regret**: at each ply, compare the model's
move against the best available move under the exact solver's evaluation
of the *actual* resulting position, weighted by the probability the game
reaches a win from each branch.

- Artifact: `src/c4/regret.py`, metric `regret_per_ply`.
- Test: on a position where the model picks a losing move, regret > 0 and
  the sign convention is verified against a hand-built position.

### 1.2 Classify the error modes

Bucket every disagreement into: *missed immediate win*, *missed block*,
*bad centre/edge timing*, *long-horizon threat*, *other*. Count them.

**Hypothesis to test (not an assumption):** most of the deficit is
**missed blocks and missed wins in positions the teacher never labelled
in training**, i.e. coverage rather than capacity. If the histogram is
flat, the hypothesis is wrong and the target changes.

### 1.3 Why does it lose to a 1-ply rule?

The heuristic wins by two things: it never leaves an immediate win on the
board, and it never leaves the opponent one. Measure whether the model
violates either. This is a cheap, decisive diagnostic.

**Deliverable:** `docs/STRENGTH_GAP_DIAGNOSIS.md` — a breakdown of where
the ~40 % of games are lost, with the error histogram.

---

## Phase 2 — Better supervision

**Goal:** make the target express what the game actually rewards.

### 2.1 Search-based distillation

Instead of imitating the teacher's single best move, distil the teacher's
**value function** (win probability under optimal play). Convert the value
into a policy via policy-gradient-style or advantage-weighted regression.
This is the standard correction for exactly this failure mode: per-ply
imitation teaches *what the teacher would do*, not *what wins*.

- Compare: per-ply label CE (current) vs. advantage-weighted regression
  vs. pure value regression.
- Metric: **both** teacher agreement *and* ladder strength. A change that
  improves one and not the other is itself a finding.

### 2.2 Adversarial self-play data

Generate training positions from the model's *own* games, not only from the
solver's distribution. The solver's labels are only informative on
positions the solver visits; the model visits different ones. Label
model-reached positions with the exact solver and add them to training.

### 2.3 Explicit threat features

Test whether adding a small number of hand-crafted threat features — or
simply *more context* — closes the gap. Cheap experiment, and a useful
control: if a 1-ply rule's information is enough to win, a model with
access to it should learn it.

**Controlled requirement:** every Phase-2 arm must
(a) use the same corpus, seed 32, batch 16, context 192;
(b) train to a **comparable budget** (the current corrected A/B arms ran
only 0.30 epoch-equivalents — that is why they are inconclusive);
(c) be evaluated with the same ladder + in-game matrix;
(d) be scored on the **board-disjoint holdout**.

---

## Phase 3 — Close the engineering gaps

### 3.1 Full solver coverage

Holdout coverage is **560 / 1,500 = 37.3 %**. The remaining 940 positions
need exact labels. This is a compute task (exact Connect Four solving over
a large state space) and it is worth doing because it is the main limit on
how precise the holdout claims can be.

### 3.2 Streaming full-corpus readout refit

The full-corpus readout refit died from **OOM**. Implement it as a
streaming pass over shards so it can run on the 320,000-row corpus. This
is what would let the read-out be optimised against the whole dataset
rather than a sample.

### 3.3 Rebuild the corrected split

`tools/build_full_split.py` has not been rerun since the fixes. Rerun it,
then generate and re-verify a **full mirror artifact** with the corrected
code path.

### 3.4 Rebuild the sidecar with tie preservation

`tools/relabel_perply.py` collapses ties (the strong-teacher `src=2`
path). The correct relabeller is `tools/relabel_v23.py`, which preserves
tie sets. Rebuild the production sidecar with it and re-measure.

---

## Phase 4 — Open questions the A/B never settled

These were historically "answered" before the row-index repair, so the
answers are void. Each needs a proper controlled run:

1. **Dense per-ply vs. single-label supervision.** The corrected arms
   (`fx_a`/`fx_b`/`fx_c`) reached only 0.30 epoch-equivalents and settle
   nothing. Needs a budget-matched rerun.
2. **Mirror-consistency objective.** Was measured only pre-repair.
3. **Symmetry equivariance.** Was measured only pre-repair.
4. **Pons-style graded targets.** Never trained at all.
5. **LR decay and EMA.** Both implemented and opt-in, but never given a
   controlled numerical pilot.

**Measurement discipline for all of these:** the mirrored corpus, the
mirror-consistent labels, and the mirror-consistent *behaviour* are three
different things and must be measured separately. A mirror partner and its
mirror must share one canonical solve. Self-symmetric boards are their own
mirror, so for them reflected `c → 6−c` consistency is vacuous unless the
selected move is c3.

---

## Phase 5 — Make the read-out integer (the real zero-FLOP claim)

The read-out is **8,388,608 of 8,422,496 parameters — 99.6 %**. Until it
is an integer program, "zero FLOPs" is a claim about 0.4 % of the model.

Options:
- quantised read-out with an integer accumulation path,
- a small GF(256) read-out with more channels instead of more width.

**Success criterion:** the complete model's inference executes with no
floating-point multiply-accumulate, and the language-model paper's BPC
result is re-verified under the new read-out.

This is the piece that would make the *architectural* claim whole, in
both papers.

---

## Phase 6 — Release engineering

Ordered by what unblocks the most:

| Priority | Item | Blocker |
|---:|---|---|
| 1 | **Multi-user UI** (per-session boards, bounded store, rate limits) | blocks public deployment |
| 2 | **Public GitHub repo** under `Ckal` | the current history has a 118 MB blob that GitHub rejects; use a fresh repo |
| 3 | **HF namespace + model card** | the `ckal` HF handle is a different person; needs a new account |
| 4 | **Public deploy** on Netcup | depends on 1 |
| 5 | **Video** | a scripted deterministic match, not a training montage |
| 6 | **Follow-on A/B battery** | depends on 2 and 3 (need CI) |

### 6.1 Multi-user UI — concrete requirements

Replace the module-global `_GAME` with:

1. a **per-session store** keyed by an opaque, unguessable id (HttpOnly
   cookie / signed token);
2. a **bounded store** with TTL and eviction;
3. **stateless inference** — the checkpoint is read-only and shared; only
   the board is per-session. The model is a pure function of
   (checkpoint, board), so no session state can leak into it;
4. **rate limiting** and request-size limits;
5. tests: two concurrent sessions must not see each other's boards; a
   session id must not be guessable; the store must evict under load.

### 6.2 Video — what it should show

A **deterministic scripted match**, rendered from real game records:

- the model against a fixed opponent, showing the next-move distribution
  updating each ply;
- the board and the distribution side by side;
- a caption that states the honest result — *beats random, loses to a
  1-ply heuristic* — rather than a highlight reel that implies strength.

---

## Explicit non-goals

- **No architecture-level superiority claims** until a budget-matched A/B
  exists.
- **No reuse of pre-repair conclusions** in any public artifact.
- **No public deployment** of the single-board UI.
- **No upload** of the private corpus, historical runs, or logs.
- **No headline numbers** for the 100,000-step run until it completes and
  the release battery passes.

---

## Success criteria for this phase

| Criterion | Threshold |
|---|---|
| Teacher agreement | ≥ 75 % holdout |
| Ladder vs. 1-ply heuristic | **≥ parity (net ≥ 0)** — the real milestone |
| Ladder vs. 2-ply lookahead | measurable, reported either way |
| In-game matrix vs. best constant | wins in ≥ 4 of 5 settings |
| Holdout solver coverage | ≥ 95 % |
| Ablation validity | ≥ 1 budget-matched A/B, seed 32, identical corpus |

The **first** row is a nice-to-have. The **second** is the project.
