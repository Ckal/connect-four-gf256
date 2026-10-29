---
license: mit
library_name: pytorch
tags:
  - reinforcement-learning
  - game-ai
  - connect-four
  - gf256
  - integer-arithmetic
  - research
pipeline_tag: text-generation
---

# GF(256) Connect Four — Model Card

**Author:** Christof Ckal
**Project:** ZeroFLops
**Companion paper:** *GF(256) Lagrange Field Transformer — A Discrete
Algebraic Language Model Compiled to Zero-FLOP Integer Programs* (separate
work; shares the architecture, not the findings).

---

## Model Details

A GF(2⁸) integer neural model that plays Connect Four. The trunk operates
over GF(2⁸) using XOR, learned S-box table lookups and integer additions
— **no floating-point multiply-accumulate in the trunk**. The read-out
layer is float32.

| Field | Value |
|---|---|
| Checkpoint | `best_hard.pt` |
| Size | 33,830,949 bytes (32.3 MB) |
| Step | 18,000 |
| Transcript version | v2 (move bytes `0x41..0x47`) |
| Layers | 3 |
| `d` | 32 |
| `fan_in` | 32 |
| Lags | `[0,1,2,3,4,5,6,7,8,9,10,20]` |
| Context window | 192 |
| Training seed | 32 |
| Table bias | 0.0 |
| Parameters | 8,422,496 |
| Device | CPU (GPU not required for inference) |

Parameter breakdown:

| Component | Count |
|---|---:|
| Embedding | 8,192 |
| Coefficients | 288 |
| S-boxes | 24,576 |
| Wiring | 576 |
| Read-out | 8,388,608 |
| Bias | 256 |

**"Zero FLOPs" scoping:** the *trunk* has no floating-point MAC. The
read-out is float32 and is **99.6 %** of the parameters, so the **complete
model is not** a zero-FLOP program.

---

## Intended Use

- **Research:** studying the gap between teacher-agreement and game
  strength in a small, exactly-solvable game.
- **Demonstration:** a CPU-only, single-file game server.
- **Education:** a worked example of dense per-ply supervision from an
  exact solver.

**Not intended for:** production gameplay, competitive play, or any claim
of playing strength. See *Limitations*.

---

## Training

- **Corpus:** 320,000 complete Connect Four games, 5,100,175 boards
  scanned, 4,462,571 per-ply labels.
- **Supervision:** every ply labelled with the **set of optimal moves**
  from an exact solver. Ties are preserved as sets, not collapsed.
- **Objective:** cross-entropy against the optimal-move set.
- **Protocol:** seed 32, batch 16, context 192, `--hard-start 0`,
  `--table-bias 0.0`.
- **Holdout:** 4,000 games, **board-disjoint** from the training corpus.

The training corpus itself is **not** published — it is a private research
corpus.

---

## Evaluation Results

| Metric | Value |
|---|---|
| Teacher agreement, training corpus | **59.55 %** |
| Teacher agreement, board-disjoint holdout | **41.0 %** |
| Teacher agreement, holdout per-ply subset (n=512) | 41.8 % |
| Win rate vs. random (200-game x 3-seed matrix, n=600) | 85.2 % (511-88-1) |
| Win rate vs. 1-ply heuristic | **net-negative** |
| Exact-solver holdout coverage | 560 / 1,500 = **37.3 %** |

### The honest result

The model **agrees with the exact solver on ~59 % of training positions**
and **41.0 % of held-out positions**, beats a random opponent 85.2 % of
the time, and plays a genuine board-reading game — its picks spread across
all seven columns with non-zero per-column recall for every column.

It nevertheless **loses to a 1-ply heuristic** (win-now > block-now >
centre tiebreak) at **every measured step of the entire 100,000-step run**.
The best point is 23-25 (near parity, never ahead).

An independent in-game matrix (canonical `src.c4.duel` harness, 600 games
per arm per seed × 3 seeds, model greedy ε=0, opponent explores ε=0.3)
confirms it beats the best single-column constant policy in only **2 of 5**
opponent settings.

**Teacher-agreement is not game strength.** That dissociation is the
release's central finding.

---

## Limitations

1. **Net-negative against the 1-ply heuristic** — the headline negative
   result, reproduced by two independent methods.
2. **Teacher-agreement ≠ game strength** — the central finding.
3. **Partial solver coverage.** Only 560 of 1,500 holdout positions have a
   proved label; the other 940 are **excluded**, not guessed. All holdout
   numbers are computed over the proved subset.
4. **Train/holdout gap** of ~18 pp indicates a memorisation component.
5. **Plateau.** Teacher agreement stopped improving after ~step 14,000.
6. **Architecture-level superiority is unproven.** Controlled A/B arms
   reached only 0.30 epoch-equivalents and settle nothing.
7. **Mirror/symmetry questions remain open** for the corrected model;
   earlier historical conclusions predate a critical repair and are not
   evidence about this checkpoint.
8. **This is a separate work** from the companion language-modelling paper.
   Results do not transfer between them.

---

## Provenance

Every published number traces to a command and an artifact in the release
record shipped with the repository. Key points:

- The checkpoint stores `step` at the **top level** (not under `meta`).
- Scoring requires the **owner corpus**; the scorer refuses to guess and
  exits non-zero when the corpus is ambiguous.
- The regression suite is **24 PASS, 0 FAIL**, reproduced by two
  independent sweeps.
- Resume is **bit-exact** (max weight Δ 0.000e+00, identical Adam
  moments).

---

## Ethical / Usage Notes

This is a research artifact about imitation learning, not a game-playing
agent intended to compete. No personal data is involved. The checkpoint
contains only learned model weights over Connect Four board states.

---

## License

MIT. Paper text: © 2026 Christof Kaller.
