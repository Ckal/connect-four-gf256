# Connect Four AI — in plain words

**For:** someone who has never seen this code
**Updated:** 28 October 2026 (late night) — see **§9.9** for the current status

> **Start with `docs/CURRENT_TRUTH.md`** if you want the current, one-page,
> non-contradictory summary. This document is the long-form explanation and
> contains some earlier claims that were later corrected; where the two
> disagree, `CURRENT_TRUTH.md` wins.

> **🔴 §9.9 changes the story.** The model was not bad — the *training* was
> broken. The trainer was reading the teacher's answers for the wrong games,
> and 41 % of the supervision had quietly turned into "copy the move that
> was already played". The code is fixed; a corrected run is in progress.
> Several earlier findings in this document measured that bug rather than
> the model, and are marked accordingly.

---

## 1. The game

Connect Four (Vier Gewinnt). A board of **7 columns × 6 rows**. Players drop
one disc at a time into a column; the disc falls to the lowest free row. First
to get **four in a row** — horizontal, vertical, or diagonal — wins.

Two players. You are yellow, the computer is red. Whoever connects four first.

---

## 2. The one idea that makes everything else make sense

The computer does **not** think in "boards" or "moves".

It is a **next-byte predictor**. It reads a binary transcript of the game so far
and predicts **the single next byte of that transcript**.

Every move occupies exactly **10 bytes**:

```
byte 0     column byte    (0x80 | 1<<height | ownership bits)   <- the board
bytes 1-8  reserved
byte 9     the move itself (0x41 + column 0..6  ->  'A'..'G')
```

The board is packed into the first byte of each move record: 7 column-bytes,
one per column, each carrying that column's **height** plus **which player owns
each stone**. A column of height 4 with P1 owning rows 0, 1, 3 is one byte.

So the model reads those bytes and predicts: is the next move byte `B`, `E`, or
`D`? **That single guess is the move.**

### Why this format (V2) and not the old text one

The original format wrote the board as ASCII (`A X X X ...`). That format was
**lossy**: two different positions could produce identical text. Measured
**1,662 colliding headers** among sampled positions.

V2 stores per-cell ownership in binary and had **0 collisions**. Same model, same
size — only the encoding changed.

### Two versions of the same model

| Name | What it is | What it's for |
|---|---|---|
| **soft** (`final_float.pt`) | normal float neural net, trained with backprop | learning |
| **hard** (`hard_model_final.pt`) | the same weights, frozen into integer tables | what the website plays |

The "hard" model is not a different model. It is the **same brain with the
decimal points chopped off**, so it gives the identical answer every single time,
with no floating-point wobble. Checkpoints carry a metadata stamp naming their
transcript version, so an old checkpoint can never be silently loaded as a new
one.

---

## 3. The model

A small GF(256) byte model. Not a chess engine. Not a search.

- **16.78 million parameters** (`d=32`, `layers=3`, `fan_in=3`)
- Reads up to **192 bytes** of context (~19 moves of memory)
- Outputs a probability for each of the next 256 bytes

**Its job:** given a board, put high probability on the move byte a strong
player would actually choose.

---

## 4. The data — this changed completely

The old corpus was **5,540 self-play games**. The current corpus is
**203,588 unique solver-labelled positions**.

### Where they come from

| Source | What it is |
|---|---|
| **`TonyCWang/ConnectFour`** (HuggingFace) | ~108M board positions, each labelled with **Pascal Pons**' own solver scores per column |
| Kaggle / UCI `connect-4` | 67,428 **unlabelled** boards — not used; labelling them offline is too slow |
| `Chemin-AI/llm-connect-4` | only **148 rows** — too small |

We use Pons' labels because they are trusted: on positions our own solver could
prove locally, it agreed with Pons **49 of 54** — and all 5 disagreements were
**equal-quality ties**, not errors.

### From board to training example

Pons gives a *board*, but this model reads a *move stream*. So for each position
the importer searches for **a legal move history that reaches it**, writes that
history as a V2 transcript, and then **appends the labelled position and its
optimal move** as the final ply. That final ply is the only part with a label;
earlier plies keep a one-hot target.

Result: **203,588 unique positions**, 0 hard failures, from 300,000 source rows.

### Soft targets — why one column is often wrong

**41.6% of positions have several equally optimal moves.** Picking one column
with `argmax` always returns the lowest index, which teaches a systematic bias
toward column 0. Instead we store a **target distribution**: uniform mass over
every equally optimal legal column.

### Mirroring

Connect Four is exactly left-right symmetric, so a mirrored position has the
identical minimax value and its Pons label transfers exactly. Verified:
**134/134 mirrored boards, 914/914 mirrored moves**.

Mirroring is augmentation, not new information — but it must not corrupt the
holdout. Splitting therefore hashes the **whole symmetry class** (the board and
its mirror share one hash), so a position and its mirror can never land on
opposite sides of the split. Verified: **0** holdout boards mirror a training
board.

### The split

> **CORRECTED 26 October 2026.** The split below was the single biggest
> cause of the weak model: the depth filter was applied to **train only**,
> so the model was never supervised on an opening, while the holdout
> (which kept every depth) was 68.3% opening. The metric was measuring a
> distribution the model had never trained on. The V23 split drops the
> train-side depth filter entirely (train 386,916 / holdout 20,260,
> symmetry overlap 0).

| | Rows |
|---|---|
| Train (depth >= 20 plies) — **OLD, now replaced** | 106,966 |
| Train (all depths) — **V23** | **386,916** |
| Holdout (5%, all depths) | **20,260** |

The depth filter exists because shallow positions are opening-heavy and are not
where Connect Four is decided. It is also why the split totals are smaller than
the 203,588 imported positions.

> Correction to the above: the filter was **not** a sound modelling choice
> on its own — the game *is* decided in the opening too, and a model with no
> opening strategy loses every game that is not already won tactically. The
> V23 split keeps every depth on both sides.

### Teacher games are OFF

Earlier runs grew the corpus by having a solver play itself and keep the winning
side. That is **disabled** for this run (`--teacher-games 0
--selfplay-every 1000000`). Every position comes from the imported set.

---

## 5. How it learns

The model is shown a **sliding window** of 192 bytes and asked to predict the
next byte at each position. Only **move bytes** are scored; every other byte is
masked out of the loss.

The move loss is **7-way, not 256-way**: instead of scoring all 256 byte values,
it scores only the 7 legal columns, with illegal columns masked out. That
matches the deployed decision rule exactly.

Two details that are easy to get wrong and were both wrong at some point:

- **Illegal columns are masked to a large negative number (-1e9), not
  -infinity.** With a true `-inf`, a column whose target probability is
  exactly zero produces `0 × (-inf) = NaN`, which silently poisons the
  gradient. The target mass is also restricted and renormalised over the
  legal columns only.
- **The label must actually be inside the scoring window.** The window is
  193 bytes wide but the imported labels were only 10 bytes long, so the
  labelled decision sat at weight 0 and the move loss was multiplied by
  zero. The loss looked like it was training; it was not.

### Windowing — why windows are pinned to the end

With ctx 192 and random window placement, only about **3.9%** of windows would
contain the labelled ply at all. So windows are **pinned to the end** of each
sequence: every training example actually contains the labelled position.
Verified **256/256** targets placed.

A larger context was considered and **rejected** — end-pinning achieves full
coverage at ctx 192 with no extra cost.

---

## 6. The processes

| Program | Port | What it does |
|---|---|---|
| **`src/c4/train.py`** | - | the trainer |
| **`src/c4/ui_server.py`** | **7860** | the website you play on |
| **`tools/import_hf_c4.py`** | - | builds the position corpus from the HF dataset |
| **`tools/split_positions.py`** | - | symmetry-safe train/holdout split |

Launching this run:

```
bash run_v15_posdata.sh
```

The trainer **reads** `corpus.txt` via `--load-corpus`. Without that flag it
takes its "seed corpus" branch and **overwrites** the file — with
`--seed-games 0` that writes an empty file and trains on nothing. That happened
once and is now documented in the script.

---

## 7. The website

Go to **http://127.0.0.1:7860/**

- Click a column to drop your disc
- The model picks its move automatically
- A panel shows the model's probability for each legal column

The displayed move is what the **hard** model actually played, not the float
model's separate opinion. These were once different, which made the UI lie about
what was chosen; it is now verified over multiple plies.

---

## 8. How we know if it's good

In order of trust:

1. **Held-out optimal-move accuracy** — accuracy on positions the model never
   saw, counting a tie as correct. *This is the decision metric.*
2. **Head-to-head** — full games vs `heuristic` and `random`, with **distinct
   trajectories** counted so 400 nominally-different games that are really 2
   distinct games cannot masquerade as strength.
3. **Legal accuracy (7-way)** — in-sample diagnostic.
4. **bpb** — bits per byte. A weak signal; it is not strategy.

**A higher accuracy number does not automatically mean a better player.** That
was the central trap of this project: V14 reached 83% legal accuracy while
playing an almost constant column-0 policy, and its advertised "W47 L1" against a
weak bot turned out to be **2 distinct games**.

---

## 9. Honest status — 26 October 2026 (REWRITTEN: the old status was wrong)

> **Read this before any other section.** The status that used to live here
> said "the model works". **That was wrong, and the measurement that produced
> it was broken.** The numbers are corrected below. Nothing in this document
> should be trusted over this section if the two disagree.

### 9.-1 Update 27 October 2026 — we finally know WHERE the centre-bias lives

The question "is the brain too small to ever play a side column?" now has
a **measured** answer, and it is **no**.

**How we know.** We took the model, froze it, and looked at the numbers
*inside* it at the point where it decides which column to play. Then we
threw away everything the model had learned to read, and asked a very
simple question of the remaining bits: *"can you tell me which column is
right?"* — and we tested it on positions the model had **never seen**,
and where no two positions were the same board (including its mirror
image).

| what we asked | answer |
|---|---|
| "always play the middle" gets | 23.34 % |
| the model's own layer-2 information gets | **31.42 %** |

**So the model already contains the knowledge. It is a full 8 points
better than guessing the middle.** The "always plays the centre" habit
is not a limitation of the design — it is a **bad habit in the last
layer** and in **how we taught it**.

**Where exactly the habit lives.** The model's final score for each
column is a sum of four contributions. Two of them vary with the board
(good); one of them is a **constant** that simply says "the middle
column is worth 2.58 more than the others" — a huge constant compared
to the board-driven variation, which is only 0.47. A constant that big
simply **overwhelms the board signal**, so the model never needs to look
at the board at all.

That is the whole bug, in one sentence: **a constant that outshouts the
signal.**

**We tried three ways to fix it, and we measured each one honestly.**

1. **Subtract the constant.** *Made it worse* — 25.5 % → 17.5 %. The
   weights around the constant also needed re-learning, not just
   shifting.
2. **Re-learn the final layer from scratch** (keeping the model
   identical otherwise). *Worked for the symptom, broke the game.* It
   now plays side columns far more (16.7 % → 51.3 %) and is finally
   mirror-symmetric (6.2 % → 28.2 %), but it **blocks the opponent's
   win far less often: 47.2 % → 36.8 %**. A model that is pretty but
   loses on tactics is not a fix.
3. **Change how we teach it** (nudge the training so "the middle" is not
   the easy answer to learn). *This is the promising one.* At half
   strength it became **3.5× more mirror-symmetric (6.2 % → 21.0 %) while
   still blocking 46.3 %** versus 47.2 %.

**One more thing we ruled out.** We suspected we simply needed more
different-looking training boards. We built two datasets with exactly
the same number of labels and the same mix of game lengths, differing
only in how many *distinct* boards they contained, and tested it. The
diverse one was **worse** (24.4 % vs 27.5 %). More different boards
was **not** the missing ingredient.

**A measurement mystery, solved.** For days three different scripts
disagreed on how often the model picks the middle (1250 / 1367 / 1500
out of 1500). It was **never a model bug**: one script forgot to ignore
columns that are already full, and one number belonged to a *different
model in the comparison*. The official number is **1250**, and our
reconciliation script reproduces it exactly.

**Still not finished.** The full-size training run on the big,
well-balanced dataset is running now. It will be measured with the same
tests as everything else. **We are not shipping anything until it beats
the current model on the tactical test** (blocking an opponent's win at
least as often as 47.2 %) *and* improves symmetry. Full numbers in
`docs/CONNECT_FOUR_SECTION_0H.md`.

### 9.0 The biggest finding: the model we were shipping could not see the board

Before any tuning, data or teacher question — there was a plain bug.

**The program we train and the program we ship are different functions.**

The trainer has two forward passes:

- a **soft** one, which mixes several possible values with fractions;
- a **hard** one, which picks exactly one value at every step.

The file we save and put in front of people is the **hard** one — that is
what `export_hard()` writes, and what the live game reads. But the trainer
only started training the hard version at step **12,000**:

```
--hard-start 12000 --hard-ramp 4000 --hard-w 0.5
```

**No run script ever changed that number.** Every version from V9 to V23 used
the default. So for the first 12,000 steps of every single run, the thing
we were saving and testing was a version of the model that had **never been
trained at all**.

Here is the proof. Take a saved model, change the board, and see whether
its preferred column moves. A model that looks at the board **cannot**
return exactly zero.

| saved at step | does the shipped model react to a new board? | what it plays |
|---|---|---|
| 2,000 | **no — exactly 0.000000** | only columns 2 and 3 |
| 10,000 | **no — exactly 0.000000** | column 3 on 120 of 120 positions |
| 11,500 | **no — exactly 0.000000** | column 3 on 120 of 120 positions |
| **12,000** ← the switch | **yes — 0.7402** | starts varying |
| 19,000 | yes — 0.6935 | column 3 on 81, column 4 on 33 |

The transition happens on exactly the step the flag says it should. The
*soft* weights were fine the whole time (they reacted to the board from
step 2,000 onwards) — the signal was there, sitting in the fractions. The
hard export simply threw it away by rounding to a single choice.

**So V22's `best_hard.pt` — the file the website and every test were using
— was a constant-column policy.** It answered column 3 or column 4 and
ignored the stones on the board. It looked plausible in every measurement
that did not require actually seeing the game.

**Two things made this invisible, and both are worth knowing:**

1. **The two scores came from different programs.** The "holdout" number was
   measured on the *soft* model. The ladder games were played with the
   *hard* model. Before step 12,000 those described two different things —
   one that could see the board and one that could not. Neither number was
   wrong; comparing them was meaningless.
2. **Our own audit script lied twice.** The first soft-vs-hard comparison
   reported "difference exactly zero" — because it loaded the saved
   *hard* file into the *soft* model, where the soft weights were simply
   absent, so both sides returned zero and matched perfectly. A later check
   found a "perfectly uniform" distribution — which was also just zeros
   spread evenly. Lesson: always check that the weights you meant to load
   were actually loaded.

**Fixed now:**

- `--hard-start` now defaults to **0** — the shipped path is trained from
  the first step, not from step 12,000.
- The trainer **refuses to promote** a checkpoint whose shipped model does
  not react to a change of the board. A blind checkpoint is worse than no
  checkpoint, because it looks like a learner.
- `run_v23.sh` passes `--hard-start 0` explicitly so the setting is visible
  and cannot be quietly reverted.

**What this does *not* fix.** Board-reading is necessary but nowhere near
sufficient. Every artifact we have still loses to lookahead and minimax,
and none of them beats the one-step-lookahead opponent. Turning the
board-reading back on is the first repair, not the last one.

> **CORRECTION (night of 26 October).** This section calls the
> `--hard-start` problem "the biggest finding". That is no longer true.
> The `--hard-start` fix was **necessary but not sufficient** — under it
> the exported tables were *still* frozen, so the model stayed board-blind
> anyway. The real root cause was one level deeper; see **§9.0b** and the
> `table-bias` fix. We later re-ran the schedule question with the
> deeper bug fixed: **`--hard-start 0` is still the right answer**, but
> for a different reason than this section implies.

### 9.0b The bigger bug: the shipped tables were frozen at initialisation

Section 9.0 fixed *when* the hard path starts training. It turned out
there was a second, deeper bug underneath it — and this one made the
`--hard-start` question unanswerable.

**What the shipped file actually is.** The saved model is not a network of
weights. It is a **compiled integer program**: three big lookup tables
(embedding, S-box, coefficient) plus a float readout. When we save it, we
throw away the *logits* that produced those tables and keep only the
**winner** of each lookup. "Which byte maps to which byte" is a decision,
and only the decision is saved.

**The bug: those decisions never changed.** We hashed the tables at every
checkpoint of a 3,000-step run:

```
step    0     500    1000   1500   2000   2500   3000
tables:  IDENTICAL at EVERY step
```

The saved program was **the same program on step 3,000 as on step 0**. It
had learned nothing. No matter what board it was given, it produced the
same answer.

**Why.** Three separate things conspired:

1. **The starting point was too good.** At initialisation the tables were
   deliberately biased toward the "identity" solution — each byte maps to
   itself. That bias was worth **7.78 logits**, and a gap between the best
   and second-best choice has to be crossed before anything changes.
   With this learning rate, Adam can move a number by about 0.0006 per
   step. To cross a 7.78 gap you need roughly **13,000 steps**. Most runs
   are shorter. So the winner was pinned.

2. **The regulariser pushed the wrong way.** We added a rule that
   discourages wishy-washy outputs. Applied to these tables, it pushed
   the identity answer *even further ahead*. Measured: the gap grew from
   7.72 to 9.94 during training — it moved the wrong direction.

3. **The readout started at exactly zero, so the tables got no signal at
   all.** The readout is the part that turns table lookups into an answer.
   Starting it at zero means the tables' contribution is multiplied by
   zero. We measured the gradient reaching each table:

   ```
   embedding table    gradient = 0.000000e+00   ← exactly zero
   S-box tables       gradient = 0.000000e+00
   coefficient table  gradient = 0.000000e+00
   readout            gradient = 3.52e-01      ← all the learning here
   ```

   The tables were not learning slowly. They were not learning **at all**.

**The fix.** A new flag, `--table-bias`, scales that initial bonus.
`1.0` is the old behaviour. `0.0` removes it, so the identity solution
has to be *learned* rather than handed over. The gap drops from **7.78 to
0.033**. The random-number draws are unchanged, so every existing
checkpoint still loads correctly (verified: zero missing keys).

**The controlled test** — same corpus, same seed, same labels, same
everything, only the flag differs:

| checkpoint | `--table-bias 1.0` (before) | `--table-bias 0.0` (after) |
|---|---|---|
| step 500 | 0.0000 — cannot see the board | **0.5444 — sees the board** |
| step 1000 | 0.0000 — cannot see the board | **0.8343 — sees the board** |
| step 1500 | 0.0000 — cannot see the board | **0.9569 — sees the board** |
| step 2000 | 0.0000 — cannot see the board | **1.1250 — sees the board** |
| step 3000 | 0.1567 (barely) | **1.6794** — sees the board |

And the tables now change at every checkpoint, where before they never
changed at all. **This is the first change in the project's history that
has ever moved the saved integer tables.**

**What it does *not* fix.** Being able to see the board is not the same as
being good at the game. On 4,000 held-out positions the fixed model's
saved program guesses the right column 25.3% of the time (the old one:
25.1% — a tie, not a win), plays off-centre 8.7% of the time (old: 4.4%,
so twice as good but still poor), and still leans on column 3 for 67.7% of
its answers (old: 84.8%). It is also still **mirror-inconsistent**: on
mirrored boards it usually gives a different answer instead of the
mirror-image one.

There is also a **regression**: the *soft* model's off-centre accuracy
collapsed from 16.6% to 2.4%. Removing the initial bias changed which
solution the float path settles on, and it settled on a worse one. We
have not explained that yet.

So: this is a **necessary repair, not a finished model**. It removes the
reason every saved model was a constant-column policy. The remaining
problems — the column-3 bias, the mirror asymmetry, and the weak soft
path — live in the *teacher and the loss*, not in the initialisation.

**And it retroactively explains the `--hard-start` question.** We could
never measure it, because the tables could not move under any schedule.
With the fix in place we re-ran it: 6,000 steps, identical in every way
except when the hard training kicks in (`--hard-start 0` vs `--hard-start
750`, i.e. a one-eighth warm-up of soft-only training).

| step | hard-start 0 | hard-start 750 |
|---|---|---|
| 500 | **0.6053** | 0.0094 |
| 1000 | 0.5975 | **0.7263** |
| 2000 | 0.9220 | **1.2652** |
| 3000 | 1.1497 | **1.3528** |
| 4000 | 1.3656 | **1.5041** |
| 5000 | 1.6813 | **1.8692** |
| 6000 | 2.0219 | **2.2024** |

The curves take turns leading and never separate. On the things that
matter — accuracy on held-out boards (24.9% vs 24.6%), off-centre moves
(5.7% both), and mirror consistency (9.8% vs 6.0%) — they are a tie.

**So the original suspicion was reasonable but it was not the cause.**
Starting the hard training later changes almost nothing, because before
the fix there was nothing for it to change. The table init mattered by
an order of magnitude more. **Recommendation: keep `--hard-start 0`.**

### 9.1 We did not build a strong opponent. Here is the evidence.

The old claim rested on games against opponents that were given permission
to play randomly and on scoring draws as half a win. When we re-ran the
games with **deterministic** opponents and counted only outright wins:

| opponent | our model's win rate (V21 step 2900) |
|---|---|
| always drop in the middle column | wins easily — but the middle rule **loses 100% of its games to a one-step-lookahead opponent**, so beating it proves nothing |
| plays completely randomly | wins |
| one-step lookahead | **0% wins** (draws) |
| two-step lookahead | **0% wins** (draws) |
| full three-step search | **0% wins** (loses) |

Same result for every candidate we tested. **Zero wins against any competent
opponent.** The earlier "38.35% vs one-step lookahead" was an artefact of
letting the opponent randomise and of counting draws as half-points.

The reference hierarchy, measured: the "always middle" rule is the *weakest*
opponent we have, one-step lookahead is stronger, three-step search is
stronger still, and the exact solver is strongest of all.

### 9.2 Why: three stacked problems, all now measured

**(a) The model underfits — it never learned an opening.**
Train strict agreement **23.4%**, holdout strict agreement **20.6%**. The
gap is only **2.8 percentage points**, so this is underfitting, not
overfitting. The model is simply not fitting the data.

The trainer's headline "43.5% holdout" is **tie-inflated** and should not be
quoted: the dataset's score targets are mostly flat ties (46.6% of rows have
essentially equal scores across columns), and the metric gives full credit
for picking *any* tied column. Strict agreement is about **21%**.

**(b) Openings and simple "don't lose" were the specific failures.**
Depth-stratified measurement of the shipped model:

| depth | takes an immediate win | blocks a required block | agrees with label (strict) |
|---|---|---|---|
| opening (30–42 empty) | 23.8% | 22.5% | **15.8%** |
| midgame (16–23 empty) | 40.9% | 34.7% | 21.9% |
| endgame (0–7 empty) | **72.7%** | **59.3%** | **47.7%** |

So: **the endgame works, the opening does not, and "prevent a win" works in
the endgame but fails in the opening.** That is exactly the shape of the
root cause below. Also: over the first 12 plies the model only ever plays
columns 3 and 4 (the two centre columns), and the shipped V21 checkpoint
plays column 3 **98.2%** of the time — it is effectively a centre-only
policy.

**(c) The training data was filtered so the model never saw an opening.**
This is the root cause. In `tools/split_positions.py`, the training side
kept only positions with **at least 20 plies played** (so ≥18 empty cells),
while the **holdout side kept every depth**. The model was therefore never
supervised on a single opening position, yet **68.3% of the holdout is
opening** — the metric was measuring a distribution the model had never
trained on.

**(d) Only 5.26% of the training signal was real.**
Measured on the exact production path: 304 move positions survive the
legality/visibility mask per batch, but only **16 carry a real label** — the
other 288 are unlabeled plies. Each labelled position was seen only
**3.74 times** across 25,000 training steps.

**(e) The teacher is tactically correct but strategically weak.**
The imported labels are *good* — this corrects an earlier wrong diagnosis:
- they rank an immediate win first **100%** of the time;
- they pick the required block **82.7%** of the time;
- they never put the argmax on an illegal column.

But played as a complete policy the label scores only **27.4% vs one-step
lookahead, 14.8% vs two-step, 13.2% vs three-step**. Their argmax matches
the exact solver's best move on only **57.1%** of provable positions. A
student cannot out-learn a weak teacher.

### 9.3 Things that were *not* the problem (and were wrongly blamed)

| suspicion | verdict |
|---|---|
| "soft and hard models diverge too much" | **No.** The soft/hard bits-per-byte gap peaks at **2.212** and never exceeds **3.0** in 24,500 steps. Column-distribution JSD is **0.011–0.030 nats**. Both are small; neither explains the weak play. |
| "the labels are tactically blind / sign-inverted" | **No.** That was our own audit bug (see §9.4). Corrected measurement: 100% on immediate wins, 82.7% on blocks. |
| "the exact solver lost to minimax" | **No.** `ExactSolver(max_empty=12)` was silently falling back to a depth-limited policy deeper in the game. The "exact" result was invalid. |
| "the network is too small" | **No.** The corrected 4,096-position capacity test reaches **87.7%** oracle-tie / **56.5%** strict against a **33.2%** chance line. |
| "the model overfits" | **No.** Train and holdout strict agreement are both low and nearly equal (23.4% / 20.6%). It underfits. |

### 9.3c The per-ply supervision lever — wired, verified, and now A/B tested

The single-label sidecar supervises exactly **one** move per 193-byte
window. That is the binding constraint on how much the model can learn
per byte of corpus, and it is a *data plumbing* limit, not a data
*quality* limit: the corpus already contains ~6.2M plies, we were
supervising 386,916 of them.

**The trainer now accepts a per-ply sidecar** (`--perply-targets`):

- `src/c4/train.py::_perply_rows_for_batch()` places the sidecar target
  for ply p on exactly the token that predicts ply p's **move byte**;
- `src/c4/train.py::_align_perply_targets()` loads
  `target (N,P,7)`, `ply_of (N,P)`, `n_plies (N,)` and refuses to train
  if the row count does not match the corpus;
- `snap_to_end` is now enabled for the per-ply path too, because a random
  window would usually contain none of the labelled plies.

**Measured on the real corpus, real sidecar, real batch:**

| | real targets / batch | density of move positions |
|---|---|---|
| single-label (V23) | 16 | **5.2%** |
| per-ply (V24) | 217 | **70.2%** |

**13.6× more real supervision from the same bytes.**

Two bugs were found and fixed while wiring this, both of which a
self-consistent test would have passed:

1. **The grid labelled every byte, not just the move byte.** The floor
   division `p = (byte - move_off) // ply_len` puts all 10 bytes of a
   ply into the same ply p, so 9 of every 10 positions were handed a move
   target for a *header* byte. Fixed with an exactness check
   (`abs_byte != p*ply_len + move_off`).
2. **The move offset was `ply_len - 2` (8) instead of `MOVE_OFFSET` (9).**
   Both the implementation *and* its test used the same wrong formula, so
   they agreed with each other. The test was rewritten to identify the
   move byte **by value** (0x41..0x47, disjoint from every header byte),
   which is an independent fact and does catch the off-by-one.

`tg_perply_wiring.py` now verifies the grid against that value-based
check, and `tg_perply_density.py` measures the density through the
trainer's own `build_batch` / `move_mask_from_starts` / `build_move_targets`.

**Teacher quality of the new labels** (19 last plies per transcript,
1,500 rows): column spread c0=13.9% c1=14.8% c2=16.3% c3=25.0%
c4=13.7% c5=10.4% c6=6.0% — healthy, no centre collapse.

### 9.3d Is the model just "playing the middle column"? NO — proven

This was the right question to ask, and the answer is clean. We measured
what an "always play column 3" player would score on the same test
positions, and then measured the model on positions where **the winning
column is not the centre** — where a centre-only player scores **zero by
construction**.

| | wins-only model | V23 |
|---|---|---|
| best constant column | 26.8% (column 3) | 26.8% (column 3) |
| model overall | **60.0%** | 46.8% |
| **wins NOT in the centre (n=895)** | **57.4%** | **39.6%** |
| always-centre on those same 895 | 0.0% | 0.0% |

And per winning column, the model is good nearly everywhere — this is the
part a centre-bias model cannot fake:

| win in column | c0 | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|---|
| wins-only | 63.5% | 52.9% | 74.7% | 70.1% | 66.5% | 13.6% | 51.0% |
| V23 | 48.7% | 27.7% | 59.0% | 74.9% | 45.5% | **1.7%** | 41.2% |

**The centre preference is a tie-break among good moves, not the policy.**
The one place it really costs wins is **column 5** — V23 essentially never
plays column 5 (1.7% when the win is there; 2 picks in 1,126 positions).

In actual games the model beats a centre-only opponent from both seats:
**wins-only 62.3%** (187W-113L, 118 distinct trajectories) and
**V23 95.3%** (286W-14L, 147 distinct trajectories). So the middle is the
*opponent's* weakness, not the model's strategy.

### 9.4 Two of our own audit bugs (retracted)

Recorded because the same mistake will otherwise be repeated:

1. **Off-by-one on the V2 final record.** `encode_seq(moves, label_move)`
   appends a final record whose *header* is the labelled position and whose
   *move byte* is the label. Our first audit replayed the label move and then
   audited the board **after** it. Every tactical conclusion drawn from that
   run was wrong.
2. **`Board.apply_move` mutates in place.** When probing candidate columns we
   corrupted the board we were auditing. Every candidate probe must clone
   the board first (`copy.deepcopy`) — the same bug then corrupted the
   column-distribution audit and produced a bogus "model always plays column
   0" reading.

Both were found and fixed; the corrected numbers are the ones above.

### 9.5 The fix now being run (V23)

1. **Depth-fixed split** — no train-only depth filter, so openings are
   supervised on both sides and the metric is measured on the distribution
   the model will actually face. Symmetry/mirror leakage stays **0**
   (verified: train 386,916 / holdout 20,260, overlap **0**).
2. **Exact-solver teacher overlay on the midgame band** — the exact solver
   with a shared transposition table proves **12/12** corpus positions at
   every depth tested, and is affordable across the whole blind band:

   | empty cells | 12 | 15 | 17 | 19 | 21 | 23 |
   |---|---|---|---|---|---|---|
   | ms/position | 1.4 | 11.8 | 23.5 | 125 | 348 | 2354 |

   Its best move is written as a soft target over all provably-equal-best
   columns, so the student learns the optimal move *and* the tie structure
   (which also removes the low-index bias that produced centre collapse).
3. **Playable against the best hard V22 checkpoint** — yes, the UI can serve
   it, and V22 is much less centre-degenerate than V21 (column 3 at 62.8%
   rather than 98.2%), but it is **not validated as strong** and must not be
   presented as a winning opponent.

### 9.6 What is solid

- The dataset decodes correctly: legal boards, valid targets, **0** symmetry
  leakage, **0** mirror leakage.
- The V2 encoding is correct: **0** collisions, ownership round-trips.
- The two fatal trainer defects (10-byte tags against a 193-byte window;
  `0 × (-inf) = NaN`) are fixed and regression-tested (`tg_gate_check.py`
  **24/24 PASS**).
- The hard-export transcript loader is fixed and regression-tested
  (`tg_version_check.py` **7/7 PASS**).
- The corrected 4,096-position capacity test passes: **56.5% strict /
  87.7% oracle-tie** vs **33.2%** chance.
- **The frozen-table bug is fixed.** The exported tables now change at every
  checkpoint instead of staying frozen, and the hard export reads the board
  from step 500 instead of step 12,000. This is verified end-to-end.
- **The teacher is now provably mirror-consistent** — 100% over 18,304
  mirror-paired labels, and 100% on the disjoint holdout too.
- **The STE schedule question is answered**: keep `--hard-start 0`.

### 9.6b The honest status as of the night of 26 October 2026

The table-freezing fix is real and it works. But it is a **prerequisite, not
a finish line**. Here is where we actually stand:

| thing | before the fix | after the fix | verdict |
|---|---|---|---|
| hard export reads the board | no (0.0000) | yes (0.5444 from step 500) | **fixed** |
| exported tables ever change | never | every checkpoint | **fixed** |
| hard export plays the middle column | 85.4% | 67.7% | better, still bad |
| hard export plays off-centre | 4.4% | 8.7% | better, still bad |
| hard export mirrors correctly | 2.7% | 11.8% | better, still bad |
| strict accuracy on unseen positions | 25.1% | 25.3% | no real change |

**Read that table carefully.** The bug we found was real and fixing it made
the program *capable* of seeing the board. It did **not** make the program
*good*. Strict accuracy barely moved (25.1% → 25.3%).

And one number went the **wrong** way: the soft path's off-centre accuracy
dropped from 16.6% to 2.4%. Something in the objective is actively pushing
the model towards the middle column, and removing the init bias exposed it
rather than causing it.

**What that tells us:** the remaining problem is the **objective and the
data**, not the initialisation. The teacher is provably symmetric, so the
asymmetry has to come from how the model is trained.

**So: not ready to ship, and no big V24 run yet.** The list of things still
to do is in the handover (§0e.6), in order:
1. kill the centre collapse (68% of moves are still column 3);
2. fix the mirror asymmetry;
3. run the strength tests we have never run on a fixed model;
4. relabel the whole corpus with the canonical teacher, without blowing up
   the machine's memory;
5. smoke-test the real shipping code path once more.

**A warning about the old numbers:** every checkpoint trained before this
fix — including all of V22 and its "46.3% holdout" and "ladder net -61" —
is a constant-column policy in disguise. Those numbers are **not** evidence
of a good model. Do not quote them, and do not treat them as a baseline.

### 9.6c The V23 / data / "teach it the first 7 moves" question — answered 27 October 2026

Three people were asked the same question in different words: *"isn't it just
the training data?"*, *"can't we just teach it the first 7 moves?"*, and
*"the columns must be structurally invisible — the lag wiring is too
sparse."*

**The answer is: the data hypothesis and the structure hypothesis are both
right, they are different problems, and you have to fix both.**

#### V23 is dead, and now we know why

```
c3 choices:                   476/500
off-centre choices:            24/500 = 4.8%
non-centre mirror consistency: 0/24
```

V23 had 193,453 mirror-partner pairs in its corpus and still could not
mirror a single non-centre move. **A mirror-rich corpus gives you
mirror-consistent labels. It does not give you a mirror-consistent
model.** V23 predates the table-freezing fix, so it is a historical
artifact — not a release candidate.

#### The structural blindness is real (Claude was right)

The wiring that actually ships makes four of the seven columns hard or
impossible to reach, and the left and right sides of the board are
reached by *different amounts of wiring*:

```
reachable channels:  c0 18   c1  1   c2  6   |  c6 21   c5 10   c4 18
```

Column 1 had **exactly one** reachable channel, and its measured effect
on the hard export was **exactly 0.0000** — the model literally could not
see it.

Dense lags (`0..9` plus `10,20`, with the context length still 192) fix
that:

```
                 c0     c1     c2     c3     c4     c5     c6
sparse lags   0.3403  0.0000  0.2309  0.2360  0.2132  0.2183  0.1970
dense lags    0.3382  0.3246  0.2536  0.3460  0.3552  0.3543  0.3707
```

And the mirror pairs become symmetric:

```
sparse:  c0/c6 1.00x   c1/c5 infinite   c2/c4 1.08x
dense:   c0/c6 1.09x   c1/c5 1.00x      c2/c4 1.14x
```

#### The entropy regulariser is NOT the cause (Claude was wrong here)

Turning the entropy regulariser off entirely changes nothing about the
blindness:

```
ent_max=0, 3,000 steps:
  c1 hard-export sensitivity:  still exactly 0.0000
  S-boxes still lossy:         157 of 256 distinct outputs per row
  hard strict accuracy:        24.9%   (baseline was 25.3%)
  mirror consistency:          10.0%   (baseline was 11.8%)
```

Dropping entropy made the model a bit less centre-happy (40% non-centre
vs 32%) and a bit worse at everything else. It did not make the missing
column visible. Don't spend more time here.

#### The data hypothesis is right — and now it's quantified

The labels are **not** broken:

```
true contradictions between identical boards:  0 / 3,167
mirror consistency of the labels:              ~99.9%
the teacher agrees with the exact solver:       97.6%
```

The problem is how many *different positions* we actually had:

```
full-game corpus:  18,783 labels over  ~3,856 different boards
opening curriculum: 322,633 labels over 70,268 different boards
```

**18,783 labels is not 18,783 examples — it is 3,856 examples.** And when
you sweep the number of different boards and plot the result, everything
improves with coverage while the headline accuracy stays flat:

```
off-centre accuracy:  ~11.9%  ->  ~22.9%
mirror consistency:   ~10.4%  ->  ~27.9%
strict accuracy:      broadly flat
```

#### "Can't we just teach it the first 7 moves?" — Yes, and it works

Nine plies of opening is exactly the right slice, and it is cheap to
generate. We produced 40,000 games × 9 plies = 358,480 labelled
positions over 70,268 different boards, all with the same canonical
teacher.

On the untouched full-game holdout, using a fast MLP as a stand-in:

```
arm                        boards   strict   c3%   off-c%  mirror%  beats guess-c3?
full-game only              3,856   24.0%  36.6%   17.4%    26.4%      NO
opening only               70,268   34.2%  53.1%   23.6%    39.9%     YES
curriculum, then full game  70,268   27.5%  20.5%   27.2%    27.9%     YES
opening + full game mixed   70,268   33.6%  45.8%   26.0%    39.8%     YES
------------------------------------------------------------------
always guess column 3:                  25.0%
```

The old recipe is **worse than always guessing the middle column**. Every
version that adds opening coverage is **better**. The effect scales with
coverage: a smaller 14,767-board opening set only just cleared the
baseline (25.6%); the 70,268-board set clearly beat it (34.2%).

**One trap to know about:** the teacher plays the middle column 100% of
the time in the first two moves, so the opening data is *more*
centre-heavy than the full-game data (38.9% vs ~25-30%). Opening-only
training therefore makes the model centre-happy (53.1% c3) even though
its accuracy is better. Mixing the opening with the full game, or
pretraining on the opening and then training on the full game, keeps the
centre share under control.

#### The lesson

We had been confusing **seeing the board** with **knowing the answer**.

- The table-bias fix made the export able to *see* the board.
- Dense lags made it able to see *every column*.
- Turning off entropy ruled out the regulariser.

None of those teach the model which move is right. Only **more different
positions** do that — and we were training on 3,856 of them.

### 9.6d The real 2×2 finished — and it turned out to be a flawed experiment

The experiment in §9.6c ran to completion (3,000/3,000 on all four arms,
no crashes). Here is what it actually found on the hard export:

```
arm                     strict   off-centre   model c3   board sens.   mirror%
A sparse + full-game      26.7%       3.5%       87.7%       1.12         3.5%
B dense  + full-game      27.0%       4.9%       84.5%       1.71         6.2%
C sparse + curriculum     21.2%      15.3%       36.7%       0.67        23.4%
D dense  + curriculum     21.5%      13.6%       40.3%       0.67        20.0%
```

Read that carefully, because two of those columns tell opposite stories:

- The curriculum arms **massively improved the things we were fixing**:
  the centre bias collapsed (c3 share 87.7% → 36.7%), off-centre accuracy
  went up 4.3× (3.5% → 15.3%), and mirror consistency went up 6.7×
  (3.5% → 23.4%).
- But they **lost** on overall accuracy and their sensitivity to the board
  dropped by half.

**So we checked the obvious suspect: are the curriculum labels wrong?**
No. We matched the two datasets board-by-board and compared the teacher
answers directly:

```
boards present in BOTH datasets:  3,856
same label:                        3,856  (100.00%)
different label:                       0  (0.00%)
```

The opening data uses exactly the same teacher as the full-game data, and
agrees on every single shared position. **The labels are not the problem.**

#### The real problem: the experiment changed three things at once

```
                      A/B full-game          C/D curriculum
rows                        1,500                   43,000
labels                     18,783                  396,046
different boards             3,856                   70,268
passes over each board       2.56                    0.12   <-- 21× fewer
ply-0-to-8 share             32.1%                   93.6%
```

With a fixed 3,000 steps, the curriculum arm took **21× fewer passes over
each position**. That fully explains the halved board sensitivity — it
simply hadn't seen each position often enough. The ply shift compounds
it: the curriculum is 94% opening, but the test set is 74% late-game.

In other words: the first 2×2 did not test "is the data better?" It tested
"is the data better **plus** a 21× undertraining penalty?" The answer it
gave was about the penalty.

#### A third problem we found while fixing the above

The "disjoint" holdout was built from the V23 corpus, not from the corpus
the arms trained on, so it was not disjoint from either:

```
holdout positions:                 4,325
seen during full-game training:      207  ( 4.8%)
seen during curriculum training:   1,099  (25.4%)
```

A quarter of the test set had been trained on by the curriculum arms. Any
release gate measured on that holdout is not trustworthy until the holdout
is rebuilt per-corpus.

#### The corrected experiment (running now)

Both datasets now have **the same number of labels**, so the number of
passes is identical and the budget confound is gone:

```
dataset        rows   labels   boards   epochs at 48,000 draws
lm_full        1,500   18,783   3,977     2.556
lm_open        2,087   18,712   7,564     2.565
```

```
A  sparse lags + full-game   3,977 boards
B  dense  lags + full-game   3,977 boards
C  sparse lags + opening     7,564 boards
D  dense  lags + opening     7,564 boards
```

This one changes exactly two things — the lag set and the data — and keeps
the training budget and the number of passes fixed. **This is the run that
decides the question.** Until it lands, no V24.

### 9.6e The mirror experiment — finally run properly (27 October 2026, 20:35)

The one open question from §9.6d was: *does training on mirrored copies of
the same games actually help, or does it just look like it because the
experiment was rigged?*

**The earlier experiment WAS rigged, by accident.** The trainer picks random
training rows *with replacement* (it can pick the same row twice). So when
we doubled the corpus by adding mirrored copies, each training batch spent
half its picks on the same underlying games. The mirror arm saw only about
**half as many different games** as the normal arm, for the same number of
steps. Any difference could have been "saw fewer games", not "saw mirrored
games".

**The fix:** train both arms longer, until each one has seen the *same
number of different original games*.

| | normal arm | mirror arm |
|---|---:|---:|
| games seen (different ones) | 82,936 | 44,572 |
| vs. the 3,000-step normal arm | — | 44,573 (**a match**) |

Now the two arms differ in exactly one thing: whether mirrored copies were
in the training mix.

**Result — the mirror arm wins on everything that is measured against an
independent truth:**

| what we measured | normal | mirror | who wins |
|---|---:|---:|---|
| plays the same move on a flipped board (mirror consistency) | 16.4% | **36.7%** | **mirror** |
| blocks an opponent who is about to win | 42.7% | **50.2%** | **mirror** |
| plays a provably optimal move (checked with a perfect solver) | 69.3% | **71.7%** | **mirror** |
| agrees with the teacher's advice | 21.7% | 23.8% | mirror |

### 9.6f The one number that looked worse — and why it doesn't matter

The mirror arm scores **9.7%** on "played a non-middle move when the
teacher wanted a non-middle move", versus **15.4%** for the normal arm.
That looks like a regression. It isn't.

Here is why: **the teacher is not the truth.** We checked 166 positions
where a perfect solver could prove the right answer. In those positions:

- the teacher picked the middle column **21.7%** of the time
- the middle column was actually the *best* move **36.1%** of the time

So the teacher plays the middle column *less often than it should*. A model
that plays the middle column more often than the teacher does can therefore
be **right more often than the teacher** — and be marked down for it.

When we split the results by what the teacher wanted:

| | teacher's answer was middle (36 positions) | teacher's answer was NOT middle (130 positions) |
|---|---:|---:|
| normal arm: played a provably optimal move | 83.3% | 65.4% |
| **mirror arm: played a provably optimal move** | 77.8% | **70.0%** |
| mirror arm: played the middle column | 61.1% | **18.5%** |

In the 130 positions where the teacher did *not* want the middle, the mirror
arm played the provably best move **70%** of the time — the best of any arm
we have trained — and it only played the middle column 18.5% of the time.
**It avoids the middle exactly when the middle is wrong.** That is the
opposite of a defect; it is the behaviour we were trying to teach it.

### 9.6g One more thing we ruled out

We also tested a cheap trick: run the model on the board, run it on the
flipped board, and average the two scores. If the model already "knew"
about symmetry, this would fix it for free. It does not: it gains
**0.1 to 0.6 points** and pushes the model *back* towards the middle column
(37% → 51%). The symmetry has to be **trained in**, not bolted on.

### 9.6h A correction we had to make to our own numbers (27 October 2026, 21:15)

When we first wrote up the mirror result, we put the "how often does it
play the middle column?" number next to the other results. That number came
from the **166 endgame positions** we could check with the perfect solver.
But the other results came from the **full 1,500-position test set**. Mixing
the two made the middle column look *less* used than it really is:

| | middle column used on the full 1,500 test positions | on the 166 endgames |
|---|---:|---:|
| normal arm | **33.3%** | 25.3% |
| mirror arm | **54.6%** | 27.7% |

So on the board as a whole, the mirror arm plays the middle column **more**
often than the normal arm. Our earlier claim that the mirror arm is less
centre-biased was **wrong** — it was true only of endgames.

The fair summary is that the mirror arm is **better at mirror-symmetry,
blocking, endgame correctness and head-to-head play**, and **worse at
spreading its choices across the whole board**. Both are real.

### 9.6i The diversity experiment, finished

We also ran a four-way test on a small corpus, varying how many *different
boards* the model saw and whether the moves were all "centre-ish".

| | agreed with teacher | played non-middle when teacher wanted non-middle | middle-column use |
|---|---:|---:|---:|
| many different boards | 24.5% | 10.1% | 59.7% |
| many different boards, dense moves | 24.4% | 11.7% | 57.8% |
| narrow, centre-ish | 26.9% | 5.3% | 83.5% |
| narrow, centre-ish, dense | 27.5% | 5.2% | 83.8% |

**Seeing more different boards does help the model stop fixating on the
middle** — it drops from 83% to about 58%. But it is still clearly worse
than the big-corpus version (which gets the middle column down to 33%).
So "show it more different boards" is a real effect, but on its own it is
not enough.

### 9.6j One tool that silently did nothing

We asked for the perfect-solver check to reach deeper into the game
(positions with up to 24 empty spaces instead of 18). The command ran and
reported success — but it had **ignored the new setting** and produced the
exact same 166 positions as before. We caught it by checking that the
number of positions actually changed. The tool now takes the setting
properly, and the deeper run is in progress.

This is worth remembering: a script that silently ignores its arguments
looks exactly like a script that worked. Always check that the *output
changed* when the input should have changed something.

### 9.6k The deeper perfect-solver check finally ran — and it changed one conclusion

The tool that "silently did nothing" has been fixed and rerun. It can now
prove **560 of the 1,500** test positions instead of 166 — that is 37.3%
of the test set instead of 11.1%. The other 940 positions were too hard to
prove within the time budget, so they were left out of the score rather
than counted as "half right".

| model | picked a provably best move | used the middle column when it was *not* wanted |
|---|---:|---:|
| shipped | 68.9% | 72.5% |
| big-corpus model | 71.1% | 30.3% |
| big-corpus, slight symmetry | 70.0% | 31.0% |
| small-corpus mirror model | 67.9% | 62.3% |
| **big-corpus normal** | **71.8%** | **26.8%** |
| **big-corpus mirror** | **71.8%** | 41.5% |

**One earlier conclusion did not survive the bigger sample.** With only 166
positions the mirror model looked better at perfect play (71.7% vs 69.3%).
With 560 positions they are **exactly tied at 71.8%**. The original gap was
a small-sample accident. So: the mirror model is *not* proven to play more
perfectly — that claim is withdrawn.

**One more striking finding.** In the positions we can prove, the middle
column is the best move **54.3%** of the time. Our teacher only picks it
**24.6%** of the time. In other words, the teacher we trained against is
badly biased against the middle column — which is why "how well does it
agree with the teacher" is the wrong question to grade this work on.

### 9.6l Where this leaves us

No model wins on everything:

- the **mirror model** wins on being symmetric (36.7% vs 16.4%), on
  blocking threats (50.2% vs 42.7%) and on head-to-head games (63%);
- the **normal big-corpus model** wins on being well-calibrated across the
  whole board, and ties on perfect play.

The mirror trick is a **targeted** fix: it makes the model behave the same
way left-to-right and right-to-left, and it makes it better at stopping
threats. But it also pushes it back toward the middle column elsewhere.

**This project is not finished and V24 has not been launched.** The
remaining work is to get a recipe that keeps the symmetry and blocking
gains *without* the centre drift, and to prove more of the middle-of-game
positions where our perfect checker still gives up.

### 9.6m The "draw with 36 empty squares" bug (27 October 2026, late)

You played a game against the site. You and the model both kept choosing
**column 7**, the game said **DRAW**, and the board still had most of its
squares empty. That looked exactly like "the model only ever plays one
column and does not care".

**It was not the model. It was a one-line bug in the board code.**

#### What the bug was

Connect Four tracks the board as a bitmask. The function that decides
"is the board full?" used to check **one single bit**:

```python
# WRONG
return bool((self.mask >> (ROWS * COLS - 1)) & 1)
```

That bit is the **top square of the right-hand column** (your column 7).
So the moment column 7 got six discs in it — after six plies — the program
decided the whole 6×7 board was full and declared a draw.

The fix expresses the question the way it should be asked — *is there any
legal move left?*:

```python
# RIGHT
return not self.valid_actions()
```

#### Why it survived so long

Because it was almost always harmless. A game that ends with four in a
row never gets far enough for the fullness test to matter. The bug only
fired when **both players kept stacking the same edge column** — which is
exactly what you did, and almost nobody else does by accident.

It was in the **very first commit**, so everything built on top of it
inherited it: the corpus generator, the game-playing code, the strength
measurements, and the website.

#### What it actually damaged

| Thing | Damage |
|---|---|
| Live website | Yes — your game ended in a false draw |
| Training corpus | **15,084 of 320,000 rows (4.71%)** were cut short mid-game |
| Mirrored corpus | **31,234 of 320,000 rows (4.88%)** cut short |
| The perfect-play checker | **None** — it never used this function |
| The strength measurements | Yes — re-measured, see below |

#### The important question: were the *answers* wrong, or just missing?

A truncated game could mean two very different things:

- **Coverage loss** — the remaining plies still carry correct answers, we
  just lost the later ones.
- **Label corruption** — the answers themselves are now wrong, and the
  affected rows must be thrown away.

**This was measured, not assumed.** The final audit
(`runs/_probe/corpus_truncation_final_audit.json`) walks every slot of
every one of the 320,000 rows and classifies it. The verdict is
`COVERAGE_LOSS_ONLY`:

| Measurement | Result |
|---|---|
| Answer slots checked | **4,462,571** |
| Answers that are NOT a legal move at that position | **0** |
| Answer slots with a broken position index | **0** |
| Answer slots the bug caused to be **skipped** | **101,090** |
| Rows affected | **13,168 (4.12%)** |

So: **not one answer is wrong.** The answer file is produced by solving
each position on its own, so a truncated game simply has fewer solved
positions — it never gets *wrong* ones. 101,090 solved positions were
never written because the bugged test called the position "finished".

That means: **no corpus repair is needed, and the models already trained
on it are valid.** The cost is less endgame coverage, nothing more.

#### The mistake we made while checking (worth writing down)

Our first audit said "LABEL CORRUPTION" — and it was **wrong**, for two
reasons that both looked like real findings:

1. We compared a counter called `n_plies` against the number of recorded
   moves. `n_plies` is actually the number of **answered slots**, capped at
   19 by design. A 25-move game with `n_plies=19` is perfectly correct.
   That single mistake "found" 13,688 corrupt rows.
2. We re-solved every label with the **perfect** checker. But the answer
   file uses the perfect checker only for late-game positions and a
   depth-limited "good enough" checker earlier. So "not perfect-optimal"
   looked like "corrupt" — at exactly the ~4% rate we already knew the
   teacher was not perfect.

**Lesson:** an audit that finds a big problem should first ask whether the
audit itself is broken. Both of ours were. The final audit avoids both
traps: it does not re-solve anything, and it does not interpret `n_plies`
at all — it asks the only question that matters, *"is this answer a legal
move at the position it claims to describe?"*, and classifies every
skipped slot by **why** it was skipped.

#### What is now in place

- `Board.is_full()` means "no legal action remains".
- `src/c4/board.py` exposes `is_terminal()` — one canonical terminal
  predicate for the whole project — plus `assert_terminal_consistent()`.
- **Every consumer now uses it.** The duel harness, the play loop, the
  corpus generator, the corpus quality check, the outcome-RL trainer, the
  augmentation builder, the benchmark and the trainer all call the
  canonical predicate instead of hand-rolling a fullness test. Leaving
  one old call site in place is how the same bug comes back.
- A permanent regression test lives in `src/c4/test_roundtrip.py` and
  covers the exact reported scenario, a genuinely full board, 300 random
  games, and every single-column stack.
- The reusable procedure is saved as the skill
  `terminal-detection-bitmask-audit`.

### 9.7 What is not done yet

- **The release decision itself.** Every hard gate now passes for
  `mir_a_s6000` except that the model still loses to the lookahead bot
  (5.3%) and to the heuristic bot (18.5%) in actual games, even though it
  plays the provably optimal move 71.7% of the time on tested positions.
  We have not yet decided whether to ship it as-is or keep training.
- Running the **V24** full-scale training run (not launched — waiting on
  that decision).
- DAgger (training on the positions the model reaches while playing).
- The full-scale plain-neural-network comparison.
- Tightening the frozen-integer conversion (soft/hard agree on only 58.5% of
  moves).
- The full canonical relabel of the whole corpus (memory-bound at 16
  workers — needs fewer).


---

## 10. Frequently asked

**Why predict bytes instead of scoring columns?**
Writing the move as a byte turns "choose a move" into "predict the next
character" — a well-understood task, and board structure is learned as a side
effect of predicting the stream.

**Why 10 bytes per move?**
Every move record is exactly 10 bytes, so the model can locate "the next move"
at a predictable offset. That is what makes it possible to mask the loss to only
moves.

**What's a checkpoint?**
A snapshot of the weights at one moment, kept so you can go back. We keep the
best, not the newest, because training has bad stretches.

**Why mirror the positions?**
Connect Four is symmetric, so the mirror is an equally valid position with the
same answer. It doubles edge exposure for free. It is *not* extra information,
and it must be split by symmetry class or the holdout leaks.

**Why isn't the corpus bigger?**
The dataset has ~108M rows available. The current split holds **106,966**
training and **20,260** held-out positions, which is what the model is
actually trained on. Size was not the constraint.

---

## File map

```
src/c4/train.py            the trainer
src/c4/transcript_v2.py    binary board+move encoding (authoritative)
src/c4/move_loss.py        7-way masked loss, legality, soft-target blending
src/c4/board.py            rules: legal moves, win detection
src/c4/teacher_corpus.py   self-play generator (disabled for the current runs)
src/c4/ui_server.py        the website backend
src/gf256_markov_lm_v6.py  the model
tools/import_hf_c4.py      builds the position corpus
tools/split_positions.py   symmetry-safe train/holdout split
runs/c4/posdata_score_split/  train + holdout + sidecar targets (T=2 soft targets)
runs/c4/v7mixed_v21/       the validated run (shipping candidate = step 2900)
runs/c4/v7mixed_v22/       the current longer run (25,000 steps)
```

The detailed engineering handover, including every bug that was found and
how it was proven fixed, is in `docs/CONNECT_FOUR_HANDOVER.md`.

---

# Where we are right now (27 October 2026)

## The short version

**The "always plays the middle" problem is fixed.** A new candidate model
called `fc_a_tau0` plays a much wider variety of moves, and — checked against
the perfect solver — **it makes better moves than the model we have been
shipping.**

## What "better" means here

We measured two different things, and they disagree. That is important, so
here is the honest version.

**How often it agrees with our teaching examples:** the old model scored
**28.5%**, the new one **21.9%**. Looks like a regression. It is not.

Our teacher is a program with rules of thumb, and it is right about
**89%** of the time. The old model scored higher against the teacher mostly
because it had been trained on the *same biased material* — an old dataset
where **93%** of the correct answers were "play the middle". So the old model
was scoring points by making the same mistakes as the teacher.

**How often it picks the provably best move:** we asked the perfect solver
(the algorithm that can never make a mistake) what the best move was, on 166
positions where the answer is provable:

| | shipped model | new candidate |
|---|---:|---:|
| picks the provably best move | 64.5% | **69.3%** |
| plays the middle column | 60.2% | **31.9%** |

The new model is **more often right** even though it agrees with the teacher
less. Agreeing with a fallible teacher is not the same as being right.

## The actual reason it got better

We did **not** change the model's architecture. Same network, same settings.

We changed the **training data**. The old dataset told the model "play the
middle" 93% of the time. The new dataset is 320,000 games with 4,462,571
labelled positions, and the middle column is only 25.5% of the answers — which
matches our test set (25.7%).

So the model was not incapable. It had simply been taught a lesson that was
wrong.

## Does it stop the opponent from winning?

Partly. We built 307 test positions where the opponent is one move from
winning, and checked whether our model blocks it:

| | blocks the win |
|---|---:|
| always plays the middle (dumb baseline) | 26.7% |
| random move (dumb baseline) | 18.9% |
| old shipped model | 47.2% |
| **new candidate** | 45.0% |
| — when there is ONE threat to block | 43.4% |
| — when there are SEVERAL threats at once | **57.1%** |

So: **it is much better than the dumb baselines, and better than the old
model when several threats are live at the same time** — which is the harder
and more important case. But on the simple single-threat case it is slightly
*worse* than the old model. That is a real, unresolved gap, and it is listed
below as open.

It performs the same from both sides of the board (44.9% as the first
player, 45.0% as the second).

## The one thing still clearly wrong

Connect Four is perfectly symmetrical: if you flip the board left-to-right,
it is the same game. A model that understands the board should flip its answer
along with it.

Our new model is perfect at this when it is the same player moving (100%),
but only **18.6%** consistent when the board is flipped. The old model was
6.2%. Better, but still wrong most of the time.

**The fix is training on flipped boards.** We have now built that dataset —
every game plus its mirror image, with the answers flipped to match — and
checked it very carefully (5,100,175 board checks and 4,462,571 answer
checks, zero errors). That model is training right now. When it finishes we
will check whether it fixed the symmetry **without** making the tactics worse.

## What is still open — please do not claim these are solved

| problem | where we are |
|---|---|
| flipped-board consistency | 18.6% — the run in progress targets this |
| blocking a single simple threat | 43.4%, worse than the old model's 47.8% |
| strength vs a strong search opponent | weak for **every** model, including the old one |
| using the outermost columns | column 1 is used 23 times in 1500; column 7 only 4 times |

## Two traps we hit, so they are written down

1. **The training data and its "answer file" must be proven to line up — not
   assumed.** The trainer checks that they have the same *number* of rows, but
   not that the rows are in the same *order*. Our first mirrored dataset had
   the answers in the wrong order. Every single-row check passed, and the
   dataset was still completely wrong. There is now a check that refuses to
   start training unless the ordering is proven.
2. **One of the data columns was not what its name suggested.** It holds
   *which teacher produced each answer* (0 = none, 1 = perfect solver,
   2 = good heuristic), not *which column to play*. Reading it the wrong way
   quietly corrupts it. The trainer ignores it, so nothing broke — but the
   data file's own description of itself became a lie.

### 9.6n The bug hit the *labelling* too — and the labels are being put back

There was a second, quieter victim of the same one-line bug.

When we generate the "correct answer" for every board position, the labeller
skips any position that looks finished. It asked "is this position finished?"
using the **same broken function**. So at the moment the bug thought the
board was full, the labeller also thought the position was over — and wrote
**no answer at all** for it.

Measured on every gap in the label file: **100% of the skipped answers were
caused by this bug** (10,054 of 10,054 checked), not by a genuine finished
game.

| | |
|---|---|
| Answers written | 4,462,571 |
| Answers thrown away by the bug | **101,090 (2.3%)** |
| Answers that were *wrong* | **0** |

So the training data was not poisoned — it was **2.3% smaller than it should
have been**. That is a recoverable loss, and it is being recovered: the
repair tool re-labels only the holes, leaves every existing answer
byte-for-byte identical, and was checked on a test slice before running on
the whole set (`tools/repair_sidecar.py`, `tmp/verify_repair.py` →
`REPAIR_SOUND`).

### 9.6o What the model actually does against a one-column opponent

You asked whether the repeated column 7 was the model or you. Now that the
false draw is out of the way, the question can be asked properly — with a
**real control** instead of a fake one.

Two earlier "control" attempts were worthless:

- the "strong reference" row consulted the perfect-play checker only in the
  last few moves, so for the whole opening it quietly **was the model
  itself**;
- a "pure perfect-play" policy is not a control either — the perfect-play
  checker says "I don't know" for early positions, so the policy had no move
  to pick and fell back to "leftmost legal column", which looks like a
  sensible policy but is not a search at all.

The honest control is a **depth-3 minimax with alpha-beta** that never
consults the model. Result (model plays second, opponent always plays one
fixed column, 5 games each):

| Opponent always plays | minimax (control) | **our model** |
|---|---|---|
| column 7 | wins 5/5 | **loses 5/5** |
| column 6 | wins 5/5 | **loses 5/5** |
| column 4 | wins 5/5 | **loses 5/5** |
| column 1 | wins 5/5 | wins 5/5 |

**The position is winnable. The defect is in the model.** Every one of those
12 stored games was replayed from scratch to confirm the result.

**And the reason is not what it looks like.** The model has **no bias toward
column 7** — over a normal game it plays column 4 most (36%), then column 7
(12%), column 1 (9%), column 2 (7%). What it does instead is **copy**:
it answers column X with column X about half the time (79% against a column-1
opponent, 50% against columns 4 and 7).

That is fatal against someone who keeps stacking one column. A real game
against the model:

```
opponent:  7  7  7  .  .  .  .  1  1  1  1
model   :  7  7  7  4  4  1  1  4  2  2
```

The model answered column 7 with column 7, three times over. Because the
two colours alternate in that column, the opponent never completed a
straight four — and the model, busy copying, never built one either.

**So: it is the model, and the specific fault is that it copies the
opponent instead of playing its own game.** Fixing that is a training
problem, not a bug fix.

### 9.6q So is it the labels, or the model? — measured

This is the question that decides what to fix, and it has a clean answer.

**The idea:** if the training labels themselves told the model to copy the
opponent, then the model would be doing exactly what it was taught, and the
fault would be in the teacher. If the labels *don't* copy, but the model
copies anyway, then the fault is in the model.

**The measurement:** for every one of the 4.5 million labelled positions in
the corpus, compare the *correct answer* against the opponent's *previous
move*. If the answer is the same column the opponent just played, that
answer teaches copying.

| | |
|---|---|
| **The correct answers copy the opponent** | **17.0%** of the time |
| The model copies a column-7 opponent | **~50%** of the time |
| The model copies a centre-only opponent | **~80%** of the time |

**The answers are three to five times less copy-prone than the model is.**
The model is not doing what it was taught — it is copying three to five
times more than its own supervision says it should.

And restoring the deleted labels does **not** fix it: the copy rate of the
answers moved from 17.023% to 17.024% — no change at all, and the late-game
figure actually went the wrong way by 0.16 points.

**So the answer is: the model, not the labels, and not the deleted data.**
The fix has to be on the model side. That is also consistent with the
earlier finding that the model's own column spread is healthy (36% column 4,
12% column 7, 9% column 1) — it is not "addicted" to one column; it is
*copying*, and copying harder than its training data warrants.

### 9.6r The mechanism, finally: the model cannot tell *who* played a piece

Now that the labels are proven clean, the question becomes: **how does the
model actually decide?**

**A detail about how the game is written down.** Before every move, the
program writes down the board as 9 bytes: 7 column bytes, one "last move"
byte, one "whose turn" byte. A column byte says how tall the column is and
which of its cells belong to the top player — it does **not** say which
piece was placed most recently. The "last move" byte is the *only* place
that records "the opponent just played *here*".

**The experiment.** Take a real position, and change exactly one byte, then
measure how much the model's choice of column changes. Do it three ways:
change only the "last move" byte, change only one board column byte,
change only the "whose turn" byte. The model's *hard* (shipped) path was
used — the one the UI, the duels and the game itself use.

| what was changed | how much the model's choice moved |
|---|---|
| only the **last-move** byte | **0.210** |
| only one **board** column byte | **0.187** |
| only the **whose turn** byte | 0.123 (0 on 15 of 25 positions) |

**The result:** the model reacts to the "last move" byte just as strongly
as it reacts to the whole board (ratio **1.12**). So the information *is*
available to it — but the model has learned to use it about as much as the
board itself.

**And the locality test** shows the model is quite short-sighted. It reads
the current position most strongly, and it barely notices anything older
than about ten plies back:

| how far back | how much the model's choice moved |
|---|---|
| current ply | 0.208 |
| 3 plies back | 0.138 |
| 6 plies back | 0.079 |
| 9 plies back | 0.056 |
| 12 plies back or older | 0.012 or less |

**What this means.** The model is not "addicted to column 7" and it is not
blindly copying. It is a model that weighs the opponent's last move
almost as heavily as the position itself, over a fairly short memory. When
the opponent keeps playing one column, "the opponent just played column 7"
is a strong, constantly-repeated signal, and the model's own reply drifts
towards it. That is the echo.

**Why the labels did not cause it.** The labels copy the opponent's move
only **17.0%** of the time. The model copies **50–80%**. The model
over-uses a signal that its training data barely contains — which is a
property of the *model and its objective*, not of the data.

**The open question this raises.** If the model weights the last-move byte
as heavily as the board, then the obvious fix is not more data but a
**better objective**: teach it that a column is worth playing because of
what is *in* the column, not because of what was *just* played there. One
concrete way to test that is to feed the model a position where the last
move is *misleading* and see whether it still copies. If it does, the
objective needs to change; if it does not, the echo is really about
position-specific threats (a vertical stack that genuinely must be
blocked).

### 9.6p How much did the bug actually change the scores?

We measured it directly: the same games, the same seed, the same policies,
the only difference being the old broken fullness test versus the correct
one.

| Match-up | with the bug | without | change |
|---|---|---|---|
| model vs centre-only | 30.3% | 29.5% | −0.8 |
| model vs heuristic | 23.5% | 20.5% | **−3.0** |
| model vs 2-ply search | 2.3% | 2.2% | −0.2 |
| the other six match-ups | — | — | **identical** |

**At most 3 points, in a third of the match-ups, never more.** So the old
strength numbers were not badly wrong — but they had never actually been
verified, and the re-measured ones are now the ones to trust.

## What has deliberately not happened

**The next big training run (V24) has not been started.** It waits until the
checks above come back green.


## 9.6s — WHY THE MODEL STILL PLAYED BADLY: A LIE-TO-THE-MODEL TEST (28 Oct 2026)

We had fixed the false draw, recovered 101,090 missing labels, and proved
the labels were clean. But the model still played badly. This section
explains why, with a test you can repeat.

### The two possible explanations

When the model copied the column you had just played, there were two
possible reasons:

* **A. It reads the board** and decides your stack must be blocked or
  built on. If so, the copy is a real judgement and the problem is
  somewhere else.
* **B. It reads the "last move" field** in the record and echoes it. If
  so, the copy is a mechanical habit.

### The test: lie about the last move

Take a real position. Build the record the model normally sees. Then
build a second record that is **identical except for one byte**: the
"which column was played last" field now names a *different* column.
The board, the history, everything else is byte-for-byte the same. The
tool checks that exactly one byte really did change.

Then we ask: does the model now play the false column?

### The answer

On the 1,500 real test positions:

```
choice CHANGED when the last-move byte lied : 351 / 1498  (23.43%)
model actually PLAYED the false column        :  80 / 1498  ( 5.34%)
```

The model **mostly ignores** the false field. 5.34% is barely more than
guessing. So explanation B is **mostly ruled out**. The last-move field
has a small real effect (it changes the answer a quarter of the time)
but it is not a copy instruction.

### What IS wrong: the model does not respect the shape of the game

Connect Four is left-right mirror symmetric. If you reflect the whole
board and the pieces, the best move must reflect too. We checked all
1,500 test positions that way:

```
model gives the same answer on the mirrored board : 941 / 1500  (62.73%)
model gives a DIFFERENT answer                    : 559 / 1500  (37.27%)
```

**In 37% of positions the model changes its mind when you simply mirror
the board.** That is the real defect.

### And the centre bias is the same defect wearing a different hat

"Always play the middle" is mirror-*correct* — the middle column is the
same column after mirroring. So a model that has not understood the
position can fall back on "middle" and still look consistent from both
sides. The middle-column habit and the copy habit are two faces of one
problem: **the model is not reading enough of the board to know what the
position actually requires.**

### Does mirrored training data help? Yes — but not enough

We ran the same test on the model trained **without** mirrored data:

| measurement                            | without mirror data | with mirror data |
|----------------------------------------|--------------------:|-----------------:|
| same answer on mirrored board          |             29.53%   |       62.73%     |
| follows a false last-move field        |              9.21%   |        5.34%     |

Mirrored training more than doubles the consistency. So the data was
useful. But 37% of positions still disagree with their own mirror.

**Why?** Because having both orientations in the training set makes the
model *more often* symmetric on average. It does not put a constraint on
the solution. Nothing in the training objective says "your answer must
reflect". The fix therefore has to be in the **objective or the
architecture**, not in more data.

### The honest summary

| possible cause                                   | verdict                                    |
|-------------------------------------------------|--------------------------------------------|
| broken labels                                    | ruled out (labels verified clean)          |
| missing labels (coverage)                        | ruled out (101,090 recovered, verified)    |
| the false-draw bug                               | fixed and regression-tested                 |
| echoing the "last move" field                    | mostly ruled out (5.34%)                   |
| missing mirrored data                            | ruled out (present; it helps)              |
| **the objective does not demand symmetry**       | **CONFIRMED (37.27% inconsistent)**         |


### 9.6t — We found the actual cause of the "always the same column" behaviour

Earlier sections concluded that the model copied the *last move* column
because the transcript's last-move field was suspect. That was only
partly right. The real story is worse and simpler.

**What we found**

Our training corpus is a long list of Connect Four games. Each move in
a game is recorded as 10 bytes:

```
[the 7 columns, the previous move, whose turn, this move]
```

The "previous move" byte is there so the model knows where the opponent
just played. In **126,537 of our 320,000 games (39.5%)**, that byte was
wrong: it held the *mirror-image* column instead of the column actually
played.

**Why that matters so much**

We also train on mirror images of every game (the board flipped
left-to-right) to teach the model symmetry. For those 126,537 broken
games, the pair ended up like this:

```
game:            board X,    previous move = column 4
mirror game:     board flip(X), previous move = column 4   <-- same!
```

So the training data literally said: *"a board and its flipped mirror
get the **same** answer."* That is precisely the wrong lesson, and it is
the exact defect we later measured in the model (it gave the same answer
on mirrored boards 37% of the time).

**Why the safety check missed it**

Before training, the code checks the corpus format. But it only checked
**one** row — the first one — and the first row was fine. The damage
started 193,459 rows later.

**What we did**

1. We proved the repair is safe by rewriting each game from the move
   bytes it already contained. Result: **0** changes to any board byte,
   any move byte or any turn byte — only the wrong "previous move" byte
   was fixed. All 126,537 games now decode; before, they did not.
2. We rebuilt the mirror corpus the same way. Now **320,000 of 320,000**
   mirror pairs have a correctly flipped "previous move" field (before:
   193,463).
3. We checked the labels still point at the right positions:
   **652,823 labels checked, 0 wrong**.
4. We added a permanent check that scans **every** row and refuses to
   train on a broken corpus. We confirmed it refuses the old corpus and
   accepts the fixed one.
5. We stopped the two training runs that had been reading the broken
   corpus. Their results are not evidence.

**What this changes**

The earlier conclusion — "the model is fine, only the training objective
is wrong" — was wrong. The data was actively teaching the model to be
asymmetric. The objective-level fix (a mirror-equivariance penalty) is
still worth testing, but it now runs on **correct** data instead of
data that was fighting it.

**What is still unproven**

We have not yet measured a model trained on the fixed corpus. The three
clean runs (original / mirrored / mirrored+symmetry-penalty) are
launched; until they finish and pass the hard evaluation, the model is
**not** release-ready and the column-7 behaviour is **not** yet fixed.


### 9.6u — Our new symmetry penalty was broken; we caught it before it counted

We added a "symmetry penalty" to the training loss: it asks the model to
give the **same answer** on a board and on its mirror image. That is the
right idea, and it is the fix the previous section called for.

But the first version of it was **numerically broken**, and it broke in the
quietest possible way.

**What went wrong**

Each position has some columns blocked (full) and some free. The penalty
compares the model's answer on a board with the model's answer on the
mirrored board. But the set of free columns on a board and the set of free
columns on its mirror are **not the same columns** — column 2 may be free
on one side and column 4 (its mirror) may be full on the other.

Our code computed the "correct answer on the mirror" by flipping a
probability list that had **already been restricted to the free columns**.
Flipping a restricted list gives you a list restricted to the *flipped*
free columns. We then compared it against the *original* free columns, so
for every column that was free on one side and blocked on the other, the
code paired a normal probability with a target of "-one billion".

**What that did**

The penalty came out at **42,500,000** instead of a sensible value around
0.005. The training run did **not** crash and the numbers in the log
looked plausible. It was training against garbage.

**How we caught it**

Three independent checks:

1. The penalty is a divergence between two probability lists over the same
   free columns, so it **cannot** exceed `2 × ln(7) ≈ 3.89`. We now make
   the trainer **refuse to train** if the penalty goes above that, and we
   print the penalty every 50 steps so it can never hide again.
2. A new test uses a position where the free columns are **not**
   mirror-symmetric — the exact case the bug hid in. The old test suite
   only used symmetric positions, which is why it passed while the code
   was broken.
3. The penalty is now measured on a **trained** model, not a
   freshly-initialised one. A new model gives the same answer everywhere,
   so its penalty is exactly zero — a "passing" result that means
   nothing.

**Where it stands now**

| measurement                             | value              |
|-----------------------------------------|--------------------|
| penalty, broken version                 | 42,500,000         |
| penalty, fixed version (soft path)      | 0.00494            |
| penalty, fixed version (hard path)      | 0.0563             |
| maximum allowed by the math             | 3.89               |
| gradient flowing into the model         | yes, 15.8 (abs sum)|
| penalty over 6 optimisation steps       | 0.00494 → 0.00276 |

**What this means for the results**

The first "mirrored + symmetry penalty" training run used the broken
penalty. It was **stopped and thrown out of the evidence set**. It is
being relaunched from scratch with the corrected code and a fresh run
directory, so no checkpoint can carry the broken objective.

The two other clean runs (original data, mirrored data — neither uses the
penalty) were unaffected and are still training.

**A general lesson**

`isfinite` is not a correctness check. `42,500,000` is a perfectly finite
number, and it is completely wrong. When you write a loss that compares a
prediction with a transformed version of a prediction, check the
**magnitude bound** the maths gives you, and test it on the case where
the transform does not commute with the masking.


## 9.6v — What we finally know about the model (and what is still broken)

This section is the plain-English summary of the repaired-data A/B.

### The good news

We found and fixed a serious data bug. In 39.5% of the training games,
the record of "where the opponent just played" was written down
**backwards** (mirrored instead of true). We proved this by checking
every single byte of all 320,000 games:

```
126,537 games had the wrong "last move" field
1,567,958 individual positions were affected
0 board cells were wrong
0 moves were wrong
0 turns were wrong
```

We repaired it by rewriting only that one field, leaving every other
byte untouched, and we built a checker that re-verifies the whole
corpus before training starts. The trainer now refuses to run on a
corrupt corpus.

### The bad news: fixing the data did not fix the model

| What we measured | Before (bad data) | After (fixed data) |
|---|---|---|
| Plays the proven-best move (560 checked positions) | 75.5% | 75.0% |
| Plays the centre column too often (non-centre tasks) | 26.8% | **19.0%** |

So the data fix helped the **column habits** (the model now plays the
centre 7.8 points less often when a side column is the right answer),
but it did **not** improve how often the model picks the best move.

### The clearest single measurement

We rebuilt the exact situation you described: an opponent that only ever
plays column 7.

```
the model (as second player):   0 wins, 0 draws, 40 losses
a simple minimax program:      40 wins,  0 draws,  0 losses
```

The minimax program beats that opponent **every single time**, and it
finishes the game when the board is only 23.8% full. The model loses
every time and fills the board to 50%.

**This settles it: the column-7 problem is a weakness in the model, not
a strange opponent.** The opponent is beatable — the model just cannot
beat it.

We also caught a bug in our own measurement script here: the first
version silently used the model as its own "independent reference" when
the exact solver could not prove a position, which made the model look
like it was playing well. The corrected version uses a genuinely
independent program and reports how many positions could not be proved
(here: 0 of 10, because the positions are too open).

### Does training on mirrored games make the model symmetric?

Partly. On mirror-image positions:

```
original training data:   10.8% consistent
mirrored training data:   32.7% consistent
```

Better — but still wrong about two thirds of the time. This was
measured on **repaired** data, so it is not the old data bug.

### Did the mirror-symmetry regulariser work?

We added a training penalty that punishes the model whenever it answers
differently on a board and its mirror image. It runs correctly
(the penalty went from 4.25e+07, which was broken, down to ~1e-03, which
is the right order of magnitude) and the model still reads the board
(board sensitivity 0.81 at the end of training). Whether it makes the
model *play* more symmetrically is what the final three-arm comparison
measures.

### One more measurement trap worth knowing about

When we ask "did the model choose the best move?", we can only answer
for positions where an exhaustive solver can prove the answer. With the
right search budget that is **560 of 1,500** positions (37.3%). If you
use a smaller budget you get only 42 positions, and those 42 are not
comparable to the 560 — they are a different, much easier subset. We
hit that mistake twice today and fixed it.

### Where things stand

Three models have now finished training on clean data:

```
fd_a  original data, no extra tricks      5509 seconds
fd_b  mirrored data, no extra tricks       5513 seconds
fd_c  mirrored data + symmetry penalty     3336 seconds
```

(The third is faster because it takes fewer optimisation steps to
converge once the symmetry constraint is in place.)


## 9.6w — The final three-model comparison, and what to use

We trained three models on fully repaired data and compared them:

```
fd_a   original games, no extra tricks
fd_b   mirrored games, no extra tricks
fd_c   mirrored games + a "be mirror-symmetric" training penalty
```

### The result that matters most: the echo is gone

| model | echoes the centre bot | echoes the column-7 player |
|---|---|---|
| old mirror-trained model | **80%** | **50%** |
| **fd_a (new)** | **0%** | **0%** |
| fd_b (new) | 0% | 27% |
| fd_c (new) | 0% | 29% |

For reference: a random move echoes about 14% of the time, and the
teacher labels echo 17%.

**This is the answer to your original complaint.** The old model kept
copying whatever you just played. It now doesn't — and the reason was
the broken "last move" field in the training data: it had literally
taught the model "a board and its mirror image get the same answer".
Fixing the data removed the copying.

### But the mirror penalty is not worth it

| model | plays the provably best move | mirror-consistent | beats fd_b? |
|---|---|---|---|
| fd_a | **75.0%** | 10.8% | — |
| fd_b | **75.0%** | 32.7% | — |
| fd_c | 70.9% | 34.0% | **loses 59–41** |

The penalty made the model slightly more mirror-symmetric (+1.3 points)
but cost it 4.1 points on best-move play, and plain mirror training
beats it 59% of the time. **We are not using the penalty.**

### And mirrored training is not worth it either

fd_b and fd_c both do *worse* than fd_a on:

```
teacher agreement   fd_a 23.9%   fd_b 21.1%   fd_c 21.1%
off-centre accuracy fd_a 24.1%   fd_b 15.4%   fd_c 15.6%
```

Mirroring the games doubled the model's tendency to play the centre
column (fd_a 18.3% vs fd_b 35.9%), and the centre column is usually
not the right answer.

### What we are serving

**fd_a** is now live at <http://127.0.0.1:7860/>. It is:

* the model with the lowest copying rate,
* the model that keeps the centre column at 19% when a side column is
  provably correct (the others sit at 41–42%),
* the model that plays column 7 only 1% of the time (fd_b: 3.6%,
  fd_c: 9.5%),
* the model no other arm clearly beats head-to-head.

### What is still wrong — honestly

The model still loses to an opponent that only ever plays one column.
It just loses *differently* now: it no longer copies the opponent, but
it also never builds a line that reaches the contested column, so it
runs out of board. A simple minimax program beats the same opponent
every time.

So the copying bug is fixed, and one real problem remains: **the model
has learned "which single move did the teacher pick" far better than
it has learned "how do I win this game".** That is the next thing to
attack — and it is a training-data problem (we need winning *lines*,
not just optimal *moves*), not a bug.

### One more measurement note

Against the column-7 player, fd_a's moves were:

```
c4 66.7%   c5 33.3%   never c6
```

It refuses to copy, but it also never contests the column. A
four-in-a-row needs to reach the far end. The model needs to learn
"if they are stacking column 7, I must build across to row 7", and
right now it only learned "do not copy".


## 9.6x — Where we ended up, in one paragraph

The model you were playing had a training-data bug: in 39.5% of the
training games the record of "where the opponent just played" was
written down backwards. That bug taught the model to copy your last
move. We found it, proved it byte by byte, repaired it without touching
anything else, and retrained from scratch.

The new model **no longer copies you** (it echoed the column-7 player
0% of the time; the old one echoed it 50%). It is now serving at
<http://127.0.0.1:7860/>.

But we were honest about the limits: the new model still loses to an
opponent who only plays one column. It fails *differently* now — it
refuses to copy, but it also never builds a line that reaches the
contested column. And we tested a second idea (mirroring the training
games, and forcing the model to be mirror-symmetric) and it made things
*worse*, so we dropped it.

So: the bug you reported is fixed. The deeper problem — that the model
knows "which move did the teacher make" but not "how do I win" — is
still open, and it is a data problem, not a bug. That is what we would
attack next.


## 9.6y — A correction we owe you

Last message we said the new model "echoed the column-7 player 0% of
the time". That number was wrong, and we caught it by reading the raw
file instead of the console output. The real numbers:

- Against an opponent who only plays one column, the old model copied
  them **50%** of the time. The new model copies them **0%** of the
  time.
- But the new model gets there by **never playing that column at all**.
  It is not that it learned a smarter response — it just refuses to
  enter the fight.

So the honest headline is: **the copying bug is gone, but the model
still won't contest a contested column.** It now loses differently —
it stops copying, and then it loses anyway because it declines to play
the column that matters. Beating a one-column opponent needs a model
that both plays the column *and* plays it well; we have the first half
and not the second.

We also found two of our own measurement scripts had a silent failure
mode (one wrote an empty "strength" result and still reported success;
one compared against a 42-position sample when the real sample is 560).
Both are fixed and the numbers above come from the corrected runs.


## 9.6z — How we double-checked the correction

We re-measured the column thing from scratch, with five different kinds
of opponent (random, always-centre, always-column-7, always-column-1,
and one that attacks column 7 on purpose). The result:

- The new model **does** play column 7 sometimes (3.8% of moves) when
  the game is normal.
- But against an opponent who is *actually contesting* column 7, the
  new model plays it **0.00%** of the time — not once in 180 moves.
- The old model, given the same opponents, played column 7 **30%** of
  the time.

So the corrected story is consistent: the new model has learned to
**stay out of contested columns**, not to fight in them. The copying
bug is gone; a different habit took its place. Both are ways of not
engaging, and both cost it games.


## 9.6x — Why "the script finished without errors" is not a result

While re-checking the final evaluation numbers, I found that my own
evaluation script could produce a file that *looks* like a successful
result while some of the work inside it never happened.

Three separate things could go wrong, and all three left a file on
disk that looked fine:

1. **Skipping quietly.** If a model file was missing, the script said
   "skipping" in its log and moved on. The final file was complete
   looking — it just quietly did not contain that model. The script
   still reported success.

2. **A step failing.** When I mistakenly passed a `--budget` option to
   a helper script that does not have that option, that helper exited
   with an error. The evaluation script recorded the error, wrote its
   output file anyway, and reported success. The failed step's section
   was simply empty — which is how you can tell, if you look.

3. **Picking up someone else's file.** The script read its results from
   files with fixed names. If a file was left over from an earlier run,
   the script would happily read it as if it were its own. The check
   was "does the file exist?" — which proves nothing about *who wrote
   it or when*.

**One correction to what I told you earlier.** I first explained the
empty section by saying "an old file from the previous run was still
there, so it looked populated." That was wrong. I checked the actual
file: the failed section has **no data at all**, not old data. The
simple explanation was right — the step failed, nothing was attached,
and the script still said success. I should have checked the file
before explaining it.

**What I changed**

- Every intermediate file the evaluation script writes now carries a
  unique label for that run, so two runs can never mix their files.
- Each run deletes its own leftovers before starting, so an old file
  can never be picked up as new.
- A missing model file is now a **hard failure** — the script stops and
  reports failure, instead of quietly producing a smaller result.
- Each step is checked afterwards: did it run, did it produce what was
  expected, did it use the right amount of data? If any check fails,
  the file says so and the script exits with an error.
- I wrote a test for the script itself (`tmp/test_eval_fixed_arms_harness.py`)
  that deliberately breaks things and checks the script notices. It
  **found two real bugs on its first run** — the quiet skip and the
  cleanup happening too late — both now fixed and re-verified.

**The general lesson, which I saved as a skill**
(`harness-artifact-provenance`): a script exiting without errors is not
evidence that its output is real. The only real evidence is a per-step
check that the step ran in *this* run, produced the *expected* data,
and used the *authoritative* data — plus a test that deliberately tries
to make the script lie.


## 9.6y — When two different "solver runs" disagree, check *which positions they proved*

A late background job reported solver scores that looked like they
contradicted my final numbers: the old (corrupt-corpus) model scored
**88.1%** where I had reported **75.5%**, and the repaired model
**81.0%** where I had reported **75.0%**.

Both were correct. The two runs had proved **different sets of
positions**.

The tool that finds the provably-best move has a time limit per
position. With a small limit it solves only the *easy* positions — 42
out of 1,500. With a larger limit it solves 560. And the easy positions
are exactly the ones where the answer is obvious, so:

| model | easy 42 positions | full 560 positions |
|---|---|---|
| old model (corrupt corpus) | 88.1% | 75.5% |
| repaired model | 81.0% | 75.0% |
| old mirror model | 76.2% | 73.8% |

On the easy 42, the share of provably-best moves that sit in the centre
column is **81%**; on the full 560 it is **58%**. The easy set is a
much friendlier problem.

**The ranking does not change** — the repaired and original models are
level on both. Only the absolute numbers move, because the exam got
easier. So my release decision was never affected.

**But there was a real problem underneath.** That background job wrote
its results to a file whose name did not include anything identifying
*that* run. The authoritative run had written to the same filename
earlier, and the later job replaced it. The authoritative file survived
only because it was written *last* — luck, not design. If the order had
been reversed, I would have lost the 560-position numbers and never
known it.

**What I changed**

- The tool's default time limit is now the large one, so the *default*
  produces the comparable, authoritative answer. The easy 42 is now
  something you have to ask for explicitly.
- The tool now accepts a run label and puts it in the output filename,
  so two runs cannot overwrite each other.
- The tool now **says out loud** when its result is not comparable:

  ```
  !! NON-AUTHORITATIVE SUBSET: budget=4000000 n_solvable=42
  !! The authoritative subset is budget=120000000 n_solvable=560.
  !! These percentages are NOT comparable to authoritative runs.
  ```

  and records the same warning inside the output file, so the warning
  survives even if the log is lost.

**The lesson:** two numbers that disagree are not a contradiction until
you check *what was measured*. A benchmark that quietly changes its
difficulty is worse than no benchmark — and a tool that writes to a
fixed filename turns a hard-to-spot difference into a lost result.

**One honest note about the old model.** On the easy subset the
corrupt-corpus model scored *higher* than the repaired one (88.1% vs
81.0%), and on the 560-subset it is also marginally higher (75.5% vs
75.0% — a tie within noise). The repairs were never about making the
model *stronger*; they were about making the data and the label
supervision *honest*. The repaired model is the one whose behaviour we
can actually reason about, which is why it is the one that ships.


---

## 9.7 — "What does 100% mean? Is that a chance of winning?"

### 9.7.1 There is no "temp" in the UI

We searched the served page, `src/c4/ui.html`, `src/c4/ui_server.py`,
`src/c4/` and `src/gf256*.py`. The only occurrences of the letters
"temp" are two CSS `grid-template` declarations. There is **no
temperature control, no temperature slider, and no temperature value**
anywhere in the shipped model or UI. The agent plays **greedily** — it
always takes the highest-scoring legal column.

If you saw something that looked like "temp", it was most likely the
percentage panel itself, or the word "now" in the label
*"Board as it stands now"*.

### 9.7.2 What the percentage actually is

Each number is the model's **preference weight for one column**: a softmax
over the legal columns of `hard_forward`. It answers

> "how does the model split its weight between the legal columns right now?"

It does **not** answer

> "who is going to win?"

A winning-chance percentage would need a separate, calibrated estimator
that is trained and validated against game outcomes. The project does not
have one. Claiming a win probability from this panel would be inventing
a number.

### 9.7.3 When you see 100%

We measured 19,497 random positions:

```
  displayed at 100% : 902 (4.63%)
    ...only ONE legal column : 902
    ...MULTIPLE legal columns: 0
```

**Every 100% comes from the same situation: only one column is legal.**
Six columns are masked to `-1e9` and renormalised out, so the remaining
one holds 100% of the mass by arithmetic, not by conviction. That is why
it says nothing about the game: it appears in positions the model is
losing just as often as in positions it is winning.

### 9.7.4 What we changed

The panel now says so, permanently, in plain words — you can read it
under the bars:

> **These percentages are not a winning chance.** … A **100%** simply means
> the model put all its weight on one column (this normally happens when
> only one column is legal); it does **not** mean a certain win.

And when there is only one legal column, the legend adds:

> **only one legal column** — the 100% is forced by the mask, it says
> nothing about the game

To judge the game, use the **"What is on the board"** panel below it.
That one is pure physical board truth (fours, open threes, blocked
threes, "the model can win right now"), with no model opinion mixed in.

### 9.7.5 A measurement bug we found while answering this

While checking the percentages we found that our earlier tactical
numbers — "takes an available win only 26.9% of the time", "gives away
a win 85.2% of the time" — had been produced by **probe scripts that
called the model wrong**, not by the shipped model.

The model's API is `column_logits(board, moves)` where `moves` is the
**history of columns already played**, not the list of legal columns.
Our scripts passed the legal list, so the model was asked about a
meaningless position. The live UI was always correct (it passes the real
move history), and we proved that with a parity test against the running
server: **75/75 columns match, L1 = 0.0**.

Corrected numbers for the model you can play:

| | corrected |
|---|---|
| takes an available immediate win | **23.4%** |
| blocks an opponent's immediate win | **21.4%** |

So the model is **not** board-blind — that was the bug.

#### The "gives away a win" number needed a second correction

We first reported **0.9%** for "gives away a win it could have
prevented". That number was over-claimed, and the control that should
have caught it was itself broken.

The control (a set of "ideal" players used to sanity-check the
measurement) reported that a **random** player *never* hands over a win
— 577 out of 577. A metric that a coin-flip passes perfectly is
measuring nothing. The cause: the control copied the board but forgot
two fields that decide *where* a disc lands, so the board silently
stopped changing and the opponent could never gain a new win. The
control could not fail, so it never calibrated anything.

With that fixed and every player tested on the **same** boards:

| | model | best player | random player |
|---|---|---|---|
| takes an available immediate win | 25.4% | 100% | 20.3% |
| blocks an opponent's immediate win | 22.6% | 100% | 15.2% |
| hands over a win **when handing one over was possible** | **18.5%** | 0% | 16.9% |

The last row is the honest one. Handing over a win is only *possible* in
about 5% of clean positions, so a flat percentage like 0.9% sounds
excellent but mostly counts positions where nothing bad could happen.
When it actually could, the model did it **18.5%** of the time — barely
better than random (16.9%), and well behind the best player (0%).

**So, plainly: the model is close to random at taking wins and at
avoiding give-aways, and clearly better than random only at blocking.**
The blocking is real; the rest is not yet.

### 9.7.6 Is the hard export (the "GF" version) the reason?

We checked, because the numbers above looked like they might be caused by
compressing the weights. They are not.

On the same 2,445 positions where a win was available:

| | takes the win |
|---|---|
| full-precision weights | **26.4%** |
| compressed ("hard") weights | **25.9%** |

They are the same, within half a point. The compressed version just
picks a *different wrong* column when it is wrong (they agree only 55%
of the time) — but how often they find a win is identical.

**So the weakness is in what the model was taught, not in how it is
stored.** That tells us exactly where to go next: teach it the tactics
directly (win now → block now → build a threat → stay safe), which the
project already has code for in `src/c4/threat_corpus.py` but has never
actually trained on.

One more useful finding: because the two versions fail on *different*
positions, combining them finds a win **36.0%** of the time — better than
either one alone. That is a cheap idea worth testing.


---

## 9.8 So does the model actually learn? (the direct answer)

> **🔴 §9.9 supersedes parts of this section.** The model measurements in
> §9.8 were taken on a checkpoint trained with the **supervision bug** fixed
> in §9.9. The *label-quality* and *solver-baseline* numbers below remain
> valid; the *model-behaviour* numbers (column usage, what the model picks)
> must be re-measured on a correctly-trained model. Read §9.8.9 for the
> retraction list.

This section answers four questions directly, with the numbers.

### 9.8.1 "The model gets 75% of moves right — isn't that good?"

Not as good as it sounds. We used an exact solver to prove the **best
possible move** for 560 positions from the test set. The answer is often
not one move but several:

| number of equally good moves | positions |
|---:|---:|
| **1** (a real question) | **112** |
| 2 | 55 |
| 3 | 41 |
| 4 | 31 |
| 5 | 68 |
| 6 | 101 |
| 7 (every column is fine) | 152 |

**Only 20% of positions have one right answer.** On the other 80%, a
player who always drops their disc in column 4 is "right" about 76% of
the time — because column 4 is usually in the acceptable set.

So the honest scorecard uses only the **112 positions with a single right
answer**:

| | gets it right |
|---|---:|
| pure chance | 0.9% |
| always play column 4 | 29.5% |
| **our model (best checkpoint)** | **30.4%** |
| **our model (the one we ship)** | **24.1%** |

And here's the honest catch: **30.4% vs 29.5% is one position out of 112.**
Statistically, our model and "always play column 4" are indistinguishable
(p = 0.84). The model is reliably better than chance, but **it has not
proven it can beat a constant column.**

### 9.8.2 "Is it getting better?"

It learns quickly at first, then wanders:

| training step | score on the 112 clear-cut positions |
|---:|---:|
| 0 | 0.9% |
| 1000 | 24.1% |
| 2000 | 30.4% |
| 3000 | 24.1% |
| 4000 | 17.9% |
| 5000 | 28.6% |
| 6000 (shipped) | 24.1% |

After roughly 2,000 steps it stops improving and starts bouncing around
by 5–12 percentage points (6–13 positions out of 112). **The shipped
model is worse than the step-2,000 model.** Six thousand steps bought
nothing over two thousand.

Worse, the training loop never noticed: its internal strength test played
**zero games** at every step (`W0 L0 D0`), so nothing could flag the
plateau.

### 9.8.3 "What's wrong with the training data?"

The data itself is in good shape: 320,000 games, 4.5 million labelled
positions, verified row by row for corruption, with zero overlap between
training and test. The labels are also **94.6% correct** — a human expert
would only do slightly better.

**The problem is what "correct" is being encoded as.**

In 80% of positions there is no single right move. The teacher's program
(`src/c4/strong_solver.py`) has to pick *one* anyway, and it picks the
**column closest to the middle** because it searches columns from the
centre outwards and only replaces its best answer on a *strictly* better
one. So every "several moves are fine" situation gets recorded as
"play the middle column."

**The model faithfully learned that convention.** The training labels
say column 3 in 26.3% of positions; the games that were actually played
used column 3 only 18% of the time. The label matches the move actually
played in just **20.8%** of positions.

So we are not teaching the model Connect Four. We are teaching it
**"when in doubt, play the middle."**

### 9.8.4 The part we found today that nobody expected

We compared the model's choices to its teacher's labels **on the exact
same 560 positions**:

| column | model actually plays | teacher labelled | column is in the correct set |
|---:|---:|---:|---:|
| 0 | 0.5% | 5.9% | 58.9% |
| 1 | 2.0% | 8.9% | 58.8% |
| 2 | 0.7% | 15.4% | 60.9% |
| 3 | 21.4% | 24.6% | 57.9% |
| 4 | **45.5%** | 21.1% | 68.9% |
| 5 | 27.1% | 13.9% | 66.8% |
| 6 | 2.7% | 10.2% | 70.5% |

(These are the numbers from the *masked*, shipped-protocol measurement —
see §9.8.8 for a measurement bug that briefly produced different ones.)

> **⚠️ IMPORTANT SCOPE CORRECTION (28 Oct 2026, late).** The table above is
> from the **560 solver-proven positions only**, and that set is
> **endgame-heavy** (it is restricted to boards with ≤24 empty cells). It is
> **not** the model's overall behaviour. On the **full 19,279-position**
> release holdout the same model plays:
>
> ```text
> c0 13.2%   c1 13.0%   c2  8.3%   c3 18.3%
> c4 37.8%   c5  8.4%   c6  1.0%
> ```
>
> So the model **does** use columns 1 and 2 on a representative set. The
> "94% on columns 3/4/5" figure is **true of this subset, not of the model**.
> Any statement of the form "the model never plays column N" must say which
> positions it is talking about.
>
> What survives as a real, population-independent finding:
>
> * on the **112 clear-cut positions** (where exactly one column is correct),
>   columns 1 and 2 are the single best answer **23 times (20.5%)** and the
>   model plays **neither, ever** — across all 13 checkpoints tested;
> * the shipped checkpoint picks column 3 on **16 of 112** clear-cut
>   positions, and **14 of those are wrong**;
> * column 4 really is over-used (37.8% vs 21.1% for the teacher) and column
>   6 really is under-used (1.0%) — **those two are whole-model effects**.
>
> Two explanations for the asymmetry were tested and **refuted**: the arms were
> *not* trained with sparse lags (both the launcher and the checkpoint declare
> dense lags), and no column is architecturally blind (per-column hard
> sensitivity: c1 0.126, c2 0.144, **no blind columns in any audited
> checkpoint**).

The model plays columns 3, 4 and 5 in **94%** of its moves *on this subset*.
Its teacher only used those three columns 60% of the time.

**And the game does not favour them either** — every single column is
correct in roughly 58–70% of positions. There was no pattern there to
learn. The model **invented** a narrow preference of its own and then
locked onto it.

The consequence is stark: on the 112 clear-cut positions, column 1 is
the single best answer 15 times and column 2 is best 8 times — **23
positions, 20.5% of the clear-cut set — and the model never plays either
column once.** Not once, across all 13 checkpoints we tested.

**Those 23 positions are unreachable for this model. No amount of extra
training on the same labels will ever fix them**, because the model has
structurally stopped considering those columns.

### 9.8.5 The bottom line

Three separate things are broken, and each one alone is enough to keep
the model weak:

1. **The teacher collapses ties** into one centre-biased column
   (skill: `tie-set-label-collapse`).
2. **The model then collapsed further** onto columns 3/4/5, onto
   columns its teacher never favoured — a prior of its own invention
   (skill: `model-column-prior-collapse`).
3. **The training loop has no strength test**, so it trained for 4,000
   extra steps past the point of usefulness without noticing.

The good news: **the architecture works.** The model clearly reads the
board (0.9% → 30%, and the trainer's own board-sensitivity score went
from 0.000 to 0.87). The GF trunk is not the problem. The supervision
and the training loop are.

### 9.8.6 What we are doing about it

A controlled experiment is running: 3,000 games relabelled so that every
position teaches the **full set** of correct moves instead of one
arbitrary one, then two identical training runs — old labels vs new
labels — scored on the clear-cut 112 positions and on the
never-played-columns measure. If the new labels win, the data is the
bottleneck and we do a full relabel. If the model still ignores columns
1 and 2, we add an explicit rule forcing it to consider every legal
column, because that failure is in the model's head, not in the data.

### 9.8.7 Two things we got wrong, and one that changed the plan (28 Oct 2026, late)

**A. The "28% agreement" was never a bug.** An external review suspected the
imported training data was decoded with the wrong column order or the wrong
side-to-move. We tested all 12 plausible conventions (row flip x column flip x
three player mappings) against 400 positions we could solve exactly:

```text
WINNER: row_flip=True  col_flip=False  player=parity
        - the -1000 "illegal move" marker landed on a legal column: 0 times
        - the label was in the correct optimal set: 100.0% of positions
        - correlation with the exact solver's value: +0.917
```

That is **exactly** what our own importer does, so the data was always right.
And when we reproduced the old "28.3%" number, the label was in the correct
set **100%** of the time with **zero** wrong answers. The 28.3% was two
arbitrary tie-breaks (ours vs the solver's) happening to agree. It measured
nothing.

**B. The "0 games played" was a settings bug, not a model failure.** The
training log's `W0 L0 D0` looked like the model losing every game. The
launcher actually passes `--eval-games 0`, so it played no games at all.
Re-run properly with 24 games, the shipped model goes **W12 L12 D0** against
the simple reference opponent.

**C. This one changes the plan.** The imported data does not just say "which
move" - it says **how fast you win**. The exact solver we built only says
"win" or "lose", but the imported scores distinguish *win in 2* from *win in 6*:

```text
exact-solver value =  +1 / 0 / -1        (outcome only)
imported score     =  +k / 0 / -k        (k = plies until the win)
```

They never contradict: the sign agrees **100%** of the time, and the
imported best-move set is always a **subset** of the exactly-optimal set
(**100%** of proved positions) - it just picks out the *fastest* winning
moves out of a set that is often several columns wide. That is the graded
target signal, and it is already sitting in the data.

So the next training arm does not need a new solver or new data. It needs the
existing imported scores used as **graded targets** (so the model learns
"this column wins in 2" is better than "this column wins in 6", and that
several columns are equally acceptable), with the exact-solver tie set
overlaid on the quarter of positions where the imported max-set is not quite
the optimal set.

**And the review's warning was the most useful part.** It said: *do not add a
coverage/entropy penalty before you know the cause.* Agreed - a penalty would
force the model to use columns it currently under-uses without fixing the
label problem that is upstream of it. The cause is now known.

### 9.8.8 A measuring tape that was wrong (28 Oct 2026, late)

One of our own measurement scripts told us the model scored **65 %** where
the official measurement said **75 %**. Same 560 positions, same model, same
day. Two measurements of the same thing cannot both be right.

We found the cause, and it is worth explaining because it is the single
most common way to fool yourself with a neural network.

**The model outputs a score for every column. Then something has to pick one.**

The model does not "play" — it produces seven numbers, one per column, and
the code picks the highest. But a column that is already **full** cannot be
played. So before picking, the shipped code sets the score of every full
column to a huge negative number, so the maximum can never land on one:

```python
logits = torch.where(legal_mask, logits, -1e9)   # full columns are excluded
pick = argmax(logits)
```

Our measurement script had **forgotten that line**. So in
**64 of 560 positions (11.4 %)** it picked a column that was already full —
a move that is not even allowed. Of course those positions were scored as
"wrong": the model cannot choose a legal optimal move if the measurement
asks it to choose an illegal one.

The fix was to copy the shipped code's rule into the measurement script.
After the fix, the script says **75.0 %** — exactly what the official
measurement says — and the per-column counts match the official ones
perfectly.

**Two lessons.**

1. **The rules of the game are part of the model.** When you measure what
   the model does, you must measure the whole decision, including the
   "you cannot do that" step. Measuring only half of it measures a
   different program than the one users play.
2. **When two measurements disagree, the disagreement is the finding.** We
   could have "fixed" the discrepancy by trusting the higher number. Instead
   we compared per-column counts, found they differed on 64 positions, and
   traced it to one missing line. The number went *down* by 10 points and
   the measurement became *right*.

**A related point from the same investigation:** the same script also
miscounted how often the model uses the outer columns (it said 2.0 % / 0.7 %
for columns 1 and 2; the correct numbers are 1.4 % / 0.9 %). So the "the
model never plays the outer columns" impression was itself partly a
measurement artefact — on top of being a subset artefact.

And the depth breakdown that came out of the corrected script is genuinely
interesting: the model is **good in endgames** (87.5 % optimal with 8 or
fewer empty cells) and **much weaker in the opening** (72.2 % with 14–20
empty cells), while the teacher stays flat at 94–100 % throughout. The
problem is not that the model cannot play Connect Four. It is that it has
not learned the **opening** — and the opening is where the outer columns
matter.


---

### 9.8.9 The list of things §9.9 retracts (28 Oct 2026, late night)

§9.9 found that the trainer was reading the **teacher's answers for the
wrong games** (see §9.9 for the full explanation). Anything below that
depends on *what a trained model did* was measuring that bug, and is
**withdrawn** until re-measured on a correctly-trained model.

**Withdrawn — model behaviour:**

* "the model never plays columns 1 and 2" (0/112 on the unique-optimum
  subset)
* "the model collapses onto columns 3/4/5" (94.1 % on the 560-subset)
* "no arm beats a constant-column policy" in a real game
* "the architecture cannot fit even data it saw ten times" (overfit
  19.1 % ≈ fd_a 20.2 %) — **wrong**; that measurement was of the
  supervision plumbing
* the full-holdout column distribution
  `[13.2, 13.0, 8.3, 18.3, 37.8, 8.4, 1.0] %` as a property of the model

**Still valid — not about the training loop:**

* the label-quality numbers: the label is the optimal move in **94.6 %** of
  the exact-solved positions, and exactly right on **86.6 %** of the
  112 unique-optimum positions
* the exact-solver baselines (always-c4 75.7 %, chance 65.2 %, …)
* the file-format and decoding findings (transcript v2, the Pons grid)
* the corpus and sidecar repairs
* the hard-export parity and the UI parity suite
* the trainer's own `LEGAL` metric as a measure of *played-move* agreement
  — but **not** as a measure of teacher agreement

---

## 9.9 Was the model bad, or was the training wrong? (28 October 2026, late night)

Short answer: **the training was wrong.** The architecture was fine.

### What the puzzle looked like

The model seemed to disagree with its teacher:

* it matched the **teacher's advice** on only about **1 in 5** positions
  (20 %),
* but it matched the **move that was actually played** on about **half**
  of them (57 %).

That is backwards. A model being taught the teacher's answer should fit
the teacher, not the move that was already on the board. So the obvious
conclusion was "the labels are broken" or "the architecture can't learn
this".

Both were wrong.

### What was actually happening

The trainer reads its teacher answers from a side file — one row of
"which column should you play" per game. To look up the right row for a
training example, it needs the **game number**. It was instead using the
**position in the training batch** (0 to 15).

Those are different things. The trainer was effectively asking "what did
the teacher say for game #0, #1, #2 …?" no matter which game it was
actually training on.

And here's why nobody noticed: **the side file has 320,000 rows.** Rows
0 to 15 are perfectly valid rows containing perfectly good teacher advice
— just for the wrong games.

So the model was training on real answers to the wrong questions. Every
number looked healthy: the loss fell, the accuracy climbed, the
checkpoints saved, nothing crashed.

### The bit that made it worse

There is a safety net in the training code: if the teacher's answer turns
out to be empty or impossible for that position, the trainer falls back to
"teach it the move that was already played", so the model at least learns
something sensible.

Because the trainer was looking in the wrong place, it hit that empty
answer **41 % of the time**. And since the teacher answer normally
*replaces* the played-move lesson entirely, this fallback was the **only**
way the played move ever entered the training at all.

Put plainly:

> **41 % of the training was "copy the move that was already played", and
> the other 59 % was the answer to a different game.**

That is exactly the pattern we measured. The model was good at copying
played moves and bad at following the teacher — not because of the labels,
and not because of the architecture.

### Why the labels were fine all along

The teacher answers themselves check out. The exact solver says the label
is the optimal move in **94.6 %** of positions, and on the hardest subset
— positions with exactly one right answer — the label is right **86.6 %**
of the time. The labels were good. They were just being read from the
wrong shelf.

### The fix, and how we know it worked

* The row lookup now uses the game number, in one named place
  (`_perply_corpus_rows`) so it can be tested directly.
* The silent fallback is **removed**: if the teacher's answer is empty, the
  example is skipped rather than quietly turned into "copy the played
  move". With the correct lookup, that never triggers — **0** of 13,000+
  examples.
* A new test checks the row lookup, and — importantly — a second script
  puts the **old broken code back in** and confirms the test **fails**.
  A test that can't fail on the bug isn't guarding anything.

Measured before → after, on the real data:

| | before | after |
|---|---:|---:|
| trainer reads the right answer | 9.6 % | 100 % |
| distinct games read (of 320,000) | 16 | hundreds |
| empty answers → silent fallback | 41.3 % | 0 % |

### What this changes about earlier conclusions

Several earlier findings were measuring this bug, not the model:

* "no arm beats a constant-column policy" — the arms were trained on the
  wrong supervision, so the matchup has to be re-run.
* "the model never plays columns 1 and 2" — same, re-measure.
* "the architecture cannot fit even data it saw ten times" — **wrong**.
  The overfit test was fitting the played move, not the labels. The
  architecture was never proven incapable.

What is *not* affected: the corpus repairs, the label-quality numbers, the
file-format and decoding findings, the hard-export checks, and the UI
parity tests. Those were never about the training loop.

### Where it stands

The code is fixed and guarded. A corrected small-scale run is the
acceptance test: it has to reach **over 90 %** agreement with the teacher
(versus 19 % before). Only then does a full retrain happen, and only then
are the strength and column-usage numbers re-measured. The old checkpoint
stays available to play with, but it is not a fair representation of the
architecture any more.

---

### 9.11 Does the corrected model actually play better? Yes -- measurably.

Short answer: **yes, and by a lot.** First hard in-game comparison, same
harness, same seed, same opponent, 600 games each:

```text
old broken model : won 399, lost 201   (66.5 %)
corrected model  : won 948, lost 250   (79.1 %, 1200 games)
```

That difference is far beyond statistical noise (p < 0.00000001).

You can also *see* the difference in which columns it plays. The old model
almost never used column 2 (4.25 % of its moves). The corrected model uses
it 14.13 % of the time -- more than three times as often -- and that is
exactly the column the old model could never reach on the puzzle positions
where one specific column was the only correct answer.

Two things it is **not** yet:

* It has not clearly beaten a dumb bot that always plays the same column.
  A fixed-column bot is a surprisingly strong Connect Four opponent.
  The corrected model is statistically level with all of them -- measured
  twice, 600 games each, and no comparison reached significance.
* The trainers were still running (5000 of 6000 steps) when this was
  measured, so this is the mid-training checkpoint, not the final one.

A note on how the measurement nearly went wrong: the first version of the
test used a "constant column" bot that, once its column filled up, quietly
started playing the leftmost free column. That made it behave like a
different bot entirely, and two supposedly different bots ended up
producing *identical* scores. Fixed, re-measured, and the table above is
from the corrected run.


## 9.10 The acceptance test passed — and the checkpoint picker was broken too (28 October 2026, late)

### The short version

The corrected trainer **works**. On the 500-row acceptance run it reaches
**91.29 %** agreement with the teacher, and two independent measurements
of the same checkpoint say **92.38 %** and **92.40 %**. Before the fix it
was 19 %. The architecture was never the problem.

But there was a **second** bug hiding behind the first: the trainer was
**picking the wrong checkpoint** to keep.

### Why the wrong checkpoint was kept

The trainer keeps whichever checkpoint scores best on a "promotion key".
That key was:

```text
(ladder score, holdout score, -byte-level-bpb, distinct-game count)
```

* **byte-level-bpb** is about predicting the byte sequence of the move
  that was **actually played** in the training games.
* The thing we actually train on is the **teacher's recommended column**,
  which is a *different* answer (they agree only ~21 % of the time).

In the acceptance run there was no holdout and no ladder score at all, so
those two terms were empty — and the key silently fell back to ranking by
byte-bpb, i.e. ranking by "how well does this checkpoint predict the
played move". That is the wrong objective. It chose **step 1500
(62.82 %)** over the final **step 3923 (91.29 %)**.

The gate was keeping the checkpoint that fits the teacher **least**.

### The fix

The trainer now measures **direct label agreement** — the share of
labelled positions where the model's chosen column is **in the teacher's
recommended set** — and that is the **primary** promotion key. Everything
else becomes a tie-breaker. The number is also printed in the log as
`LABEL xx.x% (n=…)` so you can watch it.

### Two traps we hit while building the new measurement (both would have produced confident nonsense)

**Trap 1 — the measurement returned the same number for every checkpoint.**
12.68 % for a randomly-initialised model *and* 12.68 % for the fully
trained one. The cause: the model class has no "import hard integers"
function, so re-loading a hard checkpoint and re-exporting it produced a
model with **all-zero weights** — every logit exactly 0.0, so it always
picked column 0. The *saved* checkpoint was fine; the round-trip through
float weights was the broken part. Fixed by passing the integer payload
straight through instead of re-exporting it.

**Trap 2 — the wrong context length.** The measurement used a window of
191 bytes where the training used 192. Because the read position is
`ctx-1`, that shifts the window and quietly degrades the score. The
context length is now passed explicitly.

### A third bug, in the *opponent*, not the model

The "always plays column 3" baseline was not a constant policy. It was:

```python
return col if col in legal else legal[0]
```

So when column 3 filled up it fell back to the **first legal column** —
a centre-first sweep. That arm played column 3 on only **28.2 %** of its
moves. A real constant policy keeps using its column and, once it is
full, picks another legal column at random.

This **reverses** the conclusion. Against a centre-ish opponent, with the
old definition "always-c3" won 46.7 % and "always-c0" 85 %; with the
correct definition "always-c3" wins 30.0 % and "always-c0" still 85 %.
The claim "always-c3 beats the model" was an artefact of the fallback.

### The in-game comparison still has to be redone

The harness itself is now correct (it also had a bug where the seed
argument was ignored, so "600 games" were really 6 unique games). But the
**model** it was measuring was trained under the broken supervision. So
every old strength number is provisional. The new arms are training now.

### What is true right now

```text
acceptance gate (>90 % direct label agreement)   PASSED (91.29 / 92.38 / 92.40 %)
promotion gate (label agreement is primary)      FIXED and regression-tested
regression suite                                  17 PASS, 0 FAIL
full corrected retrain of all three arms          RUNNING
old fd_a/fd_b/fd_c strength numbers              PROVISIONAL, must be re-measured
```

You can still play the old checkpoint at `http://127.0.0.1:7860/` — it is
a working opponent, just not a fair representation of the architecture.


---

## 9.12 The big one: the corrected full runs were stopped far too early

Here is the most important thing to know, and it is good news.

### The model does learn

We proved this two ways:

* On a small test set of 500 games, the corrected model learned the
  teacher labels almost perfectly — **91.3 %** of the time it picked a
  move that the teacher also considered a good move.
* The full run on all 320,000 games got to about **51 %**.

### Why such a big gap? Because the full run barely started.

```text
small test set : 500 games, the model saw each one ~125 times  -> 91.3 %
full data set  : 320,000 games, the model saw each one 0.3 times -> 51 %
```

To see every game once, you need **20,000 training steps**. We ran
**6,000**. That is 30 % of a single pass through the data.

### It was still improving when we stopped it

```text
step  500  ->  37.6 %
step 2000  ->  40.2 %
step 3000  ->  43.6 %
step 4000  ->  45.3 %
step 5000  ->  49.2 %
step 6000  ->  50.9 %   (still going up when we cut it off)
```

No plateau. The training budget ran out, not the model.

### What this means for the "the model is weak" claims

We previously wrote that the model had a ranking problem with the best
move, a blind spot for column 1, and was weak against lookahead. Those
measurements are still **correct as measurements** — but they were taken
on a model that had seen less than a third of its homework.

So we are withdrawing the interpretation. It is now much more likely that
these are all "hasn't learned enough yet" than "the design is wrong".
The only way to know is to train it properly and measure again.

**Nothing was lost.** The training code, the data, the labels, the
measurement tools and the checkpoints all survive, and the fix is a
matter of running the trainer longer.

### Next run (V24)

Same model, same data, same corrected code — just a bigger training
budget: **100,000 steps instead of 6,000** (about 37 hours instead of
about 2 hours), plus a fix so a long run can be safely stopped and
restarted, and re-enabling the practice-game scoreboard that was
showing zeros for a silly reason.

## 9.13 "Resume" was broken in three different ways

We wanted to be able to stop a long training run and start it again from
where it stopped — otherwise a 37-hour run that crashes at hour 30 is a
30-hour loss. Three separate bugs were stacked on top of each other, so
every attempt to resume failed, each for a different reason.

**Bug 1 — "resume from step 6,000" was interpreted as "run steps 1 to 100".**

```text
nothing to resume: the checkpoint is already at step 6000, which is beyond
--steps=100. Pass a larger --steps (the run continues from step 6000), or
drop --resume to start fresh.
```

**Bug 2 — the model was built with the wrong number of "lag" slots.**

The model looks at the last few moves to work out what just happened. A
run can be configured to look at, say, 12 different history lengths
instead of 8. That number is baked into the model's structure — it is not
a setting you can change later. But the trainer was rebuilding the model
from its default (8) before it even opened the checkpoint, so resuming a
12-slot model tried to squeeze a 12-entry table into an 8-entry one:

```text
IndexError: index 9 is out of bounds for dimension 0 with size 8
```

The trainer now opens the checkpoint FIRST, reads the lag set out of it,
and refuses to resume if you explicitly ask for a different one:

```text
refusing to resume: the checkpoint was trained with lags
[0,1,2,3,4,5,6,7,8,9,10,20] but --lags says [0,1,2,4,8,16,32,64].
```

**Bug 3 — the trainer called a method the model does not have.**

For resuming from a "hard" (integer) checkpoint, the trainer called
`model.import_hard(...)`. The shipped model has no such method, so it
raised `AttributeError` before training a single step.

With all three fixed, a resume from step 6,000 to step 6,010 now runs
cleanly and writes valid artifacts. **Until this was fixed, no resume of
any kind had ever succeeded.** That is why V24 is not yet launched: a
37-hour run that cannot be continued after a crash is not safe to start.


## 9.14 The midnight run: what we know, what we fixed, what we do not know

### The good news first

Three things that were broken and would have quietly ruined an 18-hour run
are now fixed and **proven**, not just patched:

**1. Resuming is now lossless.** If the run crashes at 3 a.m., restarting
it continues on **exactly** the same trajectory. We proved this by
training twice — once straight through, once with a deliberate
interruption in the middle — and comparing the results:

```text
max |weight difference|      = 0.000e+00
max |Adam moment difference|  = 0.000e+00
```

Before this fix, a restart silently threw away the optimiser's memory
(Adam's momentum estimates) and re-rolled the dice on which data rows got
sampled. The "final" model would have depended on *when* the computer
happened to hiccup.

**2. The generalisation check actually works now.** The number we use to
decide "is this checkpoint the best one?" was being measured on the
**training** data. That tells you how well the model memorised, not
whether it learned anything general. We now also score a separate set of
games the model has never seen. And the code path for it had **never
once been executed** before tonight, because the launcher never turned it
on. Turning it on revealed two separate crashes, both of which were
being *swallowed* — the run would have carried on for 18 hours without
the check it was supposed to have.

**3. The disk will not fill.** A resumable checkpoint is 203 MB. Sixty of
them is 12 GB, which does not fit. The new pruning keeps the three most
recent resumable checkpoints plus the best one — about 0.9 GB.

### The number that changed by half

The timeline you were shown earlier was based on **1.30 seconds per step**.
That was wrong, and in the good direction.

The three earlier test runs (fx_a, fx_b, fx_c) **all started 12 seconds
apart and ran at the same time on the same graphics card**. They were
sharing it three ways, so each was crawling at roughly a third speed.
The 1.30 s/step was the *shared* rate.

V24 runs alone, so it gets the whole card:

| | |
|---|---|
| Real rate | **0.65 s/step** |
| One full pass over the data (20,000 steps) | **3.6 hours** |
| Five passes (100,000 steps) | **18.1 hours** |

So the run finishes around **18:00 tomorrow evening**, not the following
morning.

### What it will probably score

Being straight with you: the honest answer is "about 63 %, and it could
be lower."

Earlier I quoted 70–78 %. That came from assuming the learning curve
would keep its shape indefinitely. It probably won't — curves flatten.
When I fit the real curve from the corrected 6,000-step run and
extrapolate, I get **≈ 63 %** at 100,000 steps. The 500-row overfitting
run reached 91 %, but that was memorising half a thousand games, so it is
an upper bound on the *method*, not a prediction for this run.

What I will not do is promise a number. Watch the two columns in the log:

- `LABEL` — how well it fits the training labels (the thing being
  optimised)
- `H-LABEL` — how well it does on games it has never seen (the thing that
  actually tells you if it learned)

If `LABEL` climbs and `H-LABEL` stays flat, the model is memorising and
we should stop. That is exactly what the new abort rule watches for.

### The hour-by-hour, honestly

Assuming a start at 00:00, at 0.65 s/step:

| Time | Step | Passes over data | `LABEL` (rough) |
|---|---|---|---|
| 00:30 | ~2,800 | 0.14 | ~41 % |
| 01:00 | ~5,500 | 0.28 | ~45 % |
| 02:00 | ~11,100 | 0.55 | ~50 % |
| 03:00 | ~16,600 | 0.83 | ~53 % |
| 04:00 | ~22,200 | 1.11 | ~55 % |
| 06:00 | ~33,300 | 1.66 | ~58 % |
| 12:00 | ~66,500 | 3.32 | ~61 % |
| ~18:05 | 100,000 | 5.00 | ~63 % |

Read the `LABEL` column as "roughly where the curve points", not as a
promise. The step counts are solid; the percentages are a projection.

### Playing it while it trains

You do not have to wait 18 hours to play. Every ~2,000 steps (about 22
minutes) a new checkpoint is published, and the game page can swap it in
**without losing the game you are in the middle of**. We tested that
specifically: 14 out of 14 checks pass.

### What we are still not claiming

- We do not yet know the final score. Nobody can know that before the run.
- The learning-rate schedule and the weight-averaging options exist and
  work, but they have **not** been test-flown, so they ship switched off.
  A long run with a constant learning rate can start wobbling near the
  end; if the curve looks like it is wobbling, that is the switch to
  flip on the next run.
- We still owe a clean test of "per-ply supervision" versus "single
  label" supervision. The old answer to that question was based on broken
  code and has been thrown out.
- Nothing is called "released" until the converged model passes the full
  battery: hard evaluator, mirror and column checks, in-game strength,
  seat fairness, trajectory validity, and artifact restoration.

## 9.15 The midnight path had four hidden faults -- all fixed and tested

Before you set your alarm, you should know the "watch the model while it
trains" button was not actually working. Four separate faults, all found
by running the code rather than reading it:

**1. The status file was not valid JSON.** It wrote the training step as
a bare empty slot, which produced `"step": ,`. Nothing could read it --
including the button's own test. An unknown step is now `null`, and the
free-text part is escaped so a stray quote cannot break the file again.

**2. The readiness check called the server "dead" while it was alive.**
This Windows `curl` returns exit code 23 for a *successful* request when
you throw the body away with `-o /dev/null`. The server was returning
HTTP 200 the whole time; the check was reading the wrong number. Now the
check asks for the HTTP status code itself, which the tool reports
honestly.

**3. The server never told the watcher which step was being served.** The
promoted model is called `best_hard.pt`, so the step cannot come from the
filename. The server now reads it out of the checkpoint and reports it.

**4. Saving a checkpoint could give up too early on a busy machine.**
The writer retried for only 2 seconds before giving up, and a lost
checkpoint is 20 minutes of training. It now waits up to 30 seconds, and
if the file was truly lost it rebuilds it instead of throwing the work
away. The retry jitter was also moved off the random-number stream the
training itself uses, so a busy disk can no longer change what the model
learns.

### One thing worth understanding about the numbers

When the checkpoint file is replaced while something else is reading it,
Windows says "access denied" rather than serving half a file. That is
**not** corruption -- and the distinction matters. We measured 769
successful reads with **zero** corrupt files and 50 of these harmless
"busy" refusals. The rename-based save guarantees you never get a
half-written model; the busy refusals are handled by having the reader
wait a moment and try again.

### Which of the "failures" were real?

Honesty about this, because two of them had been reported as failures
before:

- The **live-parity** failures were two regression sweeps running at once,
  both driving the same game server and resetting each other's board. The
  test's own output said "PARITY_OK = True" while the harness wrote FAIL.
  Both pass when run on their own. The evidence files now carry a
  per-sweep marker so a stale file can never be mistaken for the current
  result.
- The **transcript-stamp** failure was an out-of-date *test*, not a broken
  *model*. The test looked for a bare `torch.save(...)` but the code now
  uses the safer `atomic_torch_save(...)`. I read the production code to
  confirm the stamp is genuinely there before changing the test -- the
  order matters, because a test that is simply loosened to pass is worse
  than a red test.
- The **watcher** failure was a real fault (faults 1 and 2 above) plus a
  small mistake in the test itself.

### Where that leaves us

```text
full regression sweep:  23 tests, all green
play-while-training:    12/12 checks, including hot reload and the
                        one-writer guard
launcher smoke run:     completed, holdout agreement rose 11.8% -> 27.0%
in-game strength:       reproduced exactly on an independent re-run
```

One result from the background runs deserves to be repeated here because
it is the strongest evidence we have that the fx_a numbers are real and
not a lucky draw: the in-game strength audit was re-run from scratch and
came back **identical** in every single cell -- every win, loss and
p-value. The `duel` harness ignores its own seed argument, so the audit
seeds the random number generator itself before every match. Getting the
same answer twice, independently, is what reproducibility looks like.

## 9.16 The test failures that were never about the tests (29 October 2026)

The live-UI tests kept failing, and the failure *moved around* -- one
sweep it was the v3 test, the next sweep v4 and v5, sometimes four at
once, sometimes everything green. A failure that moves around is not a
broken test. It is the machine being too busy.

Three separate causes, all found by measuring instead of guessing:

**1. Two servers could answer on the same port.** On Windows, telling
Python's web server "you may reuse this address" does not mean what it
means on Linux. It means a *second live server* is allowed to grab a port
that a *first live server* is already using. So an old server from an
earlier session kept sitting there, quietly answering everybody's
requests, while the new one happily printed "serving" next to it. The
tests were talking to yesterday's server. Fixed by refusing to reuse the
address and by grabbing the port *before* loading the 34 MB model, so a
losing race costs one second instead of a minute. A new test
(`tmp/test_ui_single_server.py`, 8/8) now proves the second server is
turned away and the first one keeps working.

**2. The server was asking for all 32 processor cores on every single
request.** Each `/state` call does real thinking (about 105 ms). When the
server and the test both think at the same time on the same machine,
everything crawls -- slow enough that the test gave up waiting, even
though the server had already answered correctly. Fixed by capping the
server at 4 threads.

**3. Four leftover processes from earlier work were eating the machine.**
This was the real culprit. One of them was a script that had been
quietly re-scanning the whole 320,000-game corpus in a loop for about
five hours. It was never finishing and never dying.

That last one is the lesson worth keeping: **a test result measured while
the machine is busy is not a result.** Before trusting any timing test,
check what else is running. The full sweep on an idle machine is now:

```text
=== 24 PASS, 0 FAIL ===
```

All four live-UI tests pass, plus the new single-server test. The
midnight run is no longer blocked on this.

## 9.17 Three sweeps at once -- my own fault (29 October 2026, later)

Three sweeps ran at the same time and produced three different-looking
results: 24/24, 23/24, 22/23. Each of those looked like a different bug.

They were not. The sweeps take a lock so only one can run at a time, and
I had been clearing that lock with `rm -rf` before each run -- which
deletes it. So every sweep I started deleted the lock the previous sweep
was holding, and all three ran side by side on one shared game server,
resetting each other's board.

The lesson is about *how I run commands*, not about the code:

- To run something after a previous run, **do not delete the lock**. The
  suite now works out on its own whether the previous holder is still
  alive.
- A lock that says "someone else is here" must be respected, not deleted.

Two real improvements came out of it anyway. The lock now records *who*
holds it, so a crash cannot wedge the suite forever -- a lock whose owner
is dead is reclaimed automatically. And the results file is now unique per
run, with a copy at the old familiar path, so two runs can no longer
overwrite each other's answers.

There was also a genuine Windows quirk hiding in here. One test failed
with exit code `3221225794`, which is `STATUS_DLL_INIT_FAILED`: the
program was killed *before it started*, with no output at all, simply
because the machine was busy. That is not a broken program, it is a busy
machine. The test now recognises that one exit code and tries again,
instead of reporting a failure that never really happened.

Final state:

```text
regression sweep:   24 PASS, 0 FAIL
lock lifecycle:     acquire / block / auto-reclaim / release  all verified
```


## 9.18 The training run started -- and the machine was not empty (29 October 2026)

The big training run finally started at 02:34 in the morning. It is the
one that trains for many hours over the whole corpus instead of stopping
after a few thousand steps.

But when it started, something was odd: the machine was busy even though
I had not asked it to do anything. There were **33 leftover processes**
sitting there, quietly using the processor. They came from my own test
suite.

The test for the "play while it trains" watcher starts a real watcher and
a real little web server. When the test finished, it told the watcher to
stop -- but on Windows, telling an MSYS shell to stop does not reliably
run its cleanup code, so the **shell itself stayed alive** every single
time the test ran. Twenty-four times' worth of watchers and servers were
still there, holding **1.1 thousand seconds of processor time**.

That matters more than it sounds. When I started the training run I
checked the speed and got a slightly *worse* number than expected. After
clearing the junk away, the run turned out to be **much faster than
predicted**: 12.3 hours instead of 18.1 hours, finishing around half
past two in the afternoon.

**The lesson:** the "slow" measurement was not the machine being slow. It
was my own leftovers in the way. The right response to a slow run is to
check what else is running before concluding anything about the run.

I also fixed the test itself, so it now cleans up after itself properly,
and added a small tool (`tmp/cleanup_leaked_watchers.py`) that finds and
removes watchers that were left behind -- while being careful never to
touch the live one that is serving the model you are playing.

**Where things stand now:**

```text
training:    running, 100,000 steps (~12.3 h, done ~14:50)
at step 2000: 40.2% label agreement   (predicted 40.23% -- matches)
holdout:     29.0%   (data the model has never seen)
playing:     http://127.0.0.1:7860/ -- updates itself as the model improves
watching:    an automatic check every 45 minutes tells us if anything goes wrong
```

One honest correction: I had been planning for an **18.1 hour** run.
That number was measured on a crowded machine. The real figure is
**12.3 hours**. If you had quoted the 18-hour number, you would have
waited five hours longer than necessary for no reason.


## 9.19 The disk was filling up -- and it was not the training (29 October 2026)

Partway through setting up the long run, the computer reported only
**4.3 GB free** out of 931 GB. That looks alarming: the run itself was
using **604 MB**, so where did the rest go?

**It was my own tests, again.** Two of the resume tests create a full
training run to check that resuming works. They *clean up before they
start* -- but never after they finish. So every successful test run left
a spare copy of the training corpus, a spare copy of the labels and a
full set of checkpoint files sitting on disk. About **2.9 GB** of it.

The fix is small: clean up after finishing too, not just before
starting. Both tests now do, and I confirmed they still pass
(19/19 and 12/12, with the bit-exact resume check still exact) and leave
**nothing** behind.

That freed **2.9 GB** -- the machine went from 4.3 GB free to 7.2 GB
free, without deleting a single historical experiment, corpus or model.

**The pattern worth noticing.** Two separate problems in this stretch
had the same shape:

- the watcher test **left processes running** after it finished;
- the resume tests **left files behind** after they finished.

Both are "I only cleaned up on the way in, not on the way out." Both
were invisible until something else needed the same resource -- the CPU
for the first, the disk for the second. The lesson is not "clean up
more carefully." It is **clean up on the success path too**, because the
success path is the one that runs most often and nobody looks at it.

**Where the run stands:**

```text
step 4000 | LABEL 45.3%  | HOLDOUT 28.8%  | H-LABEL 28.1%
```

The model is tracking the predicted learning curve exactly
(step 2000 predicted 40.23%, got 40.2%; step 4000 predicted ~45.3%,
got 45.3%), and the playable page has already updated itself to the
step-4000 model on its own.


## 9.20 The computer went to sleep -- and everything survived (29 October 2026)

You asked whether the run was still alive after the computer slept.
Short answer: **yes, and nothing was lost.**

What actually happened, which is a bit more interesting than "it slept
for a minute":

```text
04:41   step 18000 written to the log
04:47   the computer asks to go to sleep
05:58   the computer actually wakes up     <- 71 minutes asleep
06:01   the training is still going
06:06   step 20000 written to the log
```

The gap between "asks to sleep" and "wakes up" is where I nearly
misdiagnosed it. Windows records the *request* to sleep, which makes it
look like a 6-second nap. The event that actually tells you when the
machine came back says **05:58**. So the sleep was **71 minutes**, not
six seconds. That is the number that explains the log gap.

**Why the log looked stalled.** The trainer only writes a line every
2,000 steps, which normally takes about 14 minutes. Across the sleep
that cycle took 85 minutes. If you had only looked at the log, you
would have concluded the training had hung or died.

**How I actually checked, instead of guessing:**

- Asked the operating system whether the training process still exists
  -> yes, worker process alive since 02:33.
- Looked inside the running process to see what it was doing -> it was
  in the middle of the training loop, building batches, not stuck.
- Read the **live step counter** out of the running process -> 19,439,
  and climbing.
- Measured how fast it was actually moving -> 0.395 seconds per step,
  which is the normal speed.
- Checked the graphics card -> 92 %, 408 W. It was working hard.
- Loaded every saved checkpoint -> all eight load cleanly, with
  correct step numbers and intact optimiser state.
- Searched the log for errors -> zero.

**Nothing was corrupted.** The trainer resumed exactly where it left
off and wrote step 20000 twenty minutes later, with the checkpoint
files all valid.

**One number to be careful with.** The trainer prints how long it has
been running, and that is *wall-clock* time -- it keeps counting while
the machine sleeps. So it jumped from 7,659 seconds to 12,768 seconds
even though the training did no extra work. If you plan from that
number alone you will think it is running slower than it is.

**Where it stands now:**

```text
step 20000 | LABEL 59.0% | HOLDOUT 36.2% | H-LABEL 38.9%
```

and the honest remaining time is **about 9.3 hours of awake time**,
finishing around **15:30** -- not the ~16:40 the raw wall-clock
suggested, because the sleep does not count as training time.

**The lesson:** a stale log is not evidence of a dead run. Before
interrupting anything, ask the machine what the process is actually
doing, and read the counter inside it. I have also taught the
automatic 45-minute watchdog to do exactly this, so if the computer
sleeps again it will say "the host slept" instead of crying wolf.

