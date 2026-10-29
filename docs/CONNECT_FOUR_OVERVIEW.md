# Connect Four — the whole thing, in one page

**Snapshot:** current revision

> **Note:** §1 and §7 describe the *early* project (V10/V11 era). Three
> more serious bugs were found after that and are the real story — see
> `CONNECT_FOUR_PLAIN_ENGLISH.md` §9.0/§9.0b and
> `CONNECT_FOUR_HANDOVER.md` §0b–§0e. **§9 is the current status.**

---

## 1. The short version

I built a Connect Four brain that is **just a text predictor**. It reads the board
written out as a short sentence of letters, and predicts the very next letter — which
happens to be the column you should drop a disc into. Then I spent days making it good
at *that*, and it is **not good yet**. Right now the best saved brain loses 47 games
out of 48 to a simple greedy opponent.

Two things are true at once, and both matter:

- **The numbers got much better.** Move-accuracy went 41% → 60%, loss went 2.18 → 1.48.
- **The playing did not.** It still loses almost every game.

That gap is the whole story of this project.

---

## 2. The idea: no special "how to play" brain

Most people build a Connect Four AI as a separate thing that looks at the board and
scores each of the 7 columns. **This one doesn't.** It has no idea what a board is.

It only knows how to predict the next letter of text. That's the whole trick:

```
  X X X....
.. O O...
... X...
.......

  -->  heights ABCDEFG, turn=X, last=G  -->  predicts one letter A..G
```

The board is drawn as letters `A`–`G` (one per column height) plus whose turn it is.
The *only* thing the model is ever rewarded for getting right is the single letter
naming the column to play. So:

> **The board is the sentence. The move is the next letter. Predicting the next letter IS playing the game.**

This is what you asked for — *"it should be the same as text prediction only that we
give it boards."* And it has a real payoff: everything that makes a language model good
at text (it learns patterns, it generalizes to situations it has never seen, it gets
better the more you train it) applies to playing the game **for free**.

**What it costs:** the model can spend its whole capacity on the easy parts of the
text — the header digits, the turn letter — and get a beautiful loss number while
being terrible at the one letter that matters. **That exact thing happened.** See §7.

---

## 3. The game it plays

Connect Four (Vier Gewinnt), the one everybody knows:

- 7 columns (`A`–`G`), 6 rows, discs drop from the top.
- You win by lining up **4 in a row** — horizontally, vertically, or diagonally.
- First to 4 in a row wins. If the board fills up with no winner, it's a draw.
- Two players take turns, one drops red (`X`), one drops yellow (`O`).
- **Key fact for reading the numbers:** the board has 42 slots, so a game is at most 42
  moves, and if nobody wins it is a draw.

The AI never breaks a rule and never plays a full column — illegal moves are masked out
before the model is allowed to choose.

---

## 4. The data: what the model reads

Every training example is one finished game, written as a stream of bytes. Each move
in the game gets a fixed-size record of **10 bytes**:

| # | Bytes | What it is |
|---|-------|-----------|
| 0–6 | 7 letters | Height of each column (`A`=empty … `G`=full) |
| 7 | 1 letter | The last move played (`A`–`G`) |
| 8 | 1 letter | Whose turn it is (`X` or `O`) |
| 9 | 1 letter | **The move made** — the letter the model must predict |

So a 20-move game is 200 bytes. The model sees the first 192 bytes of context and
predicts the next byte.

**The prediction target is byte #9.** Not the heights, not the turn — just the move.
That's the entire decision.

**The corpus** — the pile of all these games — lives at
`runs/c4/v7mixed_v9/corpus.txt`, currently **5,444 games**.

---

## 5. What "V10" and "V11" mean

These are just two versions of the training program. Same brain, same idea, different
training rules.

### V10 — "stop wasting capacity on the wrong letters"

**The problem it fixed.** I measured where the model's training effort was going:

- `43%` of the target bytes were **header digits** (column heights — easy, boring)
- `38%` were **padding** (newline bytes — free)
- **`12%` were actual moves** ← the only thing that matters

Only 1 byte in 8 was a real decision. The model was a superb guesser of "how many discs
are in column B" and a **beginner at Connect Four**.

**The fix.** Mask the loss. Every header digit and every padding byte now contributes
**exactly zero** gradient. All the learning pressure lands on the move letter.
Plus: moves made by the eventual winner count more than moves by the eventual loser.

**Result:** move-accuracy jumped immediately, `6% → 41%`.

### V11 — "only learn from games that were actually well-played"

**The problem it fixed.** You said the sharp thing:

> *"it should only train on good games not random"*

The old training loop had the model play **itself** at temperature 1.2, throw away the
losses, and train on whatever survived. But a 42%-accurate model doesn't play good
games — it plays *lucky* ones, full of blunders that happened to not lose immediately.
Training on those is like teaching chess from a game where both players keep hanging
their pieces.

**The fix.** Stop self-playing. Instead hire **teachers** who genuinely know how to
play, let them play, and keep only games that were *decided* (someone won):

| Who teaches | Share | Plays against | How strong |
|---|---|---|---|
| **Lookahead** (looks 2 moves ahead) | 50% | mixed | beats greedy 20–0 |
| **Heuristic** (greedy + threat-blocking) | 30% | mixed | solid but beatable |
| **Solver** (full alpha-beta search) | 10% | mixed | near-perfect, but slow |
| **Random** (the weak one) | 10% | — | only ever *loses*, never teaches |

Random is still there **only as the punching bag** — it's the opponent the good players
beat. It never supplies a move the model learns from. That's the "remove random games"
requirement, satisfied while keeping the opponents varied.

---

## 6. Who does what: the files

| File | Job |
|---|---|
| `src/c4/train.py` | **The trainer.** Reads games, masks the loss, runs the GPU loop, decides when to save. This is V10 + V11. |
| `src/c4/board.py` | The Connect Four board itself. Knows the rules, what's legal, who won. |
| `src/c4/transcript.py` | Turns a game into the 10-bytes-per-move text described in §4. |
| `src/c4/corpus.py` | Older, simpler game→training-example code. Superseded but still imported. |
| `src/c4/corpus_aug.py` | Generates extra practice games and packs them into the dense 10-byte format. |
| `src/c4/teacher_corpus.py` | **V11's brain.** Runs the four teachers, keeps only good decided games. |
| `src/c4/strong_solver.py` | A real Connect Four solver — searches ahead, uses alpha-beta pruning + a memo cache. |
| `src/c4/opponents.py` | The opponents: `heuristic_policy`, `lookahead_policy`, `random_policy`. |
| `src/c4/agent.py` | The player. Converts the board to text, asks the model, masks illegal moves, returns a column. |
| `src/gf256_markov_lm.py` | **The neural network itself.** A small GF(256) linear-algebra language model, ~59M params. |
| `src/c4/ui_server.py` | The web backend. Serves the page, makes moves, exposes probabilities. |
| `src/c4/ui.html` | The web page itself — the board you click. |
| `src/c4/trainlog.py` | A tiny web page showing the training curve. |

**Scripts** (`scripts/…`) are the debugging instruments — `diag_loss_mix.py` found the
loss-mix problem, `diag_acc_split.py` found the accuracy problem, `duel_teachers.py`
measured which teacher is strongest, `check_v11_teacher.py` proves the V11 pipeline
generates only legal, well-formed, decided games.

---

## 7. The three bugs that actually mattered

### Bug 1 — the model was wasting 88% of its effort
Header digits and padding were 88% of the training signal. **Fixed by V10's mask.**
Move-accuracy 6% → 41%.

### Bug 2 — outcome weighting was upside-down
A helper function built the "did the winner make this move?" tags using an **11-byte**
spacing when the actual data is **10 bytes**. So every tag landed one byte off — and
because the value flips between winner and loser, the model was being taught
*"reward the loser's moves, punish the winner's."* It learned backwards, for the entire
V10 run. **Found and fixed today.**

### Bug 3 — the save gate threw away the good model ← *this is why there's nothing to play*

This is the one that matters most, and nobody noticed for days.

The trainer saves a checkpoint **only when the model plays better than the last saved
one.** Look at what actually happened:

| Step | Move-accuracy | Loss | Games vs "centre column" bot | Saved? |
|---|---|---|---|---|
| 4000 | 8% | 7.78 | **6 won – 18 lost** | ✅ saved |
| 5000 | 41% | 2.18 | 8 won – 16 lost | ✅ saved |
| 6000 | 54% | 1.70 | 2 won – 21 lost | ❌ **deleted** |
| 7000 | **60%** | **1.48** | 2 won – 22 lost | ❌ **deleted** |

Look at that table. By step 7000 the model was **by far the most accurate it had ever
been** — nearly 60% on the single letter that decides the game. And because its
*score* dipped that particular thousand-step window, the trainer threw the file away.

So the reason there's no good model to play against is not that training failed.
**Training worked. The save rule destroyed the results.**

And the reason the score dipped while the accuracy climbed? Two more things:

- **The training games were made at temperature 1.2** — the model picked moves
  semi-randomly while playing, so it was graded on stumbles, not on its best play.
- **The opponent the score is measured against is "drop every disc in the middle
  column."** Beating that bot is nearly free once you can predict letters at all.

Both are wrong measurements of a good player. A 60%-accurate next-letter predictor
should not be losing to a bot that only plays the centre, and the fact that it does is
what the save gate latched onto.

**The fix, which I'm about to do:** save on **accuracy**, not on beating a weak bot.
Keep the gameplay score as information, but don't let it throw away a measurably
smarter model.

---

## 8. What the game and the UI look like

**The game:** 7 columns, drop a disc, four in a row wins. Simple rules, hard to play well.

**The UI:** a web page at `http://127.0.0.60/`
- A 7×6 board you click to drop your disc (yellow `O`).
- The model's discs are red `X`.
- It shows the board **before** your decision, the **predicted next board**, and the
  **final result**, then the bytes/tokens underneath — so you can see the AI's actual
  reasoning as letters.
- A **probability panel**: how confident the model is in each column, as a bar chart.
  Illegal columns are greyed out.
- During **your** turn it labels the top choice as *"Model recommends"*.
  During **its own** turn it separately shows *"Model picked"* — the actual move the
  hard-coded arithmetic path chose. Those two can differ, and showing both is honest.

**The trainer view:** `http://127.0.0.61/` shows the loss curve and the games being
processed, so training isn't a black box.

---

## 9. Right now, honestly

> **THIS SECTION WAS REWRITTEN.** The text it replaces
> described a step-5,000 checkpoint, a crashed trainer, and a "2–3 hours
> to a candidate" plan. **All of that is obsolete.** The project moved on
> considerably: the three bugs in §7 are long fixed, and the code has been
> through several more. **Read `CONNECT_FOUR_PLAIN_ENGLISH.md` (§9 and
> §9.6b) for the current, honest status — it is the maintained one.**
> This section is kept only so the document does not contradict itself.

**Is there a good model to play against? Still no — and we now know
exactly why.**

The honest summary as of the night of:

- **The real bug was that the shipped program could not see the board.**
  Not "played badly" — *could not see*. The file we saved and served was
  picking the same column regardless of the stones in front of it. There
  were two stacked causes: the hard path did not start training until
  step 12,000, and (deeper) the exported integer tables were **frozen at
  initialisation** — the argmax never moved at any checkpoint, and the
  table gradients were exactly zero.
- **Both are now fixed and verified.** The exported tables change at every
  checkpoint, and the hard export reads the board from step 500 onwards.
- **But it is not yet a good player.** Strict accuracy on unseen positions
  is ~25% and barely moved. The model still plays the middle column most
  of the time (67.7% of the time on unseen positions) and mirrors itself
  only 11.8% of the time. The teacher is provably symmetric, so the
  remaining problem is in the **objective**, not the data or the init.
- **No big training run until that is fixed.** The next work items, in
  order, are in `CONNECT_FOUR_HANDOVER.md` §0e.6.

**One warning worth repeating:** every checkpoint trained before the
table-bias fix — including all of V22 and its impressive-looking holdout
and ladder numbers — is a constant-column policy in disguise. Those
numbers are **not** evidence of a good model.