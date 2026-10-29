# Promotional Copy Package

**Author of record: Christof Ckal.** Never attribute to Alex.
**X:** `https://x.com/zX14_7` · **Blog:** `ckaller.blogspot.de` ·
**LinkedIn:** `de.linkedin.com/in/christof-kaller-6b043733`

> **Rules for every post below:**
> 1. Attribute to **Christof Ckal**.
> 2. **Do not claim the model is strong at Connect Four.** It is not, and the
>    ladder measures it. The interesting claim is the *negative* result.
> 3. Every number here must match `RELEASE_RECORD.md`. If a number is not
>    in the release record, it does not go in a post.
> 4. **Do not post until** the public repo, the HF model and the live game
>    URL exist — otherwise the links are dead.

---

## 1. X / Twitter

### 1a. The main thread (7 posts)

> **1/7**
> I taught a neural net to play Connect Four with an exact solver as its
> teacher — every ply of every game labelled with the set of optimal moves.
>
> Then I measured two things separately:
> - how often it agrees with the teacher: **59.6%**
> - how often it wins: it beats random 85.2%... and **loses to a 1-ply
>   heuristic.**
>
> Teacher-agreement ≠ game strength. Here's the evidence. 🧵

> **2/7**
> The setup: 320,000 games. 4,462,571 per-ply labels. Every label is the
> *set* of optimal moves from an exact solver, so ties stay ties.
>
> Architecture is GF(2⁸) — XOR, table lookups, integer adds. No float MAC in
> the trunk. 8.4M params, 32 MB, runs on CPU.

> **3/7**
> Agreement with the exact solver:
>
> step 0 → **16.3%**
> step 6000 → 50.9%
> step 18000 → **59.6%**
>
> Holdout (board-disjoint): 11.9% → **41.0%**
>
> It reads the board. It picks all 7 columns. It is not a constant policy.

> **4/7**
> And it still loses.
>
> The 1-ply heuristic is 3 lines: *win now > block now > centre.*
>
> Across the entire run the model was net-negative against it. Best point:
> 23–25. Never ahead.
>
> An independent 5-opponent matrix agrees: it beats the best constant
> column in only **2 of 5** settings.

> **5/7**
> Why?
>
> The training signal rewards copying the teacher's choice on *the
> labelled distribution*. The teacher's edge comes from elsewhere —
> threat creation and denial over horizons longer than one ply, and
> timing when to abandon the centre.
>
> Per-ply imitation doesn't express any of that.

> **6/7**
> Two more things I'd flag honestly:
>
> • the ~18pp train/holdout gap says a chunk of that 59.6% is memorisation
> • agreement plateaued after ~step 14000 — more steps stopped helping
>
> The ladder isn't a vanity metric. It's the one that caught this.

> **7/7**
> The model, the code, the measurements and the full negative result are
> all open:
>
> 🔗 repo: <GITHUB_URL>
> 🔗 model: <HF_URL>
> 🔗 play it: <GAME_URL>
>
> Paper and the full release record in the repo. The release record maps
> every published number to the command that produced it.

### 1b. Shorter single post (if the thread is too long)

> A neural net trained with an exact Connect Four solver as teacher:
> 59.6% agreement with optimal moves, beats random 85.2%...
> and loses to a 3-line heuristic.
>
> Teacher-agreement ≠ game strength. Full evidence, code and model:
> <GAME_URL>

### 1c. Follow-up posts (after the release, spaced out)

> **Post A — the bug that invalidated my own earlier conclusions**
> I published conclusions from an earlier version of this model. They were
> wrong.
>
> The trainer was reading supervision labels by *batch position* instead of
> by the *sampled corpus row*. ~41% of supervised positions had all-zero
> targets.
>
> Every "per-ply supervision is worse" claim I'd made was an artifact of
> that. Fixed, re-measured, and the old claims retracted.

> **Post B — the paper**
> I wrote up the whole thing, including the part where my model loses:
> "Reading the Board: A GF(256) Integer Model for Connect Four"
>
> The negative result is the contribution. A loss curve going down is not
> evidence of getting better at the thing.
> <PAPER_URL>

> **Post C — the demo**
> You can play the model in the browser. Watch the next-move distribution
> update each ply — it's the interesting part, not the move it picks.
> <GAME_URL>

---

## 2. Blog post — `ckaller.blogspot.de`

**Title:** *My Connect Four model agrees with the perfect player 59% of the
time and still loses to a three-line rule*

**Structure:**

1. **The setup** — why Connect Four is a clean testbed for
   "does imitation recover strength?"
2. **The model** — GF(256) integer trunk, dense per-ply supervision, the
   320k-game corpus.
3. **What it learned** — the convergence curve, the board-reading
   evidence, the honest train/holdout gap.
4. **What it didn't** — the ladder. Full curve, every step, net-negative
   throughout.
5. **Why** — supervision distribution vs. strategic advantage; the
   error-mode breakdown.
6. **The bugs** — six of them, each producing a plausible-looking wrong
   number. Including the one that invalidated my own earlier published
   claims.
7. **What I'd do next** — decision regret, search-based distillation,
   adversarial self-play data.
8. **Everything is open** — repo, model, live game, release record.

**Tone:** first person, plain, no hype. The strength deficit is stated in
the first paragraph, not buried. Every number links to the release record.

**Tags:** `machine-learning`, `connect-four`, `game-ai`, `imitation-learning`,
`gf256`, `integer-arithmetic`

---

## 3. LinkedIn

> I built a Connect Four model with an exact solver as teacher — and it
> taught me something more useful than a better model.
>
> Every ply of every game was labelled with the set of optimal moves. 320,000
> games. 4.46M per-ply labels. The architecture is GF(2⁸): XOR, table
> lookups, integer adds, no floating-point multiply-accumulate in the trunk.
>
> Two measurements, deliberately kept separate:
>
> • Agreement with the exact solver: 16.3% → 59.6% (train), 11.9% → 41.0%
>   (board-disjoint holdout)
> • Game strength: beats a random player 85.2% of the time... and stays
>   net-negative against a 1-ply heuristic at every point in the run
>
> The model reads the board. It picks all seven columns. It just isn't
> good at winning, and a loss curve going down is not the same thing as
> getting better at the game.
>
> I published the negative result as prominently as the positive one,
> because I think teacher-agreement and game strength being different
> quantities is the genuinely transferable finding — and because my own
> earlier conclusions on this project turned out to be artifacts of a
> supervision-indexing bug, which I've written up in full.
>
> Code, model weights, the paper and a release record that maps every
> published number to the command that produced it:
> <GITHUB_URL>
>
> Model: <HF_URL> · Play it: <GAME_URL>
>
> #MachineLearning #GameAI #ImitationLearning #ConnectFour #Research

---

## 4. Short demo video

**Format:** 60–90 s, 1080p, silent-with-captions (X and LinkedIn autoplay
muted).

**Storyboard — a deterministic scripted match, not a training montage:**

| Beat | Content |
|---|---|
| 0:00 | Title card: *A model that agrees with the perfect player 59% of the time — and still loses.* |
| 0:06 | The board, empty. Caption: *Every ply labelled with the set of optimal moves. 320,000 games.* |
| 0:14 | The model moves. The 7-column next-move distribution appears and **updates visibly** each ply. This is the hero shot. |
| 0:30 | Caption: *59.6% agreement with the exact solver. Holdout: 41.0%.* |
| 0:38 | Split screen vs. random: wins. Caption: *Beats random 85.2%.* |
| 0:46 | Split screen vs. the 3-line heuristic: loses. Caption: *Loses to a 1-ply heuristic. At every step of the run.* |
| 1:00 | End card: repo / model / play links. Caption: *Teacher-agreement ≠ game strength.* |

**Non-negotiable:** the video shows the loss. A highlight reel would make
the model look strong, and it isn't.

**Production notes:**
- Render from real game records via `src.c4.duel` so the footage is
  genuine, not staged.
- ffmpeg is at
  `~/AppData/Local/hermes/tools/ffmpeg-9.0.1-win32-x64/bin/ffmpeg`.
- PIL is available under `python3` (12.1.1), not under the project venv.
- Capture the board with the same rendering the UI uses so the UI and the
  video agree.

---

## 5. Release-day posting order

| When | Channel | Content |
|---|---|---|
| T-0 | Blog | Publish the post — the deepest version, and it links everything. |
| T+0h | GitHub | Repo public with README, paper, model card, release record. |
| T+0h | HF | Model + model card live. |
| T+0h | Netcup | Game live, per-session boards, tested under concurrency. |
| T+1h | X | The 7-post thread. |
| T+1h | LinkedIn | The post above. |
| T+2h | X | The video. |
| T+1d | X | Post A (the bug that invalidated my own claims). |
| T+3d | X | Post B (the paper). |
| T+5d | X | Post C (the demo). |

**Never** post the thread before the links resolve. Broken links in a
technical thread cost more credibility than a day's delay.

---

## 6. Pre-flight checklist before any post

- [ ] Every number in the post appears in `RELEASE_RECORD.md`
- [ ] Attribution says **Christof Ckal**; no mention of Alex
- [ ] The strength claim is stated as a **loss**, not softened
- [ ] Repo URL resolves and is public
- [ ] HF URL resolves and the model card is present
- [ ] Game URL resolves, and **two browser tabs show two different boards**
      (the multi-user isolation test)
- [ ] The 100,000-step final result is **not** claimed unless the run has
      finished *and* the release battery has passed
- [ ] No private corpus, logs, historical runs or credentials are attached
      or linked
