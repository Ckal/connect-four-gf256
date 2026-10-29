# Reading the Board — a GF(256) integer model that learns Connect Four

**A small neural model whose trunk does XOR and table lookups instead of
float multiply-accumulates — and an honest measurement of how good it
actually is.**

Author: **Christof Ckal**

![A recorded game against the GF(256) model](c4_001-006.gif)

*A recorded game against the released checkpoint (`best_hard`, step 18,000).
Gold is the human, blue is the model.*

## How good is it, actually?

600 games per arm, against the released checkpoint:

| Opponent | Model's win rate | Verdict |
|---|---:|---|
| Random player | **85.2 %** | beats it |
| Centre-column policy | **90.5 %** | beats it head-to-head |
| 1-ply heuristic | **37.2 %** | **loses to it** |
| 2-ply lookahead | **24.3 %** | **loses to it** |
| Minimax | **41.8 %** | **loses to it** |

That gap is the point of the project. The model agrees with its teacher on
**59.6 %** of training positions and **41.0 %** of a board-disjoint holdout —
and it still loses to a three-line rule that only checks *"can I win now?"*
and *"can I block now?"*. **Teacher-agreement and game strength are
different things**, and this repo measures both.

## ▶ Play it against the model

**https://c4.tensorgarten.de**

A real, session-isolated game — every visitor gets their own board. Deployed
on CPU. [Verified isolation + security claims](#security-verification).

Or play it inside the Hugging Face Space:
**https://huggingface.co/spaces/Chris4K/connect-four-gf256**
(the Space is static and frames the live server; the server allows that
origin explicitly — never a wildcard).

---

## The model

The trunk is a GF(2⁸) integer program: XOR, learned S-box table lookups and
integer adds. **No floating-point multiply-accumulate in the trunk.**

```
board ──► byte transcript ──► GF(256) trunk ──► read-out ──► column logits
           (v2 format)         (XOR + S-box)   (float32)      (7 columns)
```

- **Layers:** 3 · **`d`:** 32 · **lags:** `[0,1,2,3,4,5,6,7,8,9,10,20]`
- **Context:** 192 · **Batch:** 16 · **Seed:** 32
- **Params:** emb 8,192 · coef 288 · sbox 24,576 · wiring 576 · readout 8,388,608 · bias 256
- **Size:** 32.3 MB, 8,422,496 parameters, **CPU-only** inference.

**Honest scoping of "zero FLOPs":** the *trunk* has no floating-point
multiply-accumulate. The read-out is float32 and is **99.6 %** of the
parameters. The complete model is therefore *not* a zero-FLOP program.

### Supervision

Every ply of every training game is labelled with the move chosen by a
**negamax alpha-beta teacher** (and a 2-ply lookahead teacher), with ties
preserved as sets rather than collapsed.

> The teacher is a *search policy*, not a proof of optimality. Connect Four
> is a solved game, but this model was never taught the full solution — it
> imitates a bounded search. The repo's `src/c4/exact_solver.py` is a
> **separate, bounded endgame solver** (`max_empty=7`) used to *check*
> positions, not to label the corpus.

## Results

### The convergence curve

| Step | Teacher agreement (train) | Holdout (board-disjoint) | vs 1-ply heuristic | vs random |
|---:|---:|---:|---:|---:|
| 0 | 16.3 % | 11.9 % | 9-39 | 36-12 |
| 2000 | 40.2 % | 27.9 % | 10-38 | 37-11 |
| 6000 | 50.9 % | 36.7 % | 15-33 | 38-10 |
| 10000 | 56.0 % | 40.8 % | 19-29 | 37-11 |
| 14000 | 57.5 % | 38.1 % | **23-25** | 44-4 |
| 18000 | **59.6 %** | 41.0 % | 16-32 | 41-7 |
| 26000 | 58.7 % | 39.1 % | 15-33 | 38-10 |
| 30000 | 59.4 % | 42.0 % | 18-30 | 38-10 |
| 34000 | 58.5 % | 41.8 % | 14-34 | 43-5 |

### What the model learned

- It **reads the board.** Its picks spread across all seven columns
  (holdout: `[39, 55, 94, 175, 78, 48, 23]`), with non-zero per-column
  recall for every column. It is not a degenerate constant policy.
- It **beats random** 85.2 % of the time (n=600, 3 seeds).
- It **agrees with the teacher** on 59.6 % of training positions and
  41.0 % of held-out positions.

### What it did not learn

- It **does not beat a 1-ply heuristic.** Net-negative at every measured
  step of the entire run. The best point is 23-25 — near parity, never
  ahead.
- **Agreement plateaus** after ~step 14000. More steps stopped helping.
- The **~18 pp train/holdout gap** shows a real memorisation component.

### Independent strength check

The canonical `src.c4.duel` harness, 600 games per arm per seed × 3 seeds,
model greedy (ε = 0), opponent explores (ε = 0.3) so trajectories are
genuinely distinct:

- vs. random: **W1460 L337 D3** of 1800 (p ≈ 3.2 × 10⁻¹⁶⁶)
- vs. the best single-column constant: wins in **2 of 5** opponent
  settings — not a sweep
- seat check: model gap **+7.5 pp**, heuristic gap **+4.8 pp**,
  gap-vs-gap **p = 0.402** — no model-specific seat asymmetry

## Read the model yourself

The checkpoint is hosted on Hugging Face rather than committed to this
repository, so a clone stays small:

```bash
# 1. Download the checkpoint (32.3 MB)
curl -L -o best_hard.pt \
  https://huggingface.co/Chris4K/connect-four-gf256/resolve/main/best_hard.pt

# 2. Verify it before using it
sha256sum best_hard.pt
# 134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2

# 3. Play it (the server takes --model; --checkpoint does not exist)
python -m src.c4.ui_server --model best_hard.pt --port 7860 --device cpu
# open http://127.0.0.1/
```

`best_hard.pt` is the step-18,000 hard export. The UI shows the model's
next-move distribution over all seven columns (rendered from the **hard**
path), the legal-move set, and lets you undo and reset. The 100 % figure on
a column is a **probability of being optimal**, not a chance of winning.

> The public deployment uses **per-session boards**. The upstream research
> UI keeps a single board in a module global and is **not** safe for
> multi-user hosting as shipped.

## Security verification

The public deployment is gated by a script that runs **against the deployed
process over public HTTPS** — not against a local server.

```bash
python scripts/verify_public_isolation.py https://c4.tensorgarten.de
# PASSED: 19   FAILED: 0
```

### Embedding (the Hugging Face Space)

The Space at
[`Chris4K/connect-four-gf256`](https://huggingface.co/spaces/Chris4K/connect-four-gf256)
is **static** — it contains only a page that frames the live server. The
server permits exactly the framing origins it needs:

```
Content-Security-Policy: …; frame-ancestors https://chris4k-connect-four-gf256.static.hf.space https://huggingface.co
```

The allowlist is explicit and closed:

- **Default is deny.** With no `--allow-embedding` flag the response carries
  `X-Frame-Options: DENY` and `frame-ancestors 'none'`.
- **Only `https://` origins are accepted.** An `http://` ancestor could be
  intercepted; a `https://*` wildcard is refused outright, because it would
  let any site clickjack the game.
- **The allowlist replaces `X-Frame-Options`** when it is in force — XFO can
  only allow or deny a single origin, while CSP can name several.
- **State-changing requests are origin-checked.** Because the session cookie
  must be `SameSite=None` to work inside a cross-site frame, the server
  validates the `Origin` header on every mutating request instead of relying
  on `SameSite=Lax` for CSRF protection. A request from an origin that is
  not the site or an allowlisted ancestor is refused with **HTTP 403**.

Run it yourself:

```bash
python -m src.c4.ui_server --model best_hard.pt --public \
    --allow-embedding https://chris4k-connect-four-gf256.static.hf.space
```

It asserts, and all of these currently hold:

| Assertion | Result |
|---|---|
| `GET /` returns 200 with a real page | PASS |
| Content-Security-Policy, `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, HSTS | PASS |
| Two clients receive **different** session cookies | PASS |
| The session cookie is `HttpOnly` and `Secure` | PASS |
| **A move in one session is invisible in another** — no shared board | PASS |
| `/modelinfo` and `/reload` are closed in public mode (HTTP 403) | PASS |
| A mutating request from a non-allowlisted origin is refused (HTTP 403) | PASS |

The session cookie as served when cross-site embedding is enabled:

```
Set-Cookie: tg_sid=<opaque id>; Path=/; HttpOnly; SameSite=None; Secure; Max-Age=7200
```

Without an embedding allowlist it is `SameSite=Lax; Secure`.

The checkpoint served by the public container is byte-identical to the one
on Hugging Face — SHA-256
`134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2`.

## Reproduce it

```bash
git clone https://github.com/Ckal/connect-four-gf256.git
cd connect-four-gf256

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Train (seed 32, 100k steps, batch 16, ctx 192)
V24_TAG=v24 bash run_v24.sh

# Score any checkpoint against its owner corpus
python -u tools/measure_promotion_scorer.py \
  --corpus-run runs/_probe/v24 \
  runs/_probe/v24/checkpoints/best_hard.pt

# Full regression suite
bash tools/run_regression_sweep.sh
```

The scorer **refuses to guess** which corpus belongs to a checkpoint: pass
`--corpus-run` explicitly, or let it find an unambiguous ancestor owner
run. Without one it exits non-zero rather than scoring against the wrong
data.

Every published number traces to a command and an artifact — see
[`RESULTS.md`](RESULTS.md).

## Bugs that had to be fixed first

This is the part most write-ups skip. Each of these produced a
*plausible-looking* number that was wrong.

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

Fixing this also invalidated the old "always play column 3" control: it
had actually been a centre-first *staircase*, not a constant policy.

## Reproducibility guarantees

| Property | Evidence |
|---|---|
| Bit-exact resume | max weight Δ **0.000e+00**, identical Adam moments (12/12 checks) |
| Full regression suite | **27 PASS, 0 FAIL** — reproduced by independent sweeps |
| Live UI parity | 40/40 played columns, next-move distribution L1 = **0.000e+00** |
| Atomic checkpoint publication | 9/9 checks; 0 corrupt reads in 769 reads |
| Transcript stamping | 3/3 checks |

Component suites: live parity v3/v4/v5 · UI hot reload 14/14 · resume
state 19/19 · resume equivalence 12/12 · single-server UI 8/8 · atomic
save 9/9 · transcript stamp 3/3 · watcher 12/12 · sessions 47/47 ·
embedding + origin policy 18/18.

## Layout

```
.
├── README.md                  this file
├── RESULTS.md                 every published number → command + artifact
├── CONNECT_FOUR_PAPER.md     the Connect Four manuscript
├── FOLLOWUP_PLAN.md           what comes next
├── MODEL_CARD.md              the model card
├── c4_001-006.gif             the demo recording
├── requirements.txt           pinned runtime dependencies

src/
├── c4/
│   ├── train.py               trainer: dense per-ply supervision, holdout, promotion
│   ├── agent.py               hard export, prompts, greedy policy, legal masking
│   ├── board.py               board + legal moves + terminal check
│   ├── duel.py                canonical strength harness
│   ├── opponents.py           random / centre / heuristic / lookahead / minimax
│   ├── transcript_v2.py       canonical transcript decoder
│   ├── corpus_integrity.py    corpus + transcript validation
│   ├── sessions.py            per-session boards and cookies
│   └── ui_server.py           the game UI (see the multi-user note above)
├── gf256_markov_lm_v6.py      the shipped architecture
└── generate.py                text-generation driver

tests/                         the regression suite
tools/                         the authoritative measurement scripts
scripts/verify_public_isolation.py   the public deployment gate
docs/                          the full research record
```

## Limitations

1. **Net-negative vs. the 1-ply heuristic** — the headline negative result.
2. **Teacher-agreement ≠ game strength** — the central finding.
3. **Holdout solver coverage is 37.3 %** (560/1,500). The other 940
   positions have no proved label and are excluded, not guessed.
4. **~18 pp train/holdout gap** indicates memorisation in the headline
   number.
5. **Architecture-level superiority is unproven** — the controlled A/B arms
   reached only 0.30 epoch-equivalents and are provisional.
6. **Mirror/symmetry questions remain open** for the corrected model.

## Citation

```bibtex
@misc{ckal2026gf256_connectfour,
  title  = {Reading the Board: A GF(256) Integer Model for Connect Four},
  author = {Ckal, Christof},
  year   = {2026},
  url    = {https://github.com/Ckal/connect-four-gf256}
}
```

## License

Code: MIT (see `LICENSE`). Paper text: © 2026 Christof Ckal.
Model weights: MIT.
