# Release Record — GF(256) Connect Four

> Single auditable source of truth for the Connect Four release.
> Every number in the paper, README, model card, blog post and social
> posts MUST be traceable to a line in this file.
>
> Author of record: **Christof Ckal** (`@zX14_7` on X, `Ckal` on GitHub).
> This release must never be attributed to Alex.

**Status:** IN PROGRESS — training live at the time of writing.
**Record opened:** 2026-10-29
**Project:** ZeroFLops

---

## 0. Why this file exists

Chat memory is not an audit trail. Every measurement that is published
publicly is copied here with:

- the exact command that produced it,
- the artifact path it came from,
- the file mtime / commit that fixes its provenance,
- and the scope limits that must be stated alongside it.

If a number is not in this file, it is not publishable.

---

## 1. Identity and attribution

| Field | Value | Verified |
|---|---|---|
| Public name | Christof Ckal | user instruction |
| X / Twitter | `https://x.com/zX14_7` | confirmed as *Christof Kaller* |
| GitHub (public source) | **`Ckal/connect-four-gf256`** — https://github.com/Ckal/connect-four-gf256 | **PUBLISHED 2026-10-29.** Repo verified public via `gh api repos/Ckal/connect-four-gf256`. Active `gh` account switched to `Ckal` (API: login `Ckal`, name `Christof Kaller`, email `Christof.Kaller@gmail.com`, blog `ckaller.blogspot.de`). Clean-history staging via `tmp/stage_public_repo.sh` (the source history holds a 118 MB blob that exceeds GitHub's 100 MB limit). Local Windows paths sanitized to `C:\Users\<user>` before publish. |
| Blog | `ckaller.blogspot.de` | Blogger token present at `~/AppData/Local/hermes/google_token.json` |
| LinkedIn | `de.linkedin.com/in/christof-kaller-6b043733` | profile found |
| Hugging Face model | **`Chris4K/connect-four-gf256`** | **PUBLIC — live 2026-10-29**, anonymous read verified (HTTP 200). Published artifact's SHA-256 re-verified after download: `134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2` — byte-identical to the frozen release copy. |
| Hugging Face namespace | **`Chris4K`** — VERIFIED, write-tested | Email-verified `Christof.kaller@gmail.com`, fullname "CK". A repo was created and deleted under this namespace as a write probe. The `ckal` HF namespace belongs to **a different person** and must not be used. |

> **HF namespace resolved.** The live `hf_hackathon_git` token authenticates as
> **`Chris4K`** — email-verified `Christof.kaller@gmail.com`, fullname "CK" —
> with the **write** role. Write access was confirmed by creating a probe repo
> (`Chris4K/_writeprobe`) and then deleting it; the delete was confirmed with
> `RepositoryNotFoundError`. The model will publish as
> **`Chris4K/connect-four-gf256`**.
>
> Namespaces verified as belonging to *other* people and therefore off-limits:
> `ckal` (Bob Kaleugher), `christof` (Christof Roduner). The CLI's
> `hf auth whoami` reports a stale `AlexChen1403` label; the token's own
> `whoami-v2` response is the authoritative answer and says `Chris4K`.

**GitHub account note.** `gh` has two accounts: `alexchen1403-prog` (active)
and `Ckal`. The public release uses `Ckal`. `gh auth switch --user Ckal` is
the switch; verify with `gh auth status` before pushing.

---

## 2. Repository hosting decision

| Candidate | Status | Verdict |
|---|---|---|
| GitHub `Ckal/<repo>` | name free (HTTP 404 for 5 candidate names) | **CHOSEN — public canonical home** |
| Gitea `git.tensorgarten.de/Chris4k/zeroflops` | repository **does not exist**; push-to-create disabled; SSH authenticates as `Hermes` | internal/mirror only |
| Netcup host `159.195.150.113` | SSH works with `netcup_deploy_ed25519` | **host for the live game** |

**Blocker (GitHub push).** The git history contains a **118,405,919-byte**
blob (`artifacts/gf256_hard_model.pt`, introduced in commit `186d176`).
GitHub rejects any blob > 100 MB and the limit is **not** raised for
accounts by default. A plain `git push` to GitHub **will fail**.

Remedies (pick one, in preference order):

1. **Fresh repository.** Create `Ckal/<name>` as a new repo with a clean
   history containing only the Connect Four release. Keeps the 118 MB
   language-model artifact out of the public tree entirely and is the
   cleanest scholarly artifact. The text-generation paper keeps its own
   home.
2. **History rewrite.** `git filter-repo --path artifacts/gf256_hard_model.pt
   --invert-paths` (not currently installed) or `git filter-branch`. Rewrites
   a long history and invalidates the commit hashes referenced in
   `docs/CURRENT_TRUTH.md`. Only worth it if the old paper must live in the
   same repo.
3. **Git LFS.** Does not help: LFS objects still have a 100 MB per-file limit
   and the repo must be LFS-enabled before the first push.

**Decision:** option 1 — a new, dedicated public repository for the
Connect Four work, plus a separate home for the text-generation paper.
The old `paper.md` and the `artifacts/` blob stay in this private working
repository.

---

## 3. The existing text-generation paper

`paper.md` is a **text-generation** paper: *"GF(256) Lagrange Field
Transformer — A Discrete Algebraic Language Model Compiled to Zero-FLOP
Integer Programs"*, author Christof Kaller, Hard BPC 3.071 on Tiny
Shakespeare.

It contains **no Connect Four chapter**. It is therefore **preserved
unchanged**. The Connect Four work gets its **own manuscript**
(`release/CONNECT_FOUR_PAPER.md`) with a shared-architecture section that
cross-references the text-generation paper rather than duplicating it.

This avoids the failure mode of silently relabelling a language-model
paper as a game-playing paper.

---

## 4. Model artifact inventory

### 4.1 The promoted checkpoint (the release artifact)

| Field | Value |
|---|---|
| Path | `runs/_probe/v24/checkpoints/best_hard.pt` |
| Size | 33,830,949 bytes (32.3 MB) |
| Step | 18000 (stored at checkpoint top level, **not** under `meta`) |
| Transcript version | v2 (raw move bytes `0x41..0x47`) |
| Lags | `[0,1,2,3,4,5,6,7,8,9,10,20]` |
| Table bias | recorded in checkpoint (`table_bias` key) |
| Model seed | recorded in checkpoint (`model_seed` key) |
| Authoritative LABEL score | **59.55 %** |
| Parameter count | 8,422,496 elements (emb 8192, coef 288, sbox 24576, wiring 576, readout 8388608, bias 256) |

**Provenance:** produced by `run_v24.sh` (tag `v24`), 100,000 requested
steps, batch 16, context 192, seed 32, transcript v2, disjoint holdout
`runs/_probe/disjoint_holdout.corpus.txt` wired in.

### 4.2 Pinned comparison checkpoints

| Artifact | Step | Authoritative LABEL |
|---|---|---|
| `runs/_probe/compare/best_hard_step18000.pt` | 18000 | 59.55 % |
| `runs/_probe/compare/step20000_hard.pt` | 20000 | 59.00 % |
| `runs/_probe/compare/step22000_hard.pt` | 22000 | 58.52 % |
| `runs/_probe/compare/step24000_hard.pt` | 24000 | 58.55 % |
| `runs/_probe/compare/step26000_hard.pt` | 26000 | 58.70 % |

**Scoring command (authoritative):**

```bash
./.venv-c4/Scripts/python.exe -u tmp/measure_promotion_scorer.py \
  --corpus-run runs/_probe/v24 \
  runs/_probe/compare/step20000_hard.pt
```

The scorer **refuses to guess** a corpus. Without `--corpus-run` and
without an unambiguous ancestor owner run it exits non-zero. This is
deliberate: an earlier fallback silently picked a 500-row unrelated
corpus and reported an inflated 64.28 % instead of the true 59.00 %.

### 4.3 What is NOT released

- The 320,000-row training corpus and its sidecar (private corpus).
- Historical runs (~50 GB), probe scratch, logs, and the 118 MB
  text-generation artifact.
- The exact-solver relabelling tooling's intermediate data.

---

## 5. Measured results — the honest picture

### 5.0 Authoritative numbers — THE release ledger

> **Gate: these are the only numbers that may be published.** Every document
> (paper, README, model card, blog, social posts) cites this table and nothing
> else. If a number is not here, it is not publishable.

Recorded 2026-10-29 08:47. The strength rows come from the 200-game × 3-seed
matrix in `runs/_probe/ingame_strength_step*.json`; the label rows come from
the authoritative scorer against the owner corpus `runs/_probe/v24`.

| Key | Value | Source |
|---|---|---|
| `artifact_step` | **18000** | `best_hard.pt` top-level `step` |
| `artifact_path` | `runs/_probe/v24/checkpoints/best_hard.pt` | promoted checkpoint |
| `artifact_bytes` | 33,830,949 | `stat` |
| `parameters` | 8,422,496 | `sum(p.numel() for p in model.parameters())` |
| `artifact_sha256` | `134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2` | `sha256sum` of the frozen copy |
| `label_agreement_train` | **59.55 %** | `tmp/measure_promotion_scorer.py --corpus-run runs/_probe/v24`; also `metrics.jsonl` `perply_agreement_pct` at step 18000 |
| `label_agreement_holdout` | **41.02 %** | `metrics.jsonl` `holdout_perply_agreement_pct` at step 18000 (n=512) |
| `corpus_games` | 320,000 | trainer preflight row count |
| `corpus_labels` | 4,462,571 | per-ply label count |
| `corpus_boards_scanned` | 5,100,175 | corpus integrity scan |
| `corpus_violations` | 0 | preflight |
| `transcript_version` | v2 | checkpoint meta |
| `context_width` | 192 | trainer `--ctx` |
| `batch_size` | 16 | trainer `--batch` |
| `seed` | 32 | trainer `--seed` |
| `lags` | 0–10, 20 | checkpoint meta |
| `table_bias` | 0.0 | trainer `--table-bias` |
| `inference` | CPU-only | `exp_t`/`log_t` are CPU tensors |
| `solver_holdout_coverage` | 560 / 1,500 = **37.3 %** | exact-solver holdout audit |
| **strength vs random** | **85.17 %** (511-88-1, n=600) | §5.4 |
| **strength vs centre_policy (head-to-head)** | **90.50 %** (543-57-0, n=600) — model **wins** | §5.4 |
| **strength vs random, centre_policy** | **91.00 %** (546-54, n=600) | §5.4 — centre_policy scores *higher* against random than the model does |
| **strength vs 1-ply heuristic** | **37.17 %** (223-376-1, n=600) — model **loses** | §5.4 |
| **strength vs 2-ply lookahead** | **24.33 %** (146-452-2, n=600) — model **loses** | §5.4 |
| **strength vs minimax** | **41.83 %** (251-343-6, n=600) | §5.4 |
| `best_constant_baseline` | `centre_policy` (546-54 vs random) | §5.4 |
| `model_vs_centre_policy_fisher_p` | **0.00236** | **centre_policy is significantly stronger than the model** when both are measured against random. Fisher implementation self-tested against [[1,9],[11,3]] = 0.00275946. |
| `col_recall_train_c0` | 63.7 % | `metrics.jsonl` `perply_col_recall[0]` |
| `col_recall_train_c1` | 56.5 % | `perply_col_recall[1]` |
| `col_recall_train_c2` | 56.2 % | `perply_col_recall[2]` |
| `col_recall_train_c3` | 73.9 % | `perply_col_recall[3]` — the centre, the strongest |
| `col_recall_train_c4` | 50.1 % | `perply_col_recall[4]` |
| `col_recall_train_c5` | 43.3 % | `perply_col_recall[5]` |
| `col_recall_train_c6` | 43.7 % | `perply_col_recall[6]` |
| `col_recall_holdout_c1` | 42.1 % | `holdout_perply_col_recall[1]` |
| `col_recall_holdout_c5` | 18.6 % | `holdout_perply_col_recall[5]` — weakest |
| `col_recall_holdout_c6` | 7.1 % | `holdout_perply_col_recall[6]` — weakest |

**The c1–c4 finding.** The model is measurably better on the **centre column
(c3, 73.9 % recall)** than on the flanks, and its worst columns are the
**outer ones** (c5 43.3 %, c6 43.7 % on train; 18.6 % and 7.1 % on holdout).
This is the residual of the historical "always-c3" defect: the repair fixed the
*hard* centre bias, but a soft centre preference survives. The model's
weakest play is on the outer columns — exactly where a centre-column policy
is never challenged.

**The one-line truth:** the model reaches **85.2 % vs random**, **beats the
centre-column policy head-to-head (90.5 %)**, but is **net negative against a
1-ply heuristic (37.2 %)** and is **significantly weaker than the
centre-column policy when both are measured against random** (85.2 % vs.
91.0 %, p = 0.00236). It is not a strong Connect Four player.

### 5.1 The convergence curve (V24, tag `v24`)

Source: `runs/_probe/v24/metrics.jsonl`, 4000-sample holdout, 512-position
holdout per-ply subset, 48-game ladder vs. random and heuristic.

| Step | LABEL % (train) | H-LABEL % (holdout) | vs heuristic W-L | vs random W-L |
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

- **Label agreement rose from 16.3 % → ~59 %.** This is the headline
  supervised metric and it is a real, large improvement over the
  corrected short-run baseline (16.3 % at step 0).
- **Label agreement has plateaued** since roughly step 14000–18000 at
  57–60 % train / 38–42 % holdout. More steps are buying very little.
- **The model beats random decisively** (85.2 % win rate; 200-game x 3-seed matrix).
- **The model is NET NEGATIVE against the 1-ply heuristic** for the entire
  run (14-34, 18-30, 23-25 at the best points). It never reaches parity,
  let alone superiority.

### 5.2 The honest statement of strength

> This model learns the board. It is not a strong Connect Four player.

Two independent lines of evidence:

1. **Ladder:** net-negative vs. the 1-ply heuristic at every measured
   step of the entire run (best: 23-25 at step 14000).
2. **In-game matrix** (historical `fx_a`, canonical `src.c4.duel`,
   600 games/arm/seed × 3 seeds, model greedy, opponent explores):
   the model beat the *best single-column constant* in **2 of 5** opponent
   settings — it was **not** a clean sweep.

The heuristic is a 1-ply rule: *win in 1 > block opponent's win in 1 >
centre tiebreak*. A model that cannot match that is not yet a capable
player, regardless of how good its label agreement looks.

**The gap between 59 % label agreement and losing to a 1-ply rule is the
central research finding of this release**, and the honest thing to
publish. Label agreement measures *agreement with the teacher on
labelled positions*; it is not a proxy for game strength. The holdout
gap (59 % train → ~41 % holdout) shows part of the train figure is
memorisation.

### 5.3 Head-to-head: promoted 18000 vs. pinned 20000

```
W208 L192  (52.0 %, one-sided p = 0.45)
82.5 % same-column agreement
first divergence at ply 16
```

Statistically inconclusive on the direct duel. **But the 200-game strength
matrix resolves it** (§5.4): step 18000 is ahead of step 20000 on **all five**
opponent arms. Both the label metric and actual play agree — and they agree
in the same direction, which is the point.

| Opponent | step 18000 | step 20000 | Better |
|---|---:|---:|---|
| random | **85.17 %** | 84.67 % | 18000 |
| centre_policy | **90.50 %** | 86.67 % | 18000 |
| 1-ply heuristic | **37.17 %** | 30.17 % | 18000 |
| 2-ply lookahead | **24.33 %** | 24.00 % | 18000 (tie) |
| minimax | **41.83 %** | 41.33 % | 18000 (tie) |

**Promoted 18000 is the recommended playable model.**

Greedy-only 200-game comparison: 100-100 with 2 unique trajectories —
a null result by construction, reported for completeness.

### 5.3.1 Checkpoint selection is evidence-based, not "newest wins"

Three checkpoints were measured against the full five-opponent matrix
(600 games per arm). **Step 18 000 — the promoted checkpoint — wins every
single arm.** Later steps are progressively weaker:

| Checkpoint | vs random | vs heuristic | vs 2-ply | vs minimax | vs centre (head-to-head) |
|---|---:|---:|---:|---:|---:|
| **18 000 (released)** | **85.2 %** | **37.2 %** | **24.3 %** | **41.8 %** | **90.5 %** |
| 20 000 | 84.7 % | 30.2 % | 24.0 % | 41.3 % | 86.7 % |
| 40 000 | 83.8 % | 32.3 % | 18.2 % | 38.3 % | 82.7 % |

This is the same plateau seen in the teacher-agreement curve: LABEL peaked at
59.55 % at step 18 000 and then oscillated between 57.9 % and 59.4 % for
28 000 further steps, while play strength *declined*. The highest-step
checkpoint is **not** the best checkpoint, and the release ships the one
that measured best.

### 5.4 The 200-game strength matrix (canonical `src.c4.duel`)

Protocol: **200 games per arm per seed × 3 seeds (99, 1234, 4242) = 600 games
per arm.** The model is **greedy** (ε=0) on the **hard** path; the opponent
explores (ε=0.3) except against random, which is fully random. Every arm must
reach ≥ 20 genuinely distinct legal trajectories; the minimum observed was
182/600. The script seeds the global RNG itself, because `src.c4.duel.duel`
ignores its own `seed` argument.

Reproduce:

```bash
./.venv-c4/Scripts/python.exe -u tmp/audit_ingame_strength_fd_a.py \
  --ckpt runs/_probe/v24/checkpoints/best_hard.pt \
  --out runs/_probe/ingame_strength_v24_best.json --games 200
```

**Step 18000 (the release artifact)** — `runs/_probe/ingame_strength_v24_best.json`:

| Opponent | Model W-L-D | Win % | Best constant baseline | Fisher p (model vs best) | Verdict |
|---|---:|---:|---|---:|---|
| random | 511-88-1 | **85.17 %** | centre_policy (546-54, **91.00 %**) | 0.00236 | **centre_policy is stronger than the model** (both vs random) |
| heuristic (1-ply) | 223-376-1 | **37.17 %** | always-c0 | 0.00007 | model beats the constant, still loses the opponent |
| lookahead (2-ply) | 146-452-2 | **24.33 %** | centre_policy | 0.0004 | model beats the constant |
| minimax | 251-343-6 | **41.83 %** | centre_policy | 0.3165 | no significant difference |
| **centre (head-to-head vs centre_policy)** | **543-57-0** | **90.50 %** | always-c0 (470-130) | 6.9e-09 | **the model WINS the head-to-head against centre_policy** |

**What this says, precisely:**

- The model beats random **decisively** (85.2 %, p ≈ 1e-67 vs. 50 %).
- The model **beats `centre_policy` head-to-head** — 543-57-0, **90.5 %** of
  600 games. This is the direct match: model as one player, `centre_policy`
  as the opponent.
- **But** `centre_policy` scores **91.0 % against random** (546-54) versus
  the model's **85.2 %** (511-88). Fisher exact **p = 0.00236** — a
  significant difference in `centre_policy`'s favour.
- Against the 1-ply heuristic the model wins only **37.2 %** of games.
- Against 2-ply lookahead it wins **24.3 %**.

**Why both centre facts are true.** They measure different things. Head-to-
head, `centre_policy` is *deterministic* — it always plays column 3, then
the leftmost legal column — so it is exploitable: the model blocks and
punishes it. Against a *random* opponent, a deterministic policy converts
more wins than a greedy model does, because a random opponent rarely
occupies the centre. Neither number is a head-to-head, and neither alone
settles "who is the better player"; the honest summary is that the model
**wins the direct match but has the weaker overall profile**.

`centre_policy` is deliberately *not* a solver: it is a simple central-column
rule. The 1-ply heuristic loss is the headline negative result of this
release.

**Step 20000** — `runs/_probe/ingame_strength_step20000.json`: 84.67 / 86.67 /
30.17 / 24.00 / 41.33 %. Worse than 18000 on every arm.

**Step 40000** — `runs/_probe/ingame_strength_step40000.json`: see the change
log. Measured to test whether "more steps" produces a stronger player; the
LABEL curve says no, and the matrix is the check.

---

## 6. Live game — DEPLOYED AND VERIFIED

**Status: live and publicly reachable since 2026-10-29.**

- **URL:** https://c4.tensorgarten.de
- **Host:** Netcup `159.195.150.113`, service `c4-game` in
  `/opt/netcup-infra/docker-compose.yml`, image
  `netcup-infra-c4-game` (built from
  `/opt/netcup-infra/apps/c4-game/app/Dockerfile`).
- **Runtime:** `python:3.12-slim` + `torch==2.5.1+cpu` + `numpy==1.26.4`.
  CPU-only: the GF(256) trunk is integer arithmetic and the float32
  read-out needs no GPU. The model is 32 MB, ~8.4 M parameters.
- **Serving mode:** `--public`. Administrative routes are disabled and the
  session cookie is marked `Secure`.
- **Container state:** `healthy`, 0 restarts.

### 6.1 The original blocker (now resolved)

The pre-refactor server kept the game in a **module-level global `_GAME`**,
so every visitor saw **the same board**. Two visitors would have corrupted
each other's game, and the UI could not be published.

The refactor replaced `_GAME` with a **per-session store** keyed by an
opaque, unguessable session id in an `HttpOnly` cookie, with TTL + eviction
so memory stays bounded. Model inference remains stateless: the checkpoint
is read-only and shared, and only the *board* is per-session — the model is
a pure function of (checkpoint, board), so no session state leaks into it.

### 6.2 The verification that gates the announcement

`tmp/verify_public_isolation.py` runs **against the deployed process over
public HTTPS**, not against a local server. A local 200 is not evidence of
a safe public deploy.

```
PASSED: 19   FAILED: 0
```

| Assertion | Result |
|---|---|
| `GET /` returns 200 with a real page | PASS (28,234 bytes) |
| Content-Security-Policy present | PASS |
| `X-Frame-Options: DENY` | PASS |
| `X-Content-Type-Options: nosniff` | PASS |
| `Referrer-Policy` present | PASS |
| `Strict-Transport-Security` present | PASS |
| `Cache-Control: no-store` | PASS |
| Both clients receive a session cookie | PASS (`tg_sid`) |
| The two session cookies differ | PASS |
| Session cookie is `HttpOnly` | PASS |
| Session cookie is `Secure` | PASS |
| Both boards start empty | PASS |
| Client A can make a legal move | PASS |
| Client A's board shows the move | PASS |
| **Client B's board is still empty — no shared state** | PASS |
| `GET /modelinfo` blocked | PASS (HTTP 403) |
| `GET /reload` blocked | PASS (HTTP 403) |
| The move response carries the new board state | PASS |

The raw cookie as served by the deployed container:

```
Set-Cookie: tg_sid=uH5kQSTeQt0-fIz0eyKNPoxKiXDLj0-U30kFZdwonEs;
            Path=/; HttpOnly; SameSite=Lax; Secure; Max-Age=7200
```

### 6.3 Checkpoint identity on the server

The checkpoint served by the deployed container is byte-identical to the
published Hugging Face artifact:

```
134035637b637f87f3837aa8929dc5ec5db3bb0f0c9526f93fe1449be77672e2
  /opt/netcup-infra/apps/c4-game/app/model/best_hard.pt
```

---

## 7. Publication plan and status

| # | Deliverable | Where | Status |
|---:|---|---|---|
| 1 | Release record (this file) | `release/RELEASE_RECORD.md` | **done** (live-updating) |
| 2 | Connect Four manuscript | `release/CONNECT_FOUR_PAPER.md` | **done** — published on GitHub + HF |
| 3 | Public README | `release/README.md` | **done** — published on GitHub + HF |
| 4 | New public GitHub repo under `Ckal` | https://github.com/Ckal/connect-four-gf256 | **done 2026-10-29** — public, verified via API |
| 5 | Model card + checkpoint on HF | https://huggingface.co/Chris4K/connect-four-gf256 | **done 2026-10-29** — public, SHA-256 re-verified after download |
| 6 | Multi-user UI refactor | `src/c4/ui_server.py` | **done** — commit `b286aee`; 47/47 unit, 26/26 integration, 17/17 hot-reload |
| 7 | Public deployment on Netcup | https://c4.tensorgarten.de | **done 2026-10-29** — live, `healthy`, 19/19 isolation+security assertions pass |
| 8 | Post-refactor regression sweep | `tmp/run_regression_sweep.sh` | **in progress** — re-running after the host reboot; 10 PASS / 0 FAIL at last check |
| 8b | Second paper (language model) | `release/GF256_LANGUAGE_MODEL_PAPER.md` | **done** — written and committed; publication destination pending |
| 9 | X post | `https://x.com/zX14_7` | pending (no stored X session yet) |
| 10 | Blog post | `ckaller.blogspot.de` | pending |
| 11 | LinkedIn post | — | pending |
| 12 | Short demo video | `release/media/` | pending (ffmpeg available; PIL via `python3`) |
| 13 | Product Hunt | — | pending (needs a repo/account decision) |
| 14 | Follow-up research plan | `release/FOLLOWUP_PLAN.md` | **done** — published on GitHub |

### 7.1 Promotional copy — constraints

- Attribute to **Christof Ckal**.
- **Do not** claim the model is strong at Connect Four. It is not, and the
  ladder says so.
- Safe claims: it reads the board; it beat random 85.2 %; supervised
  agreement rose 16 % → 59 %; it is a **research artifact** exposing a
  measured gap between teacher-agreement and game strength; it runs on
  CPU with a 32 MB integer-model trunk.
- The **most interesting** public claim is the negative result: *label
  agreement is not game strength*. That is a genuine contribution and it
  is fully supported by the measurements above.

---

## 8. Reproducibility

```bash
# Train (the exact V24 configuration)
V24_TAG=v24 bash run_v24.sh

# Score a checkpoint against its owner corpus
./.venv-c4/Scripts/python.exe -u tmp/measure_promotion_scorer.py \
  --corpus-run runs/_probe/v24 \
  runs/_probe/v24/checkpoints/best_hard.pt

# Run the full regression sweep (24 tests)
bash tmp/run_regression_sweep.sh

# Serve the game locally
python -m src.c4.ui_server --checkpoint runs/_probe/v24/checkpoints/best_hard.pt
```

### 8.1 Regression suite status (authoritative)

```
24 PASS, 0 FAIL
```

Reproduced independently by two separate sweeps
(`proc_1ca0d0ef8895`, `proc_bea3231cc441`). Component breakdown:

| Suite | Result |
|---|---|
| live parity v3 (40 columns, L1 = 0.000e+00) | PASS |
| live parity v4 / v5 | PASS |
| UI hot reload | 14/14 |
| resume state | 19/19 |
| resume equivalence (bit-exact, Adam Δ = 0.000e+00) | 12/12 |
| single-server UI | 8/8 |
| atomic checkpoint save | 9/9 |
| transcript stamp | 3/3 |
| watcher | 12/12 |

---

## 9. Known limitations (must appear in every public artifact)

1. **Net-negative vs. the 1-ply heuristic.** Documented above.
2. **Label agreement ≠ strength.** The central finding.
3. **Holdout coverage is partial.** Exact-solver holdout coverage is
   560 / 1,500 = 37.3 %. 940 positions remain unproved and are excluded.
4. **Train/holdout gap.** ~59 % train vs ~41 % holdout indicates
   memorisation contributes to the headline number.
5. **Mirror/symmetry conclusions are not settled.** The historical
   "per-ply LOST / mirror net negative / symmetry not worth it" findings
   predate the sidecar row-index repair and must **not** be cited as
   evidence about the corrected model.
6. **Architecture-level superiority is unproven.** The corrected A/B arms
   (fx_a/fx_b/fx_c) reached only 0.30 epoch-equivalents and settle
   per-ply vs. single-label supervision. They are provisional.
7. **The 100,000-step final result is not yet known.** Training is live.
   No number may be quoted for it until the run completes and the release
   battery passes.
8. **The old paper and this release are separate works** sharing an
   architecture.

---

## 10. Change log for this record

| Date | Entry |
|---|---|
| 2026-10-29 | Record opened. Identity fixed to Christof Ckal. Hosting decided: GitHub `Ckal` for source, Netcup for the live game, new HF namespace needed. GitHub 118 MB blob blocker documented. Multi-user `_GAME` blocker documented. Strength findings recorded honestly: net-negative vs heuristic. |
