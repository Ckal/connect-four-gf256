"""Connect-Four byte-corpus experiment for the GF(256) Lagrange Field
Transformer.

The model is NOT defined here. We import `gf256_markov_lm_v6.GFMarkovLM`
unchanged and only rewire its I/O: corpus format (transcripts instead of a
.txt file), loss mask (only moves), and a play loop for interactive use.

Public surface:
  board.py          Board, valid_actions, is_win, apply_move, outcome, ROWS, COLS
  transcript.py     board <-> bytes (transcript encode/decode round-trip)
  corpus.py         random-play and self-play game generation, writes byte files
  agent.py          play_game(model, opponent) -> transcript + outcome
  train.py          training loop; imports everything V6 does, swaps load_data
  play.py           interactive CLI (you vs the trained model)
"""

ROWS, COLS = 6, 7
WIN_LEN = 4
EMPTY, P1, P2 = 0, 1, 2
TOK_EMPTY, TOK_X, TOK_O = 0x2E, 0x58, 0x4F  # '.', 'X', 'O'
