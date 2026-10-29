import torch
import math
import random
import argparse
from pathlib import Path

# ----------------------------------------------------------------------------
# GF(2^8) arithmetic tables (AES polynomial)
# ----------------------------------------------------------------------------
def build_gf_tables():
    exp = [0] * 512
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x2 = x << 1
        if x2 & 0x100: x2 ^= 0x11B
        x = x2 ^ x
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log

EXP, LOG = build_gf_tables()

class GFCipher:
    def __init__(self, pt_path="gf256_hard_model_v2.pt"):
        hard = self._load_hard_payload(pt_path)
        
        # Convert tensors to standard Python lists for speed
        self.emb = hard["emb"].tolist()
        self.coef = [c.tolist() for c in hard["coef"]]
        self.sbox = [s.tolist() for s in hard["sbox"]]
        self.readout = [r.tolist() for r in hard["readout"]]
        self.bias = hard["bias"].tolist()
        
        self.wiring_ch = [w[0].tolist() for w in hard["wiring"]]
        self.wiring_lg = [w[1].tolist() for w in hard["wiring"]]
        
        # Prefer serialized lags; keep a fallback for older artifacts.
        self.lags = hard.get("lags", [0, 1, 2, 4, 8, 16, 32, 64])
        
        self.d = len(self.emb[0])
        self.max_lag = max(self.lags)
        self.layers = len(self.sbox)
        self.fan_in = len(self.coef[0][0])
        
        print(f"Model loaded. {self.layers} layers, hidden dim {self.d}, max lag {self.max_lag}.")

    def _load_hard_payload(self, pt_path):
        path = Path(pt_path)
        print(f"Loading model payload from {path}...")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        if isinstance(payload, dict) and "model" in payload and payload.get("kind") == "hard_model":
            print(f"Detected saved hard-model artifact at step {payload.get('step', 'unknown')}.")
            return payload["model"]

        hard_keys = {"emb", "coef", "sbox", "readout", "bias", "wiring"}
        if isinstance(payload, dict) and hard_keys.issubset(payload.keys()):
            print("Detected direct hard-model payload.")
            return payload

        if isinstance(payload, dict) and {"model_state", "config"}.issubset(payload.keys()):
            print(f"Detected training checkpoint at step {payload.get('step', 'unknown')}; exporting hard model...")
            from originalGSM import GFMarkovLM

            cfg = payload["config"]["training"]
            data_info = payload.get("data", {})
            model = GFMarkovLM(
                d=cfg["d"],
                layers=cfg["layers"],
                fan_in=cfg["fan_in"],
                bigram_logp=None,
                seed=cfg["seed"],
            )
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval()
            return model.export_hard()

        raise ValueError(f"Unsupported file format for generation: {path}")
        
    def step(self, token, state_history):
        """
        Performs one forward pass for a single token using pure integer math.
        """
        # new_state[0] = embedding output
        # new_state[L+1] = output of layer L
        new_state = [[0]*self.d for _ in range(self.layers + 1)]
        new_state[0] = self.emb[token]
        
        # 1. Pass through layers
        for L in range(self.layers):
            acc = [0] * self.d
            ch = self.wiring_ch[L]
            lg = self.wiring_lg[L]
            coef = self.coef[L]
            
            for j in range(self.d):
                acc_j = 0
                for f in range(self.fan_in):
                    lag_val = self.lags[lg[j][f]]
                    src_ch = ch[j][f]
                    
                    # Fetch source value from history or current step
                    if lag_val == 0:
                        src_val = new_state[L][src_ch]
                    else:
                        if lag_val <= len(state_history):
                            src_val = state_history[-lag_val][L][src_ch]
                        else:
                            src_val = 0  # Padding for start of sequence
                            
                    c = coef[j][f]
                    # GF(256) Multiplication via Log/Exp tables
                    if c != 0 and src_val != 0:
                        prod = EXP[LOG[src_val] + LOG[c]]
                        acc_j ^= prod  # GF(256) Addition is XOR
                acc[j] = acc_j
                
            # S-box substitution (Lagrange polynomial equivalent)
            new_state[L+1] = [self.sbox[L][j][acc[j]] for j in range(self.d)]
            
        # 2. Calculate logits from readout tables
        logits = self.bias[:]
        for L_idx in range(self.layers + 1):
            h = new_state[L_idx]
            R = self.readout[L_idx]
            for j in range(self.d):
                r_row = R[j][h[j]]
                for v in range(256):
                    logits[v] += r_row[v]
                    
        return logits, new_state

    def generate(self, prompt, length=200, temperature=0.8):
        state_history = []
        tokens = [ord(c) for c in prompt]
        
        print(prompt, end="", flush=True)
        
        # Process prompt to build up initial memory state
        for tok in tokens:
            logits, new_state = self.step(tok, state_history)
            state_history.append(new_state)
            if len(state_history) > self.max_lag:
                state_history.pop(0)
                
        # Generate new text autoregressively
        for _ in range(length):
            logits, new_state = self.step(tokens[-1], state_history)
            state_history.append(new_state)
            if len(state_history) > self.max_lag:
                state_history.pop(0)
                
            # Softmax + Sampling
            if temperature > 0:
                max_logit = max(logits)
                # Subtract max for numerical stability
                probs = [math.exp((v - max_logit) / temperature) for v in logits]
                sum_p = sum(probs)
                probs = [p / sum_p for p in probs]
                
                r = random.random()
                c = 0.0
                next_tok = 255
                for i, p in enumerate(probs):
                    c += p
                    if r <= c:
                        next_tok = i
                        break
            else:
                next_tok = logits.index(max(logits))  # Greedy
                
            tokens.append(next_tok)
            char = chr(next_tok) if 32 <= next_tok < 127 else '?'
            print(char, end="", flush=True)
            
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=r"C:\Users\<user>\runs\gf64\checkpoints\step_000006000.pt",
        help="Path to a hard-model artifact, raw hard payload, or training checkpoint.",
    )
    parser.add_argument("--prompt", default="First Citizen:\n")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    cipher = GFCipher(args.model)
    print("\n--- Generation Start ---")
    cipher.generate(args.prompt, length=args.length, temperature=args.temperature)