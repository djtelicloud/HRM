Lightning Controller (HRM Swarm) — Prototype

Endpoints
- POST `/initialize` — Initialize model
  - body: `{ arch_name, arch_overrides, vocab_size, seq_len, num_puzzle_identifiers, batch_size, checkpoint_path }`
- POST `/seed` — Register branches
  - body: `{ items: [{ inputs: int[], puzzle_identifier: int }, ...] }`
- POST `/step` — Advance active branches by one ACT step (batched)
  - body: `{ max_to_step: 32 }`
- POST `/encode` — Encode atom list into input tokens
  - body: `{ atoms: [...], seq_len? }`
- GET `/best` — Get top‑k advice decoded from last logits
  - query: `k=1`
- GET `/status` — Inspect all branches and scores
- POST `/prune` — Remove weak branches by min margin
  - body/query: `min_margin=-0.1`
- GET `/fuse` — Fuse top‑K branches into one advice (union/heuristic)
  - query: `k=3`

Run locally
```
pip install -r requirements.txt
python -m services.lightning_controller.app
```

Notes
- This is a minimal scaffold. Real systems should add scheduling (UCB/Thompson), budgets, calibration, and robust encoding/decoding.
- `/best` also returns a structured JSON (`advice_struct`) suitable for prompt injection or UI.
- Scheduler supports basic modes:
  - UCB (default): score = blend(value, margin) + avg_reward + explore − cost − novelty
  - TS: Thompson sampling around observed reward + base
- Novelty penalty can use cosine (repr) or LSH (binary signature) if `lsh_bits>0` in initialize overrides.
- For Windows single‑GPU, the repo falls back to PyTorch SDPA if FlashAttention is missing.
