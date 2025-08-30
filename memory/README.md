HRM Memory‑RAG (Lightning Swarm) — Quick Notes

Pieces
- Encoding/decoding: `memory/encode_decode.py`
- Dataset builder: `dataset/build_memory_dataset.py`
- HRM config with utility/risk heads: `config/arch/hrm_memory.yaml`
- Batched controller service: `services/lightning_controller/`

Build a dataset
```
python HRM/dataset/build_memory_dataset.py --input-jsonl path/to/data.jsonl --output-dir data/memory-rag --seq-len 256 --vocab-size 256
```

Train (single GPU)
```
python HRM/pretrain.py data_path=data/memory-rag arch=arch/hrm_memory
```

Run controller
```
pip install -r HRM/requirements.txt
python -m HRM.services.lightning_controller.app
```

Basic flow (HTTP)
1) POST /initialize  — model+scheduler
2) POST /seed        — seed branches with inputs
3) POST /step        — advance selected branches (batched)
4) GET  /best        — get top advice text (placeholder)
5) GET  /status      — inspect branch scores
6) POST /prune       — remove weak branches

Notes
- Encoding/Advice DSL here are placeholders; replace with your structured schema.
- For Windows, FlashAttention is optional (PyTorch SDPA fallback is active).

