from typing import List, Dict, Tuple

import numpy as np
import torch


# Minimal placeholder vocab for Memory-RAG experiments.
# In production, replace with learned tokenizer/codebook.
class Vocab:
    def __init__(self, size: int = 256):
        # Reserve a few control tokens
        self.PAD = 0
        self.EOS = 1
        self.SEP = 2
        self.STEP = 3
        self.min_reserved = 16
        self.size = max(size, self.min_reserved)

    def clamp(self, ids: np.ndarray) -> np.ndarray:
        return np.clip(ids, 0, self.size - 1)


def encode_atoms_to_grid(atoms: List[Dict], seq_len: int, vocab: Vocab) -> np.ndarray:
    """
    Turn a list of atom dicts into a flat integer sequence of length `seq_len`.
    This is a placeholder: it hashes keys/values into token ids deterministically.
    """
    tokens: List[int] = []
    for atom in atoms:
        # simple hash-based mapping; stable but lossy
        for k, v in atom.items():
            kh = (abs(hash(str(k))) % (vocab.size - vocab.min_reserved)) + vocab.min_reserved
            vh = (abs(hash(str(v))) % (vocab.size - vocab.min_reserved)) + vocab.min_reserved
            tokens.extend([kh, vh, vocab.SEP])

        tokens.append(vocab.STEP)

    tokens.append(vocab.EOS)
    if len(tokens) < seq_len:
        tokens.extend([vocab.PAD] * (seq_len - len(tokens)))
    else:
        tokens = tokens[:seq_len]

    return np.array(tokens, dtype=np.int32)


def decode_logits_to_advice(logits: torch.Tensor, vocab: Vocab) -> Dict[str, str]:
    """
    Convert logits into a compact advice string using the Advice DSL decoder.
    """
    try:
        from HRM.memory.advice_dsl import decode_logits_to_advice as dsl_decode, advice_to_text
    except Exception:
        from memory.advice_dsl import decode_logits_to_advice as dsl_decode, advice_to_text

    adv = dsl_decode(logits)
    return {"advice_text": advice_to_text(adv)}
