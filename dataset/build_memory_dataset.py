from typing import Optional, List, Dict
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata
from memory.encode_decode import Vocab, encode_atoms_to_grid
from memory.advice_dsl import Advice as AdviceStruct, encode_advice_to_tokens


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_jsonl: str  # path to JSONL with {session_id, split, atoms: [...], advice_tokens: [...], puzzle_identifier}
    output_dir: str = "data/memory-rag"

    seq_len: int = 256
    vocab_size: int = 256

    # Optional subsample for quick experiments
    subsample_size: Optional[int] = None


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def convert_dataset(config: DataProcessConfig):
    vocab = Vocab(size=config.vocab_size)
    data = load_jsonl(config.input_jsonl)

    if config.subsample_size is not None and len(data) > config.subsample_size:
        data = data[: config.subsample_size]

    # Partition by split
    splits = {"train": [], "test": []}
    for row in data:
        split = row.get("split", "train")
        if split not in splits:
            split = "train"
        splits[split].append(row)

    num_identifiers = 1  # 0 is blank
    # Collect identifiers
    id_map: Dict[int, int] = {}
    for rows in splits.values():
        for r in rows:
            pid = int(r.get("puzzle_identifier", 0))
            if pid not in id_map:
                id_map[pid] = num_identifiers
                num_identifiers += 1

    for split_name, rows in splits.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        inputs_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        values_list: List[np.ndarray] = []
        risks_list: List[np.ndarray] = []
        puzzle_identifiers: List[int] = []
        puzzle_indices: List[int] = [0]
        group_indices: List[int] = [0]

        example_id = 0
        puzzle_id = 0
        last_sess = None

        for r in rows:
            sess = r.get("session_id", None)
            if last_sess is None:
                last_sess = sess
            # New puzzle per example (1 example per puzzle for now)
            atoms = r.get("atoms", [])
            inp = encode_atoms_to_grid(atoms, seq_len=config.seq_len, vocab=vocab)

            # Labels priority: advice_tokens > advice_dsl > inputs placeholder
            if "advice_tokens" in r:
                lab = np.array(r["advice_tokens"], dtype=np.int32)
            elif "advice_dsl" in r:
                # Best-effort encode structure into tokens
                try:
                    adv = AdviceStruct(**r["advice_dsl"])  # type: ignore
                except Exception:
                    adv = AdviceStruct(plan=r["advice_dsl"].get("plan", []))  # type: ignore
                lab = np.array(encode_advice_to_tokens(adv, seq_len=config.seq_len), dtype=np.int32)
            else:
                lab = np.array(inp.tolist(), dtype=np.int32)
            if lab.size < config.seq_len:
                lab = np.pad(lab, (0, config.seq_len - lab.size), constant_values=vocab.PAD)
            else:
                lab = lab[: config.seq_len]

            inputs_list.append(inp)
            labels_list.append(lab)

            # Optional utility/risk targets (scalar per example)
            v = r.get("value_target", r.get("value", 0.0))
            values_list.append(np.array([v], dtype=np.int32))
            rk = r.get("risk_target", r.get("risk", 0))
            risks_list.append(np.array([rk], dtype=np.int32))
            example_id += 1
            puzzle_id += 1
            puzzle_indices.append(example_id)
            puzzle_identifiers.append(id_map[int(r.get("puzzle_identifier", 0))])

            # Group by session boundary
            if sess != last_sess:
                group_indices.append(puzzle_id)
                last_sess = sess

        # Close last group
        group_indices.append(puzzle_id)

        inputs = np.stack(inputs_list, 0)
        labels = np.stack(labels_list, 0)

        np.save(os.path.join(config.output_dir, split_name, f"all__inputs.npy"), inputs)
        np.save(os.path.join(config.output_dir, split_name, f"all__labels.npy"), labels)
        np.save(os.path.join(config.output_dir, split_name, f"all__puzzle_identifiers.npy"), np.array(puzzle_identifiers, dtype=np.int32))
        np.save(os.path.join(config.output_dir, split_name, f"all__puzzle_indices.npy"), np.array(puzzle_indices, dtype=np.int32))
        np.save(os.path.join(config.output_dir, split_name, f"all__group_indices.npy"), np.array(group_indices, dtype=np.int32))
        # Save optional targets as 1-D arrays
        if values_list:
            np.save(os.path.join(config.output_dir, split_name, f"all__value_targets.npy"), np.concatenate(values_list, axis=0))
        if risks_list:
            np.save(os.path.join(config.output_dir, split_name, f"all__risk_targets.npy"), np.concatenate(risks_list, axis=0))

        metadata = PuzzleDatasetMetadata(
            seq_len=config.seq_len,
            vocab_size=config.vocab_size,
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=len(group_indices) - 1,
            mean_puzzle_examples=1,
            sets=["all"],
        )
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(), f)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w", encoding="utf-8") as f:
        json.dump(["<blank>"] + [str(k) for k, _ in sorted(id_map.items(), key=lambda x: x[1])], f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
