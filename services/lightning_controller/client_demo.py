import requests
import random


def main():
    base = "http://127.0.0.1:8000"

    # Initialize model + scheduler
    init = {
        "vocab_size": 256,
        "seq_len": 64,
        "num_puzzle_identifiers": 128,
        "batch_size": 16,
        "arch_name": "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
        "arch_overrides": {"enable_value_head": True, "enable_risk_head": True, "halt_max_steps": 8},
        "q_margin_min": -0.05,
        "beta_explore": 0.8,
        "max_per_step": 16,
    }
    print("initialize:", requests.post(f"{base}/initialize", json=init).json())

    # Seed a few branches with random inputs
    items = []
    for i in range(8):
        seq = [random.randint(16, 255) for _ in range(init["seq_len"])]
        items.append({"inputs": seq, "puzzle_identifier": random.randint(1, 3)})
    print("seed:", requests.post(f"{base}/seed", json={"items": items}).json())

    # Step waves
    for _ in range(5):
        print("step:", requests.post(f"{base}/step", json={"max_to_step": 16}).json())
        print("best:", requests.get(f"{base}/best", params={"k": 1}).json())
        print("status:", requests.get(f"{base}/status").json())


if __name__ == "__main__":
    main()

