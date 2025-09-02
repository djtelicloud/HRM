import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from HRM.memory.advice_dsl import decode_logits_to_advice as dsl_decode
    from HRM.memory.encode_decode import Vocab, decode_logits_to_advice
    from HRM.models.hrm.hrm_act_v1 import (
        HierarchicalReasoningModel_ACTV1,
        HierarchicalReasoningModel_ACTV1Carry)
    from HRM.utils.functions import load_model_class  # when run as a package
except Exception:  # fallback when executed from HRM root
    from memory.advice_dsl import decode_logits_to_advice as dsl_decode
    from memory.encode_decode import Vocab, decode_logits_to_advice
    from models.hrm.hrm_act_v1 import (HierarchicalReasoningModel_ACTV1,
                                       HierarchicalReasoningModel_ACTV1Carry)
    from utils.functions import load_model_class


app = FastAPI(title="HRM Lightning Controller", version="0.1")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Branch:
    branch_id: int
    inputs: torch.Tensor            # [seq_len]
    puzzle_identifier: int
    carry: Optional[HierarchicalReasoningModel_ACTV1Carry]
    halted: bool = True
    steps: int = 0
    visits: int = 0
    reward_sum: float = 0.0
    last_scores: Optional[Dict[str, float]] = None  # q_margin, value, risk
    last_logits: Optional[torch.Tensor] = None
    last_repr: Optional[torch.Tensor] = None  # representation vector for novelty/diversity
    lsh_sig: Optional[torch.Tensor] = None    # bit signature for novelty


@dataclass
class SchedulerConfig:
    max_per_step: int = 32
    q_margin_min: float = -0.1
    value_min: Optional[float] = None
    risk_max: Optional[float] = None
    lambda_cost: float = 0.01
    beta_explore: float = 0.8
    diversity_penalty: float = 0.0
    mode: str = "ucb"  # or "ts" (Thompson sampling)
    lsh_bits: int = 0   # 0 disables LSH; else enable bit signatures for redundancy
    max_total_waves: Optional[int] = None
    max_branch_steps: Optional[int] = None


class InitRequest(BaseModel):
    # Minimal arch+data config for inference
    arch_name: str = "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1"
    arch_overrides: Dict = {}
    vocab_size: int = 256
    seq_len: int = 256
    num_puzzle_identifiers: int = 1024
    batch_size: int = 64
    checkpoint_path: Optional[str] = None
    # Scheduler overrides (optional)
    q_margin_min: Optional[float] = None
    value_min: Optional[float] = None
    risk_max: Optional[float] = None
    lambda_cost: Optional[float] = None
    beta_explore: Optional[float] = None
    max_per_step: Optional[int] = None
    diversity_penalty: Optional[float] = None
    mode: Optional[str] = None
    lsh_bits: Optional[int] = None
    max_total_waves: Optional[int] = None
    max_branch_steps: Optional[int] = None


class SeedRequest(BaseModel):
    items: List[Dict]  # each: {"inputs": List[int], "puzzle_identifier": int}


class StepRequest(BaseModel):
    max_to_step: int = 32


class EncodeRequest(BaseModel):
    atoms: List[Dict]
    seq_len: Optional[int] = None


class Advice(BaseModel):
    branch_id: int
    q_halt: float
    q_cont: float
    q_margin: float
    value: Optional[float] = None
    risk: Optional[float] = None
    confidence: Optional[float] = None


class Controller:
    def __init__(self):
        self.model: Optional[HierarchicalReasoningModel_ACTV1] = None
        self.branches: Dict[int, Branch] = {}
        self._next_id = 1
        self.sched = SchedulerConfig()
        self._top_reprs: List[torch.Tensor] = []  # cached best representations for redundancy penalty
        self._lsh_planes: Optional[torch.Tensor] = None  # [D, bits]
        self._global_steps: int = 0

    def initialize(self, req: InitRequest):
        # Build config from provided arch_overrides safely to avoid duplicate-key errors
        cfg = dict(req.arch_overrides or {})
        cfg.update({
            "batch_size": req.batch_size,
            "vocab_size": req.vocab_size,
            "seq_len": req.seq_len,
            "num_puzzle_identifiers": req.num_puzzle_identifiers,
            # preserve any explicit halt_max_steps from arch_overrides, else default to 12
            "halt_max_steps": (req.arch_overrides.get("halt_max_steps") if req.arch_overrides else None) or 12,
            "halt_exploration_prob": 0.0,
        })
        model_cls = load_model_class(req.arch_name)
        with torch.device(device()):
            self.model = model_cls(cfg)
            if self.model is not None:
                self.model.eval()
            if req.checkpoint_path and os.path.exists(req.checkpoint_path):
                try:
                    self.model.load_state_dict(torch.load(req.checkpoint_path, map_location=device()), assign=True)  # type: ignore
                except Exception:
                    self.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(req.checkpoint_path, map_location=device()).items()}, assign=True)  # type: ignore

        # Apply scheduler overrides
        if req.q_margin_min is not None:
            self.sched.q_margin_min = req.q_margin_min
        if req.value_min is not None:
            self.sched.value_min = req.value_min
        if req.risk_max is not None:
            self.sched.risk_max = req.risk_max
        if req.lambda_cost is not None:
            self.sched.lambda_cost = req.lambda_cost
        if req.beta_explore is not None:
            self.sched.beta_explore = req.beta_explore
        if req.max_per_step is not None:
            self.sched.max_per_step = req.max_per_step
        if req.diversity_penalty is not None:
            self.sched.diversity_penalty = req.diversity_penalty
        if req.mode is not None:
            self.sched.mode = req.mode
        if req.lsh_bits is not None:
            self.sched.lsh_bits = req.lsh_bits
        if req.max_total_waves is not None:
            self.sched.max_total_waves = req.max_total_waves
        if req.max_branch_steps is not None:
            self.sched.max_branch_steps = req.max_branch_steps

        # Apply scheduler overrides
        if req.q_margin_min is not None:
            self.sched.q_margin_min = req.q_margin_min
        if req.value_min is not None:
            self.sched.value_min = req.value_min
        if req.risk_max is not None:
            self.sched.risk_max = req.risk_max
        if req.lambda_cost is not None:
            self.sched.lambda_cost = req.lambda_cost
        if req.beta_explore is not None:
            self.sched.beta_explore = req.beta_explore
        if req.max_per_step is not None:
            self.sched.max_per_step = req.max_per_step
        if req.diversity_penalty is not None:
            self.sched.diversity_penalty = req.diversity_penalty

    def seed(self, items: List[Dict]) -> List[int]:
        ids = []
        for it in items:
            inp = torch.tensor(it["inputs"], dtype=torch.int32, device=device()).view(1, -1)
            pid = int(it.get("puzzle_identifier", 0))
            b = Branch(branch_id=self._next_id, inputs=inp, puzzle_identifier=pid, carry=None, last_scores={})
            self.branches[self._next_id] = b
            ids.append(self._next_id)
            self._next_id += 1
        return ids

    def _collate(self, branches: List[Branch]):
        inputs = torch.cat([b.inputs for b in branches], dim=0)
        pids = torch.tensor([b.puzzle_identifier for b in branches], dtype=torch.int32, device=device())
        batch = {"inputs": inputs, "puzzle_identifiers": pids}

        # Build carry batch
        # Prepare per-branch carries (initialize on first use)
        carries = []
        for b in branches:
            if b.carry is None:
                with torch.device(device()):
                    single_pid = torch.tensor([b.puzzle_identifier], dtype=torch.int32, device=device())
                    c = self.model.initial_carry(batch={"inputs": b.inputs, "puzzle_identifiers": single_pid})  # type: ignore
                b.carry = c
            carries.append(b.carry)

        # Concatenate carries
        with torch.device(device()):
            inner_carry = self.model.inner.empty_carry(len(branches))  # type: ignore
        inner_carry.z_H = torch.stack([c.inner_carry.z_H.squeeze(0) for c in carries], dim=0)
        inner_carry.z_L = torch.stack([c.inner_carry.z_L.squeeze(0) for c in carries], dim=0)

        # Steps and halted
        steps = torch.tensor([int(c.steps.item()) for c in carries], dtype=torch.int32, device=device())
        halted = torch.tensor([bool(c.halted.item()) for c in carries], dtype=torch.bool, device=device())

        # Current data: if empty, use batch; else concat
        if len(carries[0].current_data):
            current_data = {k: torch.cat([c.current_data[k] for c in carries], dim=0) for k in carries[0].current_data.keys()}
        else:
            current_data = {"inputs": inputs, "puzzle_identifiers": pids}

        carry_batched = HierarchicalReasoningModel_ACTV1Carry(inner_carry=inner_carry, steps=steps, halted=halted, current_data=current_data)
        return batch, carry_batched

    def _scatter(self, branches: List[Branch], carry_batched: HierarchicalReasoningModel_ACTV1Carry):
        # Scatter carry back to branches
        for i, b in enumerate(branches):
            inner = carry_batched.inner_carry
            new_c = HierarchicalReasoningModel_ACTV1Carry(
                inner_carry=type(inner)(z_H=inner.z_H[i:i+1], z_L=inner.z_L[i:i+1]),
                steps=carry_batched.steps[i],
                halted=carry_batched.halted[i],
                current_data={k: v[i:i+1] for k, v in carry_batched.current_data.items()},
            )
            b.carry = new_c
            b.halted = bool(new_c.halted.item())
            b.steps = int(new_c.steps.item())

    def _select_branches(self, max_to_step: int) -> List[Branch]:
        pool = [b for b in self.branches.values() if not b.halted]
        if not pool and self.branches:
            # First wave: allow all to get initial scores
            for b in self.branches.values():
                b.halted = False
            pool = list(self.branches.values())
        if not pool:
            return []

        total_visits = 1 + sum(b.visits for b in pool)
        scored: List[Tuple[float, Branch]] = []
        import math
        for b in pool:
            q_margin = (b.last_scores or {}).get("q_margin", 0.0)
            value = (b.last_scores or {}).get("value", 0.0)
            avg = (b.reward_sum / max(1, b.visits)) if b.visits > 0 else 0.0
            # UCB explore bonus (used for mode==ucb)
            explore = self.sched.beta_explore * math.sqrt(max(0.0, math.log(total_visits)) / (1 + b.visits))
            cost_pen = self.sched.lambda_cost * b.steps
            base = 0.5 * q_margin + 0.5 * value
            # Redundancy penalty via cosine or LSH hamming sim
            red_pen = 0.0
            if self.sched.diversity_penalty > 0.0 and (b.last_repr is not None or b.lsh_sig is not None):
                if self.sched.lsh_bits and (b.lsh_sig is not None) and hasattr(self, "_top_sigs"):
                    sims = []
                    for sig in getattr(self, "_top_sigs", []):
                        # fraction of matching bits
                        same = (sig == b.lsh_sig).float().mean().item()
                        sims.append(same)
                    if sims:
                        red_pen = self.sched.diversity_penalty * max(sims)
                elif self._top_reprs and (b.last_repr is not None):
                    br = b.last_repr
                    brn = br / (br.norm(p=2) + 1e-6)
                    sims = []
                    for tr in self._top_reprs:
                        trn = tr / (tr.norm(p=2) + 1e-6)
                        sims.append(float(torch.dot(brn, trn).item()))
                    if sims:
                        red_pen = self.sched.diversity_penalty * max(sims)

            if self.sched.mode == "ts":
                # Thompson sampling: sample from a Gaussian around observed avg reward + base
                # Variance decays with visits
                var = 1.0 / max(1, b.visits)
                noise = torch.randn((), device=b.last_logits.device if b.last_logits is not None else None).item() * (var ** 0.5)
                score = base + avg + noise - cost_pen - red_pen
            else:
                score = base + avg + explore - cost_pen - red_pen
            scored.append((score, b))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:max_to_step]]

    def _early_stop(self, b: Branch) -> bool:
        qm = (b.last_scores or {}).get("q_margin", 0.0)
        if qm < self.sched.q_margin_min:
            return True
        v = (b.last_scores or {}).get("value", None)
        if (self.sched.value_min is not None) and (v is not None) and (v < self.sched.value_min):
            return True
        r = (b.last_scores or {}).get("risk", None)
        if (self.sched.risk_max is not None) and (r is not None) and (r > self.sched.risk_max):
            return True
        # Branch step cap (scheduler override preferred)
        max_steps = self.sched.max_branch_steps or (self.model.config.halt_max_steps if self.model is not None else None)  # type: ignore
        if max_steps is not None and b.steps >= max_steps:
            return True
        return False

    def step(self, max_to_step: int) -> List[Advice]:
        # Global waves budget
        if self.sched.max_total_waves is not None and self._global_steps >= self.sched.max_total_waves:
            return []
        batch_branches = self._select_branches(max_to_step)
        if not batch_branches:
            return []
        batch, carry = self._collate(batch_branches)

        # Forward
        with torch.inference_mode():
            new_carry, outputs = self.model(carry=carry, batch=batch)  # type: ignore

        # Scatter
        self._scatter(batch_branches, new_carry)

        # Collect advice
        adv: List[Advice] = []
        qh = outputs["q_halt_logits"].detach().float().cpu()
        qc = outputs["q_continue_logits"].detach().float().cpu()
        vals = outputs.get("value")
        risks = outputs.get("risk_logit")
        reprs = outputs.get("repr")

        for i, b in enumerate(batch_branches):
            q_hi = float(qh[i].item())
            q_ci = float(qc[i].item())
            margin = q_ci - q_hi
            v = float(vals[i].item()) if vals is not None else None
            r = float(risks[i].item()) if risks is not None else None

            b.visits += 1
            b.reward_sum += float(margin)
            b.last_scores = {"q_margin": float(margin), "value": v if v is not None else 0.0, "risk": r if r is not None else 0.0}
            # Store logits snapshot for optional decoding
            b.last_logits = outputs["logits"][i].detach().cpu()

            # Save representation for redundancy control
            if reprs is not None:
                rep = reprs[i].detach().cpu()
                b.last_repr = rep
                # Initialize LSH planes if enabled and unknown
                if self.sched.lsh_bits and self._lsh_planes is None:
                    D = rep.numel()
                    self._lsh_planes = torch.randn(D, self.sched.lsh_bits)
                if self._lsh_planes is not None and self.sched.lsh_bits:
                    sig = (rep.view(-1) @ self._lsh_planes > 0).to(torch.int8)
                    b.lsh_sig = sig

            # Controller-level early stop rules (independent of model eval batching)
            b.halted = (q_ci <= q_hi) or self._early_stop(b)

            # Confidence (sigmoid margin adjusted by risk)
            import math
            conf = 1.0 / (1.0 + math.exp(-margin))
            if r is not None:
                # risk_logit -> prob
                risk_prob = 1.0 / (1.0 + math.exp(-r))
                conf = conf * (1.0 - risk_prob)

            adv.append(Advice(branch_id=b.branch_id, q_halt=q_hi, q_cont=q_ci, q_margin=margin, value=v, risk=r, confidence=conf))

        # Update top caches for redundancy penalty (keep a few best reprs/sigs)
        ranked = sorted(
            [b for b in self.branches.values() if b.last_repr is not None],
            key=lambda x: (x.last_scores or {}).get("q_margin", -1e9),
            reverse=True,
        )[:8]
        # filter out any None representations before storing
        self._top_reprs = [b.last_repr for b in ranked if b.last_repr is not None]
        if self.sched.lsh_bits:
            setattr(self, "_top_sigs", [b.lsh_sig for b in ranked if b.lsh_sig is not None])
        self._global_steps += 1

        return adv


controller = Controller()


@app.post("/initialize")
def initialize(req: InitRequest):
    controller.initialize(req)
    return {"status": "ok"}


@app.post("/seed")
def seed(req: SeedRequest):
    ids = controller.seed(req.items)
    return {"branch_ids": ids}


@app.post("/step")
def step(req: StepRequest) -> List[Advice]:
    return controller.step(req.max_to_step)


@app.post("/encode")
def encode(req: EncodeRequest):
    # Use model config defaults if present
    seq_len = req.seq_len or (controller.model.config.seq_len if controller.model is not None else 256)  # type: ignore
    vocab_size = (controller.model.config.vocab_size if controller.model is not None else 256)  # type: ignore
    try:
        from HRM.memory.encode_decode import Vocab as _V
        from HRM.memory.encode_decode import encode_atoms_to_grid
    except Exception:
        from memory.encode_decode import Vocab as _V
        from memory.encode_decode import encode_atoms_to_grid
    vocab = _V(size=vocab_size)
    arr = encode_atoms_to_grid(req.atoms, seq_len=seq_len, vocab=vocab)
    return {"inputs": arr.tolist()}


@app.get("/best")
def best(k: int = 1):
    # Select top-k by q_margin (fallback to visits)
    items = [b for b in controller.branches.values() if b.last_logits is not None]
    items.sort(key=lambda b: (b.last_scores or {}).get("q_margin", -1e9), reverse=True)
    items = items[: max(1, k)]

    results = []
    if controller.model is None:
        return {"items": results}

    vocab = Vocab(size=controller.model.config.vocab_size)  # type: ignore
    for b in items:
        # defensive: mypy/linters can't always narrow Optional[...] from the list comprehension
        logits = b.last_logits
        if logits is None:
            continue
        advice = decode_logits_to_advice(logits, vocab)
        advice_struct = dsl_decode(logits)
        results.append({
            "branch_id": b.branch_id,
            "scores": b.last_scores,
            "advice": advice["advice_text"],
            "advice_struct": advice_struct.__dict__,
        })
    return {"items": results}


@app.get("/fuse")
def fuse(k: int = 3):
    # Take top-k by margin and merge advice (simple union with dedupe)
    items = [b for b in controller.branches.values() if b.last_logits is not None]
    items.sort(key=lambda b: (b.last_scores or {}).get("q_margin", -1e9), reverse=True)
    items = items[: max(1, k)]

    if controller.model is None or not items:
        return {"items": []}

    # Decode each
    fused = {"plan": [], "constraints": [], "tool_calls": [], "citations": set(), "risks": [], "escalation": "", "confidence": 0.0}
    weights = []
    decoded = []
    for b in items:
        # defensive: ensure logits is not None (items list filters, but help typecheckers)
        logits = b.last_logits
        if logits is None:
            continue
        adv = dsl_decode(logits)
        decoded.append(adv)
    w = max(0.0, (b.last_scores or {}).get("q_margin", 0.0))
    weights.append(w)
    # Normalize weights
    total_w = sum(weights) or 1.0
    weights = [w / total_w for w in weights]

    def dedup_add(lst: List[str], val: str):
        if val and val not in lst:
            lst.append(val)

    for adv, w in zip(decoded, weights):
        for p in adv.plan:
            dedup_add(fused["plan"], p)
        for c in adv.constraints:
            dedup_add(fused["constraints"], c)
        for t in adv.tool_calls:
            dedup_add(fused["tool_calls"], t)
        for r in adv.risks:
            dedup_add(fused["risks"], r)
        fused["citations"].update(adv.citations)
        if adv.escalation:
            fused["escalation"] = adv.escalation
        fused["confidence"] += w  # heuristic

    fused["citations"] = sorted(list(fused["citations"]))
    # to text
    try:
        from HRM.memory.advice_dsl import Advice as AdviceStruct
        from HRM.memory.advice_dsl import advice_to_text
    except Exception:
        from memory.advice_dsl import Advice as AdviceStruct
        from memory.advice_dsl import advice_to_text
    fused_struct = AdviceStruct(plan=fused["plan"], constraints=fused["constraints"], tool_calls=fused["tool_calls"], citations=fused["citations"], risks=fused["risks"], escalation=fused["escalation"], confidence=fused["confidence"])
    return {"advice_struct": fused_struct.__dict__, "advice": advice_to_text(fused_struct)}


@app.get("/status")
def status():
    items = []
    for b in controller.branches.values():
        items.append({
            "branch_id": b.branch_id,
            "halted": b.halted,
            "steps": b.steps,
            "visits": b.visits,
            "scores": b.last_scores,
        })
    return {"count": len(items), "items": items, "waves": controller._global_steps}


@app.post("/prune")
def prune(min_margin: float = -0.1):
    removed = []
    for bid, b in list(controller.branches.items()):
        if (b.last_scores or {}).get("q_margin", -1e9) < min_margin:
            controller.branches.pop(bid, None)
            removed.append(bid)
    return {"removed": removed}


@app.get("/metrics")
def metrics():
    n = len(controller.branches)
    if n == 0:
        return {"branches": 0, "waves": controller._global_steps}
    import statistics
    margins = [(b.last_scores or {}).get("q_margin", 0.0) for b in controller.branches.values() if b.last_scores]
    values = [(b.last_scores or {}).get("value", 0.0) for b in controller.branches.values() if b.last_scores]
    risks = [(b.last_scores or {}).get("risk", 0.0) for b in controller.branches.values() if b.last_scores]
    return {
        "branches": n,
        "waves": controller._global_steps,
        "avg_margin": statistics.fmean(margins) if margins else 0.0,
        "avg_value": statistics.fmean(values) if values else 0.0,
        "avg_risk": statistics.fmean(risks) if risks else 0.0,
    }


@app.post("/clear")
def clear():
    controller.branches.clear()
    controller._top_reprs = []
    if hasattr(controller, "_top_sigs"):
        setattr(controller, "_top_sigs", [])
    controller._global_steps = 0
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
