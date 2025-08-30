from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import torch


TOK_PAD = 0
TOK_EOS = 1
TOK_SEP = 2
TOK_STEP = 3

# Section markers (reserved)
TOK_PLAN = 4
TOK_CONS = 5
TOK_TOOL = 6
TOK_CITE = 7
TOK_RISK = 8
TOK_ESC  = 9
TOK_CONF = 10


@dataclass
class Advice:
    plan: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    citations: List[int] = field(default_factory=list)  # indices of atoms referenced
    risks: List[str] = field(default_factory=list)
    escalation: str = ""
    confidence: float = 0.0


def advice_to_text(a: Advice) -> str:
    parts = []
    if a.plan:
        parts.append("Plan: " + " | ".join(f"{i+1}. {s}" for i, s in enumerate(a.plan)))
    if a.constraints:
        parts.append("Constraints: " + "; ".join(a.constraints))
    if a.tool_calls:
        parts.append("Tools: " + "; ".join(a.tool_calls))
    if a.citations:
        parts.append("Refs: [" + ",".join(map(str, a.citations)) + "]")
    if a.risks:
        parts.append("Risks: " + "; ".join(a.risks))
    if a.escalation:
        parts.append("Escalate: " + a.escalation)
    parts.append(f"Confidence: {a.confidence:.2f}")
    return " | ".join(parts)


def decode_tokens_to_advice(ids: List[int]) -> Advice:
    """
    Deterministic sectioned decoder. Tokens are numeric; text fields are numeric strings.
    Grammar: [PLAN, tok*, STEP, tok*, ...] [CONS, tok*, SEP, tok*, ...] [TOOL,...] [CITE, id, id, ...]
    Sections may repeat; later ones append.
    """
    section = None
    buf: List[int] = []
    advice = Advice()

    def flush_buf_to(section_name: str):
        nonlocal buf
        if not buf:
            return
        text = " ".join(map(str, buf))
        if section_name == "plan":
            advice.plan.append(text)
        elif section_name == "constraints":
            advice.constraints.append(text)
        elif section_name == "tool_calls":
            advice.tool_calls.append(text)
        elif section_name == "risks":
            advice.risks.append(text)
        elif section_name == "escalation":
            advice.escalation = (advice.escalation + " "+ text).strip()
        buf = []

    i = 0
    while i < len(ids):
        t = ids[i]
        if t == TOK_EOS:
            break
        if t in (TOK_PAD, TOK_SEP):
            # separator: flush within current section if applicable
            if section in ("constraints", "tool_calls", "risks", "escalation"):
                flush_buf_to(section)
            i += 1
            continue
        if t == TOK_PLAN:
            if section and section != "plan":
                flush_buf_to(section)
            section = "plan"
            buf = []
            i += 1
            continue
        if t == TOK_CONS:
            flush_buf_to(section or "plan")
            section = "constraints"
            buf = []
            i += 1
            continue
        if t == TOK_TOOL:
            flush_buf_to(section or "plan")
            section = "tool_calls"
            buf = []
            i += 1
            continue
        if t == TOK_CITE:
            flush_buf_to(section or "plan")
            section = "citations"
            i += 1
            # read integer IDs until next marker
            while i < len(ids) and ids[i] not in (TOK_EOS, TOK_SEP, TOK_PLAN, TOK_CONS, TOK_TOOL, TOK_CITE, TOK_RISK, TOK_ESC, TOK_CONF):
                advice.citations.append(int(ids[i]))
                i += 1
            continue
        if t == TOK_RISK:
            flush_buf_to(section or "plan")
            section = "risks"
            buf = []
            i += 1
            continue
        if t == TOK_ESC:
            flush_buf_to(section or "plan")
            section = "escalation"
            buf = []
            i += 1
            continue
        if t == TOK_CONF:
            flush_buf_to(section or "plan")
            # next token interpreted as tenths of confidence [0..1000]
            if i + 1 < len(ids):
                advice.confidence = min(max(float(ids[i + 1]) / 1000.0, 0.0), 1.0)
                i += 2
                continue
        if t == TOK_STEP:
            # step boundary within plan
            if section == "plan":
                flush_buf_to("plan")
            i += 1
            continue
        # default: accumulate token into current section buffer
        if section is None:
            section = "plan"
        buf.append(t)
        i += 1
    flush_buf_to(section or "plan")
    return advice


def decode_logits_to_advice(logits: torch.Tensor) -> Advice:
    with torch.inference_mode():
        ids = torch.argmax(logits.to(torch.float32), dim=-1).detach().cpu().tolist()
    return decode_tokens_to_advice(ids)


def encode_advice_to_tokens(advice: Advice, seq_len: int) -> List[int]:
    out: List[int] = []
    # plan
    out.append(TOK_PLAN)
    for step in advice.plan:
        for tok in map(int_safe, step.split()):
            out.append(tok)
        out.append(TOK_STEP)
    # constraints
    if advice.constraints:
        out.append(TOK_CONS)
        for c in advice.constraints:
            for tok in map(int_safe, c.split()):
                out.append(tok)
            out.append(TOK_SEP)
    # tools
    if advice.tool_calls:
        out.append(TOK_TOOL)
        for t in advice.tool_calls:
            for tok in map(int_safe, t.split()):
                out.append(tok)
            out.append(TOK_SEP)
    # citations
    if advice.citations:
        out.append(TOK_CITE)
        out.extend(advice.citations)
    # risks
    if advice.risks:
        out.append(TOK_RISK)
        for r in advice.risks:
            for tok in map(int_safe, r.split()):
                out.append(tok)
            out.append(TOK_SEP)
    # escalation
    if advice.escalation:
        out.append(TOK_ESC)
        for tok in map(int_safe, advice.escalation.split()):
            out.append(tok)
    # confidence
    if advice.confidence > 0:
        out.extend([TOK_CONF, int(round(advice.confidence * 1000))])
    out.append(TOK_EOS)
    return out[:seq_len] + [TOK_PAD] * max(0, seq_len - len(out))


def int_safe(x: str) -> int:
    try:
        return int(x)
    except Exception:
        # fallback hash for arbitrary tokens in placeholder codec
        return (abs(hash(x)) % 200) + 16
