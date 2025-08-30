import fetch from "node-fetch";

const HRM_BASE = process.env.HRM_BASE || "http://127.0.0.1:8000";

async function api(path: string, body?: any, method = "POST") {
  const res = await fetch(`${HRM_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

export async function hrmInitialize() {
  const init = {
    vocab_size: 256,
    seq_len: 256,
    num_puzzle_identifiers: 1024,
    batch_size: 64,
    arch_name: "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
    arch_overrides: { enable_value_head: true, enable_risk_head: true, halt_max_steps: 12 },
    q_margin_min: -0.05,
    beta_explore: 0.8,
    max_per_step: 32,
    diversity_penalty: 0.2,
    lsh_bits: 64,
    mode: "ucb",
  };
  return api("/initialize", init);
}

export async function hrmEncodeAtoms(atoms: any[], seq_len?: number) {
  return api("/encode", { atoms, seq_len });
}

export async function hrmSeed(inputs: number[], puzzle_identifier = 1) {
  return api("/seed", { items: [{ inputs, puzzle_identifier }] });
}

export async function hrmStep(maxToStep = 32) {
  return api("/step", { max_to_step: maxToStep });
}

export async function hrmBest(k = 1) {
  return api(`/best?k=${k}`, undefined, "GET");
}

export async function hrmFuse(k = 3) {
  return api(`/fuse?k=${k}`, undefined, "GET");
}

export async function hrmStatus() {
  return api(`/status`, undefined, "GET");
}

export async function hrmMetrics() {
  return api(`/metrics`, undefined, "GET");
}

export async function hrmClear() {
  return api(`/clear`, {}, "POST");
}
