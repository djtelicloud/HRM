import fetch from "node-fetch";

const HRM_BASE = process.env.HRM_BASE || "http://127.0.0.1:8000";

async function api<T = any>(path: string, body?: any, method = "POST"): Promise<T> {
  const res = await fetch(`${HRM_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const data = await res.json();
  return data as T;
}

export async function hrmInitialize() {
  const init = {
    vocab_size: 256,
    seq_len: 256,
    num_puzzle_identifiers: 1024,
    batch_size: 64,
    arch_name: "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
    arch_overrides: {
      enable_value_head: true,
      enable_risk_head: true,
      halt_max_steps: 12,
      H_cycles: 2,
      L_cycles: 2,
      H_layers: 4,
      L_layers: 4,
      hidden_size: 512,
      expansion: 4,
      num_heads: 8,
      pos_encodings: "rope",
    },
    q_margin_min: -0.05,
    beta_explore: 0.8,
    max_per_step: 32,
    diversity_penalty: 0.2,
    lsh_bits: 64,
    mode: "ucb",
  };
  return api("/initialize", init);
}

export async function hrmEncodeAtoms(atoms: any[], seq_len?: number): Promise<{ inputs: number[] }> {
  return api<{ inputs: number[] }>("/encode", { atoms, seq_len });
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

export async function hrmFuse(k = 3): Promise<{ advice: string }> {
  return api<{ advice: string }>(`/fuse?k=${k}`, undefined, "GET");
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
