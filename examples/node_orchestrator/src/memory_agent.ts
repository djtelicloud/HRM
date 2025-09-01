import fetch from "node-fetch";


const BASE = process.env.HRM_BASE || "http://127.0.0.1:8000";

async function api<T = any>(path: string, opts: any = {}): Promise<T> {
  const url = `${BASE}${path}`;
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const data = await res.json();
  return data as T;
}

export async function ensureInitialized() {
  const body = {
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
  return api("/initialize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

export async function encodeAtoms(atoms: any[], seq_len?: number): Promise<{ inputs: number[] }> {
  return api<{ inputs: number[] }>("/encode", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ atoms, seq_len }) });
}

export async function seed(inputs: number[], puzzle_identifier = 1) {
  return api("/seed", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ items: [{ inputs, puzzle_identifier }] }) });
}

export async function step(maxToStep = 32) {
  return api("/step", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ max_to_step: maxToStep }) });
}

export async function best(k = 1) {
  return api(`/best?k=${k}`);
}

export async function fuse(k = 3): Promise<{ advice: string }> {
  return api<{ advice: string }>(`/fuse?k=${k}`);
}

export async function runDemoConversation(turns: string[]) {
  await ensureInitialized();
  const atoms = turns.slice(-6).map((t, i) => ({ role: i % 2 === 0 ? "user" : "assistant", content: t }));
  const enc = await encodeAtoms(atoms, 256);
  await seed(enc.inputs, 1);
  for (let i = 0; i < 6; i++) {
    await step(32);
    const fused = await fuse(3);
    console.log("HRM-Advice:", fused.advice);
  }
}
