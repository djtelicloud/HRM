import "dotenv/config";
import express from "express";
import Twilio from "twilio";
import { WebSocketServer as WSServer } from "ws";
import http from "http";

import { RealtimeBridge, type IAudioBridge } from "./realtime_bridge.js";
import { GeminiBridge } from "./gemini_bridge.js";
import { hrmInitialize, hrmEncodeAtoms, hrmSeed, hrmStep, hrmFuse, hrmStatus, hrmMetrics, hrmClear } from "./hrm_client.js";

const PORT = parseInt(process.env.PORT || "3000", 10);
const PUBLIC_URL = process.env.PUBLIC_URL || `http://localhost:${PORT}`;

// Boot HRM once on startup
hrmInitialize().catch((e) => console.warn("HRM init failed (will init lazily):", e.message));

const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static("public"));

// Twilio Voice webhook: create a media stream to our WS
app.post("/voice", async (req, res) => {
  const response = new (Twilio as any).twiml.VoiceResponse();
  const connect = response.connect();
  // Bidirectional stream if supported on your Twilio account; else one-way
  // @ts-ignore
  connect.stream({ url: `${PUBLIC_URL.replace("http", "ws")}/twilio-media`, track: "both" });
  res.type("text/xml").send(response.toString());
});

// Optional: outbound call test endpoint (requires configured Twilio creds)
app.post("/call", async (req, res) => {
  const accountSid = process.env.TWILIO_ACCOUNT_SID || "";
  const authToken = process.env.TWILIO_AUTH_TOKEN || "";
  const from = process.env.TWILIO_PHONE_NUMBER || "";
  const to = process.env.YOUR_PHONE_NUMBER || "";
  if (!accountSid || !authToken || !from || !to) {
    return res.status(400).json({ error: "Twilio env vars missing" });
  }
  const twilioClient = (await import("twilio")).default(accountSid, authToken);
  const url = `${PUBLIC_URL}/voice`;
  const call = await twilioClient.calls.create({ url, from, to });
  res.json({ sid: call.sid });
});

// Simple UI proxy endpoints (avoid CORS to HRM controller)
app.get("/ui/status", async (_req, res) => {
  try { res.json(await hrmStatus()); } catch (e) { res.status(500).json({ error: (e as any)?.message || e }); }
});
app.get("/ui/metrics", async (_req, res) => {
  try { res.json(await hrmMetrics()); } catch (e) { res.status(500).json({ error: (e as any)?.message || e }); }
});
app.get("/ui/best", async (req, res) => {
  const k = Math.max(1, parseInt((req.query.k as string) || "1", 10));
  try { res.json(await hrmFuse(k)); } catch (e) { res.status(500).json({ error: (e as any)?.message || e }); }
});
app.post("/ui/encode_and_seed", async (req, res) => {
  try {
    const atoms = Array.isArray(req.body?.atoms) ? req.body.atoms : [];
    const enc = await hrmEncodeAtoms(atoms, 256);
    const out = await hrmSeed(enc.inputs, req.body?.puzzle_identifier || 1);
    res.json(out);
  } catch (e) { res.status(500).json({ error: (e as any)?.message || e }); }
});
app.post("/ui/clear", async (_req, res) => {
  try { res.json(await hrmClear()); } catch (e) { res.status(500).json({ error: (e as any)?.message || e }); }
});

// HTTP server + WS server for Twilio Media Streams
const server = http.createServer(app);
const wss = new WSServer({ server, path: "/twilio-media" });

// μ-law <-> PCM16 (G.711) with naive resampling
// Twilio sends 8kHz μ-law; OpenAI/Gemini use 16kHz PCM16. We upsample by duplication and downsample by decimation.
const BIAS = 0x84; // 132

function muByteToPcm16Sample(mu: number): number {
  mu = (~mu) & 0xff;
  const sign = mu & 0x80;
  const exponent = (mu >> 4) & 0x07;
  const mantissa = mu & 0x0f;
  let sample = ((mantissa << 4) + 0x08) << (exponent + 3);
  sample -= BIAS;
  if (sign) sample = -sample;
  // Clamp to int16
  if (sample > 32767) sample = 32767;
  if (sample < -32768) sample = -32768;
  return sample;
}

function pcm16SampleToMuByte(sample: number): number {
  let sign = 0;
  sample = Math.max(-32768, Math.min(32767, sample));
  if (sample < 0) {
    sign = 0x80;
    sample = -sample;
  }
  if (sample > 32635) sample = 32635; // clip
  sample = sample + BIAS;

  // Determine exponent (segment)
  let exponent = 7;
  for (; exponent > 0; exponent--) {
    if (sample & (1 << (exponent + 3))) break;
  }
  const mantissa = (sample >> (exponent + 3)) & 0x0f;
  const mu = ~(sign | (exponent << 4) | mantissa) & 0xff;
  return mu;
}

function muLawToPcm16(mu: Buffer): Buffer {
  // Decode μ-law to PCM16 at 8kHz
  const n = mu.length;
  const pcm8k = new Int16Array(n);
  for (let i = 0; i < n; i++) pcm8k[i] = muByteToPcm16Sample(mu[i]);
  // Upsample to 16kHz by linear interpolation (insert midpoints)
  const up = Buffer.alloc(n * 4);
  for (let i = 0; i < n - 1; i++) {
    const s0 = pcm8k[i];
    const s1 = pcm8k[i + 1];
    const mid = (s0 + s1) >> 1;
    up.writeInt16LE(s0, i * 4);
    up.writeInt16LE(mid, i * 4 + 2);
  }
  // last sample: repeat
  const last = pcm8k[n - 1];
  up.writeInt16LE(last, (n - 1) * 4);
  up.writeInt16LE(last, (n - 1) * 4 + 2);
  return up;
}

function pcm16ToMuLaw(pcm: Buffer): Buffer {
  // Downsample 16k -> 8k by averaging pairs (simple low-pass)
  const samples16 = Math.floor(pcm.length / 2);
  const pairs = Math.floor(samples16 / 2);
  const out = Buffer.alloc(pairs);
  for (let i = 0; i < pairs; i++) {
    const s0 = pcm.readInt16LE(i * 4);
    const s1 = pcm.readInt16LE(i * 4 + 2);
    const avg = (s0 + s1) >> 1;
    out[i] = pcm16SampleToMuByte(avg);
  }
  return out;
}

wss.on("connection", async (ws, req) => {
  console.log("Twilio media stream connected");

  const provider = (process.env.REALTIME_PROVIDER || "auto").toLowerCase();
  let bridge: IAudioBridge | null = null;
  async function connectBridge() {
    if (provider === "openai" || provider === "auto") {
      try {
        const br = new RealtimeBridge({ openaiApiKey: process.env.OPENAI_API_KEY || "" });
        await br.connect();
        bridge = br;
        return;
      } catch (e) {
        console.warn("OpenAI Realtime connect error, will try Gemini:", (e as any)?.message || e);
      }
    }
    if (provider === "gemini" || provider === "auto") {
      const key = process.env.GOOGLE_API_KEY || "";
      const br = new GeminiBridge({ apiKey: key, systemInstruction: "You are a helpful assistant." });
      await br.connect();
      bridge = br;
      return;
    }
    throw new Error("No realtime provider available");
  }

  await connectBridge().catch((e) => console.error("Realtime connect error:", e));
  if (!bridge) { ws.close(); return; }

  // Keep a rolling conversation buffer for atoms
  const atoms: any[] = [];
  let seeded = false;

  bridge.onMessage(async (msg) => {
    // OpenAI Realtime events
    const t = msg?.type;
    if (t === "response.output_text.delta" && msg?.delta) {
      atoms.push({ role: "assistant", content: msg.delta });
    }
    if (t === "response.audio.delta" && msg?.audio) {
      const pcm16 = Buffer.from(msg.audio, "base64");
      const mulaw = pcm16ToMuLaw(pcm16);
      ws.send(JSON.stringify({ event: "media", media: { payload: mulaw.toString("base64") } }));
    }

    // Gemini Live events
    const sc = msg?.serverContent;
    if (sc) {
      const it = sc.inputTranscription?.text;
      const ot = sc.outputTranscription?.text;
      if (it) atoms.push({ role: "user", content: it });
      if (ot) atoms.push({ role: "assistant", content: ot });

      const parts = sc.modelTurn?.parts || [];
      for (const p of parts) {
        const inline = p.inlineData;
        if (inline && inline.data) {
          const b = Buffer.from(inline.data, "base64");
          const mulaw = pcm16ToMuLaw(b);
          ws.send(JSON.stringify({ event: "media", media: { payload: mulaw.toString("base64") } }));
        }
      }
    }
  });

  ws.on("message", async (data) => {
    try {
      const msg = JSON.parse(data.toString());
      const event = msg.event;
      if (event === "start") {
        console.log("Twilio stream start", msg.start);
      } else if (event === "media") {
        // Twilio sends base64 mu-law 8k audio
        const payload = msg.media?.payload;
        if (payload) {
          const mu = Buffer.from(payload, "base64");
          const pcm16 = muLawToPcm16(mu);
          bridge.sendPcm16Audio(pcm16);
        }
      } else if (event === "stop") {
        console.log("Twilio stream stop");
        bridge.commitAudio();
        ws.close();
        bridge.close();
      } else if (event === "mark") {
        // ignore
      }
    } catch (e) {
      console.warn("WS parse error", e);
    }
  });

  // Clamp advice to a safe size
  function clipAdviceText(text: string, maxChars = 800): string {
    if (!text) return "";
    let t = text.replace(/[\r\n\t]+/g, " ").trim();
    if (t.length > maxChars) t = t.slice(0, maxChars) + " …";
    return t;
  }

  // Periodically update HRM advice and inject as system hint
  const timer = setInterval(async () => {
    try {
      // Build compact atoms (last few turns); here we only show assistant deltas collected above
      const enc = await hrmEncodeAtoms(atoms.slice(-8), 256);
      if (!seeded) {
        await hrmSeed(enc.inputs, 1);
        seeded = true;
      }
      await hrmStep(32);
      const fused = await hrmFuse(3);
      const advisory = clipAdviceText(fused?.advice || "");
      if (advisory) {
        if (bridge.setInstructions) {
          bridge.setInstructions(`HRM-Advice: ${advisory}`);
        } else {
          // Fallback: send as an event for OpenAI bridge
          bridge.sendEvent({ type: "session.update", session: { instructions: `HRM-Advice: ${advisory}` } });
        }
      }
    } catch (e) {
      // ignore transient errors
    }
  }, 700);

  ws.once("close", () => {
    clearInterval(timer);
    bridge.close();
  });
});

server.listen(PORT, () => {
  console.log(`Voice server listening on http://localhost:${PORT}`);
  console.log(`Twilio webhook -> ${PUBLIC_URL}/voice`);
});
