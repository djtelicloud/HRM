import { GoogleGenAI, Modality, type Session, type LiveServerMessage } from "@google/genai";

export type GeminiBridgeOpts = {
  apiKey: string;
  model?: string;
  systemInstruction?: string;
};

function encodeBase64(bytes: Uint8Array) {
  return Buffer.from(bytes).toString("base64");
}

export class GeminiBridge {
  private client: GoogleGenAI;
  private session?: Session;
  private opts: GeminiBridgeOpts;
  private msgCb?: (msg: any) => void;

  constructor(opts: GeminiBridgeOpts) {
    this.opts = opts;
    this.client = new GoogleGenAI({ apiKey: opts.apiKey });
  }

  async connect(): Promise<void> {
    const model = this.opts.model || "gemini-live-2.5-flash-preview";
    this.session = await this.client.live.connect({
      model,
      callbacks: {
        onopen: () => {},
        onmessage: (message: LiveServerMessage) => {
          // Forward raw server message to consumer
          this.msgCb?.(message);
        },
        onerror: (e: any) => {
          console.error("Gemini error:", e?.message || e);
        },
        onclose: (e: any) => {
          console.log("Gemini closed:", e?.reason || "");
        },
      },
      config: {
        outputAudioTranscription: {},
        systemInstruction: this.opts.systemInstruction || "You are a helpful assistant.",
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: "Orus" } },
        },
      },
    });
  }

  close() {
    try {
      // @ts-ignore — live sessions close on GC; no explicit close
      this.session?.close?.();
    } catch {}
  }

  onMessage(cb: (msg: any) => void) {
    this.msgCb = cb;
  }

  sendPcm16Audio(pcm16: Buffer) {
    if (!this.session) return;
    const blob = {
      data: encodeBase64(new Uint8Array(pcm16.buffer, pcm16.byteOffset, pcm16.byteLength)),
      mimeType: "audio/pcm;rate=16000",
    } as any; // Blob shape used by @google/genai
    // @ts-ignore
    this.session.sendRealtimeInput({ media: blob });
  }

  commitAudio() {
    // Gemini Live streams continuously; no explicit commit op required.
  }

  setInstructions(text: string) {
    // No direct session.update equivalent; push as user hint mid-stream
    try {
      // @ts-ignore
      this.session?.sendClientContent({ turns: [{ role: "user", parts: [{ text }] }], turnComplete: false });
    } catch {}
  }
}
