import {
  EndSensitivity,
  GoogleGenAI,
  Modality,
  type LiveConnectConfig,
  type LiveServerMessage,
  type Session,
} from "@google/genai";

export type GeminiBridgeOpts = {
  apiKey: string;
  model?: string;
  systemInstruction?: string;
};

function encodeBase64(bytes: Uint8Array) {
  return Buffer.from(bytes).toString("base64");
}

export class GeminiBridge {
  // Add sendEvent to satisfy IAudioBridge interface
  sendEvent(_evt: any): void {
    // No-op for Gemini; OpenAI bridge uses this for session updates
  }
  private client: GoogleGenAI;
  private session?: Session;
  private opts: GeminiBridgeOpts;
  private msgCb?: (msg: any) => void;

  constructor(opts: GeminiBridgeOpts) {
    this.opts = opts;
    this.client = new GoogleGenAI({ apiKey: opts.apiKey });
  }

  async connect(): Promise<void> {
    const model = this.opts.model || "gemini-2.5-flash-preview-native-audio-dialog";
    const liveConnectConfig: LiveConnectConfig = {
      outputAudioTranscription: {},
      inputAudioTranscription: {},
      systemInstruction: this.opts.systemInstruction || "You are a helpful assistant.",
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: "Orus" } },
      },
      realtimeInputConfig: {
        automaticActivityDetection: {
          disabled: false,
          endOfSpeechSensitivity: EndSensitivity.END_SENSITIVITY_LOW,
          silenceDurationMs: 100,
        },
      },
    };
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
      config: liveConnectConfig,
    });
  }

  close() {
    this.session?.close();
  }

  onMessage(cb: (msg: any) => void) {
    this.msgCb = cb;
  }

  sendPcm16Audio(pcm16: Buffer) {
    if (!this.session) return;
    const base64Data = encodeBase64(new Uint8Array(pcm16.buffer, pcm16.byteOffset, pcm16.byteLength));
    this.session.sendRealtimeInput({
      audio: {
        data: base64Data,
        mimeType: "audio/pcm;rate=16000",
      },
    });
  }

  commitAudio() {
    // Gemini Live streams continuously; no explicit commit op required.
  }

  setInstructions(text: string) {
    // No direct session.update equivalent; push as user hint mid-stream
    this.session?.sendClientContent({ turns: [{ role: "user", parts: [{ text }] }] });
  }
}
