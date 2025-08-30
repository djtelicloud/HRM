import WebSocket from "ws";

type RealtimeBridgeOpts = {
  openaiApiKey: string;
  model?: string;
};

export interface IAudioBridge {
  connect(): Promise<void>;
  close(): void;
  sendEvent(evt: any): void;
  onMessage(cb: (msg: any) => void): void;
  sendPcm16Audio(pcm16: Buffer): void;
  commitAudio(): void;
  setInstructions?(text: string): void;
}

export class RealtimeBridge implements IAudioBridge {
  private ws?: WebSocket;
  private model: string;
  private key: string;

  constructor(opts: RealtimeBridgeOpts) {
    this.key = opts.openaiApiKey;
    this.model = opts.model || "gpt-realtime";
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `wss://api.openai.com/v1/realtime?model=${encodeURIComponent(this.model)}`;
      const headers = {
        Authorization: `Bearer ${this.key}`,
        "OpenAI-Beta": "realtime=v1",
      } as any;
      this.ws = new WebSocket(url, { headers });
      this.ws.once("open", () => resolve());
      this.ws.once("error", (e) => reject(e));
    });
  }

  close() {
    this.ws?.close();
  }

  sendEvent(evt: any) {
    if (!this.ws) return;
    this.ws.send(JSON.stringify(evt));
  }

  onMessage(cb: (msg: any) => void) {
    this.ws?.on("message", (data) => {
      try {
        const msg = JSON.parse(data.toString());
        cb(msg);
      } catch {}
    });
  }

  // Placeholder audio bridge helpers
  sendPcm16Audio(pcm16: Buffer) {
    this.sendEvent({ type: "input_audio_buffer.append", audio: pcm16.toString("base64") });
  }

  commitAudio() {
    this.sendEvent({ type: "input_audio_buffer.commit" });
  }
}
