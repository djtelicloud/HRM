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

function generateId(prefix = "evt_") {
  return `${prefix}${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 9)}`;
}

export class RealtimeBridge implements IAudioBridge {
  private ws?: WebSocket;
  private model: string;
  private key: string;
  // queue messages sent before socket is open
  private messageQueue: any[] = [];
  private messageCb?: (msg: any) => void;

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

      const onOpen = () => {
        // flush queued messages
        while (this.messageQueue.length && this.ws && this.ws.readyState === WebSocket.OPEN) {
          const queued = this.messageQueue.shift();
          try {
            this.ws.send(typeof queued === "string" ? queued : JSON.stringify(queued));
          } catch {}
        }
        resolve();
      };

      const onError = (e: any) => {
        reject(e);
      };

      this.ws.once("open", onOpen);
      this.ws.once("error", onError);
      this.ws.on("message", (data: WebSocket.Data, isBinary: boolean) => {
        // forward parsed JSON messages and raw binary to consumer
        if (isBinary) {
          // pass Buffer for binary
          this.messageCb?.({ type: "binary", data: Buffer.from(data as Buffer) });
          return;
        }

        try {
          const text = (data as Buffer).toString("utf-8");
          const obj = JSON.parse(text);
          this.messageCb?.(obj);
        } catch (e) {
          // non-JSON textual messages are forwarded as raw text
          this.messageCb?.({ type: "text", data: (data as Buffer).toString("utf-8") });
        }
      });

      // attach basic close handler to clear queue
      this.ws.on("close", () => {
        this.messageQueue = [];
      });
    });
  }

  close() {
    try {
      this.ws?.close();
    } catch {}
    this.ws = undefined;
    this.messageQueue = [];
  }

  sendEvent(evt: any) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.messageQueue.push(evt);
      return;
    }

    try {
      this.ws.send(JSON.stringify(evt));
    } catch (e) {
      // if send fails, buffer the event for a short time
      this.messageQueue.push(evt);
    }
  }

  onMessage(cb: (msg: any) => void) {
    this.messageCb = cb;
  }

  // Helper to send audio as base64 using OpenAI realtime compatible events
  sendPcm16Audio(pcm16: Buffer) {
    const event = {
      event_id: generateId("evt_"),
      type: "input_audio_buffer.append",
      audio: pcm16.toString("base64"),
    };
    this.sendEvent(event);
  }

  commitAudio() {
    const event = {
      event_id: generateId("evt_"),
      type: "input_audio_buffer.commit",
    };
    this.sendEvent(event);
  }

  setInstructions?(text: string) {
    const event = {
      event_id: generateId("evt_"),
      type: "conversation.item.create",
      previous_item_id: "root",
      item: {
        type: "message",
        role: "system",
        content: [{ type: "input_text", text }],
      },
    };
    this.sendEvent(event);
  }
}
