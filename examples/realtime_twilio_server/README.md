Realtime Voice Orchestrator (Twilio ↔ OpenAI Realtime ↔ HRM)

Overview
- Accepts Twilio Programmable Voice calls, starts a Media Stream over WebSocket.
- Bridges audio to OpenAI Realtime (WebSocket) and streams TTS back to Twilio (stubbed transcoding).
- Periodically queries the HRM Lightning Controller to inject fused "HRM‑Advice" into the LLM session.

Setup
1) Copy `.env.sample` to `.env` and fill secrets (do not commit your real keys):
   - `OPENAI_API_KEY`, `HRM_BASE` (e.g., http://127.0.0.1:8000)
   - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`
   - `YOUR_PHONE_NUMBER` (optional, for quick outbound test)
2) Start the HRM Lightning Controller (from repo root):
   - `python -m HRM.services.lightning_controller.app`
3) Install and start this server:
   - `cd HRM/examples/realtime_twilio_server`
   - `npm install`
   - `npm start`
   - It will listen on `PORT` (default 3000)

Twilio Configuration
- Set your Twilio Voice webhook to `http://<your-host>/voice` (use a tunnel like ngrok for local dev).
- Optional outbound test: POST `http://localhost:3000/call` (uses YOUR_PHONE_NUMBER env).

Notes & TODOs
- μ-law/PCM16 transcoding and 8k↔16k resampling are stubbed — replace with a DSP lib (e.g., `sox` bindings, `prism-media`, or a native addon) to get full duplex audio.
- OpenAI Realtime event types vary; adjust handlers for transcripts and audio deltas to your model version.
- The HRM cadence runs every ~700 ms; tune cadence and scheduler thresholds for latency/cost.
- Never commit real secrets; `.env.sample` is provided for safe placeholders.

