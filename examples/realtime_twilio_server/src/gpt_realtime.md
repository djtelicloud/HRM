Realtime API
============
Model: "gpt-realtime"
Build low-latency, multimodal LLM applications with the Realtime API.

The OpenAI Realtime API enables low-latency communication with [models](/docs/models) that natively support speech-to-speech interactions as well as multimodal inputs (audio, images, and text) and outputs (audio and text). These APIs can also be used for [realtime audio transcription](/docs/guides/realtime-transcription).

Voice agents
------------

One of the most common use cases for the Realtime API is building voice agents for speech-to-speech model interactions in the browser. Our recommended starting point for these types of applications is the [Agents SDK for TypeScript](https://openai.github.io/openai-agents-js/guides/voice-agents/), which uses a [WebRTC connection](/docs/guides/realtime-webrtc) to the Realtime model in the browser, and [WebSocket](/docs/guides/realtime-websocket) when used on the server.

```js
import { RealtimeAgent, RealtimeSession } from "@openai/agents/realtime";

const agent = new RealtimeAgent({
    name: "Assistant",
    instructions: "You are a helpful assistant.",
});

const session = new RealtimeSession(agent);

// Automatically connects your microphone and audio output
await session.connect({
    apiKey: "<client-api-key>",
});
```

[

Voice Agent Quickstart

Follow the voice agent quickstart to build Realtime agents in the browser.

](https://openai.github.io/openai-agents-js/guides/voice-agents/quickstart/)

To use the Realtime API directly outside the context of voice agents, check out the other connection options below.

Connection methods
------------------

While building [voice agents with the Agents SDK](https://openai.github.io/openai-agents-js/guides/voice-agents/) is the fastest path to one specific type of application, the Realtime API provides an entire suite of flexible tools for a variety of use cases.

There are three primary supported interfaces for the Realtime API:

[

WebRTC connection

Ideal for browser and client-side interactions with a Realtime model.

](/docs/guides/realtime-webrtc)[

WebSocket connection

Ideal for middle tier server-side applications with consistent low-latency network connections.

](/docs/guides/realtime-websocket)[

SIP connection

Ideal for VoIP telephony connections.

](/docs/guides/realtime-sip)

Depending on how you'd like to connect to a Realtime model, check out one of the connection guides above to get started. You'll learn how to initialize a Realtime session, and how to interact with a Realtime model using client and server events.

API Usage
---------

Once connected to a realtime model using one of the methods above, learn how to interact with the model in these usage guides.

*   **[Prompting guide](/docs/guides/realtime-models-prompting):** learn tips and best practices for prompting and steering Realtime models.
*   **[Inputs and outputs](/docs/guides/realtime-inputs-outputs):** Learn how to pass audio, text, and image inputs to the model, and how to receive audio and text back.
*   **[Managing conversations](/docs/guides/realtime-conversations):** Learn about the Realtime session lifecycle and the key events that happen during a conversation.
*   **[Webhooks and server-side controls](/docs/guides/realtime-server-controls):** Learn how you can control a Realtime session on the server to call tools and implement guardrails.
*   **[Function calling](/docs/guides/realtime-function-calling):** Give the realtime model access to call custom code in your own application when appropriate.
*   **[MCP servers](/docs/guides/realtime-mcp):** Give realtime models access to new capabilities via Model Context Protocol (MCP) servers.
*   **[Realtime audio transcription](/docs/guides/realtime-transcription):** Transcribe audio streams in real time over a WebSocket connection.

Voice Agents Quickstart
Create a project

In this quickstart we will create a voice agent you can use in the browser. If you want to check out a new project, you can try out Next.js or Vite.

Terminal window
npm create vite@latest my-project --template vanilla-ts

Install the Agents SDK

Terminal window
npm install @openai/agents zod@3

Alternatively you can install @openai/agents-realtime for a standalone browser package.

Generate a client ephemeral token

As this application will run in the user’s browser, we need a secure way to connect to the model through the Realtime API. For this we can use an ephemeral client key that should be generated on your backend server. For testing purposes you can also generate a key using curl and your regular OpenAI API key.

Terminal window
curl -X POST https://api.openai.com/v1/realtime/client_secrets \
   -H "Authorization: Bearer $OPENAI_API_KEY" \
   -H "Content-Type: application/json" \
   -d '{
     "session": {
       "type": "realtime",
       "model": "gpt-realtime"
     }
   }'

The response will contain a client_secret.value value that you can use to connect later on. Note that this key is only valid for a short period of time and will need to be regenerated.

Create your first Agent

Creating a new RealtimeAgent is very similar to creating a regular Agent.

import { RealtimeAgent } from '@openai/agents-realtime';

const agent = new RealtimeAgent({
  name: 'Assistant',
  instructions: 'You are a helpful assistant.',
});

Create a session

Unlike a regular agent, a Voice Agent is continuously running and listening inside a RealtimeSession that handles the conversation and connection to the model over time. This session will also handle the audio processing, interruptions, and a lot of the other lifecycle functionality we will cover later on.

import { RealtimeSession } from '@openai/agents-realtime';

const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
});

The RealtimeSession constructor takes an agent as the first argument. This agent will be the first agent that your user will be able to interact with.

Connect to the session

To connect to the session you need to pass the client ephemeral token you generated earlier on.

await session.connect({ apiKey: '<client-api-key>' });

This will connect to the Realtime API using WebRTC in the browser and automatically configure your microphone and speaker for audio input and output. If you are running your RealtimeSession on a backend server (like Node.js) the SDK will automatically use WebSocket as a connection. You can learn more about the different transport layers in the Realtime Transport Layer guide.

Putting it all together

import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Assistant',
  instructions: 'You are a helpful assistant.',
});

const session = new RealtimeSession(agent);

// Automatically connects your microphone and audio output
// in the browser via WebRTC.
await session.connect({
  apiKey: '<client-api-key>',
});

Fire up the engines and start talking

Start up your webserver and navigate to the page that includes your new Realtime Agent code. You should see a request for microphone access. Once you grant access you should be able to start talking to your agent.

Terminal window
npm run dev

Next Steps
From here you can start designing and building your own voice agent. Voice agents include a lot of the same features as regular agents, but have some of their own unique features.

Learn how to give your voice agent:

Tools
Handoffs
Guardrails
Handle audio interruptions
Manage session history
Learn more about the different transport layers.

WebRTC
WebSocket
Building your own transport mechanism

Building Voice Agents
Audio handling
Some transport layers like the default OpenAIRealtimeWebRTC will handle audio input and output automatically for you. For other transport mechanisms like OpenAIRealtimeWebSocket you will have to handle session audio yourself:

import {
  RealtimeAgent,
  RealtimeSession,
  TransportLayerAudio,
} from '@openai/agents/realtime';

const agent = new RealtimeAgent({ name: 'My agent' });
const session = new RealtimeSession(agent);
const newlyRecordedAudio = new ArrayBuffer(0);

session.on('audio', (event: TransportLayerAudio) => {
  // play your audio
});

// send new audio to the agent
session.sendAudio(newlyRecordedAudio);

Session configuration
You can configure your session by passing additional options to either the RealtimeSession during construction or when you call connect(...).

import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Greeter',
  instructions: 'Greet the user with cheer and answer questions.',
});

const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
  config: {
    inputAudioFormat: 'pcm16',
    outputAudioFormat: 'pcm16',
    inputAudioTranscription: {
      model: 'gpt-4o-mini-transcribe',
    },
  },
});

These transport layers allow you to pass any parameter that matches session.

For parameters that are new and don’t have a matching parameter in the RealtimeSessionConfig you can use providerData. Anything passed in providerData will be passed directly as part of the session object.

Handoffs
Similarly to regular agents, you can use handoffs to break your agent into multiple agents and orchestrate between them to improve the performance of your agents and better scope the problem.

import { RealtimeAgent } from '@openai/agents/realtime';

const mathTutorAgent = new RealtimeAgent({
  name: 'Math Tutor',
  handoffDescription: 'Specialist agent for math questions',
  instructions:
    'You provide help with math problems. Explain your reasoning at each step and include examples',
});

const agent = new RealtimeAgent({
  name: 'Greeter',
  instructions: 'Greet the user with cheer and answer questions.',
  handoffs: [mathTutorAgent],
});

Unlike regular agents, handoffs behave slightly differently for Realtime Agents. When a handoff is performed, the ongoing session will be updated with the new agent configuration. Because of this, the agent automatically has access to the ongoing conversation history and input filters are currently not applied.

Additionally, this means that the voice or model cannot be changed as part of the handoff. You can also only connect to other Realtime Agents. If you need to use a different model, for example a reasoning model like o4-mini, you can use delegation through tools.

Tools
Just like regular agents, Realtime Agents can call tools to perform actions. You can define a tool using the same tool() function that you would use for a regular agent.

import { tool, RealtimeAgent } from '@openai/agents/realtime';
import { z } from 'zod';

const getWeather = tool({
  name: 'get_weather',
  description: 'Return the weather for a city.',
  parameters: z.object({ city: z.string() }),
  async execute({ city }) {
    return `The weather in ${city} is sunny.`;
  },
});

const weatherAgent = new RealtimeAgent({
  name: 'Weather assistant',
  instructions: 'Answer weather questions.',
  tools: [getWeather],
});

You can only use function tools with Realtime Agents and these tools will be executed in the same place as your Realtime Session. This means if you are running your Realtime Session in the browser, your tool will be executed in the browser. If you need to perform more sensitive actions, you can make an HTTP request within your tool to your backend server.

While the tool is executing the agent will not be able to process new requests from the user. One way to improve the experience is by telling your agent to announce when it is about to execute a tool or say specific phrases to buy the agent some time to execute the tool.

Accessing the conversation history
Additionally to the arguments that the agent called a particular tool with, you can also access a snapshot of the current conversation history that is tracked by the Realtime Session. This can be useful if you need to perform a more complex action based on the current state of the conversation or are planning to use tools for delegation.

import {
  tool,
  RealtimeContextData,
  RealtimeItem,
} from '@openai/agents/realtime';
import { z } from 'zod';

const parameters = z.object({
  request: z.string(),
});

const refundTool = tool<typeof parameters, RealtimeContextData>({
  name: 'Refund Expert',
  description: 'Evaluate a refund',
  parameters,
  execute: async ({ request }, details) => {
    // The history might not be available
    const history: RealtimeItem[] = details?.context?.history ?? [];
    // making your call to process the refund request
  },
});

Note

The history passed in is a snapshot of the history at the time of the tool call. The transcription of the last thing the user said might not be available yet.

Approval before tool execution
If you define your tool with needsApproval: true the agent will emit a tool_approval_requested event before executing the tool.

By listening to this event you can show a UI to the user to approve or reject the tool call.

import { session } from './agent';

session.on('tool_approval_requested', (_context, _agent, request) => {
  // show a UI to the user to approve or reject the tool call
  // you can use the `session.approve(...)` or `session.reject(...)` methods to approve or reject the tool call

  session.approve(request.approvalItem); // or session.reject(request.rawItem);
});

Note

While the voice agent is waiting for approval for the tool call, the agent won’t be able to process new requests from the user.

Guardrails
Guardrails offer a way to monitor whether what the agent has said violated a set of rules and immediately cut off the response. These guardrail checks will be performed based on the transcript of the agent’s response and therefore requires that the text output of your model is enabled (it is enabled by default).

The guardrails that you provide will run asynchronously as a model response is returned, allowing you to cut off the response based a predefined classification trigger, for example “mentions a specific banned word”.

When a guardrail trips the session emits a guardrail_tripped event. The event also provides a details object containing the itemId that triggered the guardrail.

import { RealtimeOutputGuardrail, RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Greeter',
  instructions: 'Greet the user with cheer and answer questions.',
});

const guardrails: RealtimeOutputGuardrail[] = [
  {
    name: 'No mention of Dom',
    async execute({ agentOutput }) {
      const domInOutput = agentOutput.includes('Dom');
      return {
        tripwireTriggered: domInOutput,
        outputInfo: { domInOutput },
      };
    },
  },
];

const guardedSession = new RealtimeSession(agent, {
  outputGuardrails: guardrails,
});

By default guardrails are run every 100 characters or at the end of the response text has been generated. Since speaking out the text normally takes longer it means that in most cases the guardrail should catch the violation before the user can hear it.

If you want to modify this behavior you can pass a outputGuardrailSettings object to the session.

import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Greeter',
  instructions: 'Greet the user with cheer and answer questions.',
});

const guardedSession = new RealtimeSession(agent, {
  outputGuardrails: [
    /*...*/
  ],
  outputGuardrailSettings: {
    debounceTextLength: 500, // run guardrail every 500 characters or set it to -1 to run it only at the end
  },
});

Turn detection / voice activity detection
The Realtime Session will automatically detect when the user is speaking and trigger new turns using the built-in voice activity detection modes of the Realtime API.

You can change the voice activity detection mode by passing a turnDetection object to the session.

import { RealtimeSession } from '@openai/agents/realtime';
import { agent } from './agent';

const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
  config: {
    turnDetection: {
      type: 'semantic_vad',
      eagerness: 'medium',
      createResponse: true,
      interruptResponse: true,
    },
  },
});

Modifying the turn detection settings can help calibrate unwanted interruptions and dealing with silence. Check out the Realtime API documentation for more details on the different settings

Interruptions
When using the built-in voice activity detection, speaking over the agent automatically triggers the agent to detect and update its context based on what was said. It will also emit an audio_interrupted event. This can be used to immediately stop all audio playback (only applicable to WebSocket connections).

import { session } from './agent';

session.on('audio_interrupted', () => {
  // handle local playback interruption
});

If you want to perform a manual interruption, for example if you want to offer a “stop” button in your UI, you can call interrupt() manually:

import { session } from './agent';

session.interrupt();
// this will still trigger the `audio_interrupted` event for you
// to cut off the audio playback when using WebSockets

In either way, the Realtime Session will handle both interrupting the generation of the agent, truncate its knowledge of what was said to the user, and update the history.

If you are using WebRTC to connect to your agent, it will also clear the audio output. If you are using WebSocket, you will need to handle this yourself by stopping audio playack of whatever has been queued up to be played.

Text input
If you want to send text input to your agent, you can use the sendMessage method on the RealtimeSession.

This can be useful if you want to enable your user to interface in both modalities with the agent, or to provide additional context to the conversation.

import { RealtimeSession, RealtimeAgent } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Assistant',
});

const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
});

session.sendMessage('Hello, how are you?');

Conversation history management
The RealtimeSession automatically manages the conversation history in a history property:

You can use this to render the history to the customer or perform additional actions on it. As this history will constantly change during the course of the conversation you can listen for the history_updated event.

If you want to modify the history, like removing a message entirely or updating its transcript, you can use the updateHistory method.

import { RealtimeSession, RealtimeAgent } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Assistant',
});

const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
});

await session.connect({ apiKey: '<client-api-key>' });

// listening to the history_updated event
session.on('history_updated', (history) => {
  // returns the full history of the session
  console.log(history);
});

// Option 1: explicit setting
session.updateHistory([
  /* specific history */
]);

// Option 2: override based on current state like removing all agent messages
session.updateHistory((currentHistory) => {
  return currentHistory.filter(
    (item) => !(item.type === 'message' && item.role === 'assistant'),
  );
});

Limitations
You can currently not update/change function tool calls after the fact
Text output in the history requires transcripts and text modalities to be enabled
Responses that were truncated due to an interruption do not have a transcript
Delegation through tools
Delegation through tools

By combining the conversation history with a tool call, you can delegate the conversation to another backend agent to perform a more complex action and then pass it back as the result to the user.

import {
  RealtimeAgent,
  RealtimeContextData,
  tool,
} from '@openai/agents/realtime';
import { handleRefundRequest } from './serverAgent';
import z from 'zod';

const refundSupervisorParameters = z.object({
  request: z.string(),
});

const refundSupervisor = tool<
  typeof refundSupervisorParameters,
  RealtimeContextData
>({
  name: 'escalateToRefundSupervisor',
  description: 'Escalate a refund request to the refund supervisor',
  parameters: refundSupervisorParameters,
  execute: async ({ request }, details) => {
    // This will execute on the server
    return handleRefundRequest(request, details?.context?.history ?? []);
  },
});

const agent = new RealtimeAgent({
  name: 'Customer Support',
  instructions:
    'You are a customer support agent. If you receive any requests for refunds, you need to delegate to your supervisor.',
  tools: [refundSupervisor],
});

The code below will then be executed on the server. In this example through a server actions in Next.js.

// This runs on the server
import 'server-only';

import { Agent, run } from '@openai/agents';
import type { RealtimeItem } from '@openai/agents/realtime';
import z from 'zod';

const agent = new Agent({
  name: 'Refund Expert',
  instructions:
    'You are a refund expert. You are given a request to process a refund and you need to determine if the request is valid.',
  model: 'o4-mini',
  outputType: z.object({
    reasong: z.string(),
    refundApproved: z.boolean(),
  }),
});

export async function handleRefundRequest(
  request: string,
  history: RealtimeItem[],
) {
  const input = `
The user has requested a refund.

The request is: ${request}

Current conversation history:
${JSON.stringify(history, null, 2)}
`.trim();

  const result = await run(agent, input);

  return JSON.stringify(result.finalOutput, null, 2);
}

Using Realtime Agents with Twilio
Twilio offers a Media Streams API that sends the raw audio from a phone call to a WebSocket server. This set up can be used to connect your voice agents to Twilio. You can use the default Realtime Session transport in websocket mode to connect the events coming from Twilio to your Realtime Session. However, this requires you to set the right audio format and adjust your own interruption timing as phone calls will naturally introduce more latency than a web-based conversation.

To improve the set up experience, we’ve created a dedicated transport layer that handles the connection to Twilio for you, including handling interruptions and audio forwarding for you.

Caution

This adapter is still in beta. You may run into edge case issues or bugs. Please report any issues via GitHub issues and we’ll fix quickly.

Setup
Make sure you have a Twilio account and a Twilio phone number.

Set up a WebSocket server that can receive events from Twilio.

If you are developing locally, this will require you to configure a local tunnel like this will require you to configure a local tunnel like ngrok or Cloudflare Tunnel to make your local server accessible to Twilio. You can use the TwilioRealtimeTransportLayer to connect to Twilio.

Install the Twilio adapter by installing the extensions package:

Terminal window
npm install @openai/agents-extensions

Import the adapter and model to connect to your RealtimeSession:

import { TwilioRealtimeTransportLayer } from '@openai/agents-extensions';
import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'My Agent',
});

// Create a new transport mechanism that will bridge the connection between Twilio and
// the OpenAI Realtime API.
const twilioTransport = new TwilioRealtimeTransportLayer({
  twilioWebSocket: websocketConnection,
});

const session = new RealtimeSession(agent, {
  // set your own transport
  transport: twilioTransport,
});

Connect your RealtimeSession to Twilio:

session.connect({ apiKey: 'your-openai-api-key' });

Any event and behavior that you would expect from a RealtimeSession will work as expected including tool calls, guardrails, and more. Read the voice agents guide for more information on how to use the RealtimeSession with voice agents.

Tips and Considerations
Speed is the name of the game.

In order to receive all the necessary events and audio from Twilio, you should create your TwilioRealtimeTransportLayer instance as soon as you have a reference to the WebSocket connection and immediately call session.connect() afterwards.

Access the raw Twilio events.

If you want to access the raw events that are being sent by Twilio, you can listen to the transport_event event on your RealtimeSession instance. Every event from Twilio will have a type of twilio_message and a message property that contains the raw event data.

Watch debug logs.

Sometimes you may run into issues where you want more information on what’s going on. Using a DEBUG=openai-agents* environment variable will show all the debug logs from the Agents SDK. Alternatively, you can enable just debug logs for the Twilio adapter using DEBUG=openai-agents:extensions:twilio*.

Full example server
Below is an example of a full end-to-end example of a WebSocket server that receives requests from Twilio and forwards them to a RealtimeSession.

Example server using Fastify
import Fastify from 'fastify';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';
import { TwilioRealtimeTransportLayer } from '@openai/agents-extensions';

// Load environment variables from .env file
dotenv.config();

// Retrieve the OpenAI API key from environment variables. You must have OpenAI Realtime API access.
const { OPENAI_API_KEY } = process.env;
if (!OPENAI_API_KEY) {
  console.error('Missing OpenAI API key. Please set it in the .env file.');
  process.exit(1);
}
const PORT = +(process.env.PORT || 5050);

// Initialize Fastify
const fastify = Fastify();
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

const agent = new RealtimeAgent({
  name: 'Triage Agent',
  instructions:
    'You are a helpful assistant that starts every conversation with a creative greeting.',
});

// Root Route
fastify.get('/', async (request, reply) => {
  reply.send({ message: 'Twilio Media Stream Server is running!' });
});

// Route for Twilio to handle incoming and outgoing calls
// <Say> punctuation to improve text-to-speech translation
fastify.all('/incoming-call', async (request, reply) => {
  const twimlResponse = `
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>O.K. you can start talking!</Say>
    <Connect>
        <Stream url="wss://${request.headers.host}/media-stream" />
    </Connect>
</Response>`.trim();
  reply.type('text/xml').send(twimlResponse);
});

// WebSocket route for media-stream
fastify.register(async (fastify) => {
  fastify.get('/media-stream', { websocket: true }, async (connection) => {
    const twilioTransportLayer = new TwilioRealtimeTransportLayer({
      twilioWebSocket: connection,
    });

    const session = new RealtimeSession(agent, {
      transport: twilioTransportLayer,
    });

    await session.connect({
      apiKey: OPENAI_API_KEY,
    });
    console.log('Connected to the OpenAI Realtime API');
  });
});

fastify.listen({ port: PORT }, (err) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`Server is listening on port ${PORT}`);
});

process.on('SIGINT', () => {
  fastify.close();
  process.exit(0);
});



Realtime Transport Layer
Default transport layers
Connecting over WebRTC
The default transport layer uses WebRTC. Audio is recorded from the microphone and played back automatically.

To use your own media stream or audio element, provide an OpenAIRealtimeWebRTC instance when creating the session.

import { RealtimeAgent, RealtimeSession, OpenAIRealtimeWebRTC } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Greeter',
  instructions: 'Greet the user with cheer and answer questions.',
});

async function main() {
  const transport = new OpenAIRealtimeWebRTC({
    mediaStream: await navigator.mediaDevices.getUserMedia({ audio: true }),
    audioElement: document.createElement('audio'),
  });

  const customSession = new RealtimeSession(agent, { transport });
}

Connecting over WebSocket
Pass transport: 'websocket' or an instance of OpenAIRealtimeWebSocket when creating the session to use a WebSocket connection instead of WebRTC. This works well for server-side use cases, for example building a phone agent with Twilio.

import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Greeter',
  instructions: 'Greet the user with cheer and answer questions.',
});

const myRecordedArrayBuffer = new ArrayBuffer(0);

const wsSession = new RealtimeSession(agent, {
  transport: 'websocket',
  model: 'gpt-realtime',
});
await wsSession.connect({ apiKey: process.env.OPENAI_API_KEY! });

wsSession.on('audio', (event) => {
  // event.data is a chunk of PCM16 audio
});

wsSession.sendAudio(myRecordedArrayBuffer);

Use any recording/playback library to handle the raw PCM16 audio bytes.

Building your own transport mechanism
If you want to use a different speech-to-speech API or have your own custom transport mechanism, you can create your own by implementing the RealtimeTransportLayer interface and emit the RealtimeTransportEventTypes events.

Interacting with the Realtime API more directly
If you want to use the OpenAI Realtime API but have more direct access to the Realtime API, you have two options:

Option 1 - Accessing the transport layer
If you still want to benefit from all of the capabilities of the RealtimeSession you can access your transport layer through session.transport.

The transport layer will emit every event it receives under the * event and you can send raw events using the sendEvent() method.

import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Greeter',
  instructions: 'Greet the user with cheer and answer questions.',
});

const session = new RealtimeSession(agent, {
  model: 'gpt-realtime',
});

session.transport.on('*', (event) => {
  // JSON parsed version of the event received on the connection
});

// Send any valid event as JSON. For example triggering a new response
session.transport.sendEvent({
  type: 'response.create',
  // ...
});

Option 2 — Only using the transport layer
If you don’t need automatic tool execution, guardrails, etc. you can also use the transport layer as a “thin” client that just manages connection and interruptions.

import { OpenAIRealtimeWebRTC } from '@openai/agents/realtime';

const client = new OpenAIRealtimeWebRTC();
const audioBuffer = new ArrayBuffer(0);

await client.connect({
  apiKey: '<api key>',
  model: 'gpt-4o-mini-realtime-preview',
  initialSessionConfig: {
    instructions: 'Speak like a pirate',
    voice: 'ash',
    modalities: ['text', 'audio'],
    inputAudioFormat: 'pcm16',
    outputAudioFormat: 'pcm16',
  },
});

// optionally for WebSockets
client.on('audio', (newAudio) => {});

client.sendAudio(audioBuffer);


Agent
The class representing an AI agent configured with instructions, tools, guardrails, handoffs and more.

We strongly recommend passing instructions, which is the “system prompt” for the agent. In addition, you can pass handoffDescription, which is a human-readable description of the agent, used when the agent is used inside tools/handoffs.

Agents are generic on the context type. The context is a (mutable) object you create. It is passed to tool functions, handoffs, guardrails, etc.

Extends
AgentHooks<TContext, TOutput>
Extended by
RealtimeAgent
Type Parameters
Type Parameter	Default type
TContext

UnknownContext

TOutput extends AgentOutputType

TextOutput

Implements
AgentConfiguration<TContext, TOutput>
Constructors
Constructor
new Agent<TContext, TOutput>(config): Agent<TContext, TOutput>;

Parameters
Parameter	Type	Description
config

{ handoffDescription?: string; handoffOutputTypeWarningEnabled?: boolean; handoffs?: ( | Agent<any, any> | Handoff<any, TOutput>)[]; inputGuardrails?: InputGuardrail[]; instructions?: string | (runContext, agent) => string | Promise<string>; mcpServers?: MCPServer[]; model?: string | Model; modelSettings?: ModelSettings; name: string; outputGuardrails?: OutputGuardrail<TOutput>[]; outputType?: TOutput; prompt?: Prompt | (runContext, agent) => Prompt | Promise<Prompt>; resetToolChoice?: boolean; tools?: Tool<TContext>[]; toolUseBehavior?: ToolUseBehavior; }

‐

config.handoffDescription?

string

A description of the agent. This is used when the agent is used as a handoff, so that an LLM knows what it does and when to invoke it.

config.handoffOutputTypeWarningEnabled?

boolean

The warning log would be enabled when multiple output types by handoff agents are detected.

config.handoffs?

( | Agent<any, any> | Handoff<any, TOutput>)[]

Handoffs are sub-agents that the agent can delegate to. You can provide a list of handoffs, and the agent can choose to delegate to them if relevant. Allows for separation of concerns and modularity.

config.inputGuardrails?

InputGuardrail[]

A list of checks that run in parallel to the agent’s execution, before generating a response. Runs only if the agent is the first agent in the chain.

config.instructions?

string | (runContext, agent) => string | Promise<string>

The instructions for the agent. Will be used as the “system prompt” when this agent is invoked. Describes what the agent should do, and how it responds.

Can either be a string, or a function that dynamically generates instructions for the agent. If you provide a function, it will be called with the context and the agent instance. It must return a string.

config.mcpServers?

MCPServer[]

A list of Model Context Protocol servers the agent can use. Every time the agent runs, it will include tools from these servers in the list of available tools.

NOTE: You are expected to manage the lifecycle of these servers. Specifically, you must call server.connect() before passing it to the agent, and server.cleanup() when the server is no longer needed.

config.model?

string | Model

The model implementation to use when invoking the LLM.

By default, if not set, the agent will use the default model returned by getDefaultModel (currently “gpt-4.1”).

config.modelSettings?

ModelSettings

Configures model-specific tuning parameters (e.g. temperature, top_p, etc.)

config.name

string

The name of the agent.

config.outputGuardrails?

OutputGuardrail<TOutput>[]

A list of checks that run on the final output of the agent, after generating a response. Runs only if the agent produces a final output.

config.outputType?

TOutput

The type of the output object. If not provided, the output will be a string.

config.prompt?

Prompt | (runContext, agent) => Prompt | Promise<Prompt>

The prompt template to use for the agent (OpenAI Responses API only).

Can either be a prompt template object, or a function that returns a prompt template object. If a function is provided, it will be called with the run context and the agent instance. It must return a prompt template object.

config.resetToolChoice?

boolean

Whether to reset the tool choice to the default value after a tool has been called. Defaults to true. This ensures that the agent doesn’t enter an infinite loop of tool usage.

config.tools?

Tool<TContext>[]

A list of tools the agent can use.

config.toolUseBehavior?

ToolUseBehavior

This lets you configure how tool use is handled.

run_llm_again: The default behavior. Tools are run, and then the LLM receives the results and gets to respond.
stop_on_first_tool: The output of the first tool call is used as the final output. This means that the LLM does not process the result of the tool call.
A list of tool names: The agent will stop running if any of the tools in the list are called. The final output will be the output of the first matching tool call. The LLM does not process the result of the tool call.
A function: if you pass a function, it will be called with the run context and the list of tool results. It must return a ToolsToFinalOutputResult, which determines whether the tool call resulted in a final output.
NOTE: This configuration is specific to FunctionTools. Hosted tools, such as file search, web search, etc. are always processed by the LLM

Returns
Agent<TContext, TOutput>

Overrides
AgentHooks.constructor

Properties
handoffDescription
handoffDescription: string;

A description of the agent. This is used when the agent is used as a handoff, so that an LLM knows what it does and when to invoke it.

Implementation of
AgentConfiguration.handoffDescription

handoffs
handoffs: (
  | Handoff<any, TOutput>
  | Agent<any, TOutput>)[];

Handoffs are sub-agents that the agent can delegate to. You can provide a list of handoffs, and the agent can choose to delegate to them if relevant. Allows for separation of concerns and modularity.

Implementation of
AgentConfiguration.handoffs

inputGuardrails
inputGuardrails: InputGuardrail[];

A list of checks that run in parallel to the agent’s execution, before generating a response. Runs only if the agent is the first agent in the chain.

Implementation of
AgentConfiguration.inputGuardrails

instructions
instructions: string | (runContext, agent) => string | Promise<string>;

The instructions for the agent. Will be used as the “system prompt” when this agent is invoked. Describes what the agent should do, and how it responds.

Can either be a string, or a function that dynamically generates instructions for the agent. If you provide a function, it will be called with the context and the agent instance. It must return a string.

Implementation of
AgentConfiguration.instructions

mcpServers
mcpServers: MCPServer[];

A list of Model Context Protocol servers the agent can use. Every time the agent runs, it will include tools from these servers in the list of available tools.

NOTE: You are expected to manage the lifecycle of these servers. Specifically, you must call server.connect() before passing it to the agent, and server.cleanup() when the server is no longer needed.

Implementation of
AgentConfiguration.mcpServers

model
model: string | Model;

The model implementation to use when invoking the LLM.

By default, if not set, the agent will use the default model returned by getDefaultModel (currently “gpt-4.1”).

Implementation of
AgentConfiguration.model

modelSettings
modelSettings: ModelSettings;

Configures model-specific tuning parameters (e.g. temperature, top_p, etc.)

Implementation of
AgentConfiguration.modelSettings

name
name: string;

The name of the agent.

Implementation of
AgentConfiguration.name

outputGuardrails
outputGuardrails: OutputGuardrail<AgentOutputType<unknown>>[];

A list of checks that run on the final output of the agent, after generating a response. Runs only if the agent produces a final output.

Implementation of
AgentConfiguration.outputGuardrails

outputType
outputType: TOutput;

The type of the output object. If not provided, the output will be a string.

Implementation of
AgentConfiguration.outputType

prompt?
optional prompt: Prompt | (runContext, agent) => Prompt | Promise<Prompt>;

The prompt template to use for the agent (OpenAI Responses API only).

Can either be a prompt template object, or a function that returns a prompt template object. If a function is provided, it will be called with the run context and the agent instance. It must return a prompt template object.

Implementation of
AgentConfiguration.prompt

resetToolChoice
resetToolChoice: boolean;

Whether to reset the tool choice to the default value after a tool has been called. Defaults to true. This ensures that the agent doesn’t enter an infinite loop of tool usage.

Implementation of
AgentConfiguration.resetToolChoice

tools
tools: Tool<TContext>[];

A list of tools the agent can use.

Implementation of
AgentConfiguration.tools

toolUseBehavior
toolUseBehavior: ToolUseBehavior;

This lets you configure how tool use is handled.

run_llm_again: The default behavior. Tools are run, and then the LLM receives the results and gets to respond.
stop_on_first_tool: The output of the first tool call is used as the final output. This means that the LLM does not process the result of the tool call.
A list of tool names: The agent will stop running if any of the tools in the list are called. The final output will be the output of the first matching tool call. The LLM does not process the result of the tool call.
A function: if you pass a function, it will be called with the run context and the list of tool results. It must return a ToolsToFinalOutputResult, which determines whether the tool call resulted in a final output.
NOTE: This configuration is specific to FunctionTools. Hosted tools, such as file search, web search, etc. are always processed by the LLM

Implementation of
AgentConfiguration.toolUseBehavior

DEFAULT_MODEL_PLACEHOLDER
static DEFAULT_MODEL_PLACEHOLDER: string;

Accessors
outputSchemaName
Get Signature
get outputSchemaName(): string;

Output schema name.

Returns
string

Methods
asTool()
asTool(options): FunctionTool;

Transform this agent into a tool, callable by other agents.

This is different from handoffs in two ways:

In handoffs, the new agent receives the conversation history. In this tool, the new agent receives generated input.
In handoffs, the new agent takes over the conversation. In this tool, the new agent is called as a tool, and the conversation is continued by the original agent.
Parameters
Parameter	Type	Description
options

{ customOutputExtractor?: (output) => string | Promise<string>; toolDescription?: string; toolName?: string; }

Options for the tool.

options.customOutputExtractor?

(output) => string | Promise<string>

A function that extracts the output text from the agent. If not provided, the last message from the agent will be used.

options.toolDescription?

string

The description of the tool, which should indicate what the tool does and when to use it.

options.toolName?

string

The name of the tool. If not provided, the name of the agent will be used.

Returns
FunctionTool

A tool that runs the agent and returns the output text.

clone()
clone(config): Agent<TContext, TOutput>;

Makes a copy of the agent, with the given arguments changed. For example, you could do:

const newAgent = agent.clone({ instructions: 'New instructions' })

Parameters
Parameter	Type	Description
config

Partial<AgentConfiguration<TContext, TOutput>>

A partial configuration to change.

Returns
Agent<TContext, TOutput>

A new agent with the given changes.

emit()
emit<K>(type, ...args): boolean;

Type Parameters
Type Parameter
K extends keyof AgentHookEvents<TContext, TOutput>

Parameters
Parameter	Type
type

K

…args

AgentHookEvents<TContext, TOutput>[K]

Returns
boolean

Inherited from
AgentHooks.emit

getAllTools()
getAllTools(runContext): Promise<Tool<TContext>[]>;

ALl agent tools, including the MCPl and function tools.

Parameters
Parameter	Type
runContext

RunContext<TContext>

Returns
Promise<Tool<TContext>[]>

all configured tools

getMcpTools()
getMcpTools(runContext): Promise<Tool<TContext>[]>;

Fetches the available tools from the MCP servers.

Parameters
Parameter	Type
runContext

RunContext<TContext>

Returns
Promise<Tool<TContext>[]>

the MCP powered tools

getPrompt()
getPrompt(runContext): Promise<undefined | Prompt>;

Returns the prompt template for the agent, if defined.

If the agent has a function as its prompt, this function will be called with the runContext and the agent instance.

Parameters
Parameter	Type
runContext

RunContext<TContext>

Returns
Promise<undefined | Prompt>

getSystemPrompt()
getSystemPrompt(runContext): Promise<undefined | string>;

Returns the system prompt for the agent.

If the agent has a function as its instructions, this function will be called with the runContext and the agent instance.

Parameters
Parameter	Type
runContext

RunContext<TContext>

Returns
Promise<undefined | string>

off()
off<K>(type, listener): EventEmitter<EventTypes>;

Type Parameters
Type Parameter
K extends keyof AgentHookEvents<TContext, TOutput>

Parameters
Parameter	Type
type

K

listener

(…args) => void

Returns
EventEmitter<EventTypes>

Inherited from
AgentHooks.off

on()
on<K>(type, listener): EventEmitter<EventTypes>;

Type Parameters
Type Parameter
K extends keyof AgentHookEvents<TContext, TOutput>

Parameters
Parameter	Type
type

K

listener

(…args) => void

Returns
EventEmitter<EventTypes>

Inherited from
AgentHooks.on

once()
once<K>(type, listener): EventEmitter<EventTypes>;

Type Parameters
Type Parameter
K extends keyof AgentHookEvents<TContext, TOutput>

Parameters
Parameter	Type
type

K

listener

(…args) => void

Returns
EventEmitter<EventTypes>

Inherited from
AgentHooks.once

processFinalOutput()
processFinalOutput(output): ResolvedAgentOutput<TOutput>;

Processes the final output of the agent.

Parameters
Parameter	Type	Description
output

string

The output of the agent.

Returns
ResolvedAgentOutput<TOutput>

The parsed out.

toJSON()
toJSON(): object;

Returns a JSON representation of the agent, which is serializable.

Returns
object

A JSON object containing the agent’s name.

name
name: string;

create()
static create<TOutput, Handoffs>(config): Agent<unknown, TOutput | HandoffsOutputUnion<Handoffs>>;

Create an Agent with handoffs and automatically infer the union type for TOutput from the handoff agents’ output types.

Type Parameters
Type Parameter	Default type
TOutput extends AgentOutputType<unknown>

"text"

Handoffs extends readonly ( | Agent<any, any> | Handoff<any, any>)[]

[]

Parameters
Parameter	Type
config

AgentConfigWithHandoffs<TOutput, Handoffs>

Returns
Agent<unknown, TOutput | HandoffsOutputUnion<Handoffs>>
