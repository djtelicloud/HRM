import { best, encodeAtoms, ensureInitialized, fuse, runDemoConversation, seed, step } from "./memory_agent.js";

async function main() {
  await ensureInitialized();
  // Demo: text turns -> atoms -> HRM advice
  await runDemoConversation([
    "User: Need to triage a Twilio call drop issue.",
    "Agent: I can help. What's your call SID?",
    "User: CA123... and it drops after IVR.",
    "Agent: Checking logs and SIP traces.",
    "User: We also changed our webhook yesterday.",
    "Agent: Noted. Reviewing webhook responses and media stream health.",
  ]);

  // Or seed arbitrary encoded inputs
  const enc = await encodeAtoms([{ role: "user", content: "How to escalate a PCI incident?" }], 256);
  await seed(enc.inputs, 2);
  await step(32);
  console.log("best", await best(1));
  console.log("fuse", await fuse(3));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
