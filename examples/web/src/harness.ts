import { BrowserClient } from "./client";
import { decimalSeed } from "./protocol";
import { renderTree } from "./render";
import records from "../fixtures/recordings.json";
const harness = { BrowserClient, decimalSeed, renderTree, records };
declare global {
  interface Window {
    rcswxHarness: typeof harness;
  }
}
window.rcswxHarness = harness;
