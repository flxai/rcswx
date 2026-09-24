import "./styles.css";
import rawRecords from "../fixtures/recordings.json";
import type {
  Analysis,
  Applied,
  ExampleManifest,
  Recording,
  TracePage,
} from "../../../crates/rcswx-wasm/browser";
import { BrowserClient } from "./client";
import { browserError, decimalSeed } from "./protocol";
import {
  element,
  renderTree,
  renderTokens,
  renderMatrix,
  renderPath,
  subproblems,
} from "./render";
const records = rawRecords as unknown as ExampleManifest;
if (
  records.schema !== 1 ||
  records.engine.api !== 1 ||
  records.engine.trace_schema !== 1
)
  throw new Error("Unsupported example recording schema.");
const byId = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;
const pairSelect = byId<HTMLSelectElement>("pair");
const stepInput = byId<HTMLInputElement>("step");
const cursor = byId<HTMLInputElement>("cursor");
const problemSelect = byId<HTMLSelectElement>("subproblem");
const liveButton = byId<HTMLButtonElement>("live");
const sampleButton = byId<HTMLButtonElement>("sample");
const replayButton = byId<HTMLButtonElement>("replay");
const client = new BrowserClient();
let pair = records.pairs[0];
let analysis: Omit<Analysis, "plan_id"> = pair.analysis;
let recording: Recording = pair.recording;
let livePlan: string | null = null;
let revision = 0;
let stepRevision = 0;
let timer: number | undefined;
let liveFrames = new Map<number, Applied>();
let displayed: Omit<Applied, "plan_id"> = pair.frames[0];
const status = (text: string) => {
  byId("status").textContent = text;
};
function renderRecording() {
  byId("cursor-label").textContent =
    `Event ${cursor.value} of ${recording.events.length}`;
  renderMatrix(
    byId("matrix"),
    byId("cell-details"),
    recording,
    analysis,
    Number(cursor.value),
    Number(problemSelect.value),
  );
}
function setRecording(next: Recording) {
  recording = next;
  const problems = subproblems(recording);
  problemSelect.replaceChildren(
    ...problems.map((p) => {
      const option = element(
        "option",
        `#${p.id} · ${p.parent === null ? "root" : `child of #${p.parent}`} · ${p.rows} × ${p.columns}`,
      );
      option.value = String(p.id);
      return option;
    }),
  );
  cursor.max = String(recording.events.length);
  cursor.value = cursor.max;
  cursor.disabled = recording.events.length === 0;
  byId("trace-status").textContent =
    recording.level !== "full"
      ? "A full computation recording is unavailable."
      : recording.complete
        ? `Complete recording · ${recording.bytes.toLocaleString()} serialized event bytes. Shared identities are preserved across matrices.`
        : `Recording truncated (${recording.truncation_reason}); the algorithm completed independently. Unrecorded cells are not inferred.`;
  renderRecording();
}
function showResult(result: Omit<Applied, "plan_id">, source: string) {
  displayed = result;
  renderTree(byId("child"), result.display, result.origins);
  byId("selection").textContent =
    `${source} · requested raw-path indices [${result.selected_indices.join(", ")}] · cost ${result.cost}`;
  renderPath(byId("path"), analysis, result.selected_indices);
  const details = JSON.stringify(
    {
      requested_indices: result.selected_indices,
      origins: result.origins,
      application: result.recording,
    },
    null,
    2,
  );
  byId("application").textContent =
    details.length > 24_000
      ? details.slice(0, 24_000) +
        "\n… display excerpt; export retains the complete bounded result."
      : details;
}
async function setStep(k: number) {
  const generation = ++stepRevision;
  k = Math.max(0, Math.min(k, analysis.nontrivial.length));
  stepInput.value = String(k);
  byId("step-label").textContent = `Step ${k} of ${analysis.nontrivial.length}`;
  byId<HTMLButtonElement>("previous").disabled = k === 0;
  byId<HTMLButtonElement>("next").disabled = k === analysis.nontrivial.length;
  const started = performance.now();
  try {
    if (!livePlan) showResult(pair.frames[k], "Cached native result");
    else {
      const cached = liveFrames.get(k);
      if (cached) showResult(cached, "Cached live result");
      else {
        byId("selection").textContent = `Computing requested step ${k}…`;
        byId("child").replaceChildren(
          element("p", "No child is displayed until this request completes."),
        );
        const plan = livePlan;
        const result = await client.request(
          "preview_step",
          { plan_id: plan, k },
          { figure_id: "explorer", revision },
        );
        if (generation !== stepRevision || plan !== livePlan) return;
        liveFrames.set(k, result);
        showResult(result, "Live WASM result");
      }
    }
    byId("metrics").textContent =
      `Last frame update: ${(performance.now() - started).toFixed(2)} ms. This is a local measurement, not a performance guarantee.`;
  } catch (error) {
    if (generation !== stepRevision) return;
    const failure = browserError(error);
    if (["superseded", "stale_result", "cancelled"].includes(failure.code))
      return;
    byId("child").replaceChildren(
      element("p", `No valid child was produced for requested step ${k}.`),
    );
    byId("selection").textContent = `${failure.code}: ${failure.message}`;
  }
}
function choosePair() {
  clearInterval(timer);
  timer = undefined;
  replayButton.textContent = replayButton.disabled
    ? "Use the recording slider (reduced motion)"
    : "Play recording";
  pair = records.pairs.find((p) => p.id === pairSelect.value)!;
  analysis = pair.analysis;
  livePlan = null;
  liveFrames = new Map();
  revision++;
  stepRevision++;
  client.setRevision("explorer", revision);
  liveButton.disabled = false;
  sampleButton.disabled = true;
  byId("description").textContent = pair.description;
  byId("sample-status").textContent = "";
  renderTree(byId("parent1"), analysis.parents.first);
  renderTree(byId("parent2"), analysis.parents.second);
  renderTokens(byId("tokens1"), analysis.tokens.first, 1);
  renderTokens(byId("tokens2"), analysis.tokens.second, 2);
  stepInput.max = String(analysis.nontrivial.length);
  stepInput.disabled = analysis.nontrivial.length === 0;
  setRecording(pair.recording);
  void setStep(0);
  status(
    `Cached native recording · distance ${analysis.distance} · selected history ${analysis.path_index} of ${analysis.history_count}. Live WASM has not analyzed this pair.`,
  );
}
async function enableLive() {
  const requestedRevision = revision;
  liveButton.disabled = true;
  status("Initializing the worker and analyzing the exact recorded inputs…");
  const started = performance.now();
  try {
    for (const [text, expected] of [
      [pair.parent1_json, pair.input_hashes.first],
      [pair.parent2_json, pair.input_hashes.second],
    ]) {
      const hash = await crypto.subtle.digest(
        "SHA-256",
        new TextEncoder().encode(text),
      );
      const actual = [...new Uint8Array(hash)]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
      if (actual !== expected)
        throw new Error(
          "Cached recording does not match the exact input text.",
        );
    }
    if (requestedRevision !== revision) return;
    const result = await client.request(
      "analyze",
      {
        parent1_json: pair.parent1_json,
        parent2_json: pair.parent2_json,
        options: pair.options,
      },
      { figure_id: "explorer", revision },
    );
    if (requestedRevision !== revision) return;
    analysis = result;
    livePlan = result.plan_id;
    liveFrames.clear();
    const events: Recording["events"] = [];
    let offset: number | null = 0;
    while (offset !== null) {
      const page: TracePage = await client.request(
        "trace_page",
        { plan_id: result.plan_id, offset, limit: 256 },
        { figure_id: "explorer", revision: requestedRevision },
      );
      events.push(...page.events);
      offset = page.next_offset;
    }
    if (requestedRevision !== revision || result.plan_id !== livePlan) return;
    setRecording({
      schema: 1,
      level: result.trace.level,
      events,
      complete: result.trace.complete,
      bytes: result.trace.bytes,
      truncation_reason: result.trace.truncation_reason,
      last_complete_event: events.length ? events.length - 1 : null,
    });
    sampleButton.disabled = false;
    await setStep(Number(stepInput.value));
    if (requestedRevision !== revision || result.plan_id !== livePlan) return;
    status(
      `Live WASM · distance ${result.distance} · analysis and trace transfer ${(performance.now() - started).toFixed(1)} ms · selected history ${result.path_index} of ${result.history_count}.`,
    );
  } catch (error) {
    if (requestedRevision !== revision) return;
    livePlan = null;
    sampleButton.disabled = true;
    liveButton.disabled = false;
    status(
      `Live computation unavailable (${browserError(error).message}). Exact-pair cached playback remains available.`,
    );
    setRecording(pair.recording);
    await setStep(Number(stepInput.value));
  }
}
pairSelect.replaceChildren(
  ...records.pairs.map((p) => {
    const option = element("option", p.title);
    option.value = p.id;
    return option;
  }),
);
pairSelect.addEventListener("change", choosePair);
stepInput.addEventListener(
  "input",
  () => void setStep(Number(stepInput.value)),
);
byId("previous").addEventListener(
  "click",
  () => void setStep(Number(stepInput.value) - 1),
);
byId("next").addEventListener(
  "click",
  () => void setStep(Number(stepInput.value) + 1),
);
liveButton.addEventListener("click", () => void enableLive());
byId("cancel").addEventListener("click", () => client.cancelAll());
client.addEventListener("invalidated", (event) => {
  livePlan = null;
  liveFrames.clear();
  sampleButton.disabled = true;
  liveButton.disabled = false;
  stepRevision++;
  status(
    `${(event as CustomEvent).detail.message} Displayed data remains a read-only snapshot; cached playback is available.`,
  );
});
cursor.addEventListener("input", renderRecording);
problemSelect.addEventListener("change", renderRecording);
replayButton.addEventListener("click", () => {
  if (timer !== undefined) {
    clearInterval(timer);
    timer = undefined;
    replayButton.textContent = "Play recording";
    return;
  }
  if (Number(cursor.value) === recording.events.length) cursor.value = "0";
  replayButton.textContent = "Pause recording";
  timer = window.setInterval(() => {
    cursor.value = String(
      Math.min(Number(cursor.value) + 1, recording.events.length),
    );
    renderRecording();
    if (Number(cursor.value) === recording.events.length) {
      clearInterval(timer);
      timer = undefined;
      replayButton.textContent = "Play recording";
    }
  }, 100);
});
if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
  replayButton.disabled = true;
  replayButton.textContent = "Use the recording slider (reduced motion)";
}
sampleButton.addEventListener("click", async () => {
  if (!livePlan) return;
  const plan = livePlan;
  const generation = ++stepRevision;
  sampleButton.disabled = true;
  try {
    const seed = byId<HTMLInputElement>("seed").value;
    const rawSkew = byId<HTMLInputElement>("skewness").value;
    if (rawSkew.trim() === "") throw new Error("Enter a finite skewness.");
    const result = await client.request(
      "sample",
      { plan_id: plan, seed_hex: decimalSeed(seed), skewness: Number(rawSkew) },
      { figure_id: "explorer", revision },
    );
    if (plan !== livePlan || generation !== stepRevision) return;
    showResult(result, "Seeded live sample");
    byId("sample-status").textContent =
      `Seed ${seed} (${result.seed_hex}); mask ${result.mask || "(empty)"}; selected [${result.selected_indices.join(", ")}]. The slider still denotes its separate prefix.`;
  } catch (error) {
    if (plan !== livePlan || generation !== stepRevision) return;
    const failure = browserError(error);
    byId("child").replaceChildren(
      element("p", "No valid sampled child was produced."),
    );
    byId("sample-status").textContent = `${failure.code}: ${failure.message}`;
  } finally {
    sampleButton.disabled = livePlan === null;
  }
});
byId("export").addEventListener("click", () => {
  const exported = {
    schema: 1,
    engine: records.engine,
    pairs: [
      {
        ...pair,
        analysis: { ...analysis, plan_id: undefined },
        recording,
        last_result: { ...displayed, plan_id: undefined },
      },
    ],
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(exported)], { type: "application/json" }),
  );
  const anchor = element("a");
  anchor.href = url;
  anchor.download = `rcswx-${pair.id}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
});
window.addEventListener("pagehide", () => {
  clearInterval(timer);
  client.close();
});
choosePair();
