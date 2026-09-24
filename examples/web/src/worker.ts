/// <reference lib="webworker" />
import init, {
  WasmSession,
} from "../../../crates/rcswx-wasm/pkg/rcswx_wasm.js";
import wasmUrl from "../../../crates/rcswx-wasm/pkg/rcswx_wasm_bg.wasm?url";
import {
  browserError,
  failure,
  validEnvelope,
  type Request,
  type Response,
} from "./protocol";
const scope = self as unknown as DedicatedWorkerGlobalScope;
let session: WasmSession | undefined;
let sessionId: string | undefined;
let initialized: Promise<unknown> | undefined;
let chain = Promise.resolve();
function validate(request: Request): void {
  if (
    !validEnvelope(request) ||
    typeof request.kind !== "string" ||
    !request.payload ||
    typeof request.payload !== "object"
  )
    throw failure("invalid_input", "Malformed request envelope.", "protocol");
  const payload = request.payload as unknown as Record<string, unknown>;
  if (
    request.kind !== "version" &&
    request.kind !== "analyze" &&
    (typeof payload.plan_id !== "string" || payload.plan_id.length > 160)
  )
    throw failure(
      "invalid_input",
      "A valid plan identifier is required.",
      "protocol",
    );
  const number = (field: string) => {
    if (typeof payload[field] !== "number")
      throw failure("invalid_input", `${field} must be a number.`, "protocol");
  };
  switch (request.kind) {
    case "analyze":
      if (
        typeof payload.parent1_json !== "string" ||
        typeof payload.parent2_json !== "string" ||
        !payload.options ||
        typeof payload.options !== "object"
      )
        throw failure(
          "invalid_input",
          "Analysis requires JSON text parents and options.",
          "protocol",
        );
      if (
        payload.parent1_json.length > 256 * 1024 ||
        payload.parent2_json.length > 256 * 1024
      )
        throw failure(
          "input_limit",
          "Parent JSON exceeds browser policy.",
          "input",
        );
      try {
        JSON.stringify(payload.options);
      } catch {
        throw failure(
          "invalid_input",
          "Options must be JSON-serializable.",
          "options",
        );
      }
      break;
    case "preview_step":
      number("k");
      break;
    case "sample":
      number("skewness");
      if (typeof payload.seed_hex !== "string")
        throw failure(
          "invalid_seed",
          "Seed must be hexadecimal text.",
          "protocol",
        );
      break;
    case "trace_page":
      number("offset");
      number("limit");
      break;
    case "apply_selection":
      if (
        !Array.isArray(payload.path_indices) ||
        !payload.path_indices.every(
          (i) =>
            typeof i === "number" &&
            Number.isInteger(i) &&
            i >= 0 &&
            i <= 0xffffffff,
        )
      )
        throw failure(
          "invalid_selection",
          "Path indices must be nonnegative wasm32 integers.",
          "protocol",
        );
      break;
    case "version":
    case "inspect":
    case "dispose":
      break;
    default:
      throw failure("unsupported_operation", "Unknown operation.", "protocol");
  }
}
async function dispatch(request: Request): Promise<Response> {
  const envelope = {
    protocol: 1 as const,
    session_id: request.session_id,
    request_id: request.request_id,
    figure_id: request.figure_id,
    revision: request.revision,
  };
  try {
    validate(request);
    if (sessionId && sessionId !== request.session_id)
      throw failure(
        "wrong_session",
        "Worker belongs to a different session.",
        "protocol",
      );
    sessionId = request.session_id;
    try {
      initialized ??= init({ module_or_path: wasmUrl });
      await initialized;
      session ??= new WasmSession(sessionId);
    } catch (error) {
      throw browserError(error, "initialization_failure");
    }
    let text: string;
    switch (request.kind) {
      case "version":
        text = session.version();
        break;
      case "analyze":
        text = session.analyze(
          request.payload.parent1_json,
          request.payload.parent2_json,
          JSON.stringify(request.payload.options),
        );
        break;
      case "inspect":
        text = session.inspect(request.payload.plan_id);
        break;
      case "preview_step":
        text = session.preview_step(request.payload.plan_id, request.payload.k);
        break;
      case "apply_selection":
        text = session.apply_selection(
          request.payload.plan_id,
          JSON.stringify(request.payload.path_indices),
        );
        break;
      case "sample":
        text = session.sample(
          request.payload.plan_id,
          request.payload.seed_hex,
          request.payload.skewness,
        );
        break;
      case "trace_page":
        text = session.trace_page(
          request.payload.plan_id,
          request.payload.offset,
          request.payload.limit,
        );
        break;
      case "dispose":
        text = session.dispose(request.payload.plan_id);
        break;
    }
    // Parse only the transport DTO. architecture_json remains untouched text.
    return { ...envelope, ok: true, result: JSON.parse(text) };
  } catch (error) {
    return { ...envelope, ok: false, error: browserError(error) };
  }
}
scope.onmessage = (event: MessageEvent<Request>) => {
  chain = chain
    .then(async () => {
      scope.postMessage(await dispatch(event.data));
    })
    .catch((error) => {
      scope.postMessage({
        protocol: 1,
        session_id: sessionId,
        request_id: event.data?.request_id,
        figure_id: event.data?.figure_id,
        revision: event.data?.revision,
        ok: false,
        error: browserError(error),
      });
    });
};
