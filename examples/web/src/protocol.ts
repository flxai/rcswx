import type {
  Analysis,
  Applied,
  AnalyzeOptions,
  BrowserError,
  TracePage,
  Version,
} from "../../../crates/rcswx-wasm/browser";
export type {
  Analysis,
  Applied,
  AnalyzeOptions,
  BrowserError,
  TracePage,
  Version,
};
export interface Operations {
  version: { payload: Record<string, never>; result: Version };
  analyze: {
    payload: {
      parent1_json: string;
      parent2_json: string;
      options: AnalyzeOptions;
    };
    result: Analysis;
  };
  inspect: { payload: { plan_id: string }; result: Analysis };
  preview_step: { payload: { plan_id: string; k: number }; result: Applied };
  apply_selection: {
    payload: { plan_id: string; path_indices: number[] };
    result: Applied;
  };
  sample: {
    payload: { plan_id: string; seed_hex: string; skewness: number };
    result: Applied;
  };
  trace_page: {
    payload: { plan_id: string; offset: number; limit: number };
    result: TracePage;
  };
  dispose: { payload: { plan_id: string }; result: { disposed: boolean } };
}
export type Kind = keyof Operations;
export interface Route {
  figure_id: string;
  revision: number;
}
export interface Envelope extends Route {
  protocol: 1;
  session_id: string;
  request_id: string;
}
export type Request = {
  [K in Kind]: Envelope & { kind: K; payload: Operations[K]["payload"] };
}[Kind];
export type Response = Envelope &
  (
    | { ok: true; result: Operations[Kind]["result"] }
    | { ok: false; error: BrowserError }
  );
export const failure = (
  code: string,
  message: string,
  stage = "worker",
): BrowserError => ({
  code,
  message,
  stage,
  retryable: [
    "cancelled",
    "timeout",
    "worker_crash",
    "initialization_failure",
  ].includes(code),
});
export function validEnvelope(value: unknown): value is Envelope {
  if (!value || typeof value !== "object") return false;
  const v = value as Partial<Envelope>;
  return (
    v.protocol === 1 &&
    typeof v.session_id === "string" &&
    v.session_id.length > 0 &&
    v.session_id.length <= 128 &&
    typeof v.request_id === "string" &&
    v.request_id.length <= 128 &&
    typeof v.figure_id === "string" &&
    v.figure_id.length <= 128 &&
    Number.isSafeInteger(v.revision) &&
    v.revision! >= 0
  );
}
export function browserError(
  value: unknown,
  code = "worker_crash",
): BrowserError {
  if (
    value &&
    typeof value === "object" &&
    "code" in value &&
    "stage" in value &&
    "message" in value &&
    "retryable" in value &&
    typeof value.code === "string" &&
    typeof value.stage === "string" &&
    typeof value.message === "string" &&
    typeof value.retryable === "boolean"
  )
    return value as BrowserError;
  return failure(code, value instanceof Error ? value.message : String(value));
}
/** Little-endian integer seed encoding without passing through a JS number. */
export function decimalSeed(text: string): string {
  if (!/^[0-9]+$/.test(text) || text.length > 78)
    throw failure(
      "invalid_seed",
      "Enter a decimal integer in [0, 2^256).",
      "input",
    );
  let value = BigInt(text);
  if (value >= 1n << 256n)
    throw failure("invalid_seed", "Seed exceeds 256 bits.", "input");
  let hex = "";
  for (let i = 0; i < 32; i++) {
    hex += Number(value & 255n)
      .toString(16)
      .padStart(2, "0");
    value >>= 8n;
  }
  return hex;
}
