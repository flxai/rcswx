/** Public JSON DTOs. Architecture documents remain opaque JSON text. */
export type ApiVersion = 1;
export type ArchitectureSchema = 1;
export type TraceSchema = 1;
export type TraceLevel = "none" | "summary" | "full";
export interface AnalyzeOptions {
  api: 1;
  collapse_corners?: boolean;
  trace_level?: TraceLevel;
  limits?: {
    max_work?: number;
    max_output?: number;
    max_allocation_bytes?: number;
  };
  trace_limits?: { max_events: number; max_bytes: number };
}
export interface BrowserError {
  code: string;
  stage: string;
  message: string;
  retryable: boolean;
  requested_step?: number;
}
export interface DisplayNode {
  occurrence_index: number;
  id: string;
  name: string;
  children: number[];
}
export interface DisplaySnapshot {
  snapshot_key: string;
  root: number;
  nodes: DisplayNode[];
}
export interface TokenView {
  index: number;
  id: string;
  name: string;
  children: string[];
  parent_arity: number;
  occurrence: number | null;
}
export interface Parents<T> {
  first: T;
  second: T;
}
export interface Edit {
  id: number;
  op_type: string;
  node1_id: string | null;
  node2_id: string | null;
  i: number;
  j: number;
  ii: number | null;
  jj: number | null;
  value: number;
  i_swapped: boolean;
  j_swapped: boolean;
  enabler_ops: (number | number[])[];
  disabler_ops: (number | number[])[];
}
export type TraceNumber =
  | number
  | { state: "uncomputed" | "nan" | "positive_infinity" | "negative_infinity" };
export type TraceEdit = Omit<Edit, "value"> & { value: TraceNumber };
export interface HistoryValue {
  previous: number | null;
  len: number;
  step: {
    id: number;
    kind: string;
    node1_id: string | null;
    node2_id: string | null;
    i: number;
    j: number;
    value: TraceNumber;
    i_swapped: boolean;
    j_swapped: boolean;
  };
}
export interface CellValue {
  value: TraceNumber;
  top: TraceNumber[];
  left: TraceNumber[];
  corner: TraceNumber[];
  histories: number[];
}
export interface Subproblem {
  id: number;
  parent: number | null;
  rows: number;
  columns: number;
  start_i: number;
  start_j: number;
  first_indices: number[];
  second_indices: number[];
}
export interface TraceData {
  header: {
    api: 1;
    architecture_schema: 1;
    engine_version: string;
    build: string;
    options: AnalyzeOptions;
    direction: string;
    input_hashes: Parents<string>;
  };
  preparation: {
    collapse_corners: boolean;
    direction: string;
    identities: string[];
    first_tokens: TokenView[];
    second_tokens: TokenView[];
  };
  subproblem: Subproblem;
  subproblem_end: { id: number };
  history: { id: number; revision: number; value: HistoryValue };
  cell: { id: number; revision: number; value: CellValue };
  history_clone: { source: number; target: number };
  cell_clone: { source: number; target: number };
  matrix: { subproblem: number; phase: string; cells: number[][] };
  computed: {
    subproblem: number;
    i: number;
    j: number;
    cell: number;
    predecessors: (number | null)[];
  };
  collapse: {
    left: number;
    right: number;
    i: number;
    j: number;
    first: boolean;
  };
  clean: { cell: number; completely: boolean };
  retained_histories: { histories: number[] };
  plan: {
    path_index: number;
    paths: TraceEdit[][];
    operations: number[];
    operations_unordered: number[];
    nontrivial: number[];
    distance: TraceNumber;
    stats: Record<string, number>;
  };
  requested: { selected_indices: number[] };
  execute: {
    path_index: number;
    operation_id: number;
    operation: string;
    already_performed: boolean;
  };
  actions: {
    path_index: number | null;
    recipe_offset: number;
    actions: Materialization[];
    ok: boolean;
    error?: string | null;
  };
  materialized: {
    handles: number[];
    origins: OccurrenceOrigin[];
    architecture_json: string;
  };
  outcome: { ok: boolean; error: string | null };
}
export type Materialization =
  | { kind: "deep_copy"; source: number; mapping: [number, number][] }
  | {
      kind: "new_sequence";
      node: number;
      template: number;
      parent: number | null;
      id: string;
    }
  | { kind: "set_children"; node: number; children: number[] }
  | { kind: "replace_child"; node: number; index: number; child: number }
  | { kind: "set_parent"; node: number; parent: number | null }
  | { kind: "set_id"; node: number; id: string };
export type TraceEvent = {
  [K in keyof TraceData]: { seq: number; kind: K; data: TraceData[K] };
}[keyof TraceData];
export interface TraceSummary {
  level: TraceLevel;
  complete: boolean;
  event_count: number;
  bytes: number;
  truncation_reason: string | null;
}
export interface Recording {
  schema: 1;
  level: TraceLevel;
  events: TraceEvent[];
  complete: boolean;
  bytes: number;
  truncation_reason: string | null;
  last_complete_event: number | null;
}
export interface Analysis {
  api: 1;
  plan_id: string;
  distance: number;
  path_index: number;
  history_count: number;
  path: Edit[];
  operations: number[];
  operations_unordered: number[];
  nontrivial: number[];
  /** Dependency-compatible prefix lengths, not raw path indices. */
  prefix_steps: number[];
  parents: Parents<DisplaySnapshot>;
  tokens: Parents<TokenView[]>;
  trace: TraceSummary;
}
export interface SourceOccurrence {
  parent: 1 | 2;
  occurrence_index: number;
}
export interface OccurrenceOrigin {
  occurrence_index: number;
  sources: SourceOccurrence[];
  synthesized: boolean;
}
export interface Applied {
  api: 1;
  plan_id: string;
  selected_indices: number[];
  cost: number;
  step?: number;
  architecture_json: string;
  display: DisplaySnapshot;
  origins: OccurrenceOrigin[];
  recording: Recording;
  seed_hex?: string;
  skewness?: number;
  mask?: string;
}
export interface TracePage {
  api: 1;
  plan_id: string;
  offset: number;
  next_offset: number | null;
  total: number;
  complete: boolean;
  bytes: number;
  truncation_reason: string | null;
  events: TraceEvent[];
}
export interface Version {
  package: string;
  version: string;
  build: string;
  api: 1;
  architecture_schema: 1;
  trace_schema: 1;
  sampler: string;
  limits: Record<string, unknown>;
}
export interface ExamplePair {
  id: string;
  title: string;
  description: string;
  parent1_json: string;
  parent2_json: string;
  input_hashes: { first: string; second: string };
  options: AnalyzeOptions;
  analysis: Omit<Analysis, "plan_id">;
  recording: Recording;
  frames: Omit<Applied, "plan_id">[];
  samples: Omit<Applied, "plan_id">[];
  endpoint_projection: string;
}
export interface ExampleManifest {
  schema: 1;
  engine: Version;
  pairs: ExamplePair[];
}
