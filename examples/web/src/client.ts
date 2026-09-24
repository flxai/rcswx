import {
  browserError,
  failure,
  validEnvelope,
  type BrowserError,
  type Kind,
  type Operations,
  type Request,
  type Response,
  type Route,
} from "./protocol";
interface Job {
  request: Request;
  resolve: (value: unknown) => void;
  reject: (error: BrowserError) => void;
  cleanup?: boolean;
}
export interface ClientOptions {
  timeout_ms?: number;
  initialization_timeout_ms?: number;
  max_queued?: number;
}
/** One page-owned instance; each figure supplies a revision and releases itself. */
export class BrowserClient extends EventTarget {
  private worker: Worker | null = null;
  private session = crypto.randomUUID();
  private next = 0;
  private queue: Job[] = [];
  private active: Job | null = null;
  private timer: number | undefined;
  private figures = new Map<string, number>();
  private plans = new Map<string, string>();
  private disposal = new Set<string>();
  private closed = false;
  private initialized = false;
  private readonly timeout: number;
  private readonly initializationTimeout: number;
  private readonly maxQueued: number;
  constructor(options: ClientOptions = {}) {
    super();
    this.timeout = options.timeout_ms ?? 10_000;
    this.initializationTimeout = options.initialization_timeout_ms ?? 20_000;
    this.maxQueued = options.max_queued ?? 16;
    if (
      ![this.timeout, this.initializationTimeout, this.maxQueued].every(
        (n) => Number.isSafeInteger(n) && n > 0,
      ) ||
      this.maxQueued > 64 ||
      this.timeout > 0x7fffffff ||
      this.initializationTimeout > 0x7fffffff
    )
      throw failure(
        "invalid_input",
        "Client limits must be positive bounded integers.",
        "client",
      );
  }
  get state() {
    return {
      session_id: this.session,
      queued: this.queue.length,
      active: this.active !== null,
      plans: this.plans.size,
      pending_disposals: this.disposal.size,
      initialized: this.initialized,
    };
  }
  setRevision(figure: string, revision: number): void {
    if (
      !figure ||
      figure.length > 128 ||
      !Number.isSafeInteger(revision) ||
      revision < 0
    )
      throw failure("invalid_input", "Invalid figure revision.", "client");
    const current = this.figures.get(figure);
    if (current !== undefined && revision < current)
      throw failure("stale_result", "Figure revision is obsolete.", "client");
    if (current === revision) return;
    this.figures.set(figure, revision);
    this.dropQueued(
      (job) => job.request.figure_id === figure,
      failure("stale_result", "Figure inputs changed.", "client"),
    );
    for (const [id, owner] of this.plans)
      if (owner === figure) {
        this.disposal.add(id);
        this.plans.delete(id);
      }
    this.pump();
  }
  request<K extends Kind>(
    kind: K,
    payload: Operations[K]["payload"],
    route: Route,
  ): Promise<Operations[K]["result"]> {
    const { promise, resolve, reject } =
      Promise.withResolvers<Operations[K]["result"]>();
    try {
      if (this.closed)
        throw failure("cancelled", "Client has been closed.", "client");
      this.setRevision(route.figure_id, route.revision);
      if (
        "plan_id" in payload &&
        kind !== "dispose" &&
        !this.plans.has(payload.plan_id)
      )
        throw failure(
          "expired_plan",
          "Plan is not live in this worker session.",
          "client",
        );
      if (kind === "preview_step")
        this.dropQueued(
          (job) =>
            job.request.kind === "preview_step" &&
            job.request.figure_id === route.figure_id,
          failure(
            "superseded",
            "A newer slider request replaced this queued request.",
            "client",
          ),
        );
      if (this.queue.length >= this.maxQueued)
        throw failure("queue_limit", "Worker queue is full.", "client");
      const request = {
        protocol: 1,
        session_id: this.session,
        request_id: String(++this.next),
        ...route,
        kind,
        payload,
      } as Request;
      this.queue.push({
        request,
        resolve: (value) => resolve(value as Operations[K]["result"]),
        reject,
      });
      this.pump();
    } catch (error) {
      reject(browserError(error));
    }
    return promise;
  }
  private dropQueued(
    predicate: (job: Job) => boolean,
    error: BrowserError,
  ): void {
    const kept: Job[] = [];
    for (const job of this.queue) {
      if (predicate(job)) job.reject(error);
      else kept.push(job);
    }
    this.queue = kept;
  }
  private ensureWorker(): Worker {
    if (!this.worker) {
      this.worker = new Worker(new URL("./worker.ts", import.meta.url), {
        type: "module",
        name: "rcswx",
      });
      this.worker.onmessage = (event) => this.receive(event.data);
      this.worker.onerror = (event) => {
        event.preventDefault();
        this.invalidate(
          failure("worker_crash", event.message || "Worker failed."),
        );
      };
      this.worker.onmessageerror = () =>
        this.invalidate(
          failure("worker_crash", "Worker response could not be decoded."),
        );
    }
    return this.worker;
  }
  private pump(): void {
    if (this.active || this.closed) return;
    const disposed = this.disposal.values().next();
    let job: Job | undefined;
    if (!disposed.done) {
      this.disposal.delete(disposed.value);
      job = {
        request: {
          protocol: 1,
          session_id: this.session,
          request_id: String(++this.next),
          figure_id: "__cleanup",
          revision: 0,
          kind: "dispose",
          payload: { plan_id: disposed.value },
        },
        cleanup: true,
        resolve: () => {},
        reject: () => {},
      };
    } else job = this.queue.shift();
    if (!job) return;
    this.active = job;
    try {
      const worker = this.ensureWorker();
      this.timer = setTimeout(
        () =>
          this.invalidate(
            failure(
              "timeout",
              "Computation timed out; every plan in the shared session expired.",
            ),
          ),
        this.initialized ? this.timeout : this.initializationTimeout,
      );
      worker.postMessage(job.request);
    } catch (error) {
      this.invalidate(browserError(error));
    }
  }
  private receive(value: unknown): void {
    const job = this.active;
    if (!job || !validEnvelope(value)) {
      if (job)
        this.invalidate(failure("worker_crash", "Malformed worker response."));
      return;
    }
    const response = value as Response;
    if (response.session_id !== this.session) return;
    const request = job.request;
    if (
      response.request_id !== request.request_id ||
      response.figure_id !== request.figure_id ||
      response.revision !== request.revision ||
      typeof response.ok !== "boolean"
    ) {
      this.invalidate(
        failure("worker_crash", "Worker response routing mismatch."),
      );
      return;
    }
    clearTimeout(this.timer);
    this.timer = undefined;
    if (
      !response.ok &&
      ["worker_crash", "initialization_failure"].includes(response.error.code)
    ) {
      this.invalidate(response.error);
      return;
    }
    this.active = null;
    this.initialized = true;
    const stale =
      !job.cleanup && this.figures.get(request.figure_id) !== request.revision;
    if (response.ok && request.kind === "analyze") {
      const id = (response.result as Operations["analyze"]["result"]).plan_id;
      if (stale) this.disposal.add(id);
      else this.plans.set(id, request.figure_id);
    }
    if (response.ok && request.kind === "dispose")
      this.plans.delete(request.payload.plan_id);
    if (stale)
      job.reject(
        failure(
          "stale_result",
          "Response belongs to obsolete figure inputs.",
          "client",
        ),
      );
    else if (response.ok) job.resolve(response.result);
    else job.reject(response.error);
    this.pump();
  }
  private invalidate(error: BrowserError): void {
    clearTimeout(this.timer);
    this.timer = undefined;
    this.worker?.terminate();
    this.worker = null;
    const active = this.active;
    this.active = null;
    const queued = this.queue;
    this.queue = [];
    this.plans.clear();
    this.disposal.clear();
    this.initialized = false;
    this.session = crypto.randomUUID();
    active?.reject(error);
    for (const job of queued) job.reject(error);
    this.dispatchEvent(new CustomEvent("invalidated", { detail: error }));
  }
  releaseFigure(figure: string): void {
    this.figures.delete(figure);
    this.dropQueued(
      (job) => job.request.figure_id === figure,
      failure("cancelled", "Figure was released.", "client"),
    );
    for (const [id, owner] of this.plans)
      if (owner === figure) {
        this.disposal.add(id);
        this.plans.delete(id);
      }
    if (this.figures.size === 0) {
      this.invalidate(failure("cancelled", "No figures remain.", "client"));
      return;
    }
    this.pump();
  }
  cancelAll(): void {
    this.invalidate(
      failure(
        "cancelled",
        "Shared worker cancelled; all live plans expired.",
        "client",
      ),
    );
  }
  restart(): void {
    this.invalidate(
      failure(
        "cancelled",
        "Worker restarted; rebuild plans from their input documents.",
        "client",
      ),
    );
    this.closed = false;
  }
  close(): void {
    this.closed = true;
    this.figures.clear();
    this.invalidate(
      failure("cancelled", "Page computation session closed.", "client"),
    );
  }
}
