import { test, expect } from "@playwright/test";
import { setTimeout as delay } from "node:timers/promises";

test("exact worker artifact matches native plans, traces, every prefix and seeds", async ({
  page,
}) => {
  await page.goto("harness.html");
  await page.waitForFunction(() => !!window.rcswxHarness);
  const result = await page.evaluate(async () => {
    const { BrowserClient, records } = window.rcswxHarness;
    const client = new BrowserClient();
    const clean = (value: unknown): unknown => {
      if (Array.isArray(value)) return value.map(clean);
      if (value && typeof value === "object")
        return Object.fromEntries(
          Object.entries(value)
            .filter(([key]) => !["plan_id", "build"].includes(key))
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([key, value]) => [key, clean(value)]),
        );
      return value;
    };
    const equal = (actual: unknown, expected: unknown, label: string) => {
      if (JSON.stringify(clean(actual)) !== JSON.stringify(clean(expected)))
        throw new Error(`Native/browser mismatch: ${label}`);
    };
    let frames = 0;
    const measured = [];
    for (const [revision, pair] of records.pairs.entries()) {
      const route = { figure_id: "parity", revision };
      const start = performance.now();
      const plan = await client.request(
        "analyze",
        {
          parent1_json: pair.parent1_json,
          parent2_json: pair.parent2_json,
          options: pair.options,
        },
        route,
      );
      const analysis_ms = performance.now() - start;
      equal(plan, pair.analysis, `${pair.id} analysis`);
      const events = [];
      let offset: number | null = 0;
      while (offset !== null) {
        const trace = await client.request(
          "trace_page",
          { plan_id: plan.plan_id, offset, limit: 64 },
          route,
        );
        events.push(...trace.events);
        offset = trace.next_offset;
      }
      equal(events, pair.recording.events, `${pair.id} trace`);
      const applyStart = performance.now();
      for (let k = 0; k <= plan.nontrivial.length; k++) {
        equal(
          await client.request(
            "preview_step",
            { plan_id: plan.plan_id, k },
            route,
          ),
          pair.frames[k],
          `${pair.id} frame ${k}`,
        );
        frames++;
      }
      for (const sample of pair.samples)
        equal(
          await client.request(
            "sample",
            {
              plan_id: plan.plan_id,
              seed_hex: sample.seed_hex!,
              skewness: sample.skewness!,
            },
            route,
          ),
          sample,
          `${pair.id} sample ${sample.seed_hex}`,
        );
      measured.push({
        pair: pair.id,
        analysis_ms,
        applications_and_samples_ms: performance.now() - applyStart,
        trace_bytes: plan.trace.bytes,
      });
      await client.request("dispose", { plan_id: plan.plan_id }, route);
    }
    client.close();
    return { frames, measured };
  });
  expect(result.frames).toBe(10);
  await test.info().attach("browser-measurements", {
    body: JSON.stringify(result.measured, null, 2),
    contentType: "application/json",
  });
});

test("stale analyses are disposed and queued previews are coalesced", async ({
  page,
}) => {
  await page.goto("harness.html");
  await page.waitForFunction(() => !!window.rcswxHarness);
  const result = await page.evaluate(async () => {
    const { BrowserClient, records } = window.rcswxHarness;
    const client = new BrowserClient({ max_queued: 3 });
    const pair = records.pairs[0];
    const input = {
      parent1_json: pair.parent1_json,
      parent2_json: pair.parent2_json,
      options: pair.options,
    };
    await client.request("version", {}, { figure_id: "f", revision: 0 });
    const stale = [];
    for (let revision = 0; revision < 8; revision++) {
      const old = client
        .request("analyze", input, { figure_id: "f", revision })
        .catch((e) => e.code);
      client.setRevision("f", revision + 1);
      stale.push(await old);
      await client.request(
        "version",
        {},
        { figure_id: "f", revision: revision + 1 },
      );
    }
    const route = { figure_id: "f", revision: 9 };
    const plan = await client.request("analyze", input, route);
    const active = client.request("inspect", { plan_id: plan.plan_id }, route);
    const previews = Array.from({ length: 20 }, (_, i) =>
      client
        .request("preview_step", { plan_id: plan.plan_id, k: i % 3 }, route)
        .then(
          (value) => value.step,
          (error) => error.code,
        ),
    );
    const queued = client.state.queued;
    await active;
    const completed = await Promise.all(previews);
    const hold = client.request("inspect", { plan_id: plan.plan_id }, route);
    const overload = Array.from({ length: 10 }, () =>
      client.request("inspect", { plan_id: plan.plan_id }, route).then(
        () => "ok",
        (e) => e.code,
      ),
    );
    await hold;
    const overflow = await Promise.all(overload);
    const plans = client.state.plans;
    client.close();
    return { stale, queued, completed, overflow, plans };
  });
  expect(result.stale).toEqual(Array(8).fill("stale_result"));
  expect(result.queued).toBeLessThanOrEqual(3);
  expect(result.completed.slice(0, -1)).toEqual(Array(19).fill("superseded"));
  expect(result.completed.at(-1)).toBe(1);
  expect(result.overflow.filter((code) => code === "queue_limit").length).toBe(
    7,
  );
  expect(result.plans).toBe(1);
});

test("figure release preserves peers; cancellation invalidates the whole session", async ({
  page,
}) => {
  await page.goto("harness.html");
  await page.waitForFunction(() => !!window.rcswxHarness);
  const result = await page.evaluate(async () => {
    const { BrowserClient, records } = window.rcswxHarness;
    const client = new BrowserClient();
    const pair = records.pairs[0];
    const input = {
      parent1_json: pair.parent1_json,
      parent2_json: pair.parent2_json,
      options: pair.options,
    };
    await client.request("analyze", input, { figure_id: "a", revision: 0 });
    const b = await client.request("analyze", input, {
      figure_id: "b",
      revision: 0,
    });
    const original = client.state.session_id;
    client.releaseFigure("a");
    const inspected = await client.request(
      "inspect",
      { plan_id: b.plan_id },
      { figure_id: "b", revision: 0 },
    );
    const stillSameSession = original === client.state.session_id;
    const pending = [
      client.request(
        "preview_step",
        { plan_id: b.plan_id, k: 1 },
        { figure_id: "b", revision: 0 },
      ),
      client.request("version", {}, { figure_id: "c", revision: 0 }),
    ].map((p) =>
      p.then(
        () => "unexpected_success",
        (e) => e.code,
      ),
    );
    client.cancelAll();
    const errors = await Promise.all(pending);
    const expired = await client
      .request(
        "inspect",
        { plan_id: b.plan_id },
        { figure_id: "b", revision: 0 },
      )
      .catch((e) => e.code);
    client.restart();
    const rebuilt = await client.request("analyze", input, {
      figure_id: "b",
      revision: 1,
    });
    const newSession = original !== client.state.session_id;
    client.close();
    return {
      stillSameSession,
      distance: inspected.distance,
      errors,
      expired,
      rebuilt: rebuilt.distance,
      newSession,
    };
  });
  expect(result.stillSameSession).toBe(true);
  expect(result.errors).toEqual(["cancelled", "cancelled"]);
  expect(result.expired).toBe("expired_plan");
  expect(result.newSession).toBe(true);
  expect(result.rebuilt).toBe(result.distance);
});

test("initialization timeout rejects every waiter and permits a fresh worker", async ({
  page,
}) => {
  await page.route("**/*.wasm", async (route) => {
    await delay(150);
    await route.continue();
  });
  await page.goto("harness.html");
  await page.waitForFunction(() => !!window.rcswxHarness);
  const invalidLimit = await page.evaluate(async () => {
    let client;
    try {
      client = new window.rcswxHarness.BrowserClient({
        initialization_timeout_ms: 2 ** 31,
      });
      await client.request(
        "version",
        {},
        { figure_id: "overflow", revision: 0 },
      );
      return "unexpected_success";
    } catch (error) {
      if (
        error &&
        typeof error === "object" &&
        "code" in error &&
        typeof error.code === "string"
      )
        return error.code;
      throw error;
    } finally {
      client?.close();
    }
  });
  expect(invalidLimit).toBe("invalid_input");
  const errors = await page.evaluate(async () => {
    const client = new window.rcswxHarness.BrowserClient({
      initialization_timeout_ms: 20,
    });
    const calls = ["a", "b"].map((figure_id) =>
      client.request("version", {}, { figure_id, revision: 0 }).then(
        () => "unexpected_success",
        (e) => e.code,
      ),
    );
    const errors = await Promise.all(calls);
    client.close();
    return errors;
  });
  expect(errors).toEqual(["timeout", "timeout"]);
  await page.unroute("**/*.wasm");
  const api = await page.evaluate(async () => {
    const client = new window.rcswxHarness.BrowserClient();
    const version = await client.request(
      "version",
      {},
      { figure_id: "new", revision: 0 },
    );
    client.close();
    return version.api;
  });
  expect(api).toBe(1);
});

test("real worker startup failure rejects active and queued requests", async ({
  page,
}) => {
  await page.route("**/worker-*.js", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body:
        (await response.text()) +
        '\nthrow new Error("injected worker startup failure");',
    });
  });
  await page.goto("harness.html");
  await page.waitForFunction(() => !!window.rcswxHarness);
  const errors = await page.evaluate(async () => {
    const client = new window.rcswxHarness.BrowserClient();
    const calls = ["a", "b"].map((figure_id) =>
      client.request("version", {}, { figure_id, revision: 0 }).then(
        () => "unexpected_success",
        (e) => e.code,
      ),
    );
    const errors = await Promise.all(calls);
    client.close();
    return errors;
  });
  expect(errors).toEqual(["worker_crash", "worker_crash"]);
});

test("non-root deployment serves WASM correctly and UI keeps controls independent", async ({
  page,
}) => {
  await page.goto("./");
  await expect(page.getByTestId("step-label")).toHaveText("Step 0 of 2");
  await page
    .getByRole("button", { name: "Next offspring step", exact: true })
    .click();
  await expect(page.getByTestId("selection")).toContainText("[2]");
  const wasm = page.waitForResponse((response) =>
    response.url().endsWith(".wasm"),
  );
  await page.getByTestId("live").click();
  expect((await wasm).headers()["content-type"]).toContain("application/wasm");
  await expect(page.getByTestId("sample")).toBeEnabled();
  await page.getByTestId("sample").click();
  await expect(page.locator("#sample-status")).toContainText("Seed 42");
  await expect(page.getByTestId("step-label")).toHaveText("Step 1 of 2");
  await page.getByTestId("pair").selectOption("recursive");
  expect(await page.locator("#subproblem option").count()).toBeGreaterThan(1);
  await page.locator("#cursor").fill("0");
  await expect(page.getByTestId("matrix")).toContainText(
    "has not been entered",
  );
  await page.getByTestId("pair").selectOption("identical");
  await expect(page.getByTestId("step")).toBeDisabled();
  await expect(page.getByTestId("step-label")).toHaveText("Step 0 of 0");
});

for (const mode of ["missing", "csp"] as const)
  test(`${mode} WASM leaves exact cached playback available`, async ({
    page,
  }) => {
    if (mode === "missing")
      await page.route("**/*.wasm", (route) => route.abort());
    else
      await page.route("**/worker-*.js", async (route) => {
        const response = await route.fetch();
        await route.fulfill({
          response,
          headers: {
            ...response.headers(),
            "content-security-policy":
              "default-src 'self'; script-src 'self'; connect-src 'self'",
          },
        });
      });
    await page.goto("./");
    await page.getByTestId("live").click();
    await expect(page.getByTestId("status")).toContainText("unavailable");
    await page
      .getByRole("button", { name: "Next offspring step", exact: true })
      .click();
    await expect(page.getByTestId("step-label")).toHaveText("Step 1 of 2");
    await expect(page.getByTestId("selection")).toContainText(
      "Cached native result",
    );
    await expect(page.getByTestId("sample")).toBeDisabled();
  });

test("labels are inert text and seeds retain all 256 bits", async ({
  page,
}) => {
  await page.goto("harness.html");
  await page.waitForFunction(() => !!window.rcswxHarness);
  const result = await page.evaluate(() => {
    const { renderTree, decimalSeed } = window.rcswxHarness;
    renderTree(document.getElementById("surface")!, {
      snapshot_key: "parent1",
      root: 0,
      nodes: [
        {
          occurrence_index: 0,
          id: "1",
          name: "<img src=x onerror=alert(1)>",
          children: [],
        },
      ],
    });
    return {
      images: document.querySelectorAll("#surface img").length,
      max: decimalSeed(((1n << 256n) - 1n).toString()),
      large: decimalSeed("9007199254740993"),
      title: document.querySelector("#surface title")?.textContent,
    };
  });
  expect(result.images).toBe(0);
  expect(result.title).toContain("<img src=x onerror=alert(1)>");
  expect(result.max).toBe("ff".repeat(32));
  expect(result.large).toBe("0100000000002000" + "00".repeat(24));
});

test("worker transport preserves opaque numbers and occurrence origins across failures", async ({
  page,
}) => {
  await page.goto("harness.html");
  await page.waitForFunction(() => !!window.rcswxHarness);
  const result = await page.evaluate(async () => {
    const client = new window.rcswxHarness.BrowserClient();
    const route = { figure_id: "lossless", revision: 0 };
    const text =
      '{"schema":1,"grammar":"x","grammar_version":"1","root":1,"input_spec":{"n":999999999999999999999999999999999999,"fraction":1.000000000000000000000001},"nodes":[{"id":"9007199254740993","name":"relu","children":[],"parameters":{"n":9007199254740993123456789}},{"id":"9007199254740993","name":"computation","children":[0]}]}';
    const input = {
      parent1_json: text,
      parent2_json: text,
      options: {
        api: 1 as const,
        trace_level: "full" as const,
        trace_limits: { max_events: 1, max_bytes: 4096 },
      },
    };
    const schemaError = await client
      .request(
        "analyze",
        { ...input, parent1_json: text.replace('"schema":1', '"schema":2') },
        route,
      )
      .catch((error) => error.code);
    const plan = await client.request("analyze", input, route);
    const before = await client.request(
      "preview_step",
      { plan_id: plan.plan_id, k: 0 },
      route,
    );
    const indexErrors = [];
    for (const k of [-1, 0.5, Infinity, Number.MAX_SAFE_INTEGER])
      indexErrors.push(
        await client
          .request("preview_step", { plan_id: plan.plan_id, k }, route)
          .catch((error) => error.code),
      );
    const quotaError = await client
      .request(
        "analyze",
        {
          ...input,
          options: {
            ...input.options,
            limits: { max_work: 0 },
            trace_limits: { max_events: 0, max_bytes: 0 },
          },
        },
        route,
      )
      .catch((error) => error.code);
    const after = await client.request(
      "preview_step",
      { plan_id: plan.plan_id, k: 0 },
      route,
    );
    client.close();
    return {
      schemaError,
      indexErrors,
      quotaError,
      before,
      after,
      trace: plan.trace,
      ids: plan.parents.first.nodes.map((node) => node.id),
    };
  });
  expect(result.schemaError).toBe("invalid_input");
  expect(result.indexErrors).toEqual(Array(4).fill("invalid_index"));
  expect(result.quotaError).toBe("core_quota");
  expect(result.trace.complete).toBe(false);
  expect(result.before.architecture_json).toContain(
    "9007199254740993123456789",
  );
  expect(result.before.architecture_json).toContain(
    "999999999999999999999999999999999999",
  );
  expect(result.before.architecture_json).toContain(
    "1.000000000000000000000001",
  );
  expect(result.ids).toEqual(["9007199254740993", "9007199254740993"]);
  expect(result.before.origins.map((origin) => origin.sources)).toEqual([
    [{ parent: 2, occurrence_index: 1 }],
    [{ parent: 2, occurrence_index: 0 }],
  ]);
  expect(result.after).toEqual(result.before);
});
