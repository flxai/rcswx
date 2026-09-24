import type {
  Analysis,
  Applied,
  CellValue,
  DisplaySnapshot,
  HistoryValue,
  Recording,
  Subproblem,
  TokenView,
  TraceNumber,
} from "../../../crates/rcswx-wasm/browser";
const ns = "http://www.w3.org/2000/svg";
export function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text?: string,
) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}
function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
  text?: string,
) {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs))
    node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function highlight(keys: string[]) {
  for (const item of document.querySelectorAll<HTMLElement | SVGElement>(
    "[data-links]",
  ))
    item.classList.toggle(
      "linked",
      item.dataset.links!.split(" ").some((k) => keys.includes(k)),
    );
}
function link(
  node: HTMLElement | SVGElement,
  keys: string[],
  inspect?: () => void,
) {
  node.dataset.links = keys.join(" ");
  const enter = () => {
    highlight(keys);
    inspect?.();
  };
  node.addEventListener("pointerenter", enter);
  node.addEventListener("focus", enter);
  node.addEventListener("pointerleave", () => highlight([]));
  node.addEventListener("blur", () => highlight([]));
}
export function renderTree(
  container: HTMLElement,
  snapshot: DisplaySnapshot,
  origins?: Applied["origins"],
) {
  const positions = new Map<number, { x: number; y: number }>();
  const pending = [{ index: snapshot.root, depth: 0 }];
  let row = 0;
  while (pending.length && row < 256) {
    const current = pending.pop()!;
    positions.set(current.index, {
      x: 20 + current.depth * 24,
      y: 24 + row++ * 42,
    });
    for (const child of [...snapshot.nodes[current.index].children].reverse())
      pending.push({ index: child, depth: current.depth + 1 });
  }
  const width = Math.max(340, ...[...positions.values()].map((p) => p.x + 240));
  const drawing = svg("svg", {
    viewBox: `0 0 ${width} ${Math.max(68, row * 42 + 20)}`,
    role: "group",
    "aria-label": `${snapshot.snapshot_key} occurrence tree`,
    class: "tree",
  });
  for (const node of snapshot.nodes) {
    const p = positions.get(node.occurrence_index);
    if (!p) continue;
    for (const child of node.children) {
      const c = positions.get(child);
      if (c)
        drawing.append(
          svg("path", {
            d: `M ${p.x + 8} ${p.y + 14} V ${c.y} H ${c.x}`,
            class: "edge",
          }),
        );
    }
  }
  for (const node of snapshot.nodes) {
    const p = positions.get(node.occurrence_index);
    if (!p) continue;
    const origin = origins?.[node.occurrence_index];
    const keys = origin
      ? origin.sources.map(
          (source) => `parent${source.parent}:${source.occurrence_index}`,
        )
      : [`${snapshot.snapshot_key}:${node.occurrence_index}`];
    const label = `${node.name}; occurrence ${node.occurrence_index}; logical ID ${node.id}${origin?.synthesized ? "; synthesized from multiple source occurrences" : ""}`;
    const group = svg("g", {
      transform: `translate(${p.x},${p.y})`,
      tabindex: 0,
      role: "img",
      "aria-label": label,
      class: origin?.synthesized ? "node synthesized" : "node",
    });
    group.append(
      svg("rect", { x: 0, y: -15, width: 238, height: 30, rx: 5 }),
      svg(
        "text",
        { x: 9, y: 5 },
        `${node.occurrence_index} · ${node.name.length > 25 ? node.name.slice(0, 24) + "…" : node.name}`,
      ),
      svg("title", {}, label),
    );
    link(group, keys);
    drawing.append(group);
  }
  container.replaceChildren(drawing);
}
export function renderTokens(
  container: HTMLElement,
  tokens: TokenView[],
  parent: 1 | 2,
) {
  container.replaceChildren(
    ...tokens.map((token) => {
      const button = element("button", `${token.index}: ${token.name}`);
      button.className = "token";
      button.title = `Occurrence ${token.occurrence ?? "synthetic start"}; logical ID ${token.id}`;
      link(
        button,
        token.occurrence === null
          ? []
          : [`parent${parent}:${token.occurrence}`],
      );
      return button;
    }),
  );
}
export function formatNumber(value: TraceNumber | undefined) {
  if (value === undefined) return "·";
  if (typeof value === "number") return String(value);
  return {
    uncomputed: "·",
    nan: "NaN",
    positive_infinity: "+∞",
    negative_infinity: "−∞",
  }[value.state];
}
export function subproblems(recording: Recording) {
  return recording.events
    .filter((e) => e.kind === "subproblem")
    .map((e) => e.data);
}
interface Location {
  problem: number;
  i: number;
  j: number;
}
export function replay(
  recording: Recording,
  cursor: number,
  selectedPath: number,
) {
  const cells = new Map<number, CellValue>();
  const histories = new Map<number, HistoryValue>();
  const matrices = new Map<number, number[][]>();
  const definitions = new Map<number, Subproblem>();
  const locations = new Map<number, Location>();
  const clones = new Map<number, number>();
  let retained: number[] = [];
  for (const event of recording.events.slice(0, cursor)) {
    switch (event.kind) {
      case "subproblem":
        definitions.set(event.data.id, event.data);
        break;
      case "history":
        histories.set(event.data.id, event.data.value);
        break;
      case "cell":
        cells.set(event.data.id, event.data.value);
        break;
      case "history_clone":
        clones.set(event.data.target, event.data.source);
        break;
      case "matrix":
        if (["enter", "complete"].includes(event.data.phase))
          matrices.set(
            event.data.subproblem,
            event.data.cells.map((row) => [...row]),
          );
        break;
      case "computed": {
        const d = event.data;
        const matrix = matrices.get(d.subproblem);
        if (matrix?.[d.i]) matrix[d.i][d.j] = d.cell;
        for (const history of cells.get(d.cell)?.histories ?? [])
          locations.set(history, { problem: d.subproblem, i: d.i, j: d.j });
        break;
      }
      case "retained_histories":
        retained = event.data.histories;
        break;
    }
  }
  const path: Location[] = [];
  let tail: number | null = retained[selectedPath] ?? null;
  while (tail !== null) {
    let source = tail;
    const visited = new Set<number>();
    while (
      !locations.has(source) &&
      clones.has(source) &&
      !visited.has(source)
    ) {
      visited.add(source);
      source = clones.get(source)!;
    }
    const location = locations.get(source);
    if (location) path.push(location);
    tail = histories.get(tail)?.previous ?? null;
  }
  return { cells, histories, matrices, definitions, path };
}
export function renderMatrix(
  container: HTMLElement,
  details: HTMLElement,
  recording: Recording,
  analysis: Omit<Analysis, "plan_id">,
  cursor: number,
  problemId: number,
) {
  const state = replay(recording, cursor, analysis.path_index);
  const problem = state.definitions.get(problemId);
  const matrix = state.matrices.get(problemId);
  if (!problem || !matrix) {
    container.replaceChildren(
      element(
        "p",
        "This subproblem has not been entered at the current recording cursor.",
      ),
    );
    return;
  }
  const table = element("table");
  table.setAttribute(
    "aria-label",
    `Recorded matrix for subproblem ${problemId}`,
  );
  const caption = element(
    "caption",
    `Subproblem ${problemId}${problem.parent === null ? " (root)" : `, parent ${problem.parent}`} · rows: Parent 1; columns: Parent 2. · = uncomputed; outlined cells belong to the recovered history.`,
  );
  table.append(caption);
  const heading = element("tr");
  heading.append(element("th", "P1 / P2"));
  for (const index of problem.second_indices.slice(0, 20)) {
    const th = element(
      "th",
      `${index}: ${analysis.tokens.second[index]?.name ?? "?"}`,
    );
    th.scope = "col";
    heading.append(th);
  }
  table.append(heading);
  for (let i = 0; i < Math.min(problem.rows, 20); i++) {
    const row = element("tr");
    const index1 = problem.first_indices[i];
    const token1 = analysis.tokens.first[index1];
    const th = element("th", `${index1}: ${token1?.name ?? "?"}`);
    th.scope = "row";
    row.append(th);
    for (let j = 0; j < Math.min(problem.columns, 20); j++) {
      const id = matrix[i]?.[j];
      const value = state.cells.get(id);
      const token2 = analysis.tokens.second[problem.second_indices[j]];
      const cell = element("td");
      const button = element("button", formatNumber(value?.value));
      button.setAttribute(
        "aria-label",
        `Cell ${i}, ${j}; shared identity ${id}; cost ${formatNumber(value?.value)}`,
      );
      button.className = state.path.some(
        (p) => p.problem === problemId && p.i === i && p.j === j,
      )
        ? "path-cell"
        : "";
      const keys = [
        token1?.occurrence === null || token1?.occurrence === undefined
          ? ""
          : `parent1:${token1.occurrence}`,
        token2?.occurrence === null || token2?.occurrence === undefined
          ? ""
          : `parent2:${token2.occurrence}`,
      ].filter(Boolean);
      link(button, keys, () => {
        details.textContent = JSON.stringify(
          {
            subproblem: problemId,
            local: [i, j],
            prepared_tokens: [index1, problem.second_indices[j]],
            shared_cell: id,
            ...value,
            retained_histories: value?.histories.map((h) => ({
              id: h,
              ...state.histories.get(h),
            })),
          },
          null,
          2,
        );
      });
      button.addEventListener("click", () => button.focus());
      cell.append(button);
      row.append(cell);
    }
    table.append(row);
  }
  container.replaceChildren(table);
  if (problem.rows > 20 || problem.columns > 20)
    container.append(
      element(
        "p",
        "Rendering is bounded to the first 20 rows and columns; export contains the complete recording.",
      ),
    );
}
export function renderPath(
  container: HTMLElement,
  analysis: Omit<Analysis, "plan_id">,
  selected: number[],
) {
  container.replaceChildren(
    ...analysis.path.map((edit, index) => {
      const item = element("li");
      item.value = index;
      item.textContent = `[${index}] ${edit.op_type} · cost ${edit.value}${analysis.nontrivial.includes(index) ? ` · selectable stop ${analysis.nontrivial.indexOf(index) + 1}` : " · bookkeeping / zero-cost"}${selected.includes(index) ? " · selected" : ""}`;
      item.className = selected.includes(index) ? "selected-edit" : "";
      const detail = element("details");
      detail.append(
        element("summary", "Indices and grouped restrictions"),
        element("pre", JSON.stringify(edit, null, 2)),
      );
      item.append(detail);
      return item;
    }),
  );
}
