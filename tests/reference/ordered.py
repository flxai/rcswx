"""Tiny uncompressed history oracle; deliberately independent of production code."""


def transition_cost(left, right):
    return 0 if left == right else (1 if left[0] == right[0] else 2)


def optimum(left, right):
    if not left or not right or max(len(left), len(right)) > 6:
        raise ValueError("reference domain is nonempty sequences of at most six operations")
    best = None

    def visit(i, j, cost, history):
        nonlocal best
        if i == len(left) and j == len(right):
            candidate = cost, history
            if best is None or candidate < best:
                best = candidate
            return
        if i < len(left) and j < len(right):
            charge = transition_cost(left[i], right[j])
            visit(i + 1, j + 1, cost + charge, history + ((0 if charge == 0 else 1, i, j),))
        if i < len(left):
            visit(i + 1, j, cost + 4, history + ((2, i, -1),))
        if j < len(right):
            visit(i, j + 1, cost + 4, history + ((3, -1, j),))

    visit(0, 0, 0, ())
    return best


def projection(left, right, witness, mask):
    output, origins, cost, edit = [], [], 0, 0
    for tag, source, target in witness:
        selected = tag != 0 and bool(mask & (1 << edit))
        if tag:
            edit += 1
            if selected:
                cost += transition_cost(left[source], right[target]) if tag == 1 else 4
        if tag == 0 or tag in (1, 2) and not selected:
            output.append(left[source])
            origins.append((0, source))
        elif tag in (1, 3) and selected:
            output.append(right[target])
            origins.append((1, target))
    if mask >= 1 << edit:
        raise ValueError("unknown edit bit")
    return tuple(output), tuple(origins), cost
