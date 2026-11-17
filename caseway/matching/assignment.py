"""Step 3: assignment algorithms.

Real-time mode (one case at a time): greedy with a priority queue. Push every
eligible lawyer into a max-heap keyed by S(c,l); pop the best whose capacity is
not full. O(L log L) per case.

Batch mode (e.g. every 30 minutes): min-cost flow, so a strong lawyer is not
used up by the first case of the day when a harder case arrives later.
    source -> case (cap 1) -> lawyer (cap 1, cost = -S(c,l)) -> sink
The lawyer -> sink link is split into parallel unit edges with INCREASING cost
for the 1st, 2nd, 3rd new case today; that convex cost load-balances without a
hard rule. Solved here with successive shortest paths (Bellman-Ford / SPFA, so
negative edge costs are fine). Google OR-Tools SimpleMinCostFlow is a drop-in
replacement for large batches.
"""

from __future__ import annotations

import heapq
from collections import deque

from ..models import LawyerMatch

COST_SCALE = 10_000


def greedy_rank(matches: list[LawyerMatch], remaining_capacity: dict[str, int], top_n: int = 3) -> list[LawyerMatch]:
    """Pop lawyers from a max-heap by score, skipping anyone with no capacity left."""
    heap = [(-m.score, m.lawyer_id, m) for m in matches]
    heapq.heapify(heap)
    out = []
    while heap and len(out) < top_n:
        _, lid, m = heapq.heappop(heap)
        if remaining_capacity.get(lid, 1) > 0:
            out.append(m)
    return out


class MinCostFlow:
    """Successive-shortest-path min-cost max-flow with SPFA."""

    def __init__(self, n: int):
        self.n = n
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]  # edge: [to, cap, cost, rev_index]

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

    def solve(self, s: int, t: int) -> tuple[int, int]:
        flow = cost = 0
        while True:
            dist = [float("inf")] * self.n
            in_q = [False] * self.n
            prev: list[tuple[int, int] | None] = [None] * self.n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                in_q[u] = False
                for i, (v, cap, c, _) in enumerate(self.graph[u]):
                    if cap > 0 and dist[u] + c < dist[v]:
                        dist[v] = dist[u] + c
                        prev[v] = (u, i)
                        if not in_q[v]:
                            in_q[v] = True
                            q.append(v)
            if dist[t] == float("inf"):
                return flow, cost
            # bottleneck along the path
            f, v = float("inf"), t
            while v != s:
                u, i = prev[v]
                f = min(f, self.graph[u][i][1])
                v = u
            v = t
            while v != s:
                u, i = prev[v]
                e = self.graph[u][i]
                e[1] -= f
                self.graph[v][e[3]][1] += f
                v = u
            flow += f
            cost += f * dist[t]


def batch_assign(scores: dict[str, dict[str, float]], remaining_capacity: dict[str, int],
                 load_step: float = 0.05) -> dict[str, str | None]:
    """Assign many cases at once.

    scores: case id -> {lawyer id -> S(c,l)} for ELIGIBLE pairs only.
    remaining_capacity: lawyer id -> how many new cases they can still take.
    load_step: convex extra cost for the k-th new case of a lawyer: load_step * k^2.
    Returns case id -> lawyer id (or None if no feasible lawyer remained).
    """
    cases = list(scores)
    lawyers = sorted({l for per in scores.values() for l in per})
    src, sink = 0, 1 + len(cases) + len(lawyers)
    case_node = {c: 1 + i for i, c in enumerate(cases)}
    law_node = {l: 1 + len(cases) + j for j, l in enumerate(lawyers)}
    mcf = MinCostFlow(sink + 1)
    for c in cases:
        mcf.add_edge(src, case_node[c], 1, 0)
        for l, s in scores[c].items():
            mcf.add_edge(case_node[c], law_node[l], 1, -int(round(s * COST_SCALE)))
    for l in lawyers:
        for k in range(1, max(0, remaining_capacity.get(l, 1)) + 1):
            mcf.add_edge(law_node[l], sink, 1, int(round(load_step * k * k * COST_SCALE)))
    mcf.solve(src, sink)
    result: dict[str, str | None] = {c: None for c in cases}
    for c in cases:
        for v, cap, cost, _ in mcf.graph[case_node[c]]:
            if v in law_node.values() and cap == 0 and cost <= 0:
                result[c] = lawyers[v - 1 - len(cases)]
    return result
