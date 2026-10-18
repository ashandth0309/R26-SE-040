"""Dijkstra shortest-path planning on a 2D occupancy grid.

Only the mathematics of the earlier TurtleBot3 maze work is reused: a
uniform-cost search over grid cells.  Nothing here knows about wheels,
motors or any particular robot, so the planner can be tested on its own
and reused by any controller.

Cell coordinates are ``(column, row)``.  Costs are in cell units; multiply by
the grid resolution for metres.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum

from buddy_navigation.core.grid import Cell, PlanningGrid

SQRT2 = math.sqrt(2.0)

ORTHOGONAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))


class PlanStatus(str, Enum):
    """Outcome of a planning request."""

    OK = "ok"
    START_OUT_OF_BOUNDS = "start_out_of_bounds"
    GOAL_OUT_OF_BOUNDS = "goal_out_of_bounds"
    START_BLOCKED = "start_blocked"
    GOAL_BLOCKED = "goal_blocked"
    NO_PATH = "no_path"


@dataclass(frozen=True)
class PlannerConfig:
    """Planner options."""

    connectivity: int = 8
    allow_corner_cutting: bool = False

    def __post_init__(self) -> None:
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")


@dataclass(frozen=True)
class PlanResult:
    """Result of a planning request."""

    status: PlanStatus
    cells: tuple[Cell, ...] = field(default_factory=tuple)
    cost_cells: float = math.inf
    expanded: int = 0

    @property
    def success(self) -> bool:
        """Whether a path was found."""

        return self.status is PlanStatus.OK

    @property
    def reason(self) -> str:
        """Human readable explanation of the status."""

        return {
            PlanStatus.OK: "Path found.",
            PlanStatus.START_OUT_OF_BOUNDS: "Start is outside the map.",
            PlanStatus.GOAL_OUT_OF_BOUNDS: "Goal is outside the map.",
            PlanStatus.START_BLOCKED: "Start cell is blocked.",
            PlanStatus.GOAL_BLOCKED: "Goal cell is blocked.",
            PlanStatus.NO_PATH: "No path: goal is unreachable.",
        }[self.status]


def _neighbours(config: PlannerConfig):
    if config.connectivity == 4:
        return ORTHOGONAL

    return ORTHOGONAL + DIAGONAL


def dijkstra(
    grid: PlanningGrid,
    start: Cell,
    goal: Cell,
    config: PlannerConfig | None = None,
) -> PlanResult:
    """Find the lowest-cost path from ``start`` to ``goal``.

    Returns a failed ``PlanResult`` with a specific status instead of
    raising when the request is impossible, so callers can report why.
    """

    config = config or PlannerConfig()

    if not grid.in_bounds(start):
        return PlanResult(PlanStatus.START_OUT_OF_BOUNDS)

    if not grid.in_bounds(goal):
        return PlanResult(PlanStatus.GOAL_OUT_OF_BOUNDS)

    if not grid.is_free(start):
        return PlanResult(PlanStatus.START_BLOCKED)

    if not grid.is_free(goal):
        return PlanResult(PlanStatus.GOAL_BLOCKED)

    if start == goal:
        return PlanResult(PlanStatus.OK, (start,), 0.0, 0)

    blocked = grid.blocked.tolist()
    width, height = grid.width, grid.height
    moves = _neighbours(config)

    best = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    heap: list[tuple[float, int, Cell]] = [(0.0, 0, start)]
    counter = 0
    expanded = 0
    settled: set[Cell] = set()

    while heap:
        cost, _, cell = heapq.heappop(heap)

        if cell in settled:
            continue

        settled.add(cell)
        expanded += 1

        if cell == goal:
            path = [goal]

            while path[-1] != start:
                path.append(parent[path[-1]])

            path.reverse()

            return PlanResult(PlanStatus.OK, tuple(path), cost, expanded)

        column, row = cell

        for dc, dr in moves:
            nc, nr = column + dc, row + dr

            if not (0 <= nc < width and 0 <= nr < height):
                continue

            if blocked[nr][nc]:
                continue

            diagonal = dc != 0 and dr != 0

            if (
                diagonal
                and not config.allow_corner_cutting
                and (blocked[row][nc] or blocked[nr][column])
            ):
                continue

            neighbour = (nc, nr)

            if neighbour in settled:
                continue

            new_cost = cost + (SQRT2 if diagonal else 1.0)

            if new_cost < best.get(neighbour, math.inf):
                best[neighbour] = new_cost
                parent[neighbour] = cell
                counter += 1
                heapq.heappush(heap, (new_cost, counter, neighbour))

    return PlanResult(PlanStatus.NO_PATH, (), math.inf, expanded)


def simplify_path(cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    """Drop intermediate cells that lie on a straight line."""

    if len(cells) <= 2:
        return tuple(cells)

    kept = [cells[0]]

    for previous, current, following in zip(cells, cells[1:], cells[2:]):
        before = (current[0] - previous[0], current[1] - previous[1])
        after = (following[0] - current[0], following[1] - current[1])

        if before != after:
            kept.append(current)

    kept.append(cells[-1])

    return tuple(kept)


def path_length_m(cells: tuple[Cell, ...], resolution_m: float) -> float:
    """Euclidean length of a cell path in metres."""

    return resolution_m * sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(cells, cells[1:])
    )


def nearest_free_cell(
    grid: PlanningGrid,
    cell: Cell,
    max_radius: int = 5,
) -> Cell | None:
    """Return the closest free cell to ``cell`` within ``max_radius``."""

    if grid.is_free(cell):
        return cell

    best: Cell | None = None
    best_distance = math.inf

    for dr in range(-max_radius, max_radius + 1):
        for dc in range(-max_radius, max_radius + 1):
            candidate = (cell[0] + dc, cell[1] + dr)

            if not grid.is_free(candidate):
                continue

            distance = math.hypot(dc, dr)

            if distance < best_distance:
                best, best_distance = candidate, distance

    return best
