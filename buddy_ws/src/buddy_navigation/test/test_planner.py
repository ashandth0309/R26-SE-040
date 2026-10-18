"""Tests for the grid structures and the Dijkstra planner."""

import math

import numpy as np
import pytest

from buddy_navigation.core.grid import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid2D,
    PlanningGrid,
    inflate_mask,
)
from buddy_navigation.core.planner import (
    PlannerConfig,
    PlanStatus,
    dijkstra,
    nearest_free_cell,
    path_length_m,
    simplify_path,
)

# Simple 6x6 teaching grid (not Buddy's real map).  '#' is blocked.
# Row 5 is printed first so the picture matches the (column, row) axes.
SIX_BY_SIX = [
    "......",
    ".##...",
    ".#..#.",
    ".#.##.",
    "...#..",
    "S.....",
]


def grid_from_rows(rows):
    height = len(rows)
    width = len(rows[0])
    blocked = set()
    start = None

    for printed, text in enumerate(rows):
        row = height - 1 - printed

        for column, char in enumerate(text):
            if char == "#":
                blocked.add((column, row))
            elif char == "S":
                start = (column, row)

    return PlanningGrid.from_blocked_cells(width, height, blocked), start


def assert_valid_path(grid, path, connectivity=8):
    for cell in path:
        assert grid.is_free(cell)

    for a, b in zip(path, path[1:]):
        step = (abs(a[0] - b[0]), abs(a[1] - b[1]))
        assert max(step) == 1

        if connectivity == 4:
            assert sum(step) == 1


def test_straight_path_on_empty_grid():
    grid = PlanningGrid.from_blocked_cells(5, 5, [])
    result = dijkstra(grid, (0, 0), (4, 0))

    assert result.success
    assert result.cells == ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
    assert result.cost_cells == pytest.approx(4.0)


def test_diagonal_is_cheaper_with_eight_connectivity():
    grid = PlanningGrid.from_blocked_cells(5, 5, [])

    eight = dijkstra(grid, (0, 0), (4, 4), PlannerConfig(connectivity=8))
    four = dijkstra(grid, (0, 0), (4, 4), PlannerConfig(connectivity=4))

    assert eight.cost_cells == pytest.approx(4 * math.sqrt(2))
    assert four.cost_cells == pytest.approx(8.0)
    assert_valid_path(grid, four.cells, connectivity=4)


def test_start_equals_goal():
    grid = PlanningGrid.from_blocked_cells(3, 3, [])
    result = dijkstra(grid, (1, 1), (1, 1))

    assert result.success
    assert result.cells == ((1, 1),)
    assert result.cost_cells == 0.0


def test_path_avoids_blocked_cells():
    grid = PlanningGrid.from_blocked_cells(
        5, 5, [(2, 0), (2, 1), (2, 2), (2, 3)]
    )
    result = dijkstra(grid, (0, 0), (4, 0))

    assert result.success
    assert (2, 4) in result.cells
    assert_valid_path(grid, result.cells)


def test_unreachable_goal_is_reported_not_raised():
    wall = [(2, row) for row in range(5)]
    grid = PlanningGrid.from_blocked_cells(5, 5, wall)
    result = dijkstra(grid, (0, 0), (4, 4))

    assert result.status is PlanStatus.NO_PATH
    assert result.cells == ()
    assert math.isinf(result.cost_cells)
    assert not result.success


def test_blocked_and_out_of_bounds_requests():
    grid = PlanningGrid.from_blocked_cells(4, 4, [(1, 1), (2, 2)])

    assert dijkstra(grid, (1, 1), (3, 3)).status is PlanStatus.START_BLOCKED
    assert dijkstra(grid, (0, 0), (2, 2)).status is PlanStatus.GOAL_BLOCKED
    assert (
        dijkstra(grid, (-1, 0), (3, 3)).status
        is PlanStatus.START_OUT_OF_BOUNDS
    )
    assert (
        dijkstra(grid, (0, 0), (4, 0)).status
        is PlanStatus.GOAL_OUT_OF_BOUNDS
    )


def test_diagonal_does_not_cut_corners_by_default():
    # Two blocked cells touch only at a corner; squeezing between them
    # diagonally would clip both.
    grid = PlanningGrid.from_blocked_cells(2, 2, [(1, 0), (0, 1)])

    strict = dijkstra(grid, (0, 0), (1, 1))
    loose = dijkstra(
        grid, (0, 0), (1, 1), PlannerConfig(allow_corner_cutting=True)
    )

    assert strict.status is PlanStatus.NO_PATH
    assert loose.success


def test_six_by_six_scenario_finds_the_shortest_route():
    grid, start = grid_from_rows(SIX_BY_SIX)
    assert start == (0, 0)

    result = dijkstra(grid, start, (5, 5), PlannerConfig(connectivity=4))

    assert result.success
    assert_valid_path(grid, result.cells, connectivity=4)
    assert result.cost_cells == pytest.approx(10.0)

    # Dijkstra optimality: no 4-connected path can beat Manhattan distance.
    assert result.cost_cells >= abs(5 - 0) + abs(5 - 0)


def test_dijkstra_matches_brute_force_on_random_grids():
    rng = np.random.default_rng(7)

    for _ in range(30):
        blocked_mask = rng.random((7, 7)) < 0.25
        blocked_mask[0, 0] = False
        blocked_mask[6, 6] = False
        grid = PlanningGrid(blocked_mask, 1.0)
        cfg = PlannerConfig(connectivity=4)
        result = dijkstra(grid, (0, 0), (6, 6), cfg)

        # Breadth-first search gives the optimal length on unit costs.
        frontier = [(0, 0)]
        seen = {(0, 0): 0}

        while frontier:
            nxt = []

            for c, r in frontier:
                for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cell = (c + dc, r + dr)

                    if cell in seen or not grid.is_free(cell):
                        continue

                    seen[cell] = seen[(c, r)] + 1
                    nxt.append(cell)

            frontier = nxt

        if (6, 6) in seen:
            assert result.success
            assert result.cost_cells == pytest.approx(seen[(6, 6)])
        else:
            assert result.status is PlanStatus.NO_PATH


def test_inflation_grows_obstacles_by_radius():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True

    grown = inflate_mask(mask, 2.0)

    assert grown[4, 4] and grown[4, 6] and grown[6, 4]
    assert not grown[4, 7]
    assert not grown[6, 6]  # corner at distance 2.83 is outside radius 2
    assert inflate_mask(mask, 0.0).sum() == 1


def test_inflation_closes_a_gap_that_is_too_narrow():
    grid = OccupancyGrid2D.empty(7, 7, 0.1, fill=FREE)
    grid.data[:, 3] = OCCUPIED
    grid.data[3, 3] = FREE  # one-cell (10 cm) doorway

    open_plan = PlanningGrid.from_occupancy(grid, inflation_radius_m=0.0)
    inflated = PlanningGrid.from_occupancy(grid, inflation_radius_m=0.15)

    assert dijkstra(open_plan, (0, 3), (6, 3)).success
    assert dijkstra(inflated, (0, 3), (6, 3)).status in (
        PlanStatus.START_BLOCKED,
        PlanStatus.NO_PATH,
    )


def test_unknown_space_is_blocked_by_default():
    grid = OccupancyGrid2D.empty(4, 4, 0.1, fill=UNKNOWN)
    grid.data[0, :] = FREE

    strict = PlanningGrid.from_occupancy(grid)
    permissive = PlanningGrid.from_occupancy(grid, unknown_is_blocked=False)

    assert not strict.is_free((1, 2))
    assert permissive.is_free((1, 2))
    assert dijkstra(strict, (0, 0), (3, 3)).status is PlanStatus.GOAL_BLOCKED


def test_world_cell_round_trip():
    grid = PlanningGrid.from_blocked_cells(
        4, 4, [], resolution_m=0.5, origin_x=-1.0, origin_y=2.0
    )
    cell = grid.world_to_cell(-0.6, 2.7)

    assert cell == (0, 1)
    assert grid.cell_to_world(cell) == pytest.approx((-0.75, 2.75))


def test_simplify_and_length_helpers():
    cells = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))

    assert simplify_path(cells) == ((0, 0), (2, 0), (2, 2))
    assert path_length_m(cells, 0.5) == pytest.approx(2.0)


def test_nearest_free_cell():
    grid = PlanningGrid.from_blocked_cells(5, 5, [(2, 2)])

    assert nearest_free_cell(grid, (0, 0)) == (0, 0)
    assert nearest_free_cell(grid, (2, 2)) in {(1, 2), (3, 2), (2, 1), (2, 3)}

    full = PlanningGrid(np.ones((3, 3), dtype=bool), 1.0)

    assert nearest_free_cell(full, (1, 1)) is None


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        PlannerConfig(connectivity=6)

    with pytest.raises(ValueError):
        PlanningGrid.from_blocked_cells(2, 2, [(5, 5)])

    with pytest.raises(ValueError):
        OccupancyGrid2D(np.zeros(4, dtype=np.int8), 0.1)
