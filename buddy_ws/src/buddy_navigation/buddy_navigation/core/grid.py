"""Occupancy grid containers and obstacle inflation.

Conventions match ``nav_msgs/OccupancyGrid``: row-major, cell (0, 0) is the
lower-left cell, ``origin`` is the world position of that cell's lower-left
corner, +x is the column direction and +y is the row direction.  Cell values
are ``-1`` for unknown and ``0..100`` for occupancy probability in percent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

Cell = tuple[int, int]


@dataclass
class OccupancyGrid2D:
    """Probabilistic grid in the ``nav_msgs/OccupancyGrid`` convention."""

    data: np.ndarray
    resolution_m: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    frame_id: str = "map"

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("Grid data must be two dimensional.")

        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive.")

    @classmethod
    def empty(
        cls,
        width: int,
        height: int,
        resolution_m: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        frame_id: str = "map",
        fill: int = UNKNOWN,
    ) -> "OccupancyGrid2D":
        """Create a grid filled with a single value."""

        return cls(
            np.full((height, width), fill, dtype=np.int8),
            resolution_m,
            origin_x,
            origin_y,
            frame_id,
        )

    @property
    def width(self) -> int:
        """Number of columns."""

        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        """Number of rows."""

        return int(self.data.shape[0])

    def in_bounds(self, cell: Cell) -> bool:
        """Return whether a (column, row) cell is inside the grid."""

        column, row = cell

        return 0 <= column < self.width and 0 <= row < self.height

    def world_to_cell(self, x: float, y: float) -> Cell:
        """Return the (column, row) cell containing a world point."""

        return (
            int(math.floor((x - self.origin_x) / self.resolution_m)),
            int(math.floor((y - self.origin_y) / self.resolution_m)),
        )

    def cell_to_world(self, cell: Cell) -> tuple[float, float]:
        """Return the world position of a cell centre."""

        column, row = cell

        return (
            self.origin_x + (column + 0.5) * self.resolution_m,
            self.origin_y + (row + 0.5) * self.resolution_m,
        )


def disc_offsets(radius_cells: float) -> list[tuple[int, int]]:
    """Cell offsets inside a disc of the given radius (in cells)."""

    reach = int(math.ceil(radius_cells))

    return [
        (dc, dr)
        for dr in range(-reach, reach + 1)
        for dc in range(-reach, reach + 1)
        if math.hypot(dc, dr) <= radius_cells + 1e-9
    ]


def inflate_mask(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Grow ``True`` cells of a boolean mask by a circular radius."""

    if radius_cells <= 0:
        return mask.copy()

    height, width = mask.shape
    result = mask.copy()

    for dc, dr in disc_offsets(radius_cells):
        if dc == 0 and dr == 0:
            continue

        src_rows = slice(max(0, -dr), min(height, height - dr))
        src_cols = slice(max(0, -dc), min(width, width - dc))
        dst_rows = slice(max(0, dr), min(height, height + dr))
        dst_cols = slice(max(0, dc), min(width, width + dc))
        result[dst_rows, dst_cols] |= mask[src_rows, src_cols]

    return result


@dataclass
class PlanningGrid:
    """Boolean blocked-cell grid used by the planner.

    ``blocked`` already includes obstacle inflation and the unknown-space
    policy.  ``raw_obstacles`` and ``unknown`` are kept for visualisation
    and diagnostics.
    """

    blocked: np.ndarray
    resolution_m: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    raw_obstacles: np.ndarray | None = None
    unknown: np.ndarray | None = field(default=None)

    @property
    def width(self) -> int:
        """Number of columns."""

        return int(self.blocked.shape[1])

    @property
    def height(self) -> int:
        """Number of rows."""

        return int(self.blocked.shape[0])

    def in_bounds(self, cell: Cell) -> bool:
        """Return whether a (column, row) cell is inside the grid."""

        column, row = cell

        return 0 <= column < self.width and 0 <= row < self.height

    def is_free(self, cell: Cell) -> bool:
        """Return whether a cell is inside the grid and not blocked."""

        return self.in_bounds(cell) and not bool(
            self.blocked[cell[1], cell[0]]
        )

    def world_to_cell(self, x: float, y: float) -> Cell:
        """Return the (column, row) cell containing a world point."""

        return (
            int(math.floor((x - self.origin_x) / self.resolution_m)),
            int(math.floor((y - self.origin_y) / self.resolution_m)),
        )

    def cell_to_world(self, cell: Cell) -> tuple[float, float]:
        """Return the world position of a cell centre."""

        column, row = cell

        return (
            self.origin_x + (column + 0.5) * self.resolution_m,
            self.origin_y + (row + 0.5) * self.resolution_m,
        )

    @classmethod
    def from_blocked_cells(
        cls,
        width: int,
        height: int,
        blocked_cells: list[Cell] | set[Cell],
        resolution_m: float = 1.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ) -> "PlanningGrid":
        """Build a grid from an explicit list of blocked (column, row)."""

        blocked = np.zeros((height, width), dtype=bool)

        for column, row in blocked_cells:
            if not (0 <= column < width and 0 <= row < height):
                raise ValueError(
                    f"Blocked cell {(column, row)} is outside the grid."
                )

            blocked[row, column] = True

        return cls(
            blocked=blocked,
            resolution_m=resolution_m,
            origin_x=origin_x,
            origin_y=origin_y,
            raw_obstacles=blocked.copy(),
        )

    @classmethod
    def from_occupancy(
        cls,
        grid: OccupancyGrid2D,
        *,
        inflation_radius_m: float = 0.0,
        occupied_threshold: int = 55,
        unknown_is_blocked: bool = True,
    ) -> "PlanningGrid":
        """Derive a planning grid from an occupancy grid.

        Cells at or above ``occupied_threshold`` percent are obstacles.
        Unknown cells are blocked unless ``unknown_is_blocked`` is False.
        Obstacles (not unknown space) are then inflated by the robot
        radius so the planner can treat the robot as a point.
        """

        unknown = grid.data == UNKNOWN
        obstacles = grid.data >= occupied_threshold

        inflated = inflate_mask(
            obstacles,
            inflation_radius_m / grid.resolution_m,
        )

        blocked = inflated | unknown if unknown_is_blocked else inflated

        return cls(
            blocked=blocked,
            resolution_m=grid.resolution_m,
            origin_x=grid.origin_x,
            origin_y=grid.origin_y,
            raw_obstacles=obstacles,
            unknown=unknown,
        )

    def with_inflation(self, radius_m: float) -> "PlanningGrid":
        """Return a copy whose blocked cells are inflated."""

        return PlanningGrid(
            blocked=inflate_mask(self.blocked, radius_m / self.resolution_m),
            resolution_m=self.resolution_m,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            raw_obstacles=self.raw_obstacles,
            unknown=self.unknown,
        )
