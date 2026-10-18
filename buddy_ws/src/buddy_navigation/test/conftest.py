"""Test configuration: make sibling source packages importable.

On a ROS machine the packages are normally installed by colcon; running the
tests straight from the source tree needs the sibling packages on the path.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]

for path in (
    SRC / "buddy_navigation",
    SRC / "buddy_core",
    SRC / "buddy_core" / "test",
):
    if str(path) not in sys.path:
        sys.path.append(str(path))
