"""Legacy module entry point for focused NumPy Blue-placement training.

TRIAD's current v4 task trains only the dynamic Blue placement policy against
an explicit reproducible Red script. Keep ``python -m triad_rl.self_play`` as
a convenience alias without pulling in the retired Torch/Ray implementation.
"""
from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from train_blue_placement import main as placement_main

    placement_main(argv)


if __name__ == "__main__":
    main()
