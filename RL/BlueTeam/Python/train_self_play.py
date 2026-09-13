"""Compatibility entry point for the supported focused Blue trainer.

Joint unfrozen self-play was intentionally replaced: Red is now a disclosed,
configurable script while the dynamic Blue sensor-placement policy trains.
Use ``train_blue_placement.py --help`` for the full interface.
"""

from train_blue_placement import main


if __name__ == "__main__":
    main()
