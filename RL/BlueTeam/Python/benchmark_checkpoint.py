"""Compatibility entry point for the dependency-free placement evaluator.

For a fair baseline comparison, run this command once with
``--initial --stochastic`` and once with ``--checkpoint PATH``, using the same
``--seed``, episode count, and Red script. Both runs write immutable parameter
hashes and aggregate metrics to ``evaluation.json``.
"""
from evaluate_checkpoint import main


if __name__ == "__main__":
    main()
