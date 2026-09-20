# Portable saved-layout preview fixtures

Exact `seed` and `placements` fields from case 0 of the archived
`Saved/WarningTraining/native-20260917-pilot` evaluations. No new inference,
optimization, timing data or performance comparison. Native deployment still
validates these placements against the current scene's surface and budget rules.
These are saved outputs, not trained model weights. Original source hashes:

| Source | SHA-256 of full original file |
| --- | --- |
| evaluation-0000.json | b4254c534ee4b226015e530ad0eaa62e1d66b37d94a395845e56df3f329db458 |
| evaluation-0128.json | 82d5e414a656c1862935a64ca931aa677238f57b27af3e6eeb7c7e40da367312 |
| evaluation-greedy.json | 69fd8e00689c36ed8d48eacb768763c492d61599778efdae701866366dde025c |

The smaller fixture files intentionally have different hashes from the full
evaluation files. They retain only what the interface needs to render layouts.
