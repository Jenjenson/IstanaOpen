# Attempt 01: technical startup failure

The parallel launch on 2026-09-24 at approximately 06:11 UTC stopped before
training episode 1. All three consoles subsequently reported zero completed
training episodes and no active training. Driver process 2224 had exited before
the execution lock and generated attempt reports were archived here.

Trial B failed because two native managers allocated the same timestamp-based
output directory within one Windows clock tick. The driver requested safe stop
for its own Trials A and C; no trial completed or published a selected model.
This is a startup infrastructure failure, not a learning result.

The protocol, compact progress, final report, results, per-console status files
and original execution lock are retained in this directory. The native run
directories were neither moved nor deleted:

- `Saved\WarningTraining\console-20260924-061100-065086`
- `Saved\WarningTraining\console-20260924-061100-066086`

The next attempt uses the same scientific parameters, seed assignments and
trial names. Only startup allocation is serialized: each console must report
an existing output directory before the next console can start. A short guard
also separates coarse wall-clock ticks. After allocation, training continues
in parallel on the three isolated native backends.
