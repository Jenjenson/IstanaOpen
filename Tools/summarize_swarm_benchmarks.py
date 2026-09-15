"""Summarize raw same-machine solver benchmarks without pooling unlike percentile samples."""
import csv
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parents[1]
for name in ("swarm-reference.csv", "swarm-level.csv"):
    path = root / "Saved" / "Benchmarks" / name
    if not path.exists():
        continue
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    print(f"\n{name}: {len(rows)} cases")
    if name == "swarm-level.csv":
        print("drones | seeds | median initial command old/new (s) | median per-case p95 old/new (ms) | worst new step (ms)")
        for count in sorted({int(r["drones"]) for r in rows}):
            group = [r for r in rows if int(r["drones"]) == count]
            med = lambda key: median(float(r[key]) for r in group)
            print(f'{count} | {len(group)} | {med("command_reference_ms")/1000:.3f}/{med("command_new_ms")/1000:.3f} | '
                  f'{med("old_p95_ms"):.3f}/{med("new_p95_ms"):.3f} | {max(float(r["new_max_ms"]) for r in group):.3f}')
    else:
        print("drones | cases | median per-case p50 old/new (ms)")
        for count in sorted({int(r["drones"]) for r in rows}):
            group = [r for r in rows if int(r["drones"]) == count]
            print(f'{count} | {len(group)} | {median(float(r["reference_p50_ms"]) for r in group):.4f}/'
                  f'{median(float(r["optimized_p50_ms"]) for r in group):.4f}')
