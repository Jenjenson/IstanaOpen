# Recorded Blue placement results

These artifacts were produced in the original TRIAD Unreal host on
**10 September 2026**. They are published here for inspection; copying them
into this repository does not constitute a fresh Istana Open training run.

## Experiment

The [configuration](DefaultTrainingConfig.json) defines one scripted drone
approaching from the north at 25 m altitude, a maximum of two sensor sites, a
2.4-unit budget, a 150 m placement radius, 30 m objective standoff, and 20 m
minimum site separation. The fixed step is 0.5 s with a 60-step horizon.
Five selectable profiles cover passive RF, radar, EO, thermal, and combined
radar/thermal sensing.

Blue learns profile selection and continuous placement. Red is scripted.
`SustainedTrackDefence` is an abstract success after sustained observed
tracking; it does not simulate physical interception. Sensors use analytical
surrogates rather than calibrated hardware.

## Evaluation

Each report contains 50 episodes using seeds `1500000000` through
`1500000049`. The scenario remains fixed across these seeds.

| Policy | Abstract defence | Detection | Mean Blue return | Invalid actions |
| --- | ---: | ---: | ---: | ---: |
| [Initialized, stochastic](Results/initial-stochastic.json) | 28/50 (56%) | 68% | 1.652 | 0 |
| [Trained, stochastic](Results/trained-stochastic.json) | 36/50 (72%) | 82% | 5.064 | 0 |
| [Trained, deterministic](Results/trained-deterministic.json) | 50/50 (100%) | 100% | 10.850 | 0 |

The sampled trained policy improves defence by 16 percentage points and
detection by 14 points in this comparison. Deterministic deployment chooses
passive RF plus search radar, with first detection at 0.5 s. Its 50 successes
are repetitions of the same threat scenario, not evidence of generalization
across 50 different threat cases. This is one run and a small evaluation, not
a multi-seed learning study.

The [native oracle](Results/native-oracle.json) separately exercises
commit-only, all five sensor profiles, north/south placement, and deterministic
repeatability. It establishes expected simulator behavior independently of
the learned policy.

## Training and checkpoint

[training.jsonl](Results/training.jsonl) records all 210 completed live
training episodes, with zero invalid actions. Abstract defence rises from
52% in episodes 1-50 to 78% in episodes 161-210; detection rises from 66% to
80% over the same windows.

The final [checkpoint metadata](Checkpoints/toy-210/checkpoint.json) records
the model, feature contract, optimizer, random-generator state, and training
settings. The policy uses a shared scorer with 32 hidden units and a
conditional tanh-Gaussian position head. The final continuation records batches
of five, learning rate `0.0003`, entropy coefficient `0.01`, and master seed
`7301`.

```text
Parameter SHA-256:
1a7d10dbab564cbea654d46f0dbea684a68d62f76453ccade94210545fa17de7

Configuration fingerprint:
md5:c7b1fcb83150ed7e5b4b47de6923142f
```

The checkpoint's `details.metrics` covers only the final 10-episode resume
segment. Use the full JSONL history for cumulative training statistics. The
MD5 value detects accidental configuration mismatch; it is not a security
guarantee or a fingerprint of the entire simulator. The running-return count
mixes earlier episode returns with later per-decision returns and is not an
episode count. Position-radius metrics also changed to actual native placements
at episode 201, so do not compare that field across the transition.

## Historical validation and limits

On 10 September, the original TRIAD project passed its editor-module build,
five native RL automation tests, 40 Python policy/environment and native
contract tests, checkpoint reload, and bytecode compilation. These counts are
historical checks of that host; this source package does not contain the
complete plugin or certify a native build inside Istana Open.

The result establishes learning of this fixed placement problem. It does not
yet establish adaptation to different threat types, relocation during an
episode, simultaneous Red/Blue learning, real sensor performance, or operation
in the Istana scene. The next study needs varied approach directions,
altitudes, emitter behavior, movement, and swarm size, with separate training
and evaluation curricula. See [integration requirements](INTEGRATION.md).
