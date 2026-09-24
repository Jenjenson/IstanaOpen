"""Three predeclared native arms. Preparation/validation never acquire a bridge."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(r'D:\triad\IstanaOpen-LearningReview')
OUT = Path(__file__).parent
PYTHON = ROOT / 'RL/BlueTeam/Python'
REGISTRY = ROOT / 'Saved/WarningTraining/models'
BASE = 61000000
COMMON = dict(episodes=200, batchSize=8, scenarioSeed=BASE,
              validationCases=8, checkpointInterval=100, headroomProbes=0,
              sensorCount=4, sensorIds=['thermal'], initialization='directional_balanced_8',
              entropyCoefficient=.01, explorationProbability=.2)
ARMS = [
    {'id': 'local-917', 'port': 8771, 'config': dict(COMMON, algorithm='local_ppo', seed=917,
        name='Local refinement pilot 2026-09-24 A917')},
    {'id': 'local-918', 'port': 8772, 'config': dict(COMMON, algorithm='local_ppo', seed=918,
        name='Local refinement pilot 2026-09-24 A918')},
    {'id': 'legacy-917', 'port': 8770, 'config': dict(COMMON, algorithm='ppo', seed=917,
        name='Legacy PPO pilot 2026-09-24 control917')},
]

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def event(kind, **values):
    row = dict(at=now(), event=kind, **values)
    with (OUT / 'driver-events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)

def prepare():
    path = OUT / 'protocol.json'
    protocol = dict(schema='istana.controlled_local_ppo_pilot.v1', declaredAt=now(),
        arms=ARMS, registry=str(REGISTRY), trainingSeeds=[BASE, BASE + 199],
        validationSeeds=list(range(BASE + 10000, BASE + 10008)),
        testSeeds=list(range(BASE + 20000, BASE + 20008)),
        objective='Mean per-drone early warning; undetected drones contribute zero.',
        selection='Use built-in fixed validation selection; retain every result and ties. Test only after selection is frozen.',
        comparison='Two action-seed repetitions of local PPO and one legacy PPO control on identical scenario panels.',
        compute='Each arm has 200 sampled training episodes and 25 updates. Local arms also replay the contractor on each training scenario. Milestone100 adds one validation panel beyond batch updates. Record actual native calls, steps and wall time.',
        expectedNativeCalls={'local-917': 641, 'local-918': 641, 'legacy-917': 441},
        limitations=['Pilot has eight final test cases, not definitive superiority.',
                    'Previous500-episode trials had25 updates; this pilot has200 episodes and25 updates. Control receives the same200 sampled training episodes.',
                    'Local PPO consumes more native calls because of paired contractor references.',
                    'All outcomes are retained. No layout, seed, or checkpoint is selected by final test outcomes.',
                    'No extra held-out cases are requested by this driver.'],
        requiredStart='Root explicitly reports focused tests pass before run mode is invoked.',
        ports='8771 local917;8772 local918;8770 legacy control. Never stop or replace Unreal processes.')
    if path.exists():
        existing = json.loads(path.read_text())
        assert existing['arms'] == ARMS, 'Existing protocol differs; never overwrite a started study.'
    else:
        with path.open('x', encoding='utf-8') as f:
            json.dump(protocol, f, indent=2)
    print(str(path), flush=True)

def imports():
    sys.path.insert(0, str(PYTHON))
    from triad_rl.training_workbench import TrainingManager, run_episode, scenario_panels
    from triad_rl.training_evidence import evaluation_summary, layout_key
    from triad_rl.trained_models import TrainedModelRegistry
    return TrainingManager, run_episode, scenario_panels, evaluation_summary, layout_key, TrainedModelRegistry

def checked_configurations():
    TrainingManager, _, scenario_panels, _, _, TrainedModelRegistry = imports()
    registry = TrainedModelRegistry(REGISTRY)
    configs = []
    for arm in ARMS:
        config = TrainingManager.validate(arm['config'])
        assert scenario_panels(config) == (BASE, list(range(BASE+10000, BASE+10008)),
                                          list(range(BASE+20000, BASE+20008)))
        registry.ensure_available(config['name'])
        configs.append(config)
    return configs

class ArmLedger:
    def __init__(self, arm, episode_runner, evaluation_summary):
        self.arm = arm; self.runner = episode_runner; self.evaluate = evaluation_summary
        self.root = OUT / 'arms' / arm['id']
        self.raw = self.root / 'native-episodes'; self.raw.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock(); self.manager = None
        self.started = 0; self.completed = 0; self.failed = 0
        self.roles = Counter(); self.steps = 0; self.native_wall = 0.

    def snapshot(self):
        with self.lock:
            return dict(started=self.started, completed=self.completed, failed=self.failed,
                        roles=dict(self.roles), nativeSteps=self.steps, nativeCallWallSeconds=self.native_wall)

    def __call__(self, client, policy, seed, **kwargs):
        fixed = type(policy).__name__ == '_FixedPlacementPolicy'
        phase = self.manager.status()['phase'] if self.manager else 'unknown'
        if BASE <= seed < BASE + 200:
            role = ('paired_contractor_training' if fixed and phase == 'training'
                    else 'training_scenario_replay' if fixed or kwargs.get('deterministic')
                    else 'sampled_training')
        elif BASE + 10000 <= seed < BASE + 10008:
            role = 'validation_contractor' if fixed else 'validation_policy'
        elif BASE + 20000 <= seed < BASE + 20008:
            role = 'final_test_contractor' if fixed else 'final_test_policy'
        else:
            raise AssertionError(f'Unplanned native scenario seed {seed}')
        with self.lock:
            self.started += 1; number = self.started
        started = time.monotonic(); stamp = now()
        try:
            run, records = self.runner(client, policy, seed, **kwargs)
            assert len(run['placements']) == 4
            assert all(p['profileId'] == 'thermal' for p in run['placements'])
            assert run['metrics']['targets'] == 5 and run['metrics']['cost'] == 4
            atomic(self.raw / f'native-{number:05}.json', run)
            case = self.evaluate([run])['cases'][0]
            elapsed = time.monotonic() - started
            row = dict(call=number, startedAt=stamp, completedAt=now(), phase=phase,
                       role=role, seed=seed, policyClass=type(policy).__name__,
                       nativeSteps=run['steps'], wallSeconds=elapsed, evidence=case)
            with self.lock:
                self.completed += 1; self.steps += run['steps']; self.native_wall += elapsed
                self.roles[role] += 1
                with (self.root / 'native-calls.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, allow_nan=False) + '\n')
                atomic(self.root / 'native-cost.json', self.snapshot())
            return run, records
        except Exception:
            with self.lock:
                self.failed += 1
            (self.root / f'native-failure-{number:05}.txt').write_text(traceback.format_exc(), encoding='utf-8')
            raise

def run_study(note):
    assert note.strip(), 'Record the root test-pass signal before acquiring ports.'
    prepare()
    configs = checked_configurations()
    TrainingManager, episode_runner, _, evaluation_summary, layout_key, TrainedModelRegistry = imports()
    registry = TrainedModelRegistry(REGISTRY)
    launch_path = OUT / 'launch.json'
    source_paths = [PYTHON / 'triad_rl' / name for name in (
        'training_workbench.py', 'training_evidence.py', 'warning_policy.py', 'warning_algorithms.py',
        'local_refinement_policy.py', 'red_policy.py', 'istana_live.py')]
    source_paths += list((PYTHON / 'triad_rl').glob('*local*ppo*.py'))
    sources = {}
    for path in source_paths:
        if path.exists():
            rel = path.relative_to(ROOT)
            saved = OUT / 'source-snapshot' / rel; saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(path.read_bytes()); sources[str(rel)] = hashlib.sha256(saved.read_bytes()).hexdigest()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    with launch_path.open('x', encoding='utf-8') as f:
        json.dump(dict(startedAt=now(), testsPassedNote=note, gitRevision=revision,
                       sourceSha256=sources, configurations=configs), f, indent=2)

    class AuditedManager(TrainingManager):
        def _set(self, **values):
            if values.get('phase') == 'failed':
                (self.output_root.parent / 'manager-failure.txt').write_text(
                    now() + '\n' + traceback.format_exc() + '\n' + str(values), encoding='utf-8')
            return super()._set(**values)

    ledgers = []; managers = []; started = time.monotonic(); checked = False; last_print = 0.
    for arm in ARMS:
        ledger = ArmLedger(arm, episode_runner, evaluation_summary)
        manager = AuditedManager(bridge_port=arm['port'], output_root=ledger.root / 'runs',
                                 episode_runner=ledger, registry=registry)
        ledger.manager = manager; ledgers.append(ledger); managers.append(manager)
    try:
        for manager, config, arm in zip(managers, configs, ARMS):
            manager.start(config); event('arm_started', arm=arm['id'], port=arm['port'])
        while True:
            statuses = [manager.status() for manager in managers]
            if not checked and all(s.get('baselineEvaluation') and s.get('initialEvaluation') for s in statuses):
                reference = statuses[0]['baselineEvaluation']
                reference_placements = None
                for status in statuses:
                    for field in ['baselineEvaluation', 'initialEvaluation']:
                        actual = status[field]
                        assert actual['seeds'] == reference['seeds']
                        for a, b in zip(actual['cases'], reference['cases']):
                            for key in ['seed', 'trajectorySha256', 'constraintsSha256', 'mean_drone_warning_s', 'detected_fraction']:
                                assert a[key] == b[key], f'Fair-start mismatch: {field} {key}'
                    directory = Path(status['outputDirectory'])
                    baseline = json.loads((directory / 'baseline-episode.json').read_text())['run']
                    initial = json.loads((directory / 'initial-episode.json').read_text())['run']
                    assert layout_key(baseline['placements']) == layout_key(initial['placements'])
                    if reference_placements is None:
                        reference_placements = layout_key(baseline['placements'])
                    assert layout_key(baseline['placements']) == reference_placements
                atomic(OUT / 'fair-start-verification.json', dict(verifiedAt=now(),
                    identicalContractorAndInitialLayouts=True, identicalMetrics=True,
                    identicalTrajectoriesAndConstraints=True, seeds=reference['seeds'],
                    baselineEvaluation=reference, placements=reference_placements))
                checked = True; event('fair_start_verified', meanWarningSeconds=reference['meanWarningSeconds'])
            compact = []
            for arm, status, ledger in zip(ARMS, statuses, ledgers):
                row = {k: status.get(k) for k in ['phase', 'running', 'episode', 'totalEpisodes',
                    'bestEpisode', 'bestValidationWarningSeconds', 'outputDirectory', 'error', 'exploration']}
                row.update(arm=arm['id'], nativeCost=ledger.snapshot())
                if not status['running'] and status['phase'] == 'complete':
                    row['testEvaluation'] = status.get('testEvaluation')
                    row['registeredModel'] = status.get('registeredModel')
                compact.append(row)
            atomic(OUT / 'status.json', dict(updatedAt=now(), elapsedSeconds=time.monotonic()-started,
                fairStartVerified=checked, arms=compact))
            if time.monotonic() - last_print >= 30:
                event('progress', arms=[{k:r[k] for k in ['arm','phase','episode','bestEpisode','nativeCost']} for r in compact])
                last_print = time.monotonic()
            if all(not s['running'] for s in statuses):
                break
            if (OUT / 'stop-request.json').exists():
                for manager in managers: manager.stop()
            time.sleep(5)
        success = checked and all(s['phase'] == 'complete' for s in statuses)
        for manager in managers:
            if manager.thread:
                manager.thread.join(timeout=5)
                assert not manager.thread.is_alive(), 'Completed manager has not released its client'
        atomic(OUT / 'completion.json', dict(completedAt=now(), complete=success,
            elapsedSeconds=time.monotonic()-started, fairStartVerified=checked, arms=compact,
            testsArePilotOnly=True, nativePortsReleased=True))
        event('study_finished', complete=success)
        if not success:
            raise RuntimeError('At least one arm did not complete; all evidence retained.')
    except BaseException:
        (OUT / 'driver-failure.txt').write_text(now() + '\n' + traceback.format_exc(), encoding='utf-8')
        for manager in managers: manager.stop()
        for manager in managers:
            if manager.thread: manager.thread.join(timeout=60)
        raise
    finally:
        for manager in managers: manager.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'validate', 'run', 'status'])
    parser.add_argument('--tests-passed-note', default='')
    args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'validate':
        prepare(); print(json.dumps(checked_configurations(), indent=2))
    elif args.mode == 'status':
        print((OUT / 'status.json').read_text() if (OUT / 'status.json').exists() else 'Not started')
    else:
        class Tee:
            def __init__(self, original, log):
                self.original = original; self.log = log; self.lock = threading.RLock()
            def write(self, value):
                with self.lock:
                    self.log.write(value); self.log.flush()
                    return self.original.write(value)
            def flush(self):
                with self.lock:
                    self.log.flush(); self.original.flush()
        original_out, original_err = sys.stdout, sys.stderr
        with (OUT / 'driver-stdout.log').open('a', encoding='utf-8') as stdout_log, \
             (OUT / 'driver-stderr.log').open('a', encoding='utf-8') as stderr_log:
            sys.stdout, sys.stderr = Tee(original_out, stdout_log), Tee(original_err, stderr_log)
            try:
                run_study(args.tests_passed_note)
            finally:
                sys.stdout, sys.stderr = original_out, original_err

if __name__ == '__main__':
    main()
