"""Run the predeclared100-case panel after every pilot arm finishes selection."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback

ROOT = Path(r'D:\triad\IstanaOpen-LearningReview')
OUT = Path(__file__).parent
FROZEN = OUT / 'frozen-runtime'
assert FROZEN.is_dir(), 'Create and verify the frozen pilot implementation before confirmation'
sys.path.insert(0, str(FROZEN))
from triad_rl.istana_live import IstanaLiveClient
from triad_rl.training_workbench import run_episode, _FixedPlacementPolicy
from triad_rl.training_evidence import evaluation_summary
from triad_rl.warning_algorithms import TRAINING_ALGORITHMS

def now(): return datetime.now(timezone.utc).isoformat()

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def statistics_summary(values):
    mean = statistics.mean(values); sd = statistics.stdev(values); se = sd / math.sqrt(len(values))
    return dict(cases=len(values), mean=mean, sd=sd, standardError=se,
                approximate95PercentInterval=[mean-1.96*se, mean+1.96*se],
                interpretation='Descriptive mean +/-1.96SE for this frozen policy; shared cases across arms.')

def main():
    protocol = json.loads((OUT / 'confirmation-protocol.json').read_text())
    pilot = json.loads((OUT / 'protocol.json').read_text())
    completion = json.loads((OUT / 'completion.json').read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert completion['complete'] and completion['fairStartVerified']
    assert protocol['seeds'] == list(range(63000000, 63000100))
    # Never silently change the loaded actor/evaluation implementation after training.
    for name, digest in launch['sourceSha256'].items():
        relative = Path(name).relative_to('RL/BlueTeam/Python')
        assert hashlib.sha256((FROZEN/relative).read_bytes()).hexdigest() == digest, f'Frozen pilot source hash mismatch: {name}'
    confirmation = OUT / 'confirmation'; confirmation.mkdir(exist_ok=True)
    frozen = []
    for arm, status in zip(pilot['arms'], completion['arms']):
        assert arm['id'] == status['arm'] and status['phase'] == 'complete'
        directory = Path(status['outputDirectory'])
        summary = json.loads((directory / 'summary.json').read_text())
        best_episode = summary['bestEpisode']
        selected = directory / ('policy-0000.json' if best_episode == 0 else f'best-policy-{best_episode:04d}.json')
        registered = ROOT / 'Saved/WarningTraining/models' / summary['modelId'] / 'best-policy.json'
        assert selected.read_bytes() == registered.read_bytes(), 'Selected run checkpoint differs from published policy'
        target = confirmation / arm['id']; target.mkdir(exist_ok=True)
        saved = target / 'frozen-selected-policy.json'
        with saved.open('xb') as f: f.write(selected.read_bytes())
        baseline = json.loads((directory / 'baseline-episode.json').read_text())['run']
        frozen.append(dict(arm=arm['id'], port=arm['port'], algorithm=arm['config']['algorithm'],
            policy=str(saved), policySha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
            selectedEpisode=summary['bestEpisode'], sourceDirectory=str(directory),
            baselinePlacements=baseline['placements'], context=baseline['context'],
            originalConstraintsSha256=evaluation_summary([baseline])['cases'][0]['constraintsSha256']))
    with (confirmation / 'frozen-selection.json').open('x', encoding='utf-8') as f:
        json.dump(dict(frozenAt=now(), beforeAnyConfirmationRun=True, policies=frozen), f, indent=2)
    statuses = {a['arm']:dict(phase='pending', cases=0, nativeCalls=0) for a in frozen}
    lock = threading.RLock(); start = time.monotonic()
    def update(arm, **values):
        with lock:
            statuses[arm].update(values)
            write(confirmation / 'status.json', dict(updatedAt=now(), elapsedSeconds=time.monotonic()-start, arms=statuses))
    def worker(arm):
        identity = arm['arm']; folder = confirmation / identity
        raw = folder / 'native-episodes'; raw.mkdir(exist_ok=True)
        policy = TRAINING_ALGORITHMS[arm['algorithm']]['factory'].load(arm['policy'], arm['context'])
        baseline_policy = _FixedPlacementPolicy(arm['baselinePlacements'])
        baseline_runs = []; policy_runs = []; native_calls = 0; steps = 0; wall = 0.
        begun = time.monotonic()
        try:
            with IstanaLiveClient(port=arm['port']) as client:
                update(identity, phase='running', port=arm['port'])
                for number, seed in enumerate(protocol['seeds'], 1):
                    pair = []
                    for label, actor in [('contractor', baseline_policy), ('policy', policy)]:
                        t = time.monotonic()
                        run, _ = run_episode(client, actor, seed, action_seed=seed+1000000, deterministic=True)
                        wall += time.monotonic()-t; native_calls += 1; steps += run['steps']
                        assert len(run['placements']) == 4 and all(p['profileId']=='thermal' for p in run['placements'])
                        assert run['metrics']['targets']==5 and run['metrics']['cost']==4
                        assert evaluation_summary([run])['cases'][0]['constraintsSha256'] == arm['originalConstraintsSha256']
                        write(raw / f'{number:03}-{label}-{seed}.json', run)
                        pair.append(run)
                    reference = evaluation_summary([pair[0]])
                    measured = evaluation_summary([pair[1]], reference)
                    baseline_runs.append(pair[0]); policy_runs.append(pair[1])
                    with (folder / 'paired-cases.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(seed=seed, contractor=reference['cases'][0],
                            policy=measured['cases'][0], warningDeltaSeconds=measured['deltaSeconds'],
                            detectionFractionDelta=pair[1]['metrics']['detected_fraction']-pair[0]['metrics']['detected_fraction']), allow_nan=False)+'\n')
                    update(identity, cases=number, nativeCalls=native_calls, nativeSteps=steps, nativeCallWallSeconds=wall)
                    if number%10==0:
                        print(json.dumps(dict(at=now(), arm=identity, completedCases=number, nativeCalls=native_calls)), flush=True)
            reference = evaluation_summary(baseline_runs)
            measured = evaluation_summary(policy_runs, reference)
            detection = [a['metrics']['detected_fraction']-b['metrics']['detected_fraction']
                         for a,b in zip(policy_runs,baseline_runs)]
            report = dict(arm=identity, complete=True, selectedEpisode=arm['selectedEpisode'],
                policySha256=arm['policySha256'], seeds=protocol['seeds'], contractor=reference,
                policy=measured, warningDelta=statistics_summary(measured['pairedDeltasSeconds']),
                detectionDelta=statistics_summary(detection), detectionFractionDeltas=detection,
                nativeCalls=native_calls, nativeSteps=steps, nativeCallWallSeconds=wall,
                elapsedSeconds=time.monotonic()-begun, identicalTrajectoriesAndConstraints=True,
                usedForSelection=False, portReleased=True)
            write(folder / 'report.json', report)
            update(identity, phase='complete', completedAt=now())
            return report
        except BaseException:
            (folder / 'failure.txt').write_text(now()+'\n'+traceback.format_exc(), encoding='utf-8')
            update(identity, phase='failed', error=traceback.format_exc())
            raise
    reports = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(worker, arm):arm['arm'] for arm in frozen}
        for future in as_completed(futures): reports.append(future.result())
    order = {a['id']:i for i,a in enumerate(pilot['arms'])}; reports.sort(key=lambda r:order[r['arm']])
    reference = reports[0]['contractor']
    for report in reports:
        assert report['seeds'] == protocol['seeds']
        for a,b in zip(report['contractor']['cases'], reference['cases']):
            for key in ['seed','trajectorySha256','constraintsSha256','mean_drone_warning_s','detected_fraction']:
                assert a[key]==b[key], f'Cross-arm contractor differs: {key}'
    final = dict(completedAt=now(), complete=True, totalNativeCalls=sum(r['nativeCalls'] for r in reports),
        sharedScenarioCount=100, crossArmContractorExactMatch=True, elapsedSeconds=time.monotonic()-start,
        allPortsReleased=True, reports=reports,
        interpretation='Three frozen policy outcomes on100 shared scenarios. Descriptive per-policy paired intervals; two local training seeds and one legacy control are not enough to establish algorithm-wide superiority.')
    write(confirmation / 'completion.json', final)
    print(json.dumps(dict(complete=True, nativeCalls=final['totalNativeCalls'], allPortsReleased=True)), flush=True)

if __name__ == '__main__':
    main()
