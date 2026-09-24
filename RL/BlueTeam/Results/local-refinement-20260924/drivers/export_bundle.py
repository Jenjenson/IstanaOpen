"""Portable evidence and weights; intentionally excludes raw replay frames."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import numpy as np

ROOT = Path(r'D:\triad\IstanaOpen-LearningReview')
SOURCE = Path(__file__).parent
TARGET = ROOT / 'RL/BlueTeam/Results/local-refinement-20260924'

def read(path): return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def copy(source, destination, compressed=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    destination.write_bytes(gzip.compress(data, mtime=0) if compressed else data)

def main():
    pilot = read(SOURCE / 'completion.json')
    confirmation = read(SOURCE / 'confirmation/completion.json')
    secondary = read(SOURCE / 'secondary/completion.json')
    protocol = read(SOURCE / 'protocol.json')
    assert pilot['complete'] and pilot['fairStartVerified'] and confirmation['complete']
    assert confirmation['totalNativeCalls'] == 600 and confirmation['sharedScenarioCount'] == 100
    assert secondary['complete'] and secondary['totalNativeCalls']==300
    if TARGET.exists(): raise FileExistsError('Do not overwrite an existing evidence bundle')
    TARGET.mkdir(parents=True)
    (TARGET/'.gitattributes').write_text('# Preserve exact archived bytes and hashes on every OS.\n* -text\n',encoding='utf-8')
    copy(SOURCE/'archive.py',TARGET/'archive.py')
    for name in ['protocol.json', 'confirmation-protocol.json', 'launch.json',
                 'fair-start-verification.json', 'source-provenance.json', 'postlaunch-source-delta.patch',
                 'learning-diagnostics.json','secondary-protocol.json','constraint-audit.json','constraint-audit-pairs.jsonl.gz']:
        copy(SOURCE / name, TARGET / name)
    for name in ['pilot.py', 'confirm.py', 'secondary.py', 'audit_constraints.py', 'export_bundle.py']:
        copy(SOURCE / name, TARGET / 'drivers' / name)
    # These Python sources are small and preserve the exact core implementation
    # used by the pilot and confirmation, with dependency provenance recorded.
    for file in (SOURCE / 'frozen-runtime').rglob('*'):
        if file.is_file() and file.suffix in ('.py','.json') and '__pycache__' not in file.parts:
            copy(file, TARGET / 'source-snapshot' / file.relative_to(SOURCE / 'frozen-runtime'))
    write(TARGET / 'pilot-completion.json', pilot)
    write(TARGET / 'confirmation-completion.json', confirmation)
    write(TARGET / 'secondary-completion.json', secondary)
    copy(SOURCE / 'confirmation/frozen-selection.json', TARGET / 'frozen-selection.json')
    rows = []
    for arm, status, final in zip(protocol['arms'], pilot['arms'], confirmation['reports']):
        assert arm['id'] == status['arm'] == final['arm']
        identity = arm['id']; destination = TARGET / 'arms' / identity
        run = Path(status['outputDirectory']); summary = read(run / 'summary.json')
        evaluation = read(run / 'evaluation-summary.json'); test = read(run / 'test-evaluation.json')
        baseline = read(run / 'baseline-episode.json')['run']
        for name in ['configuration.json','summary.json','evaluation-summary.json','test-evaluation.json','recording-manifest.json']:
            copy(run / name, destination / name)
        selected_name='policy-0000.json' if summary['bestEpisode']==0 else f"best-policy-{summary['bestEpisode']:04d}.json"
        copy(run / selected_name, destination / 'selected-policy.json')
        copy(run / 'final-policy.json', destination / 'final-policy.json')
        copy(run / 'policy-0000.json', destination / 'initial-policy.json')
        write(destination / 'baseline-public-context.json', baseline['context'])
        write(destination / 'contractor-placements.json', baseline['placements'])
        selected_layout=read(run/'best-episode.json')['run']['placements']
        final_layout=read(run/'checkpoint-000200-episode.json')['run']['placements']
        write(destination/'selected-layout.json',selected_layout)
        write(destination/'final-layout.json',final_layout)
        for name in ['training.jsonl','evaluations.jsonl']:
            copy(run / name, destination / (name+'.gz'), compressed=True)
        for name in ['native-cost.json']:
            copy(SOURCE / 'arms' / identity / name, destination / name)
        copy(SOURCE / 'arms' / identity / 'native-calls.jsonl', destination / 'native-calls.jsonl.gz', compressed=True)
        copy(SOURCE / 'confirmation' / identity / 'report.json', destination / 'confirmation.json')
        copy(SOURCE / 'confirmation' / identity / 'paired-cases.jsonl', destination / 'confirmation-paired-cases.jsonl.gz', compressed=True)
        secondary_arm=next(r for r in secondary['reports'] if r['arm']==identity)
        copy(SOURCE/'secondary'/identity/'report.json',destination/'secondary-final-confirmation.json')
        copy(SOURCE/'secondary'/identity/'paired-cases.jsonl',destination/'secondary-final-paired-cases.jsonl.gz',compressed=True)
        assert hashlib.sha256((destination/'final-policy.json').read_bytes()).hexdigest()==secondary_arm['finalPolicySha256']
        digest = hashlib.sha256((destination/'selected-policy.json').read_bytes()).hexdigest()
        assert digest == final['policySha256']
        history = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
        assert len(history) == 200
        updates = [r['update'] for r in history if 'update' in r]
        assert len(updates) == 25
        old_poses={tuple(sorted(row.items())) for row in baseline['placements']}
        new_poses={tuple(sorted(row.items())) for row in final_layout}
        final_validation=[json.loads(line) for line in (run/'evaluations.jsonl').read_text().splitlines()][-1]['evaluation']
        rows.append(dict(arm=identity, algorithm=arm['config']['algorithm'], actionSeed=arm['config']['seed'],
            sampledTrainingEpisodes=len(history), optimizerUpdates=len(updates),
            selectedEpisode=summary['bestEpisode'], selectedPolicySha256=digest,
            contractorValidationWarning=evaluation['contractor']['meanWarningSeconds'],
            selectedValidationWarning=evaluation['best']['meanWarningSeconds'],
            finalValidationWarning=final_validation['meanWarningSeconds'],
            finalValidationDelta=final_validation['deltaSeconds'],
            finalSensorsChanged=max(len(old_poses-new_poses),len(new_poses-old_poses)),
            nonzeroParameterUpdates=sum(u.get('parameter_delta_norm',0)>0 for u in updates),
            validationDelta=evaluation['best']['meanWarningSeconds']-evaluation['contractor']['meanWarningSeconds'],
            eightCaseTestDelta=test['policy']['deltaSeconds'],
            confirmationWarningDelta=final['warningDelta'], confirmationDetectionDelta=final['detectionDelta'],
            confirmationContractorWarning=final['contractor']['meanWarningSeconds'],
            confirmationSelectedWarning=final['policy']['meanWarningSeconds'],
            uniqueSampledLayouts=history[-1].get('uniqueLayouts'),
            nativePilotCost=status['nativeCost'], nativeConfirmationCalls=final['nativeCalls'],
            confirmationElapsedSeconds=final['elapsedSeconds'],
            secondaryFinalPolicySha256=secondary_arm['finalPolicySha256'],
            secondaryFinalWarningDelta=secondary_arm['warningDelta'],
            secondaryFinalWarning=secondary_arm['policy']['meanWarningSeconds'],
            secondaryFinalDetectionDelta=secondary_arm['detectionDelta'],
            weightFiles={'selected':'selected-policy.json','final':'final-policy.json','initial':'initial-policy.json'},
            sourceRunDirectory=str(run)))
    result = dict(schema='istana.local_ppo_pilot_confirmation_bundle.v1',
        createdAt=datetime.now(timezone.utc).isoformat(), arms=rows,
        pilotElapsedSeconds=pilot['elapsedSeconds'], confirmationElapsedSeconds=confirmation['elapsedSeconds'],
        pilotNativeCalls=sum(r['nativePilotCost']['completed'] for r in rows),
        confirmationNativeCalls=600, confirmationSharedScenarioCount=100,
        secondaryNativeCalls=300,secondaryElapsedSeconds=secondary['elapsedSeconds'],
        allOutcomesRetained=True, rawFramesIncluded=False,
        limitations='Two local training seeds and one legacy control; confirmation uses 100 shared cases. Per-policy intervals are descriptive. This pilot alone does not establish superiority of the algorithm.')
    write(TARGET/'results.json', result)
    lines = ['# Local refinement PPO: controlled pilot and confirmation', '',
        'All arms used the same four limited-FOV thermal sensors, five drones, contractor layout and physics. '
        'Each received 200 sampled training episodes and 25 updates. Local PPO also used matched contractor '
        'replays for training rewards. Every result is retained, including regressions.', '',
        '| Arm | Selected episode | Validation delta (s) | Pilot 8-case test delta (s) | Confirmation 100-case delta (s) | Descriptive 95% interval (s) |',
        '|---|---:|---:|---:|---:|---|']
    for row in rows:
        delta = row['confirmationWarningDelta']; low, high = delta['approximate95PercentInterval']
        lines.append(f"| {row['arm']} | {row['selectedEpisode']} | {row['validationDelta']:+.3f} | {row['eightCaseTestDelta']:+.3f} | {delta['mean']:+.3f} | [{low:+.3f}, {high:+.3f}] |")
    lines += ['',
        'Both local final greedy policies changed three of four sensor poses; the legacy final greedy layout '
        'retained the contractor poses. Each arm made 25 nonzero parameter updates. The local runs explored '
        '158 and 155 unique sampled layouts, but none of their validation checkpoints beat the contractor. '
        'All three selected policies therefore remained episode 0. Policy movement and learning updates '
        'are distinct from demonstrated performance improvement.', '',
        f"The contractor averaged {rows[0]['confirmationContractorWarning']:.10f} seconds of per-drone warning "
        'over the 100 confirmation scenarios; all three selected policies matched that absolute mean. '
        'Deltas compare mean per-drone warning against the original contractor; undetected drones contribute zero. '
        'Positive values mean earlier warning on average. The 100 confirmation scenarios are shared across arms. '
        'Each interval is mean ± 1.96 SE of the 100 paired scenario deltas for that frozen policy. '
        'Zero primary intervals arise from identical selected layouts, not certainty about the whole algorithm.', '',
        'Scenario panels and the confirmation panel were declared before pilot outcomes. Validation selected the '
        'checkpoint; neither the built-in eight-case test nor the 100-case confirmation selected a policy or tuned '
        'parameters. Two local action seeds and one legacy control remain a pilot, not definitive algorithm-wide evidence.', '',
        f"Actual native cost: {result['pilotNativeCalls']} pilot episodes plus 600 confirmation episodes. "
        f"Wall time was {result['pilotElapsedSeconds']:.1f}s for the parallel pilot and "
        f"{result['confirmationElapsedSeconds']:.1f}s for parallel confirmation. Per-arm calls, steps and timing are retained.", '',
        'A separate secondary extension was declared after pilot completion and before evaluating any final '
        'checkpoint on the confirmation panel. All three final checkpoints were frozen and evaluated on '
        'the same 100 scenarios against the retained exact contractor records. These outcomes did not select '
        'or promote a model and are not part of the original primary protocol.', '',
        '| Secondary final checkpoint | Mean warning (s) | Warning delta (s) | Descriptive 95% interval (s) |',
        '|---|---:|---:|---|',
    ]
    for row in rows:
        d=row['secondaryFinalWarningDelta'];lo,hi=d['approximate95PercentInterval']
        lines.append(f"| {row['arm']} final | {row['secondaryFinalWarning']:.3f} | {d['mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['',f"Secondary cost: 300 additional native episodes, {secondary['elapsedSeconds']:.1f}s parallel wall time. "
        'The same 100 cases are shared across primary and secondary results; they are not independent replication.', '',
        'Portable contents: selected, final and initial policy weights; baseline public context and contractor placements; '
        'full validation/test/confirmation arrays; compressed training, evaluation and native-call logs; drivers and '
        'frozen Python sources. Raw replay frames remain in the Saved experiment directories. Source paths in historical '
        'metadata identify the original run; use the relative weight filenames in results.json for this bundle.', '',
        'Source provenance: post-launch changes added a legality guard for separation > 300 m and three '
        'previously omitted fields to paired constraint validation. Exact deltas are retained as a patch. '
        'This study uses 20 m separation. A retrospective raw-context audit verifies budget_remaining and '
        'both deployment radius bounds on all 400 paired training cases and 600 primary/secondary confirmation pairs. '
        'All matched exactly at 8, 30 m and 150 m. Confirmation ran the exact launched core source from the frozen snapshot; '
        'source-provenance.json records the remaining unchanged dependencies and all hashes.', '',
        'Fairness checks: contractor and initial layouts/metrics matched across arms; every paired scenario '
        'passed Red trajectory and physical-constraint hash checks; confirmation contractor metrics matched '
        'across all three processes. The selected policy hashes match the frozen confirmation weights.',
        '', 'Offline verification: run `python archive.py` from this directory with NumPy installed. '
        'The verifier loads bundled frozen sources and public context, checks every archive hash, policy inventory '
        'and legal layout, training/update/evaluation rows, paired test/confirmation hashes and aggregates. '
        'It requires neither Saved files nor a native port. Bundle-local .gitattributes preserves exact bytes.',
        '', 'Driver copies preserve the executed procedures and original workspace paths. Those paths are '
        'historical provenance and are not required by the offline verifier.',
    ]
    (TARGET/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(TARGET/'environment.json', dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    artifacts = [dict(path=str(p.relative_to(TARGET)).replace('\\','/'), bytes=p.stat().st_size,
                      sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(TARGET.rglob('*')) if p.is_file()]
    write(TARGET/'manifest.json', dict(schema='istana.evidence_manifest.v1', files=artifacts))
    print(json.dumps(dict(bundle=str(TARGET), files=len(artifacts), bytes=sum(a['bytes'] for a in artifacts)),indent=2))

if __name__=='__main__': main()
