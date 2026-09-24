"""Separate post-pilot final-checkpoint diagnostic; never selects a model."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback

import confirm as shared

OUT=shared.OUT
ROOT=shared.ROOT
def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text(encoding='utf-8'))

def prepare():
    completion=read(OUT/'completion.json');pilot=read(OUT/'protocol.json')
    assert completion['complete']
    protocol_path=OUT/'secondary-protocol.json'
    if protocol_path.exists():
        print(str(protocol_path));return
    folder=OUT/'secondary';folder.mkdir(exist_ok=True);arms=[]
    for declared,status in zip(pilot['arms'],completion['arms']):
        assert declared['id']==status['arm']
        source=Path(status['outputDirectory'])/'final-policy.json'
        saved=folder/status['arm']/'frozen-final-policy.json';saved.parent.mkdir(exist_ok=True)
        with saved.open('xb') as f:f.write(source.read_bytes())
        arms.append(dict(arm=status['arm'],port=declared['port'],algorithm=declared['config']['algorithm'],
            finalPolicy=str(saved),finalPolicySha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
            sourceDirectory=status['outputDirectory'],trainingEpisodes=200,checkpoint='final-policy.json'))
    primary_status=read(OUT/'confirmation/status.json')
    protocol=dict(schema='istana.local_ppo_secondary_final_checkpoint.v1',declaredAt=now(),
        addedAfterPilotCompletion=True,declaredBeforeAnyFinalCheckpointConfirmation=True,
        primaryStatusAtDeclaration=primary_status,
        purpose='Secondary diagnostic of all three learned final checkpoints; separate from the original primary validation-selected policy confirmation.',
        reason='Root requested this after the pilot showed changed local final poses but validation selection retained episode0.',
        seeds=list(range(63000000,63000100)),cases=100,arms=arms,additionalNativeCalls=300,
        baseline='Reuse each arm original matched contractor native runs from primary confirmation, with identical seed/trajectory/constraint checks.',
        selection='No selection, promotion, parameter update or registry change uses this diagnostic.',
        reporting='Retain every outcome; report separately as post-pilot secondary evidence. Shared100 cases across all arms; intervals descriptive per frozen final checkpoint.')
    with protocol_path.open('x',encoding='utf-8') as f:json.dump(protocol,f,indent=2)
    print(str(protocol_path),flush=True)

def run():
    protocol=read(OUT/'secondary-protocol.json');primary=read(OUT/'confirmation/completion.json')
    assert primary['complete'] and primary['crossArmContractorExactMatch']
    assert protocol['seeds']==list(range(63000000,63000100))
    launch=read(OUT/'launch.json')
    for name,digest in launch['sourceSha256'].items():
        relative=Path(name).relative_to('RL/BlueTeam/Python')
        assert hashlib.sha256((shared.FROZEN/relative).read_bytes()).hexdigest()==digest
    folder=OUT/'secondary';start=time.monotonic();lock=threading.RLock()
    statuses={a['arm']:dict(phase='pending',cases=0,nativeCalls=0) for a in protocol['arms']}
    def update(identity,**values):
        with lock:
            statuses[identity].update(values)
            shared.write(folder/'status.json',dict(updatedAt=now(),elapsedSeconds=time.monotonic()-start,arms=statuses))
    def worker(arm):
        identity=arm['arm'];target=folder/identity;raw=target/'native-episodes';raw.mkdir(exist_ok=True)
        assert hashlib.sha256(Path(arm['finalPolicy']).read_bytes()).hexdigest()==arm['finalPolicySha256']
        source=Path(arm['sourceDirectory'])
        context=read(source/'baseline-episode.json')['run']['context']
        policy=shared.TRAINING_ALGORITHMS[arm['algorithm']]['factory'].load(arm['finalPolicy'],context)
        primary_report=next(r for r in primary['reports'] if r['arm']==identity)
        baselines=[];runs=[];steps=0;wall=0.;begun=time.monotonic()
        try:
            with shared.IstanaLiveClient(port=arm['port']) as client:
                update(identity,phase='running',port=arm['port'])
                for number,seed in enumerate(protocol['seeds'],1):
                    baseline=read(OUT/'confirmation'/identity/'native-episodes'/f'{number:03}-contractor-{seed}.json')
                    assert baseline['seed']==seed
                    baseline_summary=shared.evaluation_summary([baseline])
                    assert baseline_summary['cases'][0]==primary_report['contractor']['cases'][number-1]
                    tick=time.monotonic()
                    actual,_=shared.run_episode(client,policy,seed,action_seed=seed+1000000,deterministic=True)
                    wall+=time.monotonic()-tick;steps+=actual['steps']
                    assert len(actual['placements'])==4 and all(p['profileId']=='thermal' for p in actual['placements'])
                    measured=shared.evaluation_summary([actual],baseline_summary)
                    shared.write(raw/f'{number:03}-final-{seed}.json',actual)
                    baselines.append(baseline);runs.append(actual)
                    with (target/'paired-cases.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(seed=seed,contractor=baseline_summary['cases'][0],policy=measured['cases'][0],
                            warningDeltaSeconds=measured['deltaSeconds'],detectionFractionDelta=actual['metrics']['detected_fraction']-baseline['metrics']['detected_fraction']),allow_nan=False)+'\n')
                    update(identity,cases=number,nativeCalls=number,nativeSteps=steps,nativeCallWallSeconds=wall)
                    if number%25==0:print(json.dumps(dict(at=now(),secondaryArm=identity,completedCases=number)),flush=True)
            baseline=shared.evaluation_summary(baselines);measured=shared.evaluation_summary(runs,baseline)
            detection=[a['metrics']['detected_fraction']-b['metrics']['detected_fraction'] for a,b in zip(runs,baselines)]
            report=dict(arm=identity,complete=True,kind='secondary_final_checkpoint',checkpoint='final-policy.json',
                finalPolicySha256=arm['finalPolicySha256'],seeds=protocol['seeds'],contractor=baseline,policy=measured,
                warningDelta=shared.statistics_summary(measured['pairedDeltasSeconds']),
                detectionDelta=shared.statistics_summary(detection),detectionFractionDeltas=detection,
                nativeCalls=100,nativeSteps=steps,nativeCallWallSeconds=wall,elapsedSeconds=time.monotonic()-begun,
                contractorReusedFromPrimary=True,identicalTrajectoriesAndConstraints=True,
                usedForSelection=False,registryModified=False,portReleased=True)
            shared.write(target/'report.json',report);update(identity,phase='complete',completedAt=now())
            return report
        except BaseException:
            (target/'failure.txt').write_text(now()+'\n'+traceback.format_exc(),encoding='utf-8')
            update(identity,phase='failed',error=traceback.format_exc());raise
    reports=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,arm) for arm in protocol['arms']]
        for future in as_completed(futures):reports.append(future.result())
    order={a['arm']:i for i,a in enumerate(protocol['arms'])};reports.sort(key=lambda r:order[r['arm']])
    final=dict(complete=True,completedAt=now(),kind='secondary_final_checkpoint',reports=reports,
        totalNativeCalls=sum(r['nativeCalls'] for r in reports),sharedScenarioCount=100,
        elapsedSeconds=time.monotonic()-start,allPortsReleased=True,usedForSelection=False,
        interpretation='Extension declared after pilot completion and before final-checkpoint evaluation. No model was selected or promoted from these outcomes. Report separately from primary selected-policy confirmation.')
    shared.write(folder/'completion.json',final)
    print(json.dumps(dict(secondaryComplete=True,nativeCalls=final['totalNativeCalls'],allPortsReleased=True)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()
