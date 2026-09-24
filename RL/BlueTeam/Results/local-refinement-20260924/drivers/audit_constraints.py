"""Retrospective direct audit; never changes the frozen experiment runtime."""
from datetime import datetime, timezone
import difflib
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(r'D:\triad\IstanaOpen-LearningReview')
STUDY=Path(__file__).resolve().parent
FIELDS=('budget_remaining','deployment_min_radius','deployment_max_radius')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    assert read(STUDY/'secondary/completion.json')['complete']
    rows=[]
    def pair(arm,phase,seed,a,b):
        sides=[]
        for path in (a,b):
            raw=path.read_bytes();run=json.loads(raw)
            assert run['seed']==seed
            public=run['context']['publicSnapshot']
            values={k:public[k] for k in FIELDS}
            sides.append(dict(rawPath=path.relative_to(STUDY).as_posix(),rawSha256=sha(raw),
                              values=values,trajectorySha256=run['trajectory_sha256']))
        assert sides[0]['values']==sides[1]['values']
        assert sides[0]['trajectorySha256']==sides[1]['trajectorySha256']
        rows.append(dict(arm=arm,phase=phase,seed=seed,contractor=sides[0],policy=sides[1]))
    arms=[r['id'] for r in read(STUDY/'protocol.json')['arms']]
    for arm in arms:
        native=STUDY/'arms'/arm/'native-episodes'
        ledger=[json.loads(l) for l in (STUDY/'arms'/arm/'native-calls.jsonl').read_text().splitlines()]
        contractor={r['seed']:r for r in ledger if r['role']=='paired_contractor_training'}
        for actual in (r for r in ledger if r['role']=='sampled_training' and r['seed'] in contractor):
            reference=contractor[actual['seed']]
            pair(arm,'paired_training',actual['seed'],native/f"native-{reference['call']:05}.json",native/f"native-{actual['call']:05}.json")
        for number,seed in enumerate(range(63000000,63000100),1):
            baseline=STUDY/'confirmation'/arm/'native-episodes'/f'{number:03}-contractor-{seed}.json'
            pair(arm,'primary_confirmation',seed,baseline,STUDY/'confirmation'/arm/'native-episodes'/f'{number:03}-policy-{seed}.json')
            pair(arm,'secondary_final_confirmation',seed,baseline,STUDY/'secondary'/arm/'native-episodes'/f'{number:03}-final-{seed}.json')
    assert len(rows)==1000
    data=''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows).encode()
    (STUDY/'constraint-audit-pairs.jsonl.gz').write_bytes(gzip.compress(data,mtime=0))
    report=dict(auditedAt=datetime.now(timezone.utc).isoformat(),retrospective=True,
        reason='Postlaunch validation hardening added three previously omitted fields to the constraint hash. Direct raw public-context comparison checks them independently without altering frozen run evidence.',
        fields=list(FIELDS),pairs=len(rows),counts={phase:sum(r['phase']==phase for r in rows) for phase in sorted({r['phase'] for r in rows})},
        allPairsExact=True,valuesSeen=[dict(zip(FIELDS,values)) for values in sorted({tuple(r['policy']['values'][k] for k in FIELDS) for r in rows})],
        rawFramesIncluded=False,rawPaths='Historical provenance; each retained side has the raw source file hash and directly extracted initial public values.')
    write(STUDY/'constraint-audit.json',report)
    provenance=read(STUDY/'source-provenance.json')
    original_file=STUDY/'frozen-runtime/triad_rl/training_evidence.py'
    changed_file=ROOT/'RL/BlueTeam/Python/triad_rl/training_evidence.py'
    before=original_file.read_bytes();after=changed_file.read_bytes()
    delta=''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile='launched/triad_rl/training_evidence.py',tofile='current/triad_rl/training_evidence.py'))
    assert delta
    addition=dict(file='RL/BlueTeam/Python/triad_rl/training_evidence.py',frozenSha256=sha(before),currentSha256=sha(after),diff=delta,
        recordedAt=report['auditedAt'],purpose='Postlaunch paired constraint validation hardening; no policy/train numeric change. Exact new fields audited retrospectively in all 1000 retained paired training/confirmation cases.')
    provenance['postLaunchDeltas']=[r for r in provenance['postLaunchDeltas'] if r['file'].replace('\\','/')!=addition['file']]+[addition]
    provenance['constraintHardeningAudit']='constraint-audit.json and constraint-audit-pairs.jsonl.gz'
    write(STUDY/'source-provenance.json',provenance)
    (STUDY/'postlaunch-source-delta.patch').write_text(''.join(r['diff'] for r in provenance['postLaunchDeltas']),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
