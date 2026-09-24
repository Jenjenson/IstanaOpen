"""Verify portable pilot/confirmation evidence without Saved files or native ports."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from datetime import datetime

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-snapshot'))
from triad_rl.warning_algorithms import TRAINING_ALGORITHMS
from triad_rl.training_evidence import layout_key
from triad_rl.istana_live import public_planning_inputs
from triad_rl.directional_inputs import build_observation, apply_placement

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def lines(path): return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines() if line]
def near(a,b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8), (a,b)
def digest(value): assert isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def check_evaluation(evaluation, baseline=None):
    cases = evaluation['cases']; assert len(cases)==evaluation['caseCount']
    assert [r['seed'] for r in cases]==evaluation['seeds'] and len(set(evaluation['seeds']))==len(cases)
    for row in cases:
        digest(row['trajectorySha256']);digest(row['constraintsSha256'])
        assert row['targets']==5 and row['cost']==4 and row['unresolved']==0
        assert 0 <= row['detected_fraction'] <= 1 and row['mean_drone_warning_s'] >= 0
    warnings = [r['mean_drone_warning_s'] for r in cases]
    near(evaluation['meanWarningSeconds'],statistics.mean(warnings))
    near(evaluation['mean_drone_warning_s'],statistics.mean(warnings))
    near(evaluation['warningStdSeconds'],statistics.stdev(warnings) if len(cases)>1 else 0)
    for key in ['team_warning_s','detected_fraction']:
        near(evaluation[key],statistics.mean(r[key] for r in cases))
    if baseline is not None:
        assert evaluation['seeds']==baseline['seeds']
        deltas=[]
        for actual, reference in zip(cases,baseline['cases']):
            for key in ['seed','trajectorySha256','constraintsSha256']:assert actual[key]==reference[key],key
            deltas.append(actual['mean_drone_warning_s']-reference['mean_drone_warning_s'])
        near(evaluation['baselineMeanWarningSeconds'],baseline['meanWarningSeconds'])
        near(evaluation['deltaSeconds'],statistics.mean(deltas))
        for a,b in zip(evaluation['pairedDeltasSeconds'],deltas):near(a,b)
        assert len(evaluation['pairedDeltasSeconds'])==len(deltas)
        assert evaluation['improvedCases']==sum(d>1e-9 for d in deltas)
        assert evaluation['tiedCases']==sum(abs(d)<=1e-9 for d in deltas)
        assert evaluation['worseCases']==sum(d< -1e-9 for d in deltas)

def check_statistics(summary, values):
    assert summary['cases']==len(values)
    mean=statistics.mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(len(values))
    near(summary['mean'],mean);near(summary['sd'],sd);near(summary['standardError'],se)
    for actual,expected in zip(summary['approximate95PercentInterval'],[mean-1.96*se,mean+1.96*se]):near(actual,expected)

def check_layout(context, placements):
    assert len(placements)==4 and {r['profileId'] for r in placements}=={'thermal'}
    state,catalogue,_=public_planning_inputs(context);state['max_sites']=4
    observation=build_observation(state,catalogue)
    lookup={(r['sensor_id'],r['site_index'],r['yaw_deg'],r['pitch_deg']):i
            for i,r in enumerate(observation['options']) if not r['stop']}
    for p in placements:
        index=lookup[(p['profileId'],p['siteId'],p['yawDeg'],p['pitchDeg'])]
        assert build_observation(state,catalogue)['action_mask'][index]
        state=apply_placement(state,index,catalogue)

def main():
    manifest=read(ROOT/'manifest.json')
    for row in manifest['files']:
        path=(ROOT/row['path']).resolve();assert path.is_relative_to(ROOT)
        assert path.stat().st_size==row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
    result=read(ROOT/'results.json');protocol=read(ROOT/'protocol.json')
    confirmation_protocol=read(ROOT/'confirmation-protocol.json')
    launch=read(ROOT/'launch.json');pilot_completed=read(ROOT/'pilot-completion.json')
    frozen=read(ROOT/'frozen-selection.json');secondary_protocol=read(ROOT/'secondary-protocol.json')
    assert confirmation_protocol['declaredBeforePilotLaunch']
    assert datetime.fromisoformat(confirmation_protocol['declaredAt'])<=datetime.fromisoformat(launch['startedAt'])
    assert frozen['beforeAnyConfirmationRun'] and datetime.fromisoformat(frozen['frozenAt'])>=datetime.fromisoformat(pilot_completed['completedAt'])
    assert secondary_protocol['addedAfterPilotCompletion'] and secondary_protocol['declaredBeforeAnyFinalCheckpointConfirmation']
    assert datetime.fromisoformat(secondary_protocol['declaredAt'])>datetime.fromisoformat(pilot_completed['completedAt'])
    for name,value in launch['sourceSha256'].items():
        relative=name.replace('\\','/').split('/Python/',1)[1]
        assert hashlib.sha256((ROOT/'source-snapshot'/relative).read_bytes()).hexdigest()==value
    for row in read(ROOT/'source-provenance.json')['modules']:
        assert hashlib.sha256((ROOT/'source-snapshot'/row['file'].replace('\\','/')).read_bytes()).hexdigest()==row['sha256']
    assert confirmation_protocol['seeds']==list(range(63000000,63000100))
    assert len(result['arms'])==3 and [r['arm'] for r in result['arms']]==[a['id'] for a in protocol['arms']]
    confirmation_contractors=[];native_total=0;expected_constraint_audits={}
    for arm,declared in zip(result['arms'],protocol['arms']):
        folder=ROOT/'arms'/arm['arm'];config=read(folder/'configuration.json')
        for k,v in declared['config'].items():assert config[k]==v,(arm['arm'],k)
        assert config['sensorCount']==4 and config['sensorIds']==['thermal']
        context=read(folder/'baseline-public-context.json');contractor=read(folder/'contractor-placements.json')
        check_layout(context,contractor)
        thermal=next(p for p in context['catalogue'] if p['id']=='thermal')
        assert thermal['directional'] and 0<thermal['manufacturer_specifications']['horizontal_fov_deg']<360
        for filename in ['selected-policy.json','final-policy.json','initial-policy.json']:
            policy=TRAINING_ALGORITHMS[config['algorithm']]['factory'].load(folder/filename,context)
            assert policy.sensor_count==4 and set(policy.allowed_sensor_ids)=={'thermal'}
            placements,_=policy.plan(context,deterministic=True);check_layout(context,placements)
            expected_file={'selected-policy.json':'selected-layout.json','final-policy.json':'final-layout.json'}.get(filename)
            if expected_file:assert layout_key(placements)==layout_key(read(folder/expected_file))
            else:assert layout_key(placements)==layout_key(contractor)
            if filename=='final-policy.json':assert policy.updates==25
        selected_hash=hashlib.sha256((folder/'selected-policy.json').read_bytes()).hexdigest()
        assert selected_hash==arm['selectedPolicySha256']
        selected_declaration=next(a for a in frozen['policies'] if a['arm']==arm['arm'])
        assert selected_declaration['policySha256']==selected_hash and selected_declaration['selectedEpisode']==arm['selectedEpisode']
        history=lines(folder/'training.jsonl.gz');assert len(history)==200
        ledger=lines(folder/'native-calls.jsonl.gz');cost=read(folder/'native-cost.json')
        assert len(ledger)==cost['completed']==cost['started'] and cost['failed']==0
        assert cost==arm['nativePilotCost']
        assert [r['call'] for r in ledger]==list(range(1,len(ledger)+1))
        assert sum(r['nativeSteps'] for r in ledger)==cost['nativeSteps'];native_total+=len(ledger)
        near(sum(r['wallSeconds'] for r in ledger),cost['nativeCallWallSeconds'])
        for role,count in cost['roles'].items():assert sum(r['role']==role for r in ledger)==count
        sampled={r['seed']:r for r in ledger if r['role']=='sampled_training'}
        paired={r['seed']:r for r in ledger if r['role']=='paired_contractor_training'}
        assert len(sampled)==200
        local=config['algorithm']=='local_ppo';assert len(paired)==(200 if local else 0)
        updates=[];unique=set()
        for number,row in enumerate(history,1):
            seed=61000000+number-1
            assert row['episode']==number and row['seed']==seed and row['actionSeed']==config['seed']*100000+number
            check_layout(context,row['placements']);unique.add(layout_key(row['placements']))
            assert row['uniqueLayouts']==len(unique)
            near(row['meanWarningSeconds'],sampled[seed]['evidence']['mean_drone_warning_s'])
            if local:
                p=row['pairedTraining'];reference=paired[seed]['evidence'];actual=sampled[seed]['evidence']
                for key in ['seed','trajectorySha256','constraintsSha256']:assert actual[key]==reference[key]==p[key]
                near(p['policyWarningSeconds'],actual['mean_drone_warning_s']);near(p['contractorWarningSeconds'],reference['mean_drone_warning_s'])
                near(p['rewardSeconds'],p['policyWarningSeconds']-p['contractorWarningSeconds']);near(row['trainingReward'],p['rewardSeconds'])
                expected_constraint_audits[(arm['arm'],'paired_training',seed)]=actual['trajectorySha256']
            else:near(row['trainingReward'],row['meanWarningSeconds'])
            if 'update' in row:updates.append(row['update'])
        assert len(updates)==25 and [r['update'] for r in updates]==list(range(1,26))
        assert arm['uniqueSampledLayouts']==len(unique)
        assert arm['nonzeroParameterUpdates']==sum(u.get('parameter_delta_norm',0)>0 for u in updates)
        old=set(layout_key(contractor));new=set(layout_key(read(folder/'final-layout.json')))
        assert arm['finalSensorsChanged']==max(len(old-new),len(new-old))
        evaluation=read(folder/'evaluation-summary.json');baseline=evaluation['contractor']
        check_evaluation(baseline);assert baseline['seeds']==list(range(61010000,61010008))
        check_evaluation(evaluation['initial'],baseline);check_evaluation(evaluation['best'],baseline)
        evaluations=lines(folder/'evaluations.jsonl.gz')
        assert [e['episode'] for e in evaluations]==sorted(set(range(8,201,8))|{100})
        best=evaluation['initial'];best_episode=0
        for entry in evaluations:
            check_evaluation(entry['evaluation'],baseline)
            if entry['evaluation']['meanWarningSeconds']>best['meanWarningSeconds']+1e-9:
                best=entry['evaluation'];best_episode=entry['episode']
        assert best_episode==evaluation['bestEpisode']==arm['selectedEpisode'] and best==evaluation['best']
        near(arm['finalValidationWarning'],evaluations[-1]['evaluation']['meanWarningSeconds'])
        near(arm['finalValidationDelta'],evaluations[-1]['evaluation']['deltaSeconds'])
        test=read(folder/'test-evaluation.json');assert test['usedForSelection'] is False
        check_evaluation(test['contractor']);check_evaluation(test['policy'],test['contractor'])
        assert test['policy']['seeds']==list(range(61020000,61020008))
        final=read(folder/'confirmation.json');assert final['complete'] and not final['usedForSelection']
        assert final['policySha256']==selected_hash and final['seeds']==confirmation_protocol['seeds']
        check_evaluation(final['contractor']);check_evaluation(final['policy'],final['contractor'])
        near(arm['confirmationContractorWarning'],final['contractor']['meanWarningSeconds'])
        near(arm['confirmationSelectedWarning'],final['policy']['meanWarningSeconds'])
        check_statistics(final['warningDelta'],final['policy']['pairedDeltasSeconds'])
        detection=[a['detected_fraction']-b['detected_fraction'] for a,b in zip(final['policy']['cases'],final['contractor']['cases'])]
        check_statistics(final['detectionDelta'],detection)
        for a,b in zip(final['detectionFractionDeltas'],detection):near(a,b)
        paired_cases=lines(folder/'confirmation-paired-cases.jsonl.gz');assert len(paired_cases)==100
        for number,pair in enumerate(paired_cases):
            assert pair['seed']==final['seeds'][number] and pair['contractor']==final['contractor']['cases'][number] and pair['policy']==final['policy']['cases'][number]
            near(pair['warningDeltaSeconds'],final['policy']['pairedDeltasSeconds'][number]);near(pair['detectionFractionDelta'],detection[number])
            expected_constraint_audits[(arm['arm'],'primary_confirmation',pair['seed'])]=pair['policy']['trajectorySha256']
        near(arm['validationDelta'],best['meanWarningSeconds']-baseline['meanWarningSeconds'])
        near(arm['eightCaseTestDelta'],test['policy']['deltaSeconds'])
        assert arm['confirmationWarningDelta']==final['warningDelta'] and arm['confirmationDetectionDelta']==final['detectionDelta']
        assert final['nativeCalls']==200 and arm['nativeConfirmationCalls']==200
        secondary=read(folder/'secondary-final-confirmation.json')
        assert secondary['complete'] and secondary['kind']=='secondary_final_checkpoint'
        assert secondary['contractorReusedFromPrimary'] and not secondary['usedForSelection'] and not secondary['registryModified']
        assert secondary['contractor']==final['contractor'] and secondary['seeds']==final['seeds']
        final_hash=hashlib.sha256((folder/'final-policy.json').read_bytes()).hexdigest()
        assert final_hash==secondary['finalPolicySha256']==arm['secondaryFinalPolicySha256']
        final_declaration=next(a for a in secondary_protocol['arms'] if a['arm']==arm['arm'])
        assert final_declaration['finalPolicySha256']==final_hash and final_declaration['checkpoint']=='final-policy.json'
        check_evaluation(secondary['policy'],secondary['contractor'])
        near(arm['secondaryFinalWarning'],secondary['policy']['meanWarningSeconds'])
        check_statistics(secondary['warningDelta'],secondary['policy']['pairedDeltasSeconds'])
        d=[a['detected_fraction']-b['detected_fraction'] for a,b in zip(secondary['policy']['cases'],secondary['contractor']['cases'])]
        check_statistics(secondary['detectionDelta'],d)
        secondary_pairs=lines(folder/'secondary-final-paired-cases.jsonl.gz');assert len(secondary_pairs)==100
        for n,pair in enumerate(secondary_pairs):
            assert pair['seed']==secondary['seeds'][n] and pair['contractor']==secondary['contractor']['cases'][n] and pair['policy']==secondary['policy']['cases'][n]
            near(pair['warningDeltaSeconds'],secondary['policy']['pairedDeltasSeconds'][n]);near(pair['detectionFractionDelta'],d[n])
            near(secondary['detectionFractionDeltas'][n],d[n])
            expected_constraint_audits[(arm['arm'],'secondary_final_confirmation',pair['seed'])]=pair['policy']['trajectorySha256']
        assert secondary['nativeCalls']==100 and arm['secondaryFinalWarningDelta']==secondary['warningDelta'] and arm['secondaryFinalDetectionDelta']==secondary['detectionDelta']
        confirmation_contractors.append(final['contractor'])
        print(f"Verified {arm['arm']}: 200 training rows, 25 updates, 26 validation panels, 8 test cases, 100 primary and 100 secondary confirmation pairs")
    assert confirmation_contractors[0]==confirmation_contractors[1]==confirmation_contractors[2]
    assert native_total==result['pilotNativeCalls'] and result['confirmationNativeCalls']==600
    assert result['secondaryNativeCalls']==300
    audit=read(ROOT/'constraint-audit.json');audited=lines(ROOT/'constraint-audit-pairs.jsonl.gz')
    assert audit['retrospective'] and audit['allPairsExact'] and audit['pairs']==len(audited)==1000
    expected_values={'budget_remaining':8,'deployment_min_radius':30,'deployment_max_radius':150}
    assert audit['valuesSeen']==[expected_values] and set(audit['fields'])==set(expected_values)
    assert audit['counts']=={'paired_training':400,'primary_confirmation':300,'secondary_final_confirmation':300}
    seen=set()
    for row in audited:
        key=(row['arm'],row['phase'],row['seed']);assert key not in seen;seen.add(key)
        assert row['contractor']['values']==row['policy']['values']==expected_values
        for side in ['contractor','policy']:
            digest(row[side]['rawSha256'])
            assert row[side]['trajectorySha256']==expected_constraint_audits[key]
    assert seen==set(expected_constraint_audits)
    print('Verified retrospective direct constraint audit for 400 training and 600 confirmation pairs.')
    print(f"Verified {len(manifest['files'])} exact artifact hashes; no Saved files or native ports used.")

if __name__=='__main__':main()
