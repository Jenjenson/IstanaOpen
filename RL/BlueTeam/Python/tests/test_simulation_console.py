"""Real archived evidence and isolated local HTTP/bridge-controller checks."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import threading

import pytest

from simulation_console import ConsoleState, load_replays, make_server, replay_view


CONSOLE = Path(__file__).resolve().parents[1] / 'console'


@pytest.fixture(scope="module")
def replays():
    return load_replays()


def test_all_archived_cases_preserved(replays):
    assert len(replays) == 18
    assert {(r['profile'], r['temporal_seed'], r['case_index']) for r in replays} == {
        (p, s, c) for p in ('normal', 'stress', 'capability')
        for s in (406, 407, 408) for c in (1, 2)}
    before = deepcopy(replays)
    for replay in replays:
        view = replay_view(replay)
        assert view['frames'] == replay['frames']
        assert view['metrics'] == replay['metrics']
        assert view['mode'] == 'recorded'
        assert view['audit']['live_unreal'] is False
        assert view['frames'][0]['time'] == 0
        assert all(a['time'] <= b['time'] for a, b in zip(view['frames'], view['frames'][1:]))
    assert replays == before


@pytest.fixture
def http_server(replays):
    server = make_server(0, replays=replays)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, path, method='GET', body=None, headers=None):
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def test_http_session_replay_and_security_headers(http_server):
    status, headers, raw = request(http_server, '/api/session')
    session = json.loads(raw)
    assert status == 200 and len(session['replays']) == 18
    assert not session['status']['connected']
    assert {row['id'] for row in session['trainingInitializations']} == {
        'untrained', 'directional_balanced_5', 'directional_public_5'}
    assert {row['id'] for row in session['trainingAlgorithms']} == {'reinforce', 'ppo', 'a2c'}
    assert session['trainingLimits'] == {
        'minEpisodes': 4, 'maxEpisodes': 10000, 'minSensors': 1, 'maxSensors': 5}
    assert session['training']['phase'] == 'idle'
    assert len(session['token']) >= 32
    assert headers['Cache-Control'] == 'no-store'
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    for path in ('/', '/app.js', '/style.css', '/api/replay/0', '/api/replay/17'):
        assert request(http_server, path)[0] == 200


def test_training_tab_uses_native_status_and_exposes_no_fake_detection_claim():
    html = (CONSOLE / 'index.html').read_text(encoding='utf-8')
    script = (CONSOLE / 'app.js').read_text(encoding='utf-8')
    assert 'id="training-mode"' in html and 'id="training-initialization"' in html
    assert 'id="training-chart"' in html and 'id="training-start"' in html
    assert 'id="training-reward-chart"' in html and 'id="training-reward-log"' in html
    assert 'id="training-blue-reward"' in html and 'id="training-red-reward"' in html
    assert 'not a pretrained Red model' in html and 'same fixed spawn radius' in html
    assert 'id="training-name"' in html and 'id="training-algorithm"' in html
    assert 'id="training-sensors"' in html and 'max="10000"' in html
    assert "api('/api/training/status')" in script
    assert "api(`/api/training/${action}`,payload)" in script
    assert "name:$('training-name').value" in script
    assert "algorithm:$('training-algorithm').value" in script
    assert "sensorCount:Number($('training-sensors').value)" in script
    assert "row.blueNativeReward??row.nativeReward" in script
    assert "row.redNativeReward" in script
    assert "entry.rewardBreakdown?.components" in script
    assert "entry.redScenario.spawnBearingsDeg" in script
    assert "syncModelCatalog(session)" in script
    assert 'actual detected fraction' not in html.lower()  # results are populated from native status


def test_compare_tab_offers_measured_five_sensor_results_with_contract_caveat():
    html = (CONSOLE / 'index.html').read_text(encoding='utf-8')
    script = (CONSOLE / 'app.js').read_text(encoding='utf-8')
    assert 'id="comparison-layout"' in html and 'id="comparison-layout-note"' in html
    assert 'comparison.layouts=session.comparisonLayouts||[]' in script
    assert "layoutId=$('comparison-layout').value||'matched_common_sense'" in script
    assert "fiveSensor=result.method==='directional_balanced_5'" in script
    assert 'the archived RL placement was trained for an earlier three-sensor contract' in script
    assert "'MATCHED WORKBENCH RESULTS'" in script


def test_drone_detection_rings_are_an_opt_in_map_layer():
    html = (CONSOLE / 'index.html').read_text(encoding='utf-8')
    script = (CONSOLE / 'app.js').read_text(encoding='utf-8')
    assert '<input id="drone-rings" type="checkbox"><span>Drone detection rings</span>' in html
    assert '<input id="drone-rings" type="checkbox" checked>' not in html
    assert "if($('drone-rings').checked&&(t.detected||t.tracked||t.confirmed))circle([x,y],10" in script
    assert "if($('drone-rings').checked)circle(p,9" in script
    assert "['ranges','trails','drone-rings','sites']" in script


def test_episode_results_show_exact_warning_metrics_with_honest_fallbacks():
    html = (CONSOLE / 'index.html').read_text(encoding='utf-8')
    script = (CONSOLE / 'app.js').read_text(encoding='utf-8')
    assert '<span>Team warning</span><strong id="result-team-warning">—</strong>' in html
    assert '<span>Mean per-drone warning</span><strong id="result-mean-warning">—</strong>' in html
    assert "warningText(m?.team_warning_seconds_lower_bound)" in script
    assert "warningText(m?.mean_drone_warning_seconds_lower_bound)" in script
    assert "if(mode==='live'&&(!view||!view.ended))return 'Pending'" in script
    assert "Number.isFinite(value)?`${fmt(value,2)} s`:'Unavailable'" in script


def test_live_transport_remains_stable_and_interactive_while_stepping():
    html = (CONSOLE / 'index.html').read_text(encoding='utf-8')
    script = (CONSOLE / 'app.js').read_text(encoding='utf-8')
    assert '<span id="connection-label">Replay ready</span>' in html
    assert "const blocking=(busy&&busyOperation!=='step')||comparison.loading" in script
    assert "$('timeline').disabled=!view||blocking||draft;$('speed').disabled=!view||draft" in script
    assert "interval=1000*(Number(view.stepDurationSeconds)||.5)/speed" in script
    assert "if(mode==='live'){frameIndex=0;for(let i=1;i<view.frames.length" in script
    assert "if(canvas.width!==pixelWidth||canvas.height!==pixelHeight)" in script
    assert "ctx.setTransform(dpr,0,0,dpr,0,0)" in script


@pytest.mark.parametrize('path', ['/../simulation_console.py', '/api/replay/18', '/api/replay/-1', '/secret'])
def test_http_no_file_traversal_or_missing_replay(http_server, path):
    assert request(http_server, path)[0] == 404


def test_http_rejects_foreign_host(http_server):
    assert request(http_server, '/api/session', headers={'Host': 'attacker.invalid'})[0] == 403


@pytest.mark.parametrize('invalid', ['origin', 'token', 'missing'])
def test_mutations_require_local_origin_and_session(http_server, invalid):
    session = json.loads(request(http_server, '/api/session')[2])
    headers = {'Content-Type': 'application/json',
               'Origin': f'http://127.0.0.1:{http_server.server_port}',
               'X-Console-Token': session['token']}
    if invalid == 'origin':
        headers['Origin'] = 'https://attacker.invalid'
    elif invalid == 'token':
        headers['X-Console-Token'] = 'wrong'
    else:
        headers = {}
    assert request(http_server, '/api/action/reset', 'POST', '{}', headers)[0] == 403


@pytest.mark.parametrize('body', ['[]', '{bad', '{"extra":"' + 'x' * 4096 + '"}'])
def test_mutations_reject_invalid_body(http_server, body):
    session = json.loads(request(http_server, '/api/session')[2])
    headers = {'Content-Type': 'application/json',
               'Origin': f'http://127.0.0.1:{http_server.server_port}',
               'X-Console-Token': session['token']}
    assert request(http_server, '/api/action/reset', 'POST', body, headers)[0] == 400


def test_connection_failure_never_creates_live_data():
    def unavailable(*args, **kwargs):
        raise ConnectionRefusedError('Unreal not running')
    state = ConsoleState(client_factory=unavailable)
    with pytest.raises(ConnectionError):
        state.action('connect', {})
    assert not state.status()['connected']
    assert not state.status()['episode']
    assert state.view is None and state.context is None
    assert 'Unreal is not listening' in state.status()['error']


def test_saved_preview_deploys_exact_output_without_planning_or_red(monkeypatch, tmp_path, replays):
    import model_switch_demo
    import capture_warning_3d
    import simulation_console
    public = deepcopy(replays[0]['scenario']['public'])
    public['tracks'] = []
    placements = [{'profileId': 'eo', 'siteId': 3}]
    row = {'label': 'RL Policy', 'seed': 27, 'placements': placements}
    monkeypatch.setattr(model_switch_demo, 'ROOT', tmp_path)
    monkeypatch.setattr(model_switch_demo, 'load_layouts', lambda _: {'rl': row})
    monkeypatch.setattr(simulation_console.time, 'sleep', lambda _: None)
    png = tmp_path/'frame.png'
    png.write_bytes(b'capture-test-fixture')
    monkeypatch.setattr(capture_warning_3d, 'capture', lambda *a: {
        'path': str(png), 'sensor_screen_anchors': [{'x': 100, 'y': 100}]})
    calls = []
    class PreviewClient:
        closed = False
        completed_steps = 0
        def reset(self, seed): calls.append(('reset', seed))
        def get_blue_context(self): return {'runId': 'test-run', 'publicSnapshot': public,
            'catalogue': replays[0]['catalogue'], 'worldOriginCm': {'x': 0, 'y': 0, 'z': 0}}
        def deploy(self, rows): calls.append(('deploy', deepcopy(rows)))
        def observe_blue(self): return {'elapsedSeconds': 0, 'completedSteps': 0, 'publicSnapshot': public}
        def close(self): self.closed = True
    def forbidden_plan(*args, **kwargs): raise AssertionError('Preview must not plan')
    state = ConsoleState(planner=forbidden_plan)
    state.client = PreviewClient()
    view = state.action('preview', {'policy': 'saved-rl'})
    assert calls == [('reset', 27), ('deploy', [{**placements[0], 'yawDeg': 0., 'pitchDeg': 0.}])]
    assert view['nativePreview']['image'].startswith('data:image/png;base64,')
    assert view['metrics'] is None and view['ended']
    assert not view['frames'][0]['threats']
    assert (tmp_path/'Saved/ConsoleModelDemo/test-run.json').exists()


@pytest.mark.parametrize('selection', ['greedy', 'control', '406', 'common_sense'])
def test_controller_orders_public_plan_before_red_placement(replays, selection):
    events = []
    sample = replays[0]
    public = deepcopy(sample['scenario']['public'])
    context = {'coordinateSystem': 'unreal_xy_relative_m_z_up', 'publicSnapshot': public,
               'catalogue': sample['catalogue'], 'worldOriginCm': {'x': 100, 'y': 200, 'z': 300},
               'temporalConfig': {'objective_radius_m': 20},
               'fixedStepSeconds': .05, 'timeLimitSeconds': 96.}
    blue = {'elapsedSeconds': 0, 'completedSteps': 0, 'publicSnapshot': public}

    class FakeClient:
        closed = False
        def __init__(self, *args, **kwargs): pass
        def request(self, op): return {'context': context}
        def reset(self, seed):
            events.append('reset')
            return {'objectiveWorldCm': context['worldOriginCm'], 'groupCount': 1,
                    'minRadiusCm': 10000, 'maxRadiusCm': 20000, 'heightOffsetCm': 4500}
        def get_blue_context(self): return context
        def deploy(self, rows): events.append('deploy')
        def place_red(self, centers):
            events.append('place_red')
            return {'initialStates': [{'droneId': 4, 'positionCm': {'x': 1100, 'y': 2200, 'z': 3300}}]}
        def observe_blue(self): return blue
        def close(self): self.closed = True
        def step(self, steps): raise ConnectionError('Lost bridge during step')

    def plan(value, **kwargs):
        assert value is context
        assert kwargs['temporal_public_control'] == (selection in ('greedy', 'control'))
        assert (kwargs['checkpoint'] is None) == (selection in ('greedy', 'control', 'common_sense'))
        assert kwargs.get('common_sense', False) == (selection == 'common_sense')
        events.append('public_plan')
        return {'placements': [], 'recommendation': {'decisions': []}}
    state = ConsoleState(client_factory=FakeClient, planner=plan)
    assert state.action('connect', {})['connected']
    view = state.action('reset', {'seed': 123, 'policy': selection})
    assert ('Greedy' in view['policy']) == (selection in ('greedy', 'control'))
    assert ('Common-sense' in view['policy']) == (selection == 'common_sense')
    assert events == ['reset', 'public_plan', 'deploy', 'place_red']
    assert view['frames'][0]['threats'][0]['position'] == [10, 20, 30]
    assert view['frames'][0]['threats'][0]['observer_truth']
    assert view['fixedStepSeconds'] == .05 and view['timeLimitSeconds'] == 96.
    assert view['stepDurationSeconds'] == .5
    assert view['metrics'] is None and view['mode'] == 'live'
    with pytest.raises(ConnectionError):
        state.action('step', {})
    assert state.view is None and not state.status()['connected']
