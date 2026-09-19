"""Real archived evidence and isolated local HTTP/bridge-controller checks."""
from copy import deepcopy
import http.client
import json
import threading

import pytest

from simulation_console import ConsoleState, load_replays, make_server, replay_view


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
    assert len(session['token']) >= 32
    assert headers['Cache-Control'] == 'no-store'
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    for path in ('/', '/app.js', '/style.css', '/api/replay/0', '/api/replay/17'):
        assert request(http_server, path)[0] == 200


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
    assert calls == [('reset', 27), ('deploy', placements)]
    assert view['nativePreview']['image'].startswith('data:image/png;base64,')
    assert view['metrics'] is None and view['ended']
    assert not view['frames'][0]['threats']
    assert (tmp_path/'Saved/ConsoleModelDemo/test-run.json').exists()


@pytest.mark.parametrize('selection', ['greedy', 'control', '406'])
def test_controller_orders_public_plan_before_red_placement(replays, selection):
    events = []
    sample = replays[0]
    public = deepcopy(sample['scenario']['public'])
    context = {'coordinateSystem': 'unreal_xy_relative_m_z_up', 'publicSnapshot': public,
               'catalogue': sample['catalogue'], 'worldOriginCm': {'x': 100, 'y': 200, 'z': 300},
               'temporalConfig': {'objective_radius_m': 20}}
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
        assert (kwargs['checkpoint'] is None) == (selection in ('greedy', 'control'))
        events.append('public_plan')
        return {'placements': [], 'recommendation': {'decisions': []}}
    state = ConsoleState(client_factory=FakeClient, planner=plan)
    assert state.action('connect', {})['connected']
    view = state.action('reset', {'seed': 123, 'policy': selection})
    assert ('Greedy' in view['policy']) == (selection in ('greedy', 'control'))
    assert events == ['reset', 'public_plan', 'deploy', 'place_red']
    assert view['frames'][0]['threats'][0]['position'] == [10, 20, 30]
    assert view['frames'][0]['threats'][0]['observer_truth']
    assert view['metrics'] is None and view['mode'] == 'live'
    with pytest.raises(ConnectionError):
        state.action('step', {})
    assert state.view is None and not state.status()['connected']
