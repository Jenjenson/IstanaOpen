"""Local transport guards and public-only planning; no native-build claim."""
from copy import deepcopy
import io
import json
from pathlib import Path

import pytest

from triad_rl.istana_live import (BridgeProtocolError, BridgeRejected, IstanaLiveClient,
    make_plan, placement_world_cm, public_planning_inputs, scripted_red_centers)

BLUE = Path(__file__).resolve().parents[2]


class Socket:
    def __init__(self, data):
        self.stream, self.sent, self.closed = io.BytesIO(data), [], False
    def settimeout(self, timeout): pass
    def makefile(self, mode): return self.stream
    def sendall(self, data): self.sent.append(data)
    def close(self): self.closed = True


@pytest.mark.parametrize('data', [b'', b'{}', b'{bad}\n',
    b'{"id":1,"ok":true}\n', b'{"id":true,"ok":true}\n',
    b'{"id":0,"ok":1}\n', b'{"id":0,"id":0,"ok":true}\n',
    b'{"id":0,"ok":true,"bad":NaN}\n'])
def test_protocol_failures_close_session_without_retry(data):
    sock = Socket(data)
    client = IstanaLiveClient(connection=sock)
    with pytest.raises(BridgeProtocolError):
        client.request('context')
    assert client.closed and sock.closed and len(sock.sent) == 1
    with pytest.raises(BridgeProtocolError):
        client.request('context')
    assert len(sock.sent) == 1


def test_server_rejection_can_be_inspected_without_reconnect():
    sock = Socket(b'{"id":0,"ok":false,"error":"reset needed"}\n{"id":1,"ok":true}\n')
    with IstanaLiveClient(connection=sock) as client:
        with pytest.raises(BridgeRejected, match='reset needed'):
            client.request('step')
        assert not client.closed
        assert client.request('context')['id'] == 1
    assert sock.closed


def test_invalid_request_does_not_send_or_advance_id():
    sock = Socket(b'')
    with IstanaLiveClient(connection=sock) as client:
        for fields in ({'seed': float('nan')}, {'id': 4}, {'payload': 'x'*65536}):
            with pytest.raises(ValueError):
                client.request('reset', **fields)
        assert not sock.sent and client.next_id == 0


@pytest.fixture
def context():
    def read(name): return json.loads((BLUE / 'Examples' / name).read_text())
    public = read('public-snapshot.json')
    public.update(timestamp=0, placements=[], done=False)
    public['budget_remaining'] = public['budget_total']
    return {'runId': 'test-run', 'revision': 1, 'completedSteps': 0, 'committed': False,
            'coordinateSystem': 'unreal_xy_relative_m_z_up',
            'worldOriginCm': {'x': 100, 'y': 200, 'z': 300},
            'catalogue': read('sensor-catalogue.json'), 'publicSnapshot': public,
            'temporalConfig': read('temporal-config.json')}


def test_planning_allowlist_does_not_forward_private_envelope(context):
    before = public_planning_inputs(context)
    poisoned = deepcopy(context)
    poisoned.update(private_red_truth={'targets': [[1, 2, 3]], 'seed': 999}, reward=100)
    after = public_planning_inputs(poisoned)
    assert before == after
    assert 'private_red_truth' not in after[0]


@pytest.mark.parametrize('field,value', [('committed', True), ('completedSteps', 1),
    ('coordinateSystem', 'geographic')])
def test_planner_rejects_stale_or_wrong_coordinate_context(context, field, value):
    context[field] = value
    with pytest.raises(ValueError): public_planning_inputs(context)


@pytest.mark.parametrize('policy', ['406', '407', '408', 'control'])
@pytest.mark.parametrize('ue_whole_doubles', [False, True])
def test_explicit_published_checkpoints_and_control_create_legal_initial_plan(context, policy, ue_whole_doubles):
    if ue_whole_doubles:
        context['temporalConfig'] = {k: int(v) if isinstance(v, float) and v.is_integer() else v
                                     for k, v in context['temporalConfig'].items()}
    original = deepcopy(context)
    checkpoint = None if policy == 'control' else BLUE / f'Results/temporal-v6-pilot/training/seed-{policy}/last'
    plan = make_plan(context, checkpoint=checkpoint, temporal_public_control=policy == 'control')
    assert context == original and plan['public_only']
    assert plan['coordinateSystem'] == context['coordinateSystem']
    assert len(plan['placements']) <= context['publicSnapshot']['max_sites']
    for placement in plan['placements']:
        assert set(placement) == {'siteId', 'profileId'}
        assert placement['profileId'] in context['publicSnapshot']['available_sensor_ids']
        assert 0 <= placement['siteId'] < len(context['publicSnapshot']['sites'])


@pytest.mark.parametrize('value', [True, '20', float('nan')])
def test_config_numeric_normalization_does_not_accept_invalid_values(context, value):
    context['temporalConfig']['objective_radius_m'] = value
    with pytest.raises(ValueError):
        public_planning_inputs(context)


def test_position_units_and_scripted_red_are_explicit(context):
    site = context['publicSnapshot']['sites'][0]
    profile = context['catalogue'][0]
    position = placement_world_cm(context, {'siteId': 0, 'profileId': profile['id']})
    assert position == {'x': 100 + site[0]*100, 'y': 200 + site[1]*100,
                        'z': 300 + profile['height_m']*100}
    centers = scripted_red_centers({'objectiveWorldCm': context['worldOriginCm'],
        'minRadiusCm': 10000, 'maxRadiusCm': 20000, 'groupCount': 2, 'heightOffsetCm': 4500})
    assert centers[0] == [15100, 200, 4800]
    assert centers[1] == pytest.approx([-14900, 200, 4800])


def test_surface_position_uses_native_roof_height(context):
    site = context['publicSnapshot']['sites'][0]
    profile = context['catalogue'][0]
    context['placementRule'] = 'static_surface_mast_v1'
    context['siteSurfacesWorldCm'] = [None] * len(context['publicSnapshot']['sites'])
    context['siteSurfacesWorldCm'][0] = [100 + site[0]*100, 200 + site[1]*100, 1200]
    p = placement_world_cm(context, {'siteId': 0, 'profileId': profile['id']})
    assert p['z'] == 1200 + profile['height_m']*100
    context['siteSurfacesWorldCm'][0] = None
    with pytest.raises(ValueError, match='supporting surface'):
        placement_world_cm(context, {'siteId': 0, 'profileId': profile['id']})
    del context['siteSurfacesWorldCm']
    with pytest.raises(ValueError, match='missing surface'):
        placement_world_cm(context, {'siteId': 0, 'profileId': profile['id']})


@pytest.mark.parametrize('policy', ['406', '407', '408', 'control'])
def test_all_planners_respect_unsupported_surface_mask(context, policy):
    context['publicSnapshot']['blocked_sites'] = list(range(len(context['publicSnapshot']['sites'])))
    checkpoint = None if policy == 'control' else BLUE / f'Results/temporal-v6-pilot/training/seed-{policy}/last'
    plan = make_plan(context, checkpoint=checkpoint, temporal_public_control=policy == 'control')
    assert plan['placements'] == []
    assert plan['recommendation']['decisions'][-1]['stop']
