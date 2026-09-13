import pytest
from triad_rl.remote_control import TRIADRemoteControlClient, normalize_enum


def test_real_unreal_display_names_and_native_enum_names():
    names = ('Inactive', 'BluePlacement', 'RedDeployment', 'RedMovement', 'Terminal')
    for value in ('Blue Placement', 'BluePlacement', 'ETRIADRLPhase::BluePlacement', 1):
        assert normalize_enum(value, names) == 'BluePlacement'
    reasons = ('None', 'ProtectedZoneReached', 'AllTargetsConfirmed', 'HorizonReached',
               'ConstraintViolation', 'InvalidAction', 'SustainedTrackDefence')
    assert normalize_enum('Sustained Track Defence', reasons) == 'SustainedTrackDefence'
    for invalid in (-1, 5, True, 'Unknown', None):
        with pytest.raises(RuntimeError):
            normalize_enum(invalid, names)


def test_blue_hybrid_payload_and_local_position_validation(monkeypatch):
    captured = {}

    def fake_call(self, name, parameters):
        captured.update(name=name, parameters=parameters)
        return {"ok": True}

    monkeypatch.setattr(TRIADRemoteControlClient, "_call", fake_call)
    client = TRIADRemoteControlClient("/Game/Test.Manager")
    assert client.blue_action(2, (0.3, -0.4), stop=False) == {"ok": True}
    assert captured == {
        "name": "ApplyBlueAction",
        "parameters": {"Action": {
            "CatalogueIndex": 2,
            "NormalizedPosition": {"X": 0.3, "Y": -0.4},
            "bStopPlacement": False,
        }},
    }
    before = dict(captured)
    for invalid in (None, (), (0,), (0, 0, 0), (float("nan"), 0), (1, 1), ("x", 0)):
        with pytest.raises(ValueError):
            client.blue_action(0, invalid)
    assert captured == before
