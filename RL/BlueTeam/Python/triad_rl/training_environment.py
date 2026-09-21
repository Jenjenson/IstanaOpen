"""Training adapter for the existing native bridge, never a second simulator."""
from __future__ import annotations

import time

from .warning_policy import warning_metrics

WARNING_DEFINITION = "Mean per-threat max(0, zone entry time − first detection time), in seconds; missed threats count as zero."
SUCCESS_DEFINITION = "Fraction of threats confirmed at least the configured defence lead time before zone entry (native timely_fraction)."
NATIVE_STEP_BATCH = 20


class TrainingStopped(Exception):
    """A requested stop discarded the unfinished episode, not its predecessors."""


def require_training_runtime(context):
    if context.get("trainingConfigurationVersion") != 1:
        raise ValueError("This Unreal runtime does not support Training configuration. Build this branch and launch with -IstanaBlueLive.")
    if not any(row.get("directional") for row in context.get("catalogue", [])):
        raise ValueError("Training requires the updated native directional sensor catalogue.")


def blue_configuration(config):
    return {"budget": config["budget"], "availableSensorIds": config["enabledSensorIds"]}


def run_episode(client, policy, config, seed, stop, *, deterministic=False):
    # Reuse the existing native trainer's seeded radial Red scenario generator.
    # It receives Red-only context; the policy receives only Blue public context.
    from train_warning_live import red_centers

    started = time.monotonic()
    if stop.is_set():
        raise TrainingStopped()
    red = client.reset(seed, blue_configuration=blue_configuration(config))
    context = client.get_blue_context()
    require_training_runtime(context)
    placements, records = policy.plan(context, deterministic=deterministic)
    client.deploy(placements)
    client.place_red(red_centers(red, seed))
    for _ in range(10000):
        if stop.is_set():
            client.cancel()
            raise TrainingStopped()
        # Small native batches bound stop latency while preserving fixed-step physics.
        response = client.step(NATIVE_STEP_BATCH)
        blue = response["blueObservation"]
        if blue["terminated"] or blue["truncated"]:
            break
    else:
        client.cancel()
        raise RuntimeError("Native episode exceeded the training step bound")
    warning = warning_metrics(blue)  # also rejects unresolved arrivals / changed definitions
    evidence = blue["warningEvidenceForEvaluationOnly"]
    confirmations = [row["firstConfirmationSeconds"] for row in evidence if row.get("firstConfirmationSeconds") is not None]
    detected = sum(row["firstDetectionSeconds"] is not None for row in evidence)
    row = {
        "seed": seed, "reward": blue["reward"], "warningTime": warning["mean_drone_warning_s"],
        "teamWarningTime": warning["team_warning_s"], "firstDetectionTime": warning["first_detection_s"],
        "confirmationTime": min(confirmations) if confirmations else None,
        "confirmationWarningTime": sum(max(0., x["zoneEntrySeconds"] - x["firstConfirmationSeconds"])
            if x.get("firstConfirmationSeconds") is not None else 0. for x in evidence) / len(evidence),
        "successRate": blue["metrics"]["timely_fraction"], "threatsDetected": detected,
        "threatsMissed": len(evidence) - detected, "threatCount": len(evidence),
        "sensorsPlaced": len(placements), "budgetUsed": warning["cost"],
        "episodeDuration": blue["elapsedSeconds"], "wallDuration": time.monotonic() - started,
        "policyLoss": None, "valueLoss": None, "entropy": None,
        "nativeRunId": red["runId"], "placements": placements,
    }
    return row, records, evidence
