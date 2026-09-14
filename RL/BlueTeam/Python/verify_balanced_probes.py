"""Portable verification of the unchanged balanced-v3 public STOP probes.

Original public-input SHA-256 strings are exact byte identities, not numeric
comparisons. Platform libm differences in generated coordinates may change
those hashes despite last-bit-only numeric drift. This additive supplement
archives the complete producer observations only when their state/catalogue
hashes exactly match all three original probes. It never replaces those hashes,
probes, checkpoints, sources or the original publication manifest.

Verification separately checks (1) exact archived bytes and original identities,
(2) regenerated public observations within a declared 1e-12 float tolerance,
with every discrete field exact, and (3) actual versioned actor inference on the
archived observations. No simulator step, private scenario access or training
occurs. A sibling manifest binds the compressed corpus and all its dependencies.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np

from probe_balanced import first_action_record, source_fingerprints
from probe_stop_exploration import summarize_records
from evaluate_robust import validate_seed_range
from triad_rl.adaptive_evaluation import canonical_hash, json_safe
from triad_rl.adaptive_inputs import FEATURE_NAMES, FEATURE_SCHEMA, build_observation
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.robust_scenarios import RobustPlacementEnv


PYTHON_ROOT = Path(__file__).resolve().parent
BLUE_ROOT = PYTHON_ROOT.parent
REPO_ROOT = BLUE_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "Tools"))
from publish_robust_rl import _json_object, _canonical_bytes
from publish_balanced_rl import _same_derived


SCHEMA = "triad.balanced_probe_input_archive.v1"
MANIFEST_SCHEMA = "triad.balanced_probe_input_manifest.v1"
AUDIT_SCHEMA = "triad.balanced_probe_portable_verification.v1"
RESULTS = "Results/balanced-v3"
MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
OBSERVATION_KEYS = {"schema", "feature_schema", "feature_names", "option_features",
                    "action_mask", "options", "state", "catalogue"}
ROLES = ("frozen_v2", "balanced_initialized", "balanced_selected")
CHECKPOINTS = ("Checkpoints/robust-v2-candidate", "Checkpoints/balanced-v3-candidate-initialized",
               "Checkpoints/balanced-v3-candidate")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact(left, right) -> bool:
    """Canonical JSON identity also distinguishes bool/int and int/float."""
    return _canonical_bytes(left) == _canonical_bytes(right)


def _bounded_read(path: Path, maximum: int) -> bytes:
    if path.stat().st_size > maximum:
        raise ValueError("Portable probe artifact exceeds bounded size")
    with path.open("rb") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Portable probe artifact exceeds bounded size")
    return data


def _relative_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Evidence path must remain repository-relative")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its repository root")
    return path


def _manifest_path(path: Path) -> Path:
    if not path.name.endswith(".json.gz"):
        raise ValueError("Portable inputs require a .json.gz output path")
    return path.with_name(path.name[:-8] + ".manifest.json")


def _frozen_evidence():
    """Read and bind the original three probe/checkpoint pairs and their sources."""
    manifest_name = f"{RESULTS}/artifact-manifest.json"
    manifest_bytes = _bounded_read(BLUE_ROOT / manifest_name, MAX_COMPRESSED_BYTES)
    publication = _json_object(manifest_bytes)
    if (publication.get("schema") != "triad.balanced_publication.v1"
            or publication.get("release") != "balanced-v3" or publication.get("stage") != "validation"
            or publication.get("independent_final_test_evidence") is not False):
        raise ValueError("Unsupported original balanced publication")
    entries = {row["path"]: row for row in publication["files"]}
    if len(entries) != len(publication["files"]):
        raise ValueError("Duplicate original publication artifact paths")
    sources = publication["source_files_sha256"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Original publication source evidence missing")
    for name, expected in sources.items():
        if _sha(_relative_file(REPO_ROOT, name).read_bytes()) != expected:
            raise ValueError("Frozen original source changed")
    descriptors = publication.get("stop_probes", [])
    if len(descriptors) != 3 or [row.get("role") for row in descriptors] != list(ROLES):
        raise ValueError("Original publication requires all three ordered probe roles")
    files, probes, interval, profile = {manifest_name: _sha(manifest_bytes)}, [], None, None
    for index, (descriptor, checkpoint_name) in enumerate(zip(descriptors, CHECKPOINTS)):
        name = f"{RESULTS}/stop-probe-{index + 1}.json"
        if descriptor.get("path") != name:
            raise ValueError("Unexpected original probe path")
        raw = _bounded_read(BLUE_ROOT / name, MAX_COMPRESSED_BYTES)
        entry = entries.get(name, {})
        if entry.get("sha256") != _sha(raw) or entry.get("bytes") != len(raw):
            raise ValueError("Original probe artifact hash or size changed")
        files[name] = _sha(raw)
        probe = _json_object(raw)
        current = probe.get("seed_provenance", {}).get("validation_diagnostic", {})
        validate_seed_range(current.get("start"), current.get("count"), "validation")
        if current["count"] != 60 or len(probe.get("records", [])) != 60:
            raise ValueError("Supplement must bind the complete original 60 observations")
        if index == 0:
            interval, profile = current, probe.get("profile")
        if (probe.get("schema") != "triad.balanced_stop_probe.v1"
                or probe.get("stage") != "validation" or probe.get("profile") != profile
                or current != interval or descriptor.get("seed_range") != interval
                or descriptor.get("profile") != profile
                or probe.get("policy_kind") != ("adaptive" if index == 0 else "balanced")
                or probe.get("weights_sha256") != descriptor.get("weights_sha256")
                or any(probe.get(key) is not False for key in (
                    "training_performed", "final_test_accessed", "independent_unseen_evidence",
                    "private_truth_accessed", "simulation_rollouts_performed"))
                or [row.get("scenario_seed") for row in probe["records"]]
                   != list(range(interval["start"], interval["start"] + 60))):
            raise ValueError("Original probe role, scope, profile or seed sequence differs")
        checkpoint_files = {}
        for filename in ("checkpoint.json", "arrays.npz"):
            artifact_name = f"{checkpoint_name}/{filename}"
            data = _bounded_read(BLUE_ROOT / artifact_name, MAX_COMPRESSED_BYTES)
            digest = _sha(data)
            if index:
                archived = entries.get(artifact_name, {})
                if archived.get("sha256") != digest or archived.get("bytes") != len(data):
                    raise ValueError("Original checkpoint alias artifact changed")
            files[artifact_name] = checkpoint_files[filename] = digest
        if (probe.get("checkpoint_files_sha256_before") != checkpoint_files
                or probe.get("checkpoint_files_sha256_after") != checkpoint_files
                or probe.get("source_sha256_before") != source_fingerprints()
                or probe.get("source_sha256_after") != source_fingerprints()):
            raise ValueError("Original probe checkpoint/source fingerprint changed")
        probes.append(probe)
    hashes = [[row["public_input_sha256"] for row in probe["records"]] for probe in probes]
    if not hashes[0] == hashes[1] == hashes[2]:
        raise ValueError("Original probes do not share exact public input identities")
    metadata = {"publication_manifest": manifest_name, "frozen_artifacts_sha256": files,
                "frozen_sources_sha256": sources,
                "verifier_source_sha256": _sha(Path(__file__).read_bytes()),
                "profile": profile, "seed_range": interval,
                "feature_dtype": "float32", "action_mask_dtype": "bool",
                "tolerance": {"relative": 1e-12, "absolute": 1e-12,
                              "exact": "stored hashes, artifact bytes, schemas, IDs, masks, actions and counts"},
                "scope": "Original public initial snapshots only; no new training, simulation rollout or independent evidence"}
    return metadata, probes


def _public_hash(observation: dict) -> str:
    return canonical_hash({"state": observation["state"], "catalogue": observation["catalogue"]})


def _encode_observation(observation: dict) -> dict:
    if (set(observation) != OBSERVATION_KEYS
            or observation["option_features"].dtype != np.float32
            or observation["action_mask"].dtype != np.bool_):
        raise ValueError("Only ordinary complete public float32/bool observations can be archived")
    return json_safe(observation)


def _decode_observation(value: dict) -> dict:
    if (not isinstance(value, dict) or set(value) != OBSERVATION_KEYS
            or value.get("schema") != FEATURE_SCHEMA or value.get("feature_schema") != FEATURE_SCHEMA
            or value.get("feature_names") != list(FEATURE_NAMES)
            or not isinstance(value.get("options"), list) or not 1 <= len(value["options"]) <= 20000):
        raise ValueError("Archived public observation schema or feature names differ")
    mask, features = value.get("action_mask"), value.get("option_features")
    rows = len(value["options"])
    if (not isinstance(mask, list) or len(mask) != rows or any(type(v) is not bool for v in mask)
            or not isinstance(features, list) or len(features) != rows
            or any(not isinstance(row, list) or len(row) != len(FEATURE_NAMES)
                   or any(type(v) is not float or not np.isfinite(v) for v in row) for row in features)):
        raise ValueError("Archived feature matrix or boolean action mask shape/type differs")
    matrix = np.asarray(features, dtype=np.float32)
    if not np.isfinite(matrix).all() or matrix.astype(np.float64).tolist() != features:
        raise ValueError("Archived feature values must be exactly representable float32")
    return {**value, "feature_names": tuple(FEATURE_NAMES), "option_features": matrix,
            "action_mask": np.asarray(mask, dtype=bool)}


def _groups(records):
    result = {"all": summarize_records(records)}
    for name, threshold in (("coverage_le_0.01", .01), ("coverage_le_0.001", .001), ("coverage_zero", 0.)):
        result[name] = summarize_records([row for row in records if row["max_legal_marginal_coverage"] <= threshold])
    return result


def _verify_observations(rows, metadata, probes):
    if not isinstance(rows, list) or len(rows) != 60:
        raise ValueError("Portable corpus must contain all 60 ordered observations")
    seed = metadata["seed_range"]["start"]
    env = RobustPlacementEnv(seed=seed, profile=metadata["profile"])
    observations = []
    for offset, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != {"seed", "public_input_sha256", "observation"}
                or type(row.get("seed")) is not int or row["seed"] != seed + offset):
            raise ValueError("Portable corpus seed sequence/row schema differs")
        observation = _decode_observation(row["observation"])
        digest = _public_hash(observation)
        if digest != row["public_input_sha256"] or any(
                digest != probe["records"][offset]["public_input_sha256"] for probe in probes):
            raise ValueError("Archived public input identity differs from the exact original probe hashes")
        runtime = _encode_observation(env.reset(seed=seed + offset))
        if not _same_derived(row["observation"], runtime):
            raise ValueError("Runtime public generator differs beyond declared float tolerance/discrete identity")
        rebuilt = _encode_observation(build_observation(observation["state"], observation["catalogue"]))
        if not _same_derived(row["observation"], rebuilt):
            raise ValueError("Archived features/options do not match their exact public state/catalogue")
        observations.append(observation)
    for index, (probe, checkpoint_name) in enumerate(zip(probes, CHECKPOINTS)):
        policy = (AdaptivePolicy if index == 0 else BalancedPolicy).load(
            BLUE_ROOT / checkpoint_name, feature_names=FEATURE_NAMES)
        weights, rng = policy.weights_fingerprint(), canonical_hash(policy.rng.bit_generator.state)
        if (probe.get("policy_schema") != policy.metadata["policy_schema"]
                or probe.get("weights_sha256") != weights or probe.get("weights_sha256_after") != weights
                or probe.get("policy_rng_sha256_before") != rng or probe.get("policy_rng_sha256_after") != rng):
            raise ValueError("Original probe policy version/weights/RNG identity differs")
        records = [first_action_record(policy, observation, seed=seed + offset)
                   for offset, observation in enumerate(observations)]
        # Recommendation coordinates/cost are copied from the exact archived
        # options/catalogue, not recomputed numerical diagnostics. Keep the
        # complete action object exact, including those copied float values.
        actions_exact = all(_exact(actual["first_action"], original["first_action"])
                            for actual, original in zip(records, probe["records"]))
        if (not actions_exact or not _same_derived(records, probe["records"])
                or not _same_derived(_groups(records), probe["groups"])):
            raise ValueError("Actual versioned actor records/groups differ from the unchanged original probes")
        if weights != policy.weights_fingerprint() or rng != canonical_hash(policy.rng.bit_generator.state):
            raise ValueError("Portable inference changed policy weights or RNG")
    return {"schema": AUDIT_SCHEMA, "verified": True, "observations": 60, "probes": 3,
            "physics_rollouts": 0, "policy_rng_unchanged": True,
            "original_public_input_hashes_exact": True, "runtime_public_inputs_within_tolerance": True}


def export_inputs(*, output: str | Path) -> dict:
    """Create two new supplement files only after exact producer-identity checks."""
    output = Path(output).resolve()
    manifest_path = _manifest_path(output)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Portable supplement output/manifest already exists; neither is overwritten")
    metadata, probes = _frozen_evidence()
    seed = metadata["seed_range"]["start"]
    env = RobustPlacementEnv(seed=seed, profile=metadata["profile"])
    rows = []
    for index in range(60):
        observation = env.reset(seed=seed + index)
        digest = _public_hash(observation)
        if any(digest != probe["records"][index]["public_input_sha256"] for probe in probes):
            raise ValueError("Producer public input hash mismatch; export requires exact original observations")
        rows.append({"seed": seed + index, "public_input_sha256": digest,
                     "observation": _encode_observation(observation)})
    audit = _verify_observations(rows, metadata, probes)
    data = gzip.compress(_canonical_bytes({"schema": SCHEMA, "metadata": metadata, "observations": rows}), mtime=0)
    if len(data) > MAX_COMPRESSED_BYTES:
        raise ValueError("Portable corpus exceeds compressed size bound")
    manifest = {"schema": MANIFEST_SCHEMA, "corpus": {"path": output.name, "sha256": _sha(data), "bytes": len(data)},
                "metadata": metadata, "audit": audit}
    manifest_data = _canonical_bytes(manifest)
    if _frozen_evidence() != (metadata, probes):
        raise ValueError("Original evidence or verifier changed during export preflight")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Portable supplement destination changed during preflight")
    for target in (output, manifest_path):
        for ancestor in target.parents:
            if ancestor.exists() and not ancestor.is_dir():
                raise ValueError("Portable supplement parent is not a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(data)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_data)
    return {**audit, "corpus_sha256": _sha(data), "manifest_sha256": _sha(manifest_data)}


def verify_inputs(path: str | Path) -> dict:
    """Verify the exact corpus/manifest, generator equivalence and frozen actors."""
    path = Path(path).resolve()
    manifest_path = _manifest_path(path)
    raw_manifest = _bounded_read(manifest_path, MAX_COMPRESSED_BYTES)
    manifest = _json_object(raw_manifest)
    if set(manifest) != {"schema", "corpus", "metadata", "audit"} or manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError("Unsupported portable corpus manifest schema")
    data = _bounded_read(path, MAX_COMPRESSED_BYTES)
    if not _exact(manifest["corpus"], {"path": path.name, "sha256": _sha(data), "bytes": len(data)}):
        raise ValueError("Portable corpus compressed bytes/hash differ from its manifest")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
            expanded = handle.read(MAX_EXPANDED_BYTES + 1)
    except (OSError, EOFError) as error:
        raise ValueError("Invalid portable corpus gzip") from error
    if len(expanded) > MAX_EXPANDED_BYTES:
        raise ValueError("Portable corpus exceeds expanded size bound")
    corpus = _json_object(expanded)
    metadata, probes = _frozen_evidence()
    if (set(corpus) != {"schema", "metadata", "observations"} or corpus["schema"] != SCHEMA
            or not _exact(corpus["metadata"], metadata) or not _exact(manifest["metadata"], metadata)):
        raise ValueError("Portable corpus binding to original artifacts/sources/shape differs")
    audit = _verify_observations(corpus["observations"], metadata, probes)
    if not _exact(manifest["audit"], audit):
        raise ValueError("Portable verification audit differs from recomputation")
    if (_frozen_evidence() != (metadata, probes)
            or _bounded_read(path, MAX_COMPRESSED_BYTES) != data
            or _bounded_read(manifest_path, MAX_COMPRESSED_BYTES) != raw_manifest):
        raise ValueError("Evidence, source or portable artifact changed during verification")
    return {**audit, "corpus_sha256": _sha(data), "manifest_sha256": _sha(raw_manifest)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("export").add_argument("--output", required=True, type=Path)
    command.add_parser("verify").add_argument("path", type=Path)
    args = parser.parse_args(argv)
    result = export_inputs(output=args.output) if args.command == "export" else verify_inputs(args.path)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
