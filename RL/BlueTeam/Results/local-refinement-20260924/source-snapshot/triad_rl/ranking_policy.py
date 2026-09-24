"""Greedy-prior residual deployment/STOP ranker; no simulator or value critic.

At zero initialization the residual is exactly zero, including STOP, so joint
legal argmax is the frozen public greedy rule. Training compares local slate
rows using outcome preferences supplied by a separate, training-only collector.
The policy never reads those outcomes or private observation fields at inference.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .adaptive_inputs import FEATURE_NAMES
from .adaptive_policy import FEATURE_SCHEMA, _feature_names, _json


POLICY_SCHEMA = "triad.greedy_residual_rank_policy.v1"
CHECKPOINT_SCHEMA = "triad.ranking_checkpoint.v1"
PARAMETERS = ("w1", "b1", "wa")
PRIOR_WEIGHTS = (("marginal_coverage", 3.), ("marginal_early_coverage", 1.),
                 ("cost", -.25), ("overlap", -.15))
POLICY_CONTRACT = {
    "prior": {"coefficients": dict(PRIOR_WEIGHTS), "summation_order": [name for name, _ in PRIOR_WEIGHTS],
              "cost": "public feature actual_cost/4", "stop_prior": 0.},
    "residual": "shared tanh(features@w1+b1)@wa; zero-initialized wa; no critic",
    "scores": "prior+residual for every row, including STOP",
    "deterministic_rule": "joint-argmax-over-legal-rows;first-row-ties",
    "stochastic_rule": "softmax-legal-scores/temperature;default-temperature=0.1",
    "stop_contract": "exactly-one-binary-STOP-row;may-be-masked;at-least-one-legal-action",
    "comparison_loss": "sum(weight*softplus(-(preferred_score-rejected_score)/temperature))/sum(weight)",
    "tether": "regularization*mean(residual**2)-over-all-supplied-slate-rows",
    "optimizer": {"name": "Adam", "beta1": .9, "beta2": .999, "epsilon": 1e-8,
                  "clipping": "one-global-L2-gradient-scale-before-moments"},
}
MAX_ROWS = 16385  # 32 catalogue entries * 512 public sites, plus STOP.
MAX_STATES = 1024
MAX_COMPARISONS = 65536
MAX_FEATURE_ELEMENTS = 4_000_000


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} cannot be represented as a finite float") from error
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _names(value):
    try:
        return _feature_names(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid ordered feature names") from error


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate checkpoint JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        _json(value)  # Also rejects finite-looking literals that overflow, e.g. 1e999.
        return value
    except (UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("Invalid checkpoint JSON") from error


def _rng_state(value):
    if (not isinstance(value, dict) or set(value) != {"bit_generator", "state", "has_uint32", "uinteger"}
            or value["bit_generator"] != "PCG64" or not isinstance(value["state"], dict)
            or set(value["state"]) != {"state", "inc"}):
        raise ValueError("Invalid PCG64 checkpoint RNG state")
    for key in ("state", "inc"):
        _integer(value["state"][key], "RNG " + key, 0, 2**128 - 1)
    if value["state"]["inc"] % 2 != 1:
        raise ValueError("PCG64 increment must be odd")
    _integer(value["has_uint32"], "RNG has_uint32", 0, 1)
    _integer(value["uinteger"], "RNG uinteger", 0, 2**32 - 1)
    generator = np.random.PCG64(0)
    generator.state = value
    if _json(generator.state) != _json(value):
        raise ValueError("RNG state must round-trip exactly")
    return generator.state


class RankPolicy:
    """Permutation-equivariant public option ranker with exact greedy initialization."""

    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES, seed: int = 0,
                 hidden_size: int = 64):
        self.feature_names = _names(feature_names)
        if not {"stop", *(name for name, _ in PRIOR_WEIGHTS)} <= set(self.feature_names):
            raise ValueError("Ranking policy requires named STOP and all greedy prior features")
        self.hidden_size = _integer(hidden_size, "hidden_size", 1, 512)
        seed = _integer(seed, "seed", 0, 2**63 - 1)
        self.stop_index = self.feature_names.index("stop")
        self.rng = np.random.default_rng(seed)
        self.parameters = {"w1": self.rng.normal(0., 1 / np.sqrt(len(self.feature_names)),
                                                 (len(self.feature_names), self.hidden_size)),
                           "b1": np.zeros(self.hidden_size), "wa": np.zeros(self.hidden_size)}
        self.adam_m = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.adam_v = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.update_count = 0
        self.training_state: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}

    def _features(self, value, *, complete):
        try:
            features = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Invalid numeric feature matrix") from error
        if (features.ndim != 2 or features.shape[1] != len(self.feature_names)
                or not 1 <= len(features) <= MAX_ROWS or features.size > MAX_FEATURE_ELEMENTS
                or not np.isfinite(features).all()):
            raise ValueError("Features must be a bounded finite [rows,features] matrix")
        stops = features[:, self.stop_index]
        if not np.isin(stops, (0., 1.)).all() or np.count_nonzero(stops) > 1 or (complete and np.count_nonzero(stops) != 1):
            raise ValueError("Observation needs exactly one binary STOP; a slate may omit it")
        return features

    def _input(self, observation):
        if not isinstance(observation, Mapping):
            raise ValueError("Observation must be a mapping")
        if observation.get("feature_schema", observation.get("schema", FEATURE_SCHEMA)) != FEATURE_SCHEMA:
            raise ValueError("Observation feature semantics differ")
        if _names(observation.get("feature_names", ())) != self.feature_names:
            raise ValueError("Observation feature names/order differ")
        features = self._features(observation.get("option_features"), complete=True)
        mask = np.asarray(observation.get("action_mask"))
        if mask.dtype != np.bool_ or mask.shape != (len(features),) or not mask.any():
            raise ValueError("Mask must be boolean with at least one legal action")
        return features, mask

    def _prior(self, features):
        # Keep the frozen GreedyPublicCoverage operation order, not a dot product:
        # different floating reduction order could break exact initialized ties.
        prior = sum(weight * features[:, self.feature_names.index(name)] for name, weight in PRIOR_WEIGHTS)
        prior[features[:, self.stop_index] == 1.] = 0.
        return prior

    def _forward(self, features):
        with np.errstate(over="ignore", invalid="ignore"):
            hidden = np.tanh(features @ self.parameters["w1"] + self.parameters["b1"])
            residual = hidden @ self.parameters["wa"]
            scores = self._prior(features) + residual
        if not np.isfinite(hidden).all() or not np.isfinite(scores).all() or not np.isfinite(residual).all():
            raise ValueError("Nonfinite ranking forward pass")
        return hidden, residual, scores

    def scores(self, observation):
        """Return raw scores for all rows; callers must still honor action_mask."""
        features, _ = self._input(observation)
        return self._forward(features)[2]

    def probabilities(self, observation, temperature=.1):
        temperature = _number(temperature, "temperature", positive=True)
        features, mask = self._input(observation)
        scores = self._forward(features)[2]
        with np.errstate(over="ignore", under="ignore"):
            exp = np.exp((scores[mask] - np.max(scores[mask])) / temperature)
        probability = np.zeros(len(scores))
        probability[mask] = exp / exp.sum()
        return probability

    def act(self, observation, deterministic=True, temperature=.1):
        if not isinstance(deterministic, (bool, np.bool_)):
            raise ValueError("deterministic must be boolean")
        if deterministic:
            features, mask = self._input(observation)
            legal = np.flatnonzero(mask)
            return int(legal[np.argmax(self._forward(features)[2][legal])])
        probability = self.probabilities(observation, temperature)
        return int(self.rng.choice(len(probability), p=probability))

    def _records(self, state_records):
        if (not isinstance(state_records, Sequence) or isinstance(state_records, (str, bytes))
                or not 1 <= len(state_records) <= MAX_STATES):
            raise ValueError("Need a bounded nonempty sequence of comparison states")
        records, total_rows, total_pairs = [], 0, 0
        for state in state_records:
            if not isinstance(state, Mapping) or set(state) != {"features", "comparisons"}:
                raise ValueError("Comparison state must contain features and comparisons only")
            features = self._features(state["features"], complete=False)
            comparisons = state["comparisons"]
            if (not isinstance(comparisons, Sequence) or isinstance(comparisons, (str, bytes))
                    or not comparisons or len(comparisons) > MAX_COMPARISONS):
                raise ValueError("Each state requires a bounded nonempty comparison list; skip all-tie states")
            rows, seen = [], set()
            for pair in comparisons:
                if not isinstance(pair, Mapping) or set(pair) != {"preferred", "rejected", "weight"}:
                    raise ValueError("Invalid comparison fields")
                preferred = _integer(pair["preferred"], "preferred", 0, len(features) - 1)
                rejected = _integer(pair["rejected"], "rejected", 0, len(features) - 1)
                identity = tuple(sorted((preferred, rejected)))
                if preferred == rejected or identity in seen:
                    raise ValueError("Comparisons must be unique non-self unordered pairs")
                seen.add(identity)
                rows.append((preferred, rejected, _number(pair["weight"], "weight", positive=True)))
            total_rows += len(features)
            total_pairs += len(rows)
            if total_rows * len(self.feature_names) > MAX_FEATURE_ELEMENTS or total_pairs > MAX_COMPARISONS:
                raise ValueError("Comparison batch exceeds bounded size")
            records.append((features, rows))
        return records, total_rows, total_pairs

    def _loss_and_gradients(self, state_records, temperature=.1, regularization=.01):
        """Exact analytic gradients of weighted pairwise loss plus residual tether."""
        temperature = _number(temperature, "temperature", positive=True)
        regularization = _number(regularization, "regularization")
        records, total_rows, total_pairs = self._records(state_records)
        total_weight = sum(weight for _, pairs in records for _, _, weight in pairs)
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Total comparison weight is not finite and positive")
        gradients = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        pair_loss = residual_square = margin_sum = correct_weight = 0.
        for features, comparisons in records:
            hidden, residual, scores = self._forward(features)
            residual_square += float(residual @ residual) / total_rows
            d_scores = 2 * regularization * residual / total_rows
            for preferred, rejected, weight in comparisons:
                margin = float((scores[preferred] - scores[rejected]) / temperature)
                if not np.isfinite(margin):
                    raise ValueError("Nonfinite comparison margin")
                fraction = weight / total_weight
                pair_loss += fraction * float(np.logaddexp(0., -margin))
                margin_sum += fraction * margin
                correct_weight += fraction * (scores[preferred] > scores[rejected])
                derivative = -fraction * np.exp(-np.logaddexp(0., margin)) / temperature
                d_scores[preferred] += derivative
                d_scores[rejected] -= derivative
            gradients["wa"] += hidden.T @ d_scores
            d_z = d_scores[:, None] * self.parameters["wa"][None, :] * (1 - hidden * hidden)
            gradients["w1"] += features.T @ d_z
            gradients["b1"] += d_z.sum(axis=0)
        loss = pair_loss + regularization * residual_square
        if not np.isfinite(loss) or not all(np.isfinite(value).all() for value in gradients.values()):
            raise ValueError("Nonfinite ranking loss/gradients")
        return float(loss), gradients, {"pairwise_loss": float(pair_loss), "residual_mean_square": float(residual_square),
            "regularization_loss": float(regularization * residual_square), "weighted_accuracy": float(correct_weight),
            "mean_scaled_margin": float(margin_sum), "comparison_weight": float(total_weight),
            "states": len(records), "slate_rows": total_rows, "comparisons": total_pairs}

    def update_comparisons(self, state_records, learning_rate=.002, temperature=.1,
                           regularization=.01, max_grad_norm=1.):
        learning_rate = _number(learning_rate, "learning_rate", positive=True)
        max_grad_norm = _number(max_grad_norm, "max_grad_norm", positive=True)
        self._arrays()
        loss, gradients, metrics = self._loss_and_gradients(state_records, temperature, regularization)
        norm = float(np.sqrt(sum(np.sum(value * value) for value in gradients.values())))
        if not np.isfinite(norm):
            raise ValueError("Nonfinite ranking gradient norm; update not applied")
        scale = min(1., max_grad_norm / max(norm, 1e-12))
        count = self.update_count + 1
        parameters, moments, variances = {}, {}, {}
        for key in PARAMETERS:
            gradient = gradients[key] * scale
            moments[key] = .9 * self.adam_m[key] + .1 * gradient
            variances[key] = .999 * self.adam_v[key] + .001 * gradient * gradient
            mean = moments[key] / (1 - .9**count)
            variance = variances[key] / (1 - .999**count)
            parameters[key] = self.parameters[key] - learning_rate * mean / (np.sqrt(variance) + 1e-8)
        if not all(np.isfinite(value).all() for group in (parameters, moments, variances) for value in group.values()):
            raise ValueError("Nonfinite Adam result; update not applied")
        self.parameters, self.adam_m, self.adam_v = parameters, moments, variances
        self.update_count = count
        return {**metrics, "loss": loss, "gradient_norm": norm, "gradient_scale": float(scale),
                "clipped_gradient_norm": float(norm * scale), "updates": count}

    def _arrays(self):
        shapes = {"w1": (len(self.feature_names), self.hidden_size), "b1": (self.hidden_size,), "wa": (self.hidden_size,)}
        arrays = {}
        for prefix, group in (("param", self.parameters), ("adam_m", self.adam_m), ("adam_v", self.adam_v)):
            if set(group) != set(PARAMETERS):
                raise ValueError("Unexpected ranking parameter or optimizer names")
            for key, shape in shapes.items():
                value = group[key]
                if (not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.float64
                        or not np.isfinite(value).all() or (prefix == "adam_v" and (value < 0).any())):
                    raise ValueError("Invalid ranking parameter or optimizer array")
                arrays[f"{prefix}_{key}"] = value
        _integer(self.update_count, "update_count", 0, 2**63 - 1)
        return arrays

    def weights_fingerprint(self):
        digest = hashlib.sha256(_json({"schema": POLICY_SCHEMA, "feature_schema": FEATURE_SCHEMA,
            "policy_contract": POLICY_CONTRACT, "features": self.feature_names, "hidden_size": self.hidden_size}).encode())
        for key in PARAMETERS:
            digest.update(key.encode())
            digest.update(np.asarray(self.parameters[key], dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path, training_state=None):
        """Save JSON and non-pickled NPZ; validate all content before any write."""
        arrays = self._arrays()
        state = self.training_state if training_state is None else training_state
        if not isinstance(state, dict) or not isinstance(state.get("seed_provenance", {}), dict):
            raise ValueError("Training state and seed provenance must be dictionaries")
        state = _strict_json(_json(state))
        metadata = {"schema": CHECKPOINT_SCHEMA, "policy_schema": POLICY_SCHEMA, "policy_contract": POLICY_CONTRACT,
                    "feature_schema": FEATURE_SCHEMA, "feature_names": list(self.feature_names), "hidden_size": self.hidden_size,
                    "update_count": self.update_count, "rng_state": _rng_state(self.rng.bit_generator.state),
                    "weights_sha256": self.weights_fingerprint(), "training_state": state,
                    "seed_provenance": state.get("seed_provenance", {})}
        metadata = _strict_json(_json(metadata))  # Do not alias the module's policy contract into mutable metadata.
        archive = io.BytesIO()
        np.savez_compressed(archive, **arrays)
        blob = archive.getvalue()
        metadata["arrays_sha256"] = hashlib.sha256(blob).hexdigest()
        raw = (_json(metadata) + "\n").encode()
        if len(raw) > 4_000_000 or len(blob) > 64_000_000:
            raise ValueError("Checkpoint exceeds bounded size")
        path = Path(path)
        names = ("arrays.npz", "checkpoint.json", "arrays.tmp.npz", "checkpoint.tmp.json")
        if path.is_symlink() or any((path / name).is_symlink() for name in names):
            raise ValueError("Checkpoint targets must not be symlinks")
        if any((path / name).exists() and not (path / name).is_file() for name in names[:2]):
            raise ValueError("Existing checkpoint targets must be files")
        if any((path / name).exists() for name in names[2:]):
            raise ValueError("Checkpoint temporary files already exist")
        path.mkdir(parents=True, exist_ok=True)
        temporary_arrays, temporary_metadata = path / names[2], path / names[3]
        try:
            temporary_arrays.write_bytes(blob)
            temporary_metadata.write_bytes(raw)
            temporary_arrays.replace(path / names[0])
            temporary_metadata.replace(path / names[1])
        finally:
            for temporary in (temporary_arrays, temporary_metadata):
                if temporary.exists():
                    temporary.unlink()
        self.training_state, self.metadata = state, metadata

    @classmethod
    def load(cls, path, feature_names=None):
        path = Path(path)
        with (path / "checkpoint.json").open("rb") as handle:
            raw = handle.read(4_000_001)
        with (path / "arrays.npz").open("rb") as handle:
            blob = handle.read(64_000_001)
        if len(raw) > 4_000_000 or len(blob) > 64_000_000:
            raise ValueError("Checkpoint exceeds bounded size")
        metadata = _strict_json(raw)
        required = {"schema", "policy_schema", "policy_contract", "feature_schema", "feature_names", "hidden_size",
                    "update_count", "rng_state", "weights_sha256", "training_state", "seed_provenance", "arrays_sha256"}
        if (not isinstance(metadata, dict) or set(metadata) != required or metadata["schema"] != CHECKPOINT_SCHEMA
                or metadata["policy_schema"] != POLICY_SCHEMA or metadata["feature_schema"] != FEATURE_SCHEMA
                or _json(metadata["policy_contract"]) != _json(POLICY_CONTRACT)):
            raise ValueError("Unsupported ranking checkpoint schema or contract")
        names = _names(metadata["feature_names"])
        if feature_names is not None and tuple(feature_names) != names:
            raise ValueError("Checkpoint feature names/order differ")
        if hashlib.sha256(blob).hexdigest() != metadata["arrays_sha256"]:
            raise ValueError("Checkpoint array fingerprint mismatch")
        policy = cls(names, hidden_size=metadata["hidden_size"])
        expected = {key: value.shape for key, value in policy._arrays().items()}
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                members = archive.infolist()
                if len(members) != len(expected) or {m.filename for m in members} != {key + ".npy" for key in expected}:
                    raise ValueError("Unexpected checkpoint archive members")
                if any(m.file_size > int(np.prod(expected[m.filename[:-4]])) * 8 + 1024 for m in members):
                    raise ValueError("Checkpoint array exceeds expected expanded size")
            with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
                loaded = {key: arrays[key].copy() for key in expected}
        except (zipfile.BadZipFile, KeyError, OSError) as error:
            raise ValueError("Invalid checkpoint NPZ archive") from error
        for key in PARAMETERS:
            policy.parameters[key], policy.adam_m[key], policy.adam_v[key] = (
                loaded[f"{prefix}_{key}"] for prefix in ("param", "adam_m", "adam_v"))
        policy.update_count = _integer(metadata["update_count"], "update_count", 0, 2**63 - 1)
        policy._arrays()
        if policy.weights_fingerprint() != metadata["weights_sha256"]:
            raise ValueError("Checkpoint weights fingerprint mismatch")
        policy.rng.bit_generator.state = _rng_state(metadata["rng_state"])
        state = metadata["training_state"]
        if (not isinstance(state, dict) or not isinstance(metadata["seed_provenance"], dict)
                or _json(metadata["seed_provenance"]) != _json(state.get("seed_provenance", {}))):
            raise ValueError("Invalid checkpoint training state or seed provenance")
        policy.training_state, policy.metadata = state, metadata
        return policy


RankingPolicy = RankPolicy
