"""Dependency-light dynamic sensor-placement policy.

The policy applies one shared scorer to every catalogue row. Catalogue length
therefore never appears in a parameter shape, so one checkpoint can be used
with a different number of feature-compatible sensor options. A small
tanh-Gaussian head chooses normalized East/North controls for options which
allow dynamic positioning.

Checkpoint files contain only JSON and NumPy arrays loaded with
``allow_pickle=False``.  They deliberately bind the feature layout, but not a
particular catalogue length or list of candidate IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_SCHEMA = "triad.dynamic_placement_features.v1"
POLICY_SCHEMA = "triad.dynamic_placement_policy.v1"
CHECKPOINT_SCHEMA = "triad.dynamic_placement_checkpoint.v1"
_ARRAYS_FILENAME = "arrays.npz"
_METADATA_FILENAME = "checkpoint.json"
_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert NumPy RNG state and metadata to strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _columns(field: Mapping[str, Any], fallback: Sequence[str] = ()) -> tuple[str, ...]:
    values = field.get("columns", fallback)
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class PlacementFeatures:
    """One policy input with a variable number of catalogue options."""

    option_features: np.ndarray
    action_mask: np.ndarray
    dynamic_position_mask: np.ndarray
    feature_contract: Mapping[str, Any]

    @property
    def option_count(self) -> int:
        """Number of real catalogue rows, excluding the final stop row."""
        return int(self.option_features.shape[0] - 1)


class PlacementFeatureAdapter:
    """Decode Blue observations using ``env.observation_fields``.

    Candidate rows retain every mount value, including modality bits when the
    environment publishes them.  Variable-row placement state is pooled rather
    than flattened, keeping the model shape independent of catalogue/site count.
    """

    _DYNAMIC_POSITION_ALIASES = {
        "allowdynamicposition",
        "ballowdynamicposition",
        "dynamicposition",
        "isdynamicposition",
    }

    def __init__(self, env: Any, agent: str = "blue_placement") -> None:
        self.agent = agent
        try:
            source_fields = env.observation_fields[agent]
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError(f"Environment does not publish observation_fields for {agent}") from error
        self.fields = {
            str(name): {
                **dict(field),
                "shape": [int(value) for value in field["shape"]],
            }
            for name, field in source_fields.items()
        }
        if "mounts" not in self.fields:
            raise ValueError("Blue observation_fields must contain a mounts field")
        mount_shape = self.fields["mounts"]["shape"]
        if len(mount_shape) != 2 or mount_shape[0] < 1 or mount_shape[1] < 1:
            raise ValueError("The mounts field must be a non-empty [catalogue, features] matrix")
        self.catalogue_size_at_creation = mount_shape[0]
        self.mount_width = mount_shape[1]
        fallback_names = getattr(env, "mount_feature_names", ())
        self.mount_columns = _columns(self.fields["mounts"], fallback_names)
        if self.mount_columns and len(self.mount_columns) != self.mount_width:
            raise ValueError("Mount feature names do not match the mounts row width")

        placement_shape = self.fields.get("placements", {}).get("shape", [])
        self._aligned_placement = (
            len(placement_shape) == 1
            and placement_shape[0] == self.catalogue_size_at_creation
        )
        if not placement_shape:
            self.placement_width = 0
        elif len(placement_shape) == 1:
            self.placement_width = 1
        else:
            self.placement_width = int(np.prod(placement_shape[1:]))

        self.fixed_fields: list[tuple[str, int, tuple[int, ...]]] = []
        for name, field in self.fields.items():
            if name in {"mounts", "placements"}:
                continue
            shape = tuple(field["shape"])
            self.fixed_fields.append((name, int(np.prod(shape)), shape))

        fixed_width = sum(size for _, size, _ in self.fixed_fields)
        aligned_width = self.placement_width if self._aligned_placement else 0
        # local mount + optional aligned placement + stop bit; pooled mount
        # mean/max; pooled placement mean/max; fixed globals; two summaries.
        self.feature_size = (
            self.mount_width
            + aligned_width
            + 1
            + 2 * self.mount_width
            + 2 * self.placement_width
            + fixed_width
            + 2
        )
        self.feature_contract = {
            "schema": FEATURE_SCHEMA,
            "observation_schema": str(getattr(env, "observation_schema", "unknown")),
            "agent": agent,
            "mount_width": self.mount_width,
            "mount_columns": list(self.mount_columns),
            "placement_mode": "candidate_scalar" if self._aligned_placement else "pooled_rows",
            "placement_width": self.placement_width,
            "placement_columns": list(_columns(self.fields.get("placements", {}))),
            "fixed_fields": [
                {"name": name, "size": size, "shape": list(shape)}
                for name, size, shape in self.fixed_fields
            ],
            "pooling": "mean_and_max_v1",
            "feature_size": self.feature_size,
            "stop_is_final_mask_entry": True,
        }

        normalized = [_normalized_name(name) for name in self.mount_columns]
        matches = [
            index for index, name in enumerate(normalized)
            if name in self._DYNAMIC_POSITION_ALIASES
        ]
        if len(matches) > 1:
            raise ValueError("Mount layout contains duplicate dynamic-position features")
        self.dynamic_position_column = matches[0] if matches else None

    def _decode(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if not isinstance(observation, Mapping) or "observation" not in observation:
            raise ValueError("Blue observation must contain an observation vector")
        vector = np.asarray(observation["observation"], dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Blue observation vector must be one-dimensional and finite")
        values: dict[str, np.ndarray] = {}
        for name, field in self.fields.items():
            start, stop = int(field["start"]), int(field["stop"])
            shape = tuple(int(value) for value in field["shape"])
            if start < 0 or stop < start or stop > vector.size or stop - start != int(np.prod(shape)):
                raise ValueError(f"Invalid observation_fields entry for {name}")
            values[name] = vector[start:stop].reshape(shape)
        return values

    @staticmethod
    def _pool_rows(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
        if width == 0:
            empty = np.empty((0,), dtype=np.float64)
            return empty, empty
        rows = np.asarray(values, dtype=np.float64).reshape(-1, width)
        if rows.shape[0] == 0:
            zeros = np.zeros(width, dtype=np.float64)
            return zeros, zeros.copy()
        return rows.mean(axis=0), rows.max(axis=0)

    def extract(self, observation: Mapping[str, Any]) -> PlacementFeatures:
        values = self._decode(observation)
        mounts = np.asarray(values["mounts"], dtype=np.float64)
        option_count = mounts.shape[0]
        if mounts.shape[1] != self.mount_width or not np.isfinite(mounts).all():
            raise ValueError("Mount rows do not match the policy feature contract")

        try:
            mask = np.asarray(observation["action_mask"], dtype=np.bool_)
        except KeyError as error:
            raise ValueError("Blue observation must contain an action_mask") from error
        if mask.ndim != 1 or mask.size != option_count + 1:
            raise ValueError("Blue action_mask must contain one entry per mount plus stop")
        if not mask.any():
            raise ValueError("Cannot select from an empty Blue action mask")

        aligned = np.empty((option_count, 0), dtype=np.float64)
        placement_mean = np.empty((0,), dtype=np.float64)
        placement_max = np.empty((0,), dtype=np.float64)
        if "placements" in values:
            placements = np.asarray(values["placements"], dtype=np.float64)
            if not np.isfinite(placements).all():
                raise ValueError("Placement state contains non-finite values")
            if self._aligned_placement:
                if placements.shape != (option_count,):
                    raise ValueError("Candidate-aligned placement state changed shape")
                aligned = placements.reshape(option_count, 1)
            placement_mean, placement_max = self._pool_rows(placements, self.placement_width)

        fixed_parts: list[np.ndarray] = []
        for name, size, shape in self.fixed_fields:
            part = np.asarray(values[name], dtype=np.float64)
            if part.shape != shape or part.size != size or not np.isfinite(part).all():
                raise ValueError(f"Global observation field {name} changed shape or is non-finite")
            fixed_parts.append(part.ravel())

        mount_mean = mounts.mean(axis=0)
        mount_max = mounts.max(axis=0)
        available_fraction = float(mask[:-1].mean()) if option_count else 0.0
        global_features = np.concatenate([
            mount_mean,
            mount_max,
            placement_mean,
            placement_max,
            *fixed_parts,
            np.asarray([np.log1p(option_count), available_fraction], dtype=np.float64),
        ])
        local = np.concatenate([
            mounts,
            aligned,
            np.zeros((option_count, 1), dtype=np.float64),
        ], axis=1)
        stop_local = np.concatenate([
            np.zeros(self.mount_width + aligned.shape[1], dtype=np.float64),
            np.ones(1, dtype=np.float64),
        ])[None, :]
        local = np.concatenate([local, stop_local], axis=0)
        option_features = np.concatenate([
            local,
            np.repeat(global_features[None, :], option_count + 1, axis=0),
        ], axis=1)
        if option_features.shape != (option_count + 1, self.feature_size):
            raise ValueError("Internal placement feature size disagrees with its contract")
        if not np.isfinite(option_features).all():
            raise ValueError("Placement policy features are not finite")

        dynamic_mask = np.zeros(option_count + 1, dtype=np.bool_)
        if self.dynamic_position_column is None:
            dynamic_mask[:-1] = True
        else:
            dynamic_mask[:-1] = mounts[:, self.dynamic_position_column] > 0.5
        dynamic_mask &= mask
        dynamic_mask[-1] = False
        return PlacementFeatures(
            option_features=option_features,
            action_mask=mask,
            dynamic_position_mask=dynamic_mask,
            feature_contract=self.feature_contract,
        )


@dataclass(frozen=True)
class PlacementSample:
    """A sampled hybrid action plus sufficient statistics for REINFORCE."""

    catalogue_index: int
    position: np.ndarray
    position_latent: np.ndarray
    position_active: bool
    option_features: np.ndarray
    action_mask: np.ndarray
    log_probability: float
    entropy: float

    @property
    def stop(self) -> bool:
        return self.catalogue_index == self.option_features.shape[0] - 1

    def environment_action(self) -> dict[str, Any]:
        """Return the exact hybrid Dict expected by ``TRIADRedBlueEnv``."""
        return {
            "catalogue_index": int(self.catalogue_index),
            "position": np.asarray(self.position, dtype=np.float32).copy(),
        }


class DynamicPlacementPolicy:
    """Shared masked scorer and conditional 2-D tanh-Gaussian position head."""

    _PARAMETER_NAMES = ("input_weights", "input_bias", "score_weights", "score_bias",
                        "position_weights", "position_bias", "position_log_std")

    def __init__(
        self,
        feature_contract: Mapping[str, Any],
        *,
        hidden_size: int = 32,
        seed: int = 0,
    ) -> None:
        contract = json.loads(_canonical_json(feature_contract))
        if contract.get("schema") != FEATURE_SCHEMA:
            raise ValueError("Unsupported dynamic-placement feature schema")
        feature_size = int(contract.get("feature_size", 0))
        if feature_size < 1 or not 1 <= int(hidden_size) <= 1024:
            raise ValueError("Invalid placement policy dimensions")
        self.feature_contract = contract
        self.feature_size = feature_size
        self.hidden_size = int(hidden_size)
        self.rng = np.random.default_rng(int(seed))

        input_scale = np.sqrt(2.0 / (self.feature_size + self.hidden_size))
        self.input_weights = self.rng.normal(
            0.0, input_scale, (self.feature_size, self.hidden_size)
        ).astype(np.float64)
        self.input_bias = np.zeros(self.hidden_size, dtype=np.float64)
        self.score_weights = self.rng.normal(
            0.0, 0.01 / np.sqrt(self.hidden_size), self.hidden_size
        ).astype(np.float64)
        self.score_bias = np.zeros(1, dtype=np.float64)
        self.position_weights = self.rng.normal(
            0.0, 0.01 / np.sqrt(self.hidden_size), (self.hidden_size, 2)
        ).astype(np.float64)
        self.position_bias = np.zeros(2, dtype=np.float64)
        self.position_log_std = np.full(2, -0.5, dtype=np.float64)

        self._adam_mean = {
            name: np.zeros_like(getattr(self, name)) for name in self._PARAMETER_NAMES
        }
        self._adam_variance = {
            name: np.zeros_like(getattr(self, name)) for name in self._PARAMETER_NAMES
        }
        self.adam_step = 0
        self.update_count = 0
        self.return_count = 0
        self.return_mean = 0.0
        self.return_m2 = 0.0

    @classmethod
    def from_adapter(
        cls, adapter: PlacementFeatureAdapter, *, hidden_size: int = 32, seed: int = 0
    ) -> "DynamicPlacementPolicy":
        return cls(adapter.feature_contract, hidden_size=hidden_size, seed=seed)

    @property
    def specification(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SCHEMA,
            "feature_size": self.feature_size,
            "hidden_size": self.hidden_size,
            "position_dimensions": 2,
            "scorer": "shared_tanh_mlp",
            "position_distribution": "tanh_gaussian",
        }

    def _forward(self, option_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(option_features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.feature_size:
            raise ValueError("Option features do not match this policy")
        if not np.isfinite(features).all():
            raise ValueError("Option features contain non-finite values")
        hidden = np.tanh(features @ self.input_weights + self.input_bias)
        logits = hidden @ self.score_weights + float(self.score_bias[0])
        return hidden, logits

    @staticmethod
    def _probabilities(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        action_mask = np.asarray(mask, dtype=np.bool_)
        if action_mask.shape != logits.shape or not action_mask.any():
            raise ValueError("Action mask is empty or has the wrong shape")
        probabilities = np.zeros_like(logits, dtype=np.float64)
        valid_logits = logits[action_mask]
        shifted = valid_logits - valid_logits.max()
        weights = np.exp(shifted)
        probabilities[action_mask] = weights / weights.sum()
        return probabilities

    @staticmethod
    def _position_log_probability(
        latent: np.ndarray, mean: np.ndarray, log_std: np.ndarray
    ) -> float:
        delta = (latent - mean) / np.exp(log_std)
        normal = -0.5 * np.sum(delta * delta + 2.0 * log_std + _LOG_TWO_PI)
        # Stable log(1 - tanh(z)^2).
        jacobian = np.sum(2.0 * (np.log(2.0) - latent - np.logaddexp(0.0, -2.0 * latent)))
        return float(normal - jacobian)

    def act(self, features: PlacementFeatures, *, deterministic: bool = False) -> PlacementSample:
        if _sha256_json(features.feature_contract) != _sha256_json(self.feature_contract):
            raise ValueError("Placement features use a different checkpoint contract")
        hidden, logits = self._forward(features.option_features)
        probabilities = self._probabilities(logits, features.action_mask)
        if deterministic:
            catalogue_index = int(np.argmax(np.where(features.action_mask, logits, -np.inf)))
        else:
            catalogue_index = int(self.rng.choice(probabilities.size, p=probabilities))

        position_active = bool(features.dynamic_position_mask[catalogue_index])
        if position_active:
            mean = hidden[catalogue_index] @ self.position_weights + self.position_bias
            if deterministic:
                latent = mean
            else:
                latent = self.rng.normal(mean, np.exp(self.position_log_std))
            position = np.tanh(latent).astype(np.float32)
            position_log_probability = self._position_log_probability(
                latent, mean, self.position_log_std
            )
            position_entropy = float(
                np.sum(self.position_log_std + 0.5 * (1.0 + _LOG_TWO_PI))
            )
        else:
            latent = np.zeros(2, dtype=np.float64)
            position = np.zeros(2, dtype=np.float32)
            position_log_probability = 0.0
            position_entropy = 0.0
        selected_probability = max(float(probabilities[catalogue_index]), np.finfo(float).tiny)
        categorical_entropy = float(
            -np.sum(probabilities[features.action_mask]
                    * np.log(probabilities[features.action_mask]))
        )
        return PlacementSample(
            catalogue_index=catalogue_index,
            position=position,
            position_latent=np.asarray(latent, dtype=np.float64).copy(),
            position_active=position_active,
            option_features=np.asarray(features.option_features, dtype=np.float64).copy(),
            action_mask=np.asarray(features.action_mask, dtype=np.bool_).copy(),
            log_probability=float(np.log(selected_probability) + position_log_probability),
            entropy=categorical_entropy + position_entropy,
        )

    def deterministic_action(self, features: PlacementFeatures) -> PlacementSample:
        return self.act(features, deterministic=True)

    @property
    def return_standard_deviation(self) -> float:
        if self.return_count < 2:
            return 1.0
        return max(float(np.sqrt(self.return_m2 / (self.return_count - 1))), 1.0)

    def advantages(self, returns: Sequence[float]) -> np.ndarray:
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Returns must be a finite one-dimensional array")
        baseline = self.return_mean if self.return_count else 0.0
        return (values - baseline) / self.return_standard_deviation

    def observe_episode_returns(self, returns: Sequence[float]) -> None:
        for value in np.asarray(returns, dtype=np.float64):
            if not np.isfinite(value):
                raise ValueError("Episode return must be finite")
            self.return_count += 1
            delta = float(value) - self.return_mean
            self.return_mean += delta / self.return_count
            self.return_m2 += delta * (float(value) - self.return_mean)

    def update(
        self,
        samples: Sequence[PlacementSample],
        advantages: Sequence[float],
        *,
        learning_rate: float = 3e-4,
        entropy_coefficient: float = 0.01,
        max_gradient_norm: float = 5.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> dict[str, float | int]:
        if not samples:
            raise ValueError("At least one placement sample is required")
        advantage_values = np.asarray(advantages, dtype=np.float64)
        if advantage_values.shape != (len(samples),) or not np.isfinite(advantage_values).all():
            raise ValueError("Advantages must contain one finite value per sample")
        if not 0.0 < learning_rate <= 1.0 or entropy_coefficient < 0.0:
            raise ValueError("Invalid placement-policy optimizer settings")

        gradients = {
            name: np.zeros_like(getattr(self, name)) for name in self._PARAMETER_NAMES
        }
        objectives: list[float] = []
        entropies: list[float] = []
        for sample, advantage in zip(samples, advantage_values):
            hidden, logits = self._forward(sample.option_features)
            probabilities = self._probabilities(logits, sample.action_mask)
            action = int(sample.catalogue_index)
            if not 0 <= action < probabilities.size or not sample.action_mask[action]:
                raise ValueError("Training sample contains a masked action")

            one_hot = np.zeros_like(probabilities)
            one_hot[action] = 1.0
            categorical_entropy = float(
                -np.sum(probabilities[sample.action_mask]
                        * np.log(probabilities[sample.action_mask]))
            )
            entropy_gradient = np.zeros_like(probabilities)
            valid = sample.action_mask
            entropy_gradient[valid] = -probabilities[valid] * (
                np.log(probabilities[valid]) + categorical_entropy
            )
            logits_gradient = advantage * (one_hot - probabilities)
            logits_gradient += entropy_coefficient * entropy_gradient

            gradients["score_weights"] += hidden.T @ logits_gradient
            gradients["score_bias"] += np.asarray([logits_gradient.sum()])
            hidden_gradient = logits_gradient[:, None] * self.score_weights[None, :]
            log_probability = float(np.log(max(probabilities[action], np.finfo(float).tiny)))
            total_entropy = categorical_entropy

            if sample.position_active:
                mean = hidden[action] @ self.position_weights + self.position_bias
                latent = np.asarray(sample.position_latent, dtype=np.float64)
                if latent.shape != (2,) or not np.isfinite(latent).all():
                    raise ValueError("Training sample contains an invalid position latent")
                variance = np.exp(2.0 * self.position_log_std)
                delta = latent - mean
                mean_gradient = advantage * delta / variance
                log_std_gradient = advantage * (delta * delta / variance - 1.0)
                log_std_gradient += entropy_coefficient
                gradients["position_weights"] += np.outer(hidden[action], mean_gradient)
                gradients["position_bias"] += mean_gradient
                gradients["position_log_std"] += log_std_gradient
                hidden_gradient[action] += self.position_weights @ mean_gradient
                position_entropy = float(
                    np.sum(self.position_log_std + 0.5 * (1.0 + _LOG_TWO_PI))
                )
                total_entropy += position_entropy
                log_probability += self._position_log_probability(
                    latent, mean, self.position_log_std
                )

            preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
            gradients["input_weights"] += sample.option_features.T @ preactivation_gradient
            gradients["input_bias"] += preactivation_gradient.sum(axis=0)
            objectives.append(float(advantage) * log_probability
                              + entropy_coefficient * total_entropy)
            entropies.append(total_entropy)

        scale = 1.0 / len(samples)
        for name in gradients:
            gradients[name] *= scale
        squared_norm = sum(float(np.sum(value * value)) for value in gradients.values())
        gradient_norm = float(np.sqrt(squared_norm))
        if not np.isfinite(gradient_norm):
            raise FloatingPointError("Placement-policy gradient is not finite")
        if max_gradient_norm > 0.0 and gradient_norm > max_gradient_norm:
            clip_scale = max_gradient_norm / (gradient_norm + 1e-12)
            for name in gradients:
                gradients[name] *= clip_scale

        self.adam_step += 1
        for name, gradient in gradients.items():
            self._adam_mean[name] = beta1 * self._adam_mean[name] + (1.0 - beta1) * gradient
            self._adam_variance[name] = (
                beta2 * self._adam_variance[name] + (1.0 - beta2) * gradient * gradient
            )
            mean_hat = self._adam_mean[name] / (1.0 - beta1 ** self.adam_step)
            variance_hat = self._adam_variance[name] / (1.0 - beta2 ** self.adam_step)
            parameter = getattr(self, name)
            parameter += learning_rate * mean_hat / (np.sqrt(variance_hat) + epsilon)
        self.position_log_std[:] = np.clip(self.position_log_std, -5.0, 1.0)
        if not all(np.isfinite(getattr(self, name)).all() for name in self._PARAMETER_NAMES):
            raise FloatingPointError("Placement-policy update produced non-finite parameters")
        self.update_count += 1
        return {
            "samples": len(samples),
            "update": self.update_count,
            "objective": float(np.mean(objectives)),
            "entropy": float(np.mean(entropies)),
            "gradient_norm": gradient_norm,
            "advantage_mean": float(advantage_values.mean()),
        }

    def parameter_hash(self) -> str:
        hasher = hashlib.sha256()
        for name in self._PARAMETER_NAMES:
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            hasher.update(name.encode("utf-8"))
            hasher.update(value.tobytes())
        return hasher.hexdigest()

    def save_checkpoint(
        self, directory: str | Path, *, details: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays: dict[str, np.ndarray] = {}
        for name in self._PARAMETER_NAMES:
            arrays[f"parameter__{name}"] = np.asarray(getattr(self, name), dtype=np.float64)
            arrays[f"adam_mean__{name}"] = np.asarray(self._adam_mean[name], dtype=np.float64)
            arrays[f"adam_variance__{name}"] = np.asarray(
                self._adam_variance[name], dtype=np.float64
            )
        arrays_path = directory / _ARRAYS_FILENAME
        np.savez_compressed(arrays_path, **arrays)
        arrays_hash = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        metadata = {
            "schema": CHECKPOINT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "policy": self.specification,
            "feature_contract": self.feature_contract,
            "feature_contract_sha256": _sha256_json(self.feature_contract),
            "arrays_filename": _ARRAYS_FILENAME,
            "arrays_sha256": arrays_hash,
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "parameter_sha256": self.parameter_hash(),
            "optimizer": {"name": "adam", "step": self.adam_step},
            "training": {
                "updates": self.update_count,
                "return_count": self.return_count,
                "return_mean": self.return_mean,
                "return_m2": self.return_m2,
            },
            "rng_state": _json_value(self.rng.bit_generator.state),
            "details": _json_value(details or {}),
        }
        (directory / _METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        return metadata

    @classmethod
    def load_checkpoint(
        cls,
        directory: str | Path,
        *,
        expected_feature_contract: Mapping[str, Any] | None = None,
    ) -> tuple["DynamicPlacementPolicy", dict[str, Any]]:
        directory = Path(directory)
        metadata = json.loads((directory / _METADATA_FILENAME).read_text(encoding="utf-8"))
        if metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported dynamic-placement checkpoint schema")
        contract = metadata.get("feature_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("Checkpoint does not contain a feature contract")
        contract_hash = _sha256_json(contract)
        if metadata.get("feature_contract_sha256") != contract_hash:
            raise ValueError("Checkpoint feature contract hash mismatch")
        if expected_feature_contract is not None and contract_hash != _sha256_json(expected_feature_contract):
            raise ValueError("Checkpoint feature contract differs from this environment")
        specification = metadata.get("policy", {})
        if specification.get("schema") != POLICY_SCHEMA or specification.get("position_dimensions") != 2:
            raise ValueError("Unsupported placement-policy architecture")
        hidden_size = int(specification.get("hidden_size", 0))
        if int(specification.get("feature_size", 0)) != int(contract.get("feature_size", -1)):
            raise ValueError("Checkpoint policy and feature dimensions disagree")

        arrays_filename = metadata.get("arrays_filename")
        if arrays_filename != _ARRAYS_FILENAME:
            raise ValueError("Checkpoint arrays filename is not supported")
        arrays_path = directory / arrays_filename
        if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != metadata.get("arrays_sha256"):
            raise ValueError("Checkpoint arrays hash mismatch")
        policy = cls(contract, hidden_size=hidden_size, seed=0)
        expected_keys = {
            f"{prefix}__{name}"
            for prefix in ("parameter", "adam_mean", "adam_variance")
            for name in cls._PARAMETER_NAMES
        }
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise ValueError("Checkpoint contains missing or unexpected arrays")
            declared_shapes = metadata.get("array_shapes", {})
            for name in cls._PARAMETER_NAMES:
                parameter = np.asarray(archive[f"parameter__{name}"], dtype=np.float64)
                mean = np.asarray(archive[f"adam_mean__{name}"], dtype=np.float64)
                variance = np.asarray(archive[f"adam_variance__{name}"], dtype=np.float64)
                expected_shape = getattr(policy, name).shape
                keys = (f"parameter__{name}", f"adam_mean__{name}", f"adam_variance__{name}")
                if any(tuple(declared_shapes.get(key, ())) != expected_shape for key in keys):
                    raise ValueError("Checkpoint array-shape metadata mismatch")
                if any(value.shape != expected_shape for value in (parameter, mean, variance)):
                    raise ValueError("Checkpoint array shape mismatch")
                if not all(np.isfinite(value).all() for value in (parameter, mean, variance)):
                    raise ValueError("Checkpoint contains non-finite arrays")
                if np.any(variance < 0.0):
                    raise ValueError("Checkpoint contains a negative Adam variance")
                getattr(policy, name)[:] = parameter
                policy._adam_mean[name][:] = mean
                policy._adam_variance[name][:] = variance

        optimizer = metadata.get("optimizer", {})
        training = metadata.get("training", {})
        policy.adam_step = int(optimizer.get("step", -1))
        policy.update_count = int(training.get("updates", -1))
        policy.return_count = int(training.get("return_count", -1))
        policy.return_mean = float(training.get("return_mean", np.nan))
        policy.return_m2 = float(training.get("return_m2", np.nan))
        if (policy.adam_step < 0 or policy.update_count < 0 or policy.return_count < 0
                or not np.isfinite([policy.return_mean, policy.return_m2]).all()
                or policy.return_m2 < 0.0):
            raise ValueError("Checkpoint training state is invalid")
        try:
            policy.rng.bit_generator.state = metadata["rng_state"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Checkpoint RNG state is invalid") from error
        if policy.parameter_hash() != metadata.get("parameter_sha256"):
            raise ValueError("Loaded placement parameters do not match their hash")
        return policy, metadata


def save_placement_checkpoint(
    directory: str | Path,
    policy: DynamicPlacementPolicy,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return policy.save_checkpoint(directory, details=details)


def load_placement_checkpoint(
    directory: str | Path,
    *,
    expected_feature_contract: Mapping[str, Any] | None = None,
) -> tuple[DynamicPlacementPolicy, dict[str, Any]]:
    return DynamicPlacementPolicy.load_checkpoint(
        directory, expected_feature_contract=expected_feature_contract
    )
