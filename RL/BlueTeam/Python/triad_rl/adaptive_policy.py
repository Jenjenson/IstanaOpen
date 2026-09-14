"""Shared, variable-catalogue actor-critic for joint sensor/site decisions.

Every legal sensor/site pair (including stop) is scored by the same small MLP.
Neither sensor IDs nor the number/order of sites appears in the parameter
shapes. Training is on-policy REINFORCE with a learned state-value baseline,
discounted reward-to-go, entropy regularisation, Adam, and gradient clipping.
Only public observation features enter either network.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


POLICY_SCHEMA = "triad.adaptive_joint_actor_critic.v1"
CHECKPOINT_SCHEMA = "triad.adaptive_checkpoint.v1"
FEATURE_SCHEMA = "triad.adaptive_placement_features.v1"
PARAMETERS = ("w1", "b1", "wa", "wv", "bv")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _feature_names(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError("feature_names must be an ordered sequence of names")
    result = tuple(names)
    if (not result or len(result) > 4096
            or any(not isinstance(name, str) or not name for name in result)
            or len(set(result)) != len(result)):
        raise ValueError("Feature names must be nonempty, unique strings")
    return result


class AdaptivePolicy:
    """A sensor/site permutation-equivariant actor with a pooled critic."""

    def __init__(self, feature_names: Sequence[str], seed: int = 0,
                 hidden_size: int = 32) -> None:
        self.feature_names = _feature_names(feature_names)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or not 1 <= hidden_size <= 2048:
            raise ValueError("hidden_size must be an integer from 1 to 2048")
        self.hidden_size = hidden_size
        self.rng = np.random.default_rng(seed)
        self.parameters = {
            "w1": self.rng.normal(0, 1 / np.sqrt(len(self.feature_names)),
                                  (len(self.feature_names), hidden_size)),
            "b1": np.zeros(hidden_size),
            "wa": self.rng.normal(0, 0.05, hidden_size),
            "wv": np.zeros(hidden_size),
            "bv": np.zeros(1),
        }
        self.adam_m = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.adam_v = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.update_count = 0
        self.training_state: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}

    def _input(self, observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if observation.get("feature_schema", observation.get("schema", FEATURE_SCHEMA)) != FEATURE_SCHEMA:
            raise ValueError("Observation feature semantics version differs from checkpoint")
        if tuple(observation.get("feature_names", ())) != self.feature_names:
            raise ValueError("Observation feature schema/name/order differs from checkpoint")
        features = np.asarray(observation["option_features"], dtype=np.float64)
        mask = np.asarray(observation["action_mask"])
        if (features.ndim != 2 or features.shape[1] != len(self.feature_names)
                or features.shape[0] == 0 or not np.isfinite(features).all()):
            raise ValueError("option_features must be a finite, nonempty [options,features] matrix")
        if mask.dtype != np.bool_ or mask.shape != (features.shape[0],) or not mask.any():
            raise ValueError("action_mask must be a boolean vector with at least one legal action")
        return features, mask

    def _forward(self, features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        p = self.parameters
        hidden = np.tanh(features @ p["w1"] + p["b1"])
        logits = hidden @ p["wa"]
        legal_logits = logits[mask]
        exp = np.exp(legal_logits - np.max(legal_logits))
        probabilities = np.zeros(features.shape[0])
        probabilities[mask] = exp / np.sum(exp)
        value = float(hidden[mask].mean(axis=0) @ p["wv"] + p["bv"][0])
        return hidden, probabilities, value

    def probabilities(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return a new probability vector, exactly zero for masked actions."""
        features, mask = self._input(observation)
        return self._forward(features, mask)[1]

    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> int:
        probabilities = self.probabilities(observation)
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self.rng.choice(len(probabilities), p=probabilities))

    def sample(self, observation: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """Sample an action and detached training record; caller adds ``reward``."""
        features, mask = self._input(observation)
        _, probabilities, value = self._forward(features, mask)
        action = int(self.rng.choice(len(probabilities), p=probabilities))
        return action, {"features": features.copy(), "mask": mask.copy(),
                        "action": action, "value": value, "reward": 0.0}

    def _loss_and_gradients(self, transitions: Sequence[Mapping[str, Any]],
                            returns: np.ndarray, advantages: np.ndarray,
                            entropy_coef: float, value_coef: float
                            ) -> tuple[float, dict[str, np.ndarray], dict[str, float]]:
        """Compute exact gradients, holding sampled advantages fixed."""
        gradients = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        actor_loss = critic_loss = entropy_sum = 0.0
        for record, target, advantage in zip(transitions, returns, advantages, strict=True):
            x, mask = record["features"], record["mask"]
            hidden, probability, value = self._forward(x, mask)
            log_probability = np.zeros_like(probability)
            log_probability[mask] = np.log(np.maximum(probability[mask], np.finfo(float).tiny))
            entropy = -float(probability @ log_probability)
            action = record["action"]
            actor_loss -= float(advantage * log_probability[action])
            error = value - target
            critic_loss += 0.5 * float(error * error)
            entropy_sum += entropy

            d_logits = advantage * probability
            d_logits[action] -= advantage
            d_logits += entropy_coef * probability * (log_probability + entropy)
            d_value = value_coef * error
            gradients["wa"] += hidden.T @ d_logits
            gradients["wv"] += hidden[mask].mean(axis=0) * d_value
            gradients["bv"][0] += d_value
            d_hidden = d_logits[:, None] * self.parameters["wa"][None, :]
            d_hidden[mask] += self.parameters["wv"][None, :] * (d_value / int(mask.sum()))
            d_z = d_hidden * (1 - hidden * hidden)
            gradients["w1"] += x.T @ d_z
            gradients["b1"] += d_z.sum(axis=0)

        count = len(transitions)
        for gradient in gradients.values():
            gradient /= count
        metrics = {"actor_loss": actor_loss / count, "critic_loss": critic_loss / count,
                   "entropy": entropy_sum / count}
        loss = metrics["actor_loss"] + value_coef * metrics["critic_loss"] - entropy_coef * metrics["entropy"]
        return loss, gradients, metrics

    def update(self, episodes: Sequence[Sequence[Mapping[str, Any]]], *,
               learning_rate: float = 0.004, entropy_coef: float = 0.015,
               gamma: float = 0.99, value_coef: float = 0.5,
               max_grad_norm: float = 1.0) -> dict[str, float]:
        """Apply one on-policy update to complete episodes collected by sample()."""
        numeric = (learning_rate, entropy_coef, gamma, value_coef, max_grad_norm)
        if (not all(np.isfinite(value) for value in numeric) or learning_rate <= 0
                or entropy_coef < 0 or not 0 <= gamma <= 1 or value_coef < 0 or max_grad_norm <= 0):
            raise ValueError("Invalid finite training hyperparameters")
        transitions: list[Mapping[str, Any]] = []
        targets: list[float] = []
        for episode in episodes:
            if not episode:
                raise ValueError("Training episodes must not be empty")
            episode_targets: list[float] = []
            cumulative = 0.0
            for record in reversed(episode):
                cumulative = float(record["reward"]) + gamma * cumulative
                episode_targets.append(cumulative)
            for record, target in zip(episode, reversed(episode_targets), strict=True):
                features, mask = self._input({"feature_names": self.feature_names,
                                              "option_features": record["features"],
                                              "action_mask": record["mask"]})
                action = record["action"]
                if (isinstance(action, bool) or not isinstance(action, (int, np.integer))
                        or not 0 <= action < len(mask) or not mask[action]):
                    raise ValueError("Training action is invalid or masked")
                if not np.isfinite(target) or not np.isfinite(record["value"]):
                    raise ValueError("Training returns and values must be finite")
                transitions.append({**record, "features": features, "mask": mask})
                targets.append(target)
        if not transitions:
            raise ValueError("Need at least one training episode")
        returns = np.asarray(targets)
        advantages = returns - np.asarray([record["value"] for record in transitions])
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss, gradients, metrics = self._loss_and_gradients(
            transitions, returns, advantages, entropy_coef, value_coef)
        grad_norm = float(np.sqrt(sum(np.sum(gradient * gradient) for gradient in gradients.values())))
        if not np.isfinite(loss) or not np.isfinite(grad_norm):
            raise ValueError("Nonfinite loss/gradient; update not applied")
        scale = min(1.0, max_grad_norm / max(grad_norm, 1e-12))
        self.update_count += 1
        for key, parameter in self.parameters.items():
            gradient = gradients[key] * scale
            self.adam_m[key] = 0.9 * self.adam_m[key] + 0.1 * gradient
            self.adam_v[key] = 0.999 * self.adam_v[key] + 0.001 * gradient * gradient
            corrected_m = self.adam_m[key] / (1 - 0.9 ** self.update_count)
            corrected_v = self.adam_v[key] / (1 - 0.999 ** self.update_count)
            parameter -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        return {**metrics, "loss": float(loss), "gradient_norm": grad_norm,
                "gradient_scale": scale, "mean_return_to_go": float(returns.mean()),
                "transitions": len(transitions), "updates": self.update_count}

    train_batch = update

    def weights_fingerprint(self) -> str:
        digest = hashlib.sha256(_json({"schema": POLICY_SCHEMA, "feature_schema": FEATURE_SCHEMA,
                                      "features": self.feature_names,
                                      "hidden_size": self.hidden_size}).encode())
        for key in PARAMETERS:
            digest.update(key.encode())
            digest.update(np.asarray(self.parameters[key], dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: str | Path, training_state: Mapping[str, Any] | None = None) -> None:
        """Save strict JSON and non-pickled NPZ, including exact Adam/RNG state."""
        path = Path(path)
        arrays = {f"param_{key}": value for key, value in self.parameters.items()}
        arrays.update({f"adam_m_{key}": value for key, value in self.adam_m.items()})
        arrays.update({f"adam_v_{key}": value for key, value in self.adam_v.items()})
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("Cannot save nonfinite weights or optimizer state")
        metadata = {"schema": CHECKPOINT_SCHEMA, "policy_schema": POLICY_SCHEMA,
                    "feature_schema": FEATURE_SCHEMA,
                    "feature_names": list(self.feature_names), "hidden_size": self.hidden_size,
                    "update_count": self.update_count, "rng_state": self.rng.bit_generator.state,
                    "weights_sha256": self.weights_fingerprint(),
                    "training_state": dict(self.training_state if training_state is None else training_state)}
        metadata["seed_provenance"] = metadata["training_state"].get("seed_provenance", {})
        _json(metadata)  # Reject invalid metadata before touching an existing checkpoint.
        path.mkdir(parents=True, exist_ok=True)
        arrays_path = path / "arrays.npz"
        temporary_arrays = path / "arrays.tmp.npz"
        np.savez_compressed(temporary_arrays, **arrays)
        metadata["arrays_sha256"] = hashlib.sha256(temporary_arrays.read_bytes()).hexdigest()
        temporary_metadata = path / "checkpoint.tmp.json"
        temporary_metadata.write_text(_json(metadata) + "\n", encoding="utf-8")
        temporary_arrays.replace(arrays_path)
        temporary_metadata.replace(path / "checkpoint.json")
        self.training_state = metadata["training_state"]
        self.metadata = metadata

    @classmethod
    def load(cls, path: str | Path, feature_names: Sequence[str] | None = None) -> "AdaptivePolicy":
        path = Path(path)
        metadata_path, arrays_path = path / "checkpoint.json", path / "arrays.npz"
        if metadata_path.stat().st_size > 4_000_000 or arrays_path.stat().st_size > 256_000_000:
            raise ValueError("Checkpoint exceeds size limits")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"),
                                  parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError("Invalid checkpoint JSON") from error
        if (metadata.get("schema") != CHECKPOINT_SCHEMA or metadata.get("policy_schema") != POLICY_SCHEMA
                or metadata.get("feature_schema") != FEATURE_SCHEMA):
            raise ValueError("Unsupported adaptive checkpoint schema")
        names = _feature_names(metadata["feature_names"])
        if feature_names is not None and tuple(feature_names) != names:
            raise ValueError("Checkpoint feature schema/name/order mismatch")
        if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != metadata.get("arrays_sha256"):
            raise ValueError("Checkpoint array fingerprint mismatch")
        policy = cls(names, hidden_size=metadata["hidden_size"])
        expected = {f"{prefix}_{key}": value.shape for prefix in ("param", "adam_m", "adam_v")
                    for key, value in policy.parameters.items()}
        # Check expanded member sizes before NumPy allocates/decompresses arrays.
        try:
            with zipfile.ZipFile(arrays_path) as archive:
                members = archive.infolist()
                if len(members) != len(expected) or {m.filename for m in members} != {f"{k}.npy" for k in expected}:
                    raise ValueError("Unexpected checkpoint archive members")
                for member in members:
                    shape = expected[member.filename[:-4]]
                    if member.file_size > int(np.prod(shape)) * 8 + 1024:
                        raise ValueError("Checkpoint array exceeds expected expanded size")
        except zipfile.BadZipFile as error:
            raise ValueError("Invalid checkpoint NPZ archive") from error
        with np.load(arrays_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError("Unexpected checkpoint array names")
            loaded = {}
            for key, shape in expected.items():
                array = arrays[key]
                if array.shape != shape or array.dtype != np.float64 or not np.isfinite(array).all():
                    raise ValueError(f"Invalid checkpoint array {key}")
                if key.startswith("adam_v_") and (array < 0).any():
                    raise ValueError("Adam second moments cannot be negative")
                loaded[key] = array.copy()
        for key in PARAMETERS:
            policy.parameters[key] = loaded[f"param_{key}"]
            policy.adam_m[key] = loaded[f"adam_m_{key}"]
            policy.adam_v[key] = loaded[f"adam_v_{key}"]
        count = metadata["update_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid update_count")
        policy.update_count = count
        if policy.weights_fingerprint() != metadata.get("weights_sha256"):
            raise ValueError("Checkpoint weights fingerprint mismatch")
        try:
            policy.rng.bit_generator.state = metadata["rng_state"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid checkpoint RNG state") from error
        if not isinstance(metadata.get("training_state"), dict):
            raise ValueError("Invalid checkpoint training state")
        policy.training_state = metadata["training_state"]
        policy.metadata = metadata
        return policy
