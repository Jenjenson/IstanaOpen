"""Count-balanced deploy/stop policy with a versioned inference contract.

The shared actor still learns sensor/site scores, but deployment logits receive
``-log(number of legal deployment options)``. Thus duplicating every offered
deployment does not change the probability of deploying. Deterministic inference
first chooses deploy versus stop, then the best deployment. This intentionally
differs from taking the largest individual probability in the joint distribution.

The entropy bonus is relative entropy against a 50/50 deploy/stop reference with
uniform conditional deployments (up to the constant log(2)). It is
``H(joint) - P(deploy) * log(N)``: equivalent to gate entropy minus deployment
probability times KL(conditional deployments || uniform). It can be negative.
Neither this correction nor the actor accesses private simulation truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .adaptive_policy import (
    AdaptivePolicy, FEATURE_SCHEMA, PARAMETERS, POLICY_SCHEMA as ADAPTIVE_POLICY_SCHEMA,
    _feature_names, _json,
)


POLICY_SCHEMA = "triad.balanced_joint_actor_critic.v1"
CHECKPOINT_SCHEMA = "triad.balanced_checkpoint.v1"
DECISION_RULE = "stop-if-p-stop-ge-0.5-else-best-legal-deployment;first-row-ties"
ENTROPY_RULE = "H-joint-minus-p-deploy-times-log-legal-deployment-count"
POLICY_CONTRACT = {
    "stop_feature": "stop",
    "stop_contract": "exactly-one-binary-stop-row-and-stop-always-legal",
    "deployment_logit_adjustment": "minus-log-legal-deployment-count",
    "stochastic_rule": "sample-joint-distribution",
    "deterministic_rule": DECISION_RULE,
    "entropy_rule": ENTROPY_RULE,
    "critic_rule": "mean-hidden-over-all-legal-rows",
}


class BalancedPolicy(AdaptivePolicy):
    """Shared option actor/critic with a catalogue-count-balanced learned gate.

Network parameter shapes and the public feature schema match AdaptivePolicy;
checkpoint and weight fingerprints deliberately do not. The base policy loader
rejects these checkpoints rather than silently applying flat-softmax inference.
"""

    def __init__(self, feature_names: Sequence[str], seed: int = 0,
                 hidden_size: int = 32) -> None:
        super().__init__(feature_names, seed=seed, hidden_size=hidden_size)
        if "stop" not in self.feature_names:
            raise ValueError("Balanced policy requires the named 'stop' feature")
        self.stop_feature_index = self.feature_names.index("stop")

    def _stop_and_deployments(self, features: np.ndarray,
                              mask: np.ndarray) -> tuple[int, np.ndarray]:
        flags = features[:, self.stop_feature_index]
        if not np.all((flags == 0) | (flags == 1)) or np.count_nonzero(flags) != 1:
            raise ValueError("Balanced policy requires exactly one binary STOP row")
        stop = int(np.flatnonzero(flags)[0])
        if not mask[stop]:
            raise ValueError("STOP must always be legal")
        deployments = mask.copy()
        deployments[stop] = False
        return stop, deployments

    def _input(self, observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        features, mask = super()._input(observation)
        self._stop_and_deployments(features, mask)
        return features, mask

    def _forward(self, features: np.ndarray,
                 mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        _, deployments = self._stop_and_deployments(features, mask)
        p = self.parameters
        hidden = np.tanh(features @ p["w1"] + p["b1"])
        logits = hidden @ p["wa"]
        count = int(deployments.sum())
        if count:
            logits[deployments] -= np.log(count)
        legal_logits = logits[mask]
        exp = np.exp(legal_logits - np.max(legal_logits))
        probabilities = np.zeros(features.shape[0])
        probabilities[mask] = exp / exp.sum()
        value = float(hidden[mask].mean(axis=0) @ p["wv"] + p["bv"][0])
        return hidden, probabilities, value

    @staticmethod
    def _entropies(probability: np.ndarray, deployments: np.ndarray,
                   stop: int) -> dict[str, float]:
        tiny = np.finfo(float).tiny
        count = int(deployments.sum())
        deploy_probability = float(probability[deployments].sum())
        stop_probability = float(probability[stop])
        joint = -float(probability @ np.log(np.maximum(probability, tiny)))
        gate = -(stop_probability * np.log(max(stop_probability, tiny))
                 + deploy_probability * np.log(max(deploy_probability, tiny)))
        conditional_entropy = 0.0
        if deploy_probability > 0:
            conditional = probability[deployments] / deploy_probability
            conditional_entropy = -float(conditional @ np.log(np.maximum(conditional, tiny)))
        relative = joint - deploy_probability * np.log(max(count, 1))
        return {"stop_probability": stop_probability,
                "deployment_probability": deploy_probability,
                "joint_entropy": joint, "gate_entropy": float(gate),
                "deployment_entropy": conditional_entropy,
                "relative_entropy": float(relative), "legal_deployments": float(count)}

    def diagnostics(self, observation: Mapping[str, Any]) -> dict[str, float]:
        """Public-input-only gate and entropy diagnostics; does not consume RNG."""
        features, mask = self._input(observation)
        stop, deployments = self._stop_and_deployments(features, mask)
        return self._entropies(self._forward(features, mask)[1], deployments, stop)

    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> int:
        features, mask = self._input(observation)
        stop, deployments = self._stop_and_deployments(features, mask)
        probability = self._forward(features, mask)[1]
        if not deterministic:
            return int(self.rng.choice(len(probability), p=probability))
        if not deployments.any() or probability[stop] >= 0.5:
            return stop
        return int(np.argmax(np.where(deployments, probability, -np.inf)))

    def _loss_and_gradients(self, transitions: Sequence[Mapping[str, Any]],
                            returns: np.ndarray, advantages: np.ndarray,
                            entropy_coef: float, value_coef: float
                            ) -> tuple[float, dict[str, np.ndarray], dict[str, float]]:
        """Exact REINFORCE, relative-entropy and pooled-critic gradients."""
        if not transitions:
            raise ValueError("Need at least one transition")
        gradients = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        actor_loss = critic_loss = 0.0
        diagnostic_sums: dict[str, float] = {}
        for record, target, advantage in zip(transitions, returns, advantages, strict=True):
            x, mask = record["features"], record["mask"]
            stop, deployments = self._stop_and_deployments(x, mask)
            hidden, probability, value = self._forward(x, mask)
            log_probability = np.zeros_like(probability)
            log_probability[mask] = np.log(np.maximum(probability[mask], np.finfo(float).tiny))
            diagnostics = self._entropies(probability, deployments, stop)
            entropy = diagnostics["relative_entropy"]
            for key, diagnostic in diagnostics.items():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + diagnostic
            action = record["action"]
            actor_loss -= float(advantage * log_probability[action])
            error = value - target
            critic_loss += 0.5 * float(error * error)

            # The log-count adjustment is observation-only, so its derivative
            # with respect to every raw actor logit is zero.
            d_logits = advantage * probability
            d_logits[action] -= advantage
            log_reference_count = deployments * np.log(max(int(deployments.sum()), 1))
            d_logits += entropy_coef * probability * (log_probability + log_reference_count + entropy)
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
        metrics = {key: value / count for key, value in diagnostic_sums.items()}
        metrics.update({"actor_loss": actor_loss / count, "critic_loss": critic_loss / count,
                        "entropy": metrics["relative_entropy"]})
        loss = metrics["actor_loss"] + value_coef * metrics["critic_loss"] - entropy_coef * metrics["entropy"]
        return loss, gradients, metrics

    @classmethod
    def transfer_from_adaptive(cls, base: AdaptivePolicy, seed: int = 0) -> "BalancedPolicy":
        """Copy v1/v2 parameter values, with a fresh optimizer and sampling RNG.

        This changes inference semantics immediately; it is not an exact resume.
        The calling trainer must retain the source policy's seed exposure/lineage.
        """
        if type(base) is not AdaptivePolicy:
            raise ValueError("Transfer requires an AdaptivePolicy, not a balanced checkpoint")
        policy = cls(base.feature_names, seed=seed, hidden_size=base.hidden_size)
        for key in PARAMETERS:
            if not np.isfinite(base.parameters[key]).all():
                raise ValueError("Cannot transfer nonfinite weights")
            policy.parameters[key] = base.parameters[key].copy()
        policy.rng = np.random.default_rng(seed)
        policy.training_state = {"transfer": {"source_policy_schema": ADAPTIVE_POLICY_SCHEMA,
                                             "source_weights_sha256": base.weights_fingerprint()}}
        return policy

    def weights_fingerprint(self) -> str:
        digest = hashlib.sha256(_json({"schema": POLICY_SCHEMA, "feature_schema": FEATURE_SCHEMA,
                                      "policy_contract": POLICY_CONTRACT,
                                      "features": self.feature_names,
                                      "hidden_size": self.hidden_size}).encode())
        for key in PARAMETERS:
            digest.update(key.encode())
            digest.update(np.asarray(self.parameters[key], dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: str | Path, training_state: Mapping[str, Any] | None = None) -> None:
        """Save strict JSON/non-pickled NPZ, including exact Adam and RNG state."""
        path = Path(path)
        arrays = {f"{prefix}_{key}": value
                  for prefix, values in (("param", self.parameters), ("adam_m", self.adam_m),
                                         ("adam_v", self.adam_v))
                  for key, value in values.items()}
        expected_shapes = {"w1": (len(self.feature_names), self.hidden_size),
                           "b1": (self.hidden_size,), "wa": (self.hidden_size,),
                           "wv": (self.hidden_size,), "bv": (1,)}
        expected = {f"{prefix}_{key}": shape for prefix in ("param", "adam_m", "adam_v")
                    for key, shape in expected_shapes.items()}
        if set(arrays) != set(expected):
            raise ValueError("Unexpected parameter/optimizer names")
        for key, array in arrays.items():
            if (not isinstance(array, np.ndarray) or array.shape != expected[key]
                    or array.dtype != np.float64 or not np.isfinite(array).all()
                    or (key.startswith("adam_v_") and (array < 0).any())):
                raise ValueError("Cannot save invalid weights or optimizer state")
        if isinstance(self.update_count, bool) or not isinstance(self.update_count, int) or self.update_count < 0:
            raise ValueError("Invalid update_count")
        metadata = {"schema": CHECKPOINT_SCHEMA, "policy_schema": POLICY_SCHEMA,
                    "policy_contract": POLICY_CONTRACT, "feature_schema": FEATURE_SCHEMA,
                    "feature_names": list(self.feature_names), "hidden_size": self.hidden_size,
                    "update_count": self.update_count, "rng_state": self.rng.bit_generator.state,
                    "weights_sha256": self.weights_fingerprint(),
                    "training_state": dict(self.training_state if training_state is None else training_state)}
        metadata["seed_provenance"] = metadata["training_state"].get("seed_provenance", {})
        _json(metadata)  # Reject metadata before touching an existing checkpoint.
        path.mkdir(parents=True, exist_ok=True)
        temporary_arrays = path / "arrays.tmp.npz"
        np.savez_compressed(temporary_arrays, **arrays)
        metadata["arrays_sha256"] = hashlib.sha256(temporary_arrays.read_bytes()).hexdigest()
        temporary_metadata = path / "checkpoint.tmp.json"
        temporary_metadata.write_text(_json(metadata) + "\n", encoding="utf-8")
        temporary_arrays.replace(path / "arrays.npz")
        temporary_metadata.replace(path / "checkpoint.json")
        self.training_state = metadata["training_state"]
        self.metadata = metadata

    @classmethod
    def load(cls, path: str | Path, feature_names: Sequence[str] | None = None) -> "BalancedPolicy":
        path = Path(path)
        metadata_path, arrays_path = path / "checkpoint.json", path / "arrays.npz"
        if metadata_path.stat().st_size > 4_000_000 or arrays_path.stat().st_size > 256_000_000:
            raise ValueError("Checkpoint exceeds size limits")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate checkpoint JSON key")
                result[key] = value
            return result

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                                  parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError("Invalid checkpoint JSON") from error
        # parse_constant rejects NaN/Infinity tokens, but an ordinary exponent
        # such as 1e999 can still overflow to infinity during json.loads.
        _json(metadata)
        if (not isinstance(metadata, dict) or metadata.get("schema") != CHECKPOINT_SCHEMA
                or metadata.get("policy_schema") != POLICY_SCHEMA
                or metadata.get("feature_schema") != FEATURE_SCHEMA
                or metadata.get("policy_contract") != POLICY_CONTRACT):
            raise ValueError("Unsupported balanced checkpoint schema or policy contract")
        required = {"schema", "policy_schema", "policy_contract", "feature_schema", "feature_names",
                    "hidden_size", "update_count", "rng_state", "weights_sha256", "training_state",
                    "seed_provenance", "arrays_sha256"}
        if set(metadata) != required:
            raise ValueError("Unexpected balanced checkpoint metadata fields")
        names = _feature_names(metadata["feature_names"])
        if feature_names is not None and tuple(feature_names) != names:
            raise ValueError("Checkpoint feature schema/name/order mismatch")
        if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != metadata.get("arrays_sha256"):
            raise ValueError("Checkpoint array fingerprint mismatch")
        policy = cls(names, hidden_size=metadata["hidden_size"])
        expected = {f"{prefix}_{key}": value.shape for prefix in ("param", "adam_m", "adam_v")
                    for key, value in policy.parameters.items()}
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
        if (not isinstance(metadata["training_state"], dict)
                or metadata["seed_provenance"] != metadata["training_state"].get("seed_provenance", {})):
            raise ValueError("Invalid checkpoint training state or seed provenance")
        policy.training_state = metadata["training_state"]
        policy.metadata = metadata
        return policy
