"""Bounded temporal-prior actor and a separate public linear value baseline.

The actor never receives rewards, private simulator state, or critic features
other than its normal temporal option matrix. STOP is a learned option. Exact
greedy initialization is not evidence that reinforcement learning improved it.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .adaptive_policy import _json
from .ranking_policy import _integer, _number, _rng_state, _strict_json
from .temporal_inputs import FEATURE_NAMES, FEATURE_SCHEMA, TemporalConfig

POLICY_SCHEMA = "triad.temporal_bounded_actor_linear_critic.v1"
CHECKPOINT_SCHEMA = "triad.temporal_policy_checkpoint.v1"
ACTOR = ("w1", "b1", "wa")
CRITIC = ("wv", "bv")
PARAMETERS = (*ACTOR, *CRITIC)
MAX_ROWS, MAX_TRANSITIONS, MAX_ELEMENTS = 16385, 4096, 8_000_000
POLICY_CONTRACT = {
    "prior": "temporal_marginal_return; original expected marginal return / 20; STOP exactly zero",
    "residual": "0.25*tanh(tanh(features@w1+b1)@wa); zero-initialized wa",
    "deterministic": "highest legal raw score; STOP wins ties; other ties use first legal row",
    "stochastic": "softmax(score/temperature - deployment_indicator*log(legal_deployment_count))",
    "entropy": "Hjoint - pdeploy*log(N) = Hgate - pdeploy*KL(conditional_deployment||uniform)",
    "critic_context": "concat(mean legal DEPLOYMENT features, STOP features); zero deployment mean if none",
    "critic_prior": "16*existing_timely+2*existing_detection+existing_confirmation+3*existing_early"
                    "+2*existing_coverage-existing_early_coverage-10.5-.55*4*(budget_total_feature-budget_remaining_feature)",
    "critic_prior_limitation": "Public forecast approximation minus exact current potential; previously-paid shaping/redundancy excluded from future RTG",
    "critic": "public STOP prior + context@wv + bv; independent linear parameters; zero initialized",
    "advantages": "reward-to-go minus detached BEFORE-update sample value; gamma=1 default; no normalization default",
    "normalization": "optional batch mean/std only if count>1 and std>1e-8; divisor std+1e-8",
    "optimizer": "independent actor/critic global L2 clipping then Adam(beta1=.9,beta2=.999,epsilon=1e-8)",
    "stop": "exactly one binary STOP row, always legal",
}


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number, not boolean")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as a finite float") from error
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class TemporalPolicy:
    """Public-only variable-catalogue actor; inference requires exact config.

    Deterministic score selection intentionally differs from the count-balanced
    stochastic distribution. Critic gradients, clipping, and Adam counters never
    affect the actor update. All sample records contain detached copies.
    """

    def __init__(self, config: TemporalConfig, *, seed=0, hidden_size=64, temperature=.05):
        if type(config) is not TemporalConfig:
            raise ValueError("An explicit TemporalConfig is required")
        self.config = TemporalConfig(**asdict(config))
        self.feature_names = FEATURE_NAMES
        self.hidden_size = _integer(hidden_size, "hidden_size", 1, 512)
        self.temperature = _number(temperature, "temperature", positive=True)
        self.rng = np.random.default_rng(_integer(seed, "seed", 0, 2**63 - 1))
        width = len(self.feature_names)
        self.parameters = {"w1": self.rng.normal(0., 1. / np.sqrt(width), (width, self.hidden_size)),
                           "b1": np.zeros(self.hidden_size), "wa": np.zeros(self.hidden_size),
                           "wv": np.zeros(2 * width), "bv": np.zeros(1)}
        self.adam_m = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.adam_v = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.update_count = self.actor_update_count = self.critic_update_count = 0
        self.training_state, self.metadata = {}, {}

    def _matrix(self, features, mask):
        raw = np.asarray(features)
        if raw.dtype.kind not in "fiu":
            raise ValueError("Features must be a real numeric matrix")
        x, mask = np.asarray(raw, dtype=np.float64), np.asarray(mask)
        if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not 1 <= len(x) <= MAX_ROWS
                or x.size > MAX_ELEMENTS or not np.isfinite(x).all()):
            raise ValueError("Features must be a bounded, finite temporal feature matrix")
        if mask.dtype != np.bool_ or mask.shape != (len(x),) or not mask.any():
            raise ValueError("Action mask must be boolean, correctly shaped and nonempty")
        stops = x[:, FEATURE_NAMES.index("stop")]
        if not np.isin(stops, (0., 1.)).all() or np.count_nonzero(stops) != 1:
            raise ValueError("Exactly one binary STOP row is required")
        stop = int(np.flatnonzero(stops)[0])
        if not mask[stop] or x[stop, FEATURE_NAMES.index("temporal_marginal_return")] != 0.:
            raise ValueError("STOP must be legal and have exactly zero temporal prior")
        return x, mask, stop

    def _input(self, observation):
        if (not isinstance(observation, Mapping) or observation.get("feature_schema") != FEATURE_SCHEMA
                or tuple(observation.get("feature_names", ())) != self.feature_names
                or _json(observation.get("temporal_config")) != _json(asdict(self.config))):
            raise ValueError("Observation temporal schema, names/order or explicit config differ")
        return self._matrix(observation.get("option_features"), observation.get("action_mask"))

    def _forward(self, x):
        with np.errstate(over="ignore", invalid="ignore"):
            preactivation = x @ self.parameters["w1"] + self.parameters["b1"]
            hidden = np.tanh(preactivation)
            output = hidden @ self.parameters["wa"]
            bounded = np.tanh(output)
            scores = x[:, FEATURE_NAMES.index("temporal_marginal_return")] + .25 * bounded
        if not all(np.isfinite(value).all() for value in (preactivation, output, scores)):
            raise ValueError("Nonfinite temporal actor forward pass")
        return hidden, bounded, scores

    def _distribution(self, scores, mask, stop, temperature):
        deployments = mask.copy()
        deployments[stop] = False
        log_count = np.log(max(1, int(deployments.sum())))
        with np.errstate(over="ignore", invalid="ignore"):
            logits = scores[mask] / temperature - deployments[mask] * log_count
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite scaled temporal scores")
        with np.errstate(over="ignore", under="ignore"):
            shifted = logits - logits.max()
            exponential = np.exp(shifted)
        probability, log_probability = np.zeros(len(mask)), np.zeros(len(mask))
        probability[mask] = exponential / exponential.sum()
        log_probability[mask] = shifted - np.log(exponential.sum())
        # Zero-probability entries have zero entropy contribution/derivative.
        entropy_logs = np.where(probability > 0., log_probability, 0.)
        relative_logs = entropy_logs + deployments * log_count
        relative_entropy = -float(probability @ relative_logs)
        joint_entropy = -float(probability @ entropy_logs)
        stop_probability = float(probability[stop])
        deployment_probability = float(probability[deployments].sum())
        gate_entropy = -sum(value * np.log(value) for value in (stop_probability, deployment_probability) if value > 0.)
        diagnostics = {"entropy": relative_entropy, "joint_entropy": joint_entropy,
                       "gate_entropy": float(gate_entropy), "stop_probability": stop_probability,
                       "deployment_probability": deployment_probability}
        return probability, log_probability, relative_logs, diagnostics

    def _value(self, x, mask, stop):
        deployments = mask.copy()
        deployments[stop] = False
        mean = x[deployments].mean(axis=0) if deployments.any() else np.zeros(len(FEATURE_NAMES))
        context = np.concatenate((mean, x[stop]))
        field = lambda name: x[stop, FEATURE_NAMES.index(name)]
        prior = (16. * field("temporal_existing_timely") + 2. * field("temporal_existing_detection")
                 + field("temporal_existing_confirmation") + 3. * field("temporal_existing_early")
                 + 2. * field("existing_coverage") - field("existing_early_coverage") - 10.5
                 - .55 * 4. * (field("budget_total") - field("budget_remaining")))
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(prior + context @ self.parameters["wv"] + self.parameters["bv"][0])
        if not np.isfinite(context).all() or not np.isfinite(value):
            raise ValueError("Nonfinite temporal linear critic")
        return context, value

    def scores(self, observation):
        self._arrays()
        x, _, _ = self._input(observation)
        return self._forward(x)[2]

    def probabilities(self, observation, temperature=None):
        self._arrays()
        temperature = self.temperature if temperature is None else _number(temperature, "temperature", positive=True)
        x, mask, stop = self._input(observation)
        return self._distribution(self._forward(x)[2], mask, stop, temperature)[0]

    def value(self, observation):
        self._arrays()
        return self._value(*self._input(observation))[1]

    def act(self, observation, deterministic=True, temperature=None):
        if not isinstance(deterministic, (bool, np.bool_)):
            raise ValueError("deterministic must be boolean")
        if not deterministic:
            probability = self.probabilities(observation, temperature)
            return int(self.rng.choice(len(probability), p=probability))
        self._arrays()
        if temperature is not None:
            _number(temperature, "temperature", positive=True)
        x, mask, stop = self._input(observation)
        scores = self._forward(x)[2]
        best = int(np.flatnonzero(mask)[np.argmax(scores[mask])])
        return stop if scores[stop] >= scores[best] else best

    def sample(self, observation, temperature=None):
        self._arrays()
        temperature = self.temperature if temperature is None else _number(temperature, "temperature", positive=True)
        x, mask, stop = self._input(observation)
        probability = self._distribution(self._forward(x)[2], mask, stop, temperature)[0]
        value = self._value(x, mask, stop)[1]
        action = int(self.rng.choice(len(probability), p=probability))
        return action, {"features": x.copy(), "mask": mask.copy(), "action": action, "value": value,
                        "reward": 0., "temperature": temperature, "feature_schema": FEATURE_SCHEMA,
                        "temporal_config": asdict(self.config)}

    def _record(self, record):
        if (not isinstance(record, Mapping) or record.get("feature_schema") != FEATURE_SCHEMA
                or _json(record.get("temporal_config")) != _json(asdict(self.config))):
            raise ValueError("Sample record temporal semantics/config differ")
        x, mask, stop = self._matrix(record.get("features"), record.get("mask"))
        action = _integer(record.get("action"), "sample action", 0, len(mask) - 1)
        if not mask[action]:
            raise ValueError("Sample action is masked")
        return x, mask, stop, action, _number(record.get("temperature"), "sample temperature", positive=True)

    def _loss_and_gradients(self, transitions, returns, advantages, entropy_coef=0., value_coef=1.):
        """Analytic objective with sampled advantages held fixed, not recomputed."""
        self._arrays()
        entropy_coef, value_coef = _number(entropy_coef, "entropy_coef"), _number(value_coef, "value_coef")
        if not isinstance(transitions, Sequence) or not 1 <= len(transitions) <= MAX_TRANSITIONS:
            raise ValueError("Need a bounded nonempty transition sequence")
        returns, advantages = np.asarray(returns, dtype=float), np.asarray(advantages, dtype=float)
        if (returns.shape != (len(transitions),) or advantages.shape != returns.shape
                or not np.isfinite(returns).all() or not np.isfinite(advantages).all()):
            raise ValueError("Returns and detached advantages must be finite vectors matching transitions")
        gradients = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        sums = dict(actor_loss=0., critic_loss=0., entropy=0., joint_entropy=0., gate_entropy=0.,
                    stop_probability=0., deployment_probability=0.)
        elements = 0
        for record, target, advantage in zip(transitions, returns, advantages):
            x, mask, stop, action, temperature = self._record(record)
            elements += x.size
            if elements > MAX_ELEMENTS:
                raise ValueError("Temporal training batch exceeds feature bound")
            hidden, bounded, scores = self._forward(x)
            probability, log_probability, relative_logs, diagnostics = self._distribution(scores, mask, stop, temperature)
            if not np.isfinite(log_probability[action]):
                raise ValueError("Sample action has a nonfinite current log probability")
            context, value = self._value(x, mask, stop)
            error = value - target
            sums["actor_loss"] -= float(advantage * log_probability[action])
            sums["critic_loss"] += .5 * float(error * error)
            for key, metric in diagnostics.items():
                sums[key] += metric
            d_scores = advantage * probability
            d_scores[action] -= advantage
            d_scores += entropy_coef * probability * (relative_logs + diagnostics["entropy"])
            d_output = d_scores / temperature * .25 * (1. - bounded * bounded)
            gradients["wa"] += hidden.T @ d_output
            d_hidden = d_output[:, None] * self.parameters["wa"] * (1. - hidden * hidden)
            gradients["w1"] += x.T @ d_hidden
            gradients["b1"] += d_hidden.sum(axis=0)
            gradients["wv"] += value_coef * error * context
            gradients["bv"][0] += value_coef * error
        metrics = {key: value / len(transitions) for key, value in sums.items()}
        gradients = {key: value / len(transitions) for key, value in gradients.items()}
        loss = metrics["actor_loss"] + value_coef * metrics["critic_loss"] - entropy_coef * metrics["entropy"]
        if not np.isfinite(loss) or not all(np.isfinite(value).all() for value in gradients.values()):
            raise ValueError("Nonfinite temporal loss/gradients; no update applied")
        return float(loss), gradients, metrics

    def update(self, episodes, learning_rate=.001, critic_learning_rate=.001, *, gamma=1.,
               entropy_coef=0., value_coef=1., normalize_advantages=False, max_grad_norm=1.):
        """One complete-batch REINFORCE update; no critic-to-actor gradient path.

        Critic targets are original reward-to-go. Actor coefficients always use
        values detached by sample(), even if the critic subsequently changes.
        A zero value_coef leaves critic parameters, moments and counter intact.
        """
        learning_rate = _number(learning_rate, "learning_rate", positive=True)
        critic_learning_rate = _number(critic_learning_rate, "critic_learning_rate", positive=True)
        gamma, limit = _number(gamma, "gamma"), _number(max_grad_norm, "max_grad_norm", positive=True)
        if gamma > 1 or not isinstance(normalize_advantages, (bool, np.bool_)):
            raise ValueError("gamma must be in [0,1] and normalize_advantages boolean")
        if not isinstance(episodes, Sequence) or not 1 <= len(episodes) <= MAX_TRANSITIONS:
            raise ValueError("Need bounded nonempty complete episodes")
        transitions, targets = [], []
        for episode in episodes:
            if not isinstance(episode, Sequence) or not episode:
                raise ValueError("Each episode must be a nonempty sequence")
            if len(transitions) + len(episode) > MAX_TRANSITIONS:
                raise ValueError("Too many temporal training transitions")
            cumulative, reverse = 0., []
            for record in reversed(episode):
                if not isinstance(record, Mapping):
                    raise ValueError("Each transition must be a mapping")
                cumulative = _finite(record.get("reward"), "sample reward") + gamma * cumulative
                reverse.append(_finite(cumulative, "return-to-go"))
                _finite(record.get("value"), "sample value")
            transitions.extend(episode)
            targets.extend(reversed(reverse))
        returns = np.asarray(targets)
        advantages = returns - np.asarray([record["value"] for record in transitions])
        with np.errstate(over="ignore", invalid="ignore"):
            raw_mean, raw_std = float(advantages.mean()), float(advantages.std())
        if not np.isfinite(raw_mean) or not np.isfinite(raw_std):
            raise ValueError("Nonfinite advantage moments; no update applied")
        normalized = bool(normalize_advantages and len(advantages) > 1 and raw_std > 1e-8)
        if normalized:
            advantages = (advantages - raw_mean) / (raw_std + 1e-8)
        loss, gradients, metrics = self._loss_and_gradients(transitions, returns, advantages, entropy_coef, value_coef)
        _integer(self.update_count, "update_count before increment", 0, 2**63 - 2)
        parameters, moments, variances = (dict(group) for group in (self.parameters, self.adam_m, self.adam_v))
        counters = {"actor": self.actor_update_count, "critic": self.critic_update_count}
        for label, keys, rate in (("actor", ACTOR, learning_rate), ("critic", CRITIC, critic_learning_rate)):
            norm = float(np.sqrt(sum(np.sum(gradients[key] ** 2) for key in keys)))
            if not np.isfinite(norm):
                raise ValueError("Nonfinite temporal gradient norm; no update applied")
            scale = min(1., limit / max(norm, 1e-12))
            metrics.update({label + "_gradient_norm": norm, label + "_gradient_scale": scale})
            if label == "critic" and value_coef == 0:
                continue
            counters[label] += 1
            for key in keys:
                gradient = gradients[key] * scale
                moments[key] = .9 * self.adam_m[key] + .1 * gradient
                variances[key] = .999 * self.adam_v[key] + .001 * gradient * gradient
                mean = moments[key] / (1 - .9 ** counters[label])
                variance = variances[key] / (1 - .999 ** counters[label])
                parameters[key] = self.parameters[key] - rate * mean / (np.sqrt(variance) + 1e-8)
        if not all(np.isfinite(value).all() for group in (parameters, moments, variances) for value in group.values()):
            raise ValueError("Nonfinite Adam result; no update applied")
        self.parameters, self.adam_m, self.adam_v = parameters, moments, variances
        self.actor_update_count, self.critic_update_count = counters["actor"], counters["critic"]
        self.update_count += 1
        return {**metrics, "loss": loss, "mean_return_to_go": float(returns.mean()),
                "raw_advantage_mean": raw_mean, "raw_advantage_std": raw_std, "advantages_normalized": normalized,
                "transitions": len(transitions), "updates": self.update_count}

    def _arrays(self):
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("Temporal feature names/order cannot change")
        width = len(FEATURE_NAMES)
        shapes = {"w1": (width, self.hidden_size), "b1": (self.hidden_size,), "wa": (self.hidden_size,),
                  "wv": (2 * width,), "bv": (1,)}
        arrays = {}
        for prefix, group in (("param", self.parameters), ("adam_m", self.adam_m), ("adam_v", self.adam_v)):
            if set(group) != set(PARAMETERS):
                raise ValueError("Unexpected temporal parameter/moment names")
            for key, shape in shapes.items():
                value = group[key]
                if (not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != shape
                        or not np.isfinite(value).all() or (prefix == "adam_v" and (value < 0).any())):
                    raise ValueError("Invalid temporal parameter/moment array")
                arrays[f"{prefix}_{key}"] = value
        for name in ("update_count", "actor_update_count", "critic_update_count"):
            _integer(getattr(self, name), name, 0, 2**63 - 1)
        if self.actor_update_count != self.update_count or self.critic_update_count > self.update_count:
            raise ValueError("Inconsistent temporal optimizer counters")
        return arrays

    def _contract(self):
        return {"policy_schema": POLICY_SCHEMA, "policy_contract": POLICY_CONTRACT, "feature_schema": FEATURE_SCHEMA,
                "feature_names": list(self.feature_names), "temporal_config": asdict(self.config),
                "hidden_size": self.hidden_size, "temperature": self.temperature}

    def weights_fingerprint(self):
        self._arrays()
        digest = hashlib.sha256(_json(self._contract()).encode())
        for key in PARAMETERS:
            digest.update(key.encode())
            digest.update(np.asarray(self.parameters[key], dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def rng_fingerprint(self):
        return hashlib.sha256(_json(_rng_state(self.rng.bit_generator.state)).encode()).hexdigest()

    def save(self, path, training_state=None):
        """Write strict JSON/non-pickled NPZ to a NEW checkpoint directory only.

        Content is validated before writing; an interrupted write can leave an
        incomplete new directory. Existing checkpoints are never overwritten.
        """
        arrays = self._arrays()
        state = self.training_state if training_state is None else training_state
        if not isinstance(state, dict) or not isinstance(state.get("seed_provenance", {}), dict):
            raise ValueError("Training state and seed provenance must be dictionaries")
        metadata = {**self._contract(), "schema": CHECKPOINT_SCHEMA, "update_count": self.update_count,
                    "actor_update_count": self.actor_update_count, "critic_update_count": self.critic_update_count,
                    "rng_state": _rng_state(self.rng.bit_generator.state), "weights_sha256": self.weights_fingerprint(),
                    "training_state": state, "seed_provenance": state.get("seed_provenance", {})}
        metadata = _strict_json(_json(metadata))
        archive = io.BytesIO()
        np.savez_compressed(archive, **arrays)
        blob = archive.getvalue()
        metadata["arrays_sha256"] = hashlib.sha256(blob).hexdigest()
        raw = (_json(metadata) + "\n").encode()
        if len(raw) > 4_000_000 or len(blob) > 64_000_000:
            raise ValueError("Temporal checkpoint exceeds size bounds")
        path = Path(path)
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("Checkpoint path must not contain symlinks")
        path.mkdir(parents=True, exist_ok=False)
        with (path / "arrays.npz").open("xb") as handle:
            handle.write(blob)
        with (path / "checkpoint.json").open("xb") as handle:
            handle.write(raw)
        self.metadata, self.training_state = metadata, metadata["training_state"]

    @classmethod
    def load(cls, path, *, config: TemporalConfig):
        path = Path(path)
        with (path / "checkpoint.json").open("rb") as handle:
            raw = handle.read(4_000_001)
        with (path / "arrays.npz").open("rb") as handle:
            blob = handle.read(64_000_001)
        if len(raw) > 4_000_000 or len(blob) > 64_000_000:
            raise ValueError("Temporal checkpoint exceeds size bounds")
        metadata = _strict_json(raw)
        if not isinstance(metadata, dict) or metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported temporal checkpoint schema")
        policy = cls(config, hidden_size=metadata.get("hidden_size"), temperature=metadata.get("temperature"))
        contract = policy._contract()
        expected_fields = {*contract, "schema", "update_count", "actor_update_count", "critic_update_count",
                           "rng_state", "weights_sha256", "training_state", "seed_provenance", "arrays_sha256"}
        if set(metadata) != expected_fields or any(_json(metadata[key]) != _json(value) for key, value in contract.items()):
            raise ValueError("Temporal checkpoint policy/feature/config contract differs")
        if hashlib.sha256(blob).hexdigest() != metadata["arrays_sha256"]:
            raise ValueError("Temporal checkpoint archive hash differs")
        expected = {key: value.shape for key, value in policy._arrays().items()}
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                members = archive.infolist()
                if (len(members) != len(expected) or {m.filename for m in members} != {key + ".npy" for key in expected}
                        or any(m.file_size > int(np.prod(expected[m.filename[:-4]])) * 8 + 1024 for m in members)):
                    raise ValueError("Temporal NPZ has unexpected/oversized members")
                # Validate the NPY header BEFORE np.load can allocate from an
                # attacker-supplied shape inside an otherwise tiny ZIP member.
                for member in members:
                    with archive.open(member) as entry:
                        if np.lib.format.read_magic(entry) != (1, 0):
                            raise ValueError("Temporal checkpoint requires NPY 1.0 arrays")
                        shape, _, dtype = np.lib.format.read_array_header_1_0(entry)
                        if (shape != expected[member.filename[:-4]] or dtype != np.dtype("float64")
                                or member.file_size != entry.tell() + int(np.prod(shape)) * 8):
                            raise ValueError("Temporal NPY header shape/type/size differs")
            with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
                loaded = {key: archive[key].copy() for key in expected}
        except (zipfile.BadZipFile, KeyError, OSError, EOFError) as error:
            raise ValueError("Invalid temporal NPZ archive") from error
        for key in PARAMETERS:
            policy.parameters[key], policy.adam_m[key], policy.adam_v[key] = (
                loaded[f"{prefix}_{key}"] for prefix in ("param", "adam_m", "adam_v"))
        for name in ("update_count", "actor_update_count", "critic_update_count"):
            setattr(policy, name, _integer(metadata[name], name, 0, 2**63 - 1))
        policy._arrays()
        if policy.weights_fingerprint() != metadata["weights_sha256"]:
            raise ValueError("Temporal checkpoint weights hash differs")
        policy.rng.bit_generator.state = _rng_state(metadata["rng_state"])
        state = metadata["training_state"]
        if (not isinstance(state, dict) or not isinstance(metadata["seed_provenance"], dict)
                or _json(metadata["seed_provenance"]) != _json(state.get("seed_provenance", {}))):
            raise ValueError("Invalid temporal training state/seed provenance")
        policy.training_state, policy.metadata = state, metadata
        return policy
