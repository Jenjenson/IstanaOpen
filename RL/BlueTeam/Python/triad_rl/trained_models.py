"""Persistent local catalogue of named warning-time policies.

The catalogue contains completed runs and explicitly stopped runs that already
have a valid held-out checkpoint.  A model is published atomically after its
held-out checkpoint and all advertised native comparisons have been written,
so the console never exposes half-built entries.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata


MODEL_SCHEMA = "istana.named_warning_model.v1"
COMPARISON_SCHEMA = "istana.named_warning_model_comparison.v1"


def validate_model_name(value):
    if not isinstance(value, str):
        raise ValueError("Model name must be text")
    name = unicodedata.normalize("NFC", value).strip()
    if not 1 <= len(name) <= 64:
        raise ValueError("Model name must contain 1 to 64 characters")
    if any(unicodedata.category(character).startswith("C") for character in name):
        raise ValueError("Model name cannot contain control characters")
    return name


def _identifier(name):
    stem = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:36] or "model"
    digest = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:10]
    return f"trained-{stem}-{digest}"


class TrainedModelRegistry:
    """Read and atomically publish named model artifacts under ``Saved``."""

    def __init__(self, root):
        self.root = Path(root)

    def identifier_for(self, name):
        return _identifier(validate_model_name(name))

    def ensure_available(self, name):
        name = validate_model_name(name)
        identifier = _identifier(name)
        if (self.root / identifier).exists():
            raise ValueError(
                f'A trained model named "{name}" already exists. Choose a different name.')
        return name, identifier

    def register(self, *, name, policy_path, comparison_episode, metadata,
                 observed_episode=None):
        name, identifier = self.ensure_available(name)
        if comparison_episode.get("schema") != COMPARISON_SCHEMA:
            raise ValueError("Named model comparison evidence has the wrong schema")
        source = Path(policy_path)
        if not source.is_file():
            raise ValueError("Best policy checkpoint is missing")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".{identifier}.publishing"
        target = self.root / identifier
        if staging.exists() or target.exists():
            raise ValueError("A model with this name is already being published")
        staging.mkdir()
        try:
            shutil.copyfile(source, staging / "best-policy.json")
            (staging / "comparison.json").write_text(
                json.dumps(comparison_episode, indent=2, allow_nan=False) + "\n",
                encoding="utf-8", newline="\n")
            model = {"schema": MODEL_SCHEMA, "id": identifier, "name": name,
                     "policyFile": "best-policy.json", "comparisonFile": "comparison.json",
                     **deepcopy(metadata)}
            if observed_episode is not None:
                if (observed_episode.get("schema") != COMPARISON_SCHEMA
                        or observed_episode.get("id") != f"observed-{identifier}"):
                    raise ValueError("Best-observed episode evidence has the wrong schema")
                (staging / "best-observed-comparison.json").write_text(
                    json.dumps(observed_episode, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
                model["observedComparisonFile"] = "best-observed-comparison.json"
            (staging / "model.json").write_text(
                json.dumps(model, indent=2, allow_nan=False) + "\n",
                encoding="utf-8", newline="\n")
            staging.rename(target)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return deepcopy(model)

    def _read(self, directory):
        path = directory / "model.json"
        model = json.loads(path.read_text(encoding="utf-8"))
        if (model.get("schema") != MODEL_SCHEMA or model.get("id") != directory.name
                or _identifier(validate_model_name(model.get("name"))) != directory.name):
            raise ValueError(f"Invalid named model metadata: {directory.name}")
        for key in ("policyFile", "comparisonFile"):
            filename = model.get(key)
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"Invalid named model artifact path: {directory.name}")
            if not (directory / filename).is_file():
                raise ValueError(f"Named model artifact is missing: {directory.name}")
        if "observedComparisonFile" in model:
            filename = model["observedComparisonFile"]
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or not (directory / filename).is_file()):
                raise ValueError(f"Best-observed model artifact is missing: {directory.name}")
        return model

    def list(self):
        if not self.root.exists():
            return []
        models = [self._read(path) for path in self.root.iterdir()
                  if path.is_dir() and not path.name.startswith(".")]
        return sorted(models, key=lambda row: (row["name"].casefold(), row["id"]))

    def get(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"trained-[a-z0-9-]+", identifier):
            raise ValueError("Choose an available trained model")
        path = self.root / identifier
        if not path.is_dir():
            raise ValueError("Choose an available trained model")
        return self._read(path)

    def policy_path(self, identifier):
        model = self.get(identifier)
        return self.root / identifier / model["policyFile"]

    def comparison(self, identifier):
        model = self.get(identifier)
        row = json.loads((self.root / identifier / model["comparisonFile"]).read_text(
            encoding="utf-8"))
        if row.get("schema") != COMPARISON_SCHEMA or row.get("id") != identifier:
            raise ValueError("Named model comparison evidence is invalid")
        return row

    def observed_comparison(self, identifier):
        model = self.get(identifier)
        filename = model.get("observedComparisonFile")
        if not filename:
            raise ValueError("This trained model has no retained best-observed episode")
        row = json.loads((self.root / identifier / filename).read_text(encoding="utf-8"))
        if (row.get("schema") != COMPARISON_SCHEMA
                or row.get("id") != f"observed-{identifier}"):
            raise ValueError("Best-observed episode evidence is invalid")
        return row
