#!/usr/bin/env python3
"""Audit the publishable source tree without inspecting build/cache directories.

Usage: python Tools/audit_submission.py
Produces Docs/submission-audit.json with SHA-256 inventory and geometry checks.
Returns 1 for actionable findings. Secret values are never printed or recorded.
This is a technical provenance/configuration check, not a legal determination.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


SKIP_DIRS = {".git", ".vs", "binaries", "build", "intermediate", "saved", "releases",
             "deriveddatacache", "ddc", "__pycache__", "node_modules"}
FORBIDDEN_DIRS = {"airsim", "startercontent", "megascans", "quixel", "ultradynamicsky",
                  "ultra_dynamic_sky", "googlephotorealistic3dtiles", "engine"}
FORBIDDEN_PLUGINS = {"airsim", "cesiumforunreal", "ultradynamicsky", "androidfileserver"}
CONFIG_SUFFIXES = {".ini", ".json", ".uproject", ".uplugin", ".yaml", ".yml", ".toml", ".env"}
CODE_SUFFIXES = {".py", ".cpp", ".h", ".cs", ".ps1", ".sh", ".bat", ".cmd"}
KEY_PATTERN = re.compile(r"(?:token|api[_-]?key|secret|password|authorization|private[_-]?key)", re.I)
ASSIGNMENT = re.compile(r'''^\s*["']?([A-Za-z][A-Za-z0-9_.-]*)["']?\s*[:=]\s*(.*?)\s*,?\s*$''')
KNOWN_SECRET = re.compile(r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{30,})")
RESTRICTED_ASSET_REFERENCES = (b"/AirSim/", b"/Game/StarterContent/", b"/Game/Megascans/",
                               b"/Game/UltraDynamicSky/", b"/UltraDynamicSky/")


def placeholder(value):
    value = value.strip().strip("\"'").rstrip(",").strip()
    if value.lower() in {"", "none", "null", "false", "true", "0", "not_set", "changeme", "redacted"}:
        return True
    if value.startswith(("${", "$env:", "os.environ", "<", "YOUR_", "your_", "example", "EXAMPLE")):
        return True
    return len(value) < 8


def source_files(root, report_path):
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS and not (Path(folder) / d).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            if path.resolve() == report_path.resolve() or path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            yield path


def hash_file(path, check_asset_references=False):
    digest = hashlib.sha256()
    refs = set()
    tail = b""
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
            if check_asset_references:
                sample = tail + data
                for signature in RESTRICTED_ASSET_REFERENCES:
                    if signature in sample or signature.decode().encode("utf-16-le") in sample:
                        refs.add(signature.decode())
                tail = sample[-128:]
    return digest.hexdigest(), sorted(refs)


def inspect_config(path, relative):
    findings = []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    parsed = None
    if path.suffix.lower() in {".json", ".uproject", ".uplugin"}:
        try:
            parsed = json.loads(text)
        except ValueError:
            if path.suffix.lower() in {".uproject", ".uplugin"}:
                findings.append({"kind": "invalid_project_json", "path": relative})
    if isinstance(parsed, dict) and path.suffix.lower() in {".uproject", ".uplugin"}:
        for plugin in parsed.get("Plugins", []):
            if plugin.get("Enabled") and plugin.get("Name", "").lower() in FORBIDDEN_PLUGINS:
                findings.append({"kind": "unexpected_enabled_plugin", "path": relative, "plugin": plugin["Name"]})

    def json_credentials(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                location = prefix + "." + key if prefix else key
                if KEY_PATTERN.search(key) and isinstance(item, str) and not placeholder(item):
                    findings.append({"kind": "nonempty_credential_setting", "path": relative, "key": location})
                elif isinstance(item, (dict, list)):
                    json_credentials(item, location)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                json_credentials(item, prefix + f"[{index}]")

    json_credentials(parsed)
    # Explanatory Markdown is deliberately outside this configuration scan.
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("#", ";", "//")):
            continue
        match = ASSIGNMENT.match(line)
        if parsed is None and match and KEY_PATTERN.search(match[1]) and not placeholder(match[2]):
            findings.append({"kind": "nonempty_credential_setting", "path": relative, "line": lineno, "key": match[1]})
        if KNOWN_SECRET.search(line):
            findings.append({"kind": "credential_format_detected", "path": relative, "line": lineno})
    return findings


def obj_check(path):
    """Validate finite geometry, every OBJ index and triangle area in bounded memory."""
    vertices = array("d")
    counts = Counter()
    errors = []
    bounds_min = [float("inf")] * 3
    bounds_max = [-float("inf")] * 3
    materials = set()
    with path.open(encoding="utf-8", errors="strict") as stream:
        for lineno, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            kind = fields[0]
            if kind in {"v", "vt", "vn"}:
                counts[kind] += 1
                try:
                    values = [float(x) for x in fields[1:]]
                    if not all(math.isfinite(x) for x in values):
                        raise ValueError("non-finite")
                    if len(values) < (2 if kind == "vt" else 3):
                        raise ValueError("too few coordinates")
                except ValueError as exc:
                    errors.append({"line": lineno, "reason": str(exc)})
                    values = [0., 0., 0.]
                if kind == "v":
                    vertices.extend(values[:3])
                    for i, value in enumerate(values[:3]):
                        bounds_min[i] = min(bounds_min[i], value)
                        bounds_max[i] = max(bounds_max[i], value)
            elif kind == "usemtl" and len(fields) > 1:
                materials.add(" ".join(fields[1:]))
    seen = Counter()
    zero_area = 0
    zero_area_lines = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            kind = fields[0]
            if kind in {"v", "vt", "vn"}:
                seen[kind] += 1
                continue
            if kind != "f":
                continue
            counts["faces"] += 1
            if len(fields) < 4:
                errors.append({"line": lineno, "reason": "face has fewer than three vertices"})
                continue
            indices = []
            valid = True
            for token in fields[1:]:
                components = token.split("/")
                if len(components) > 3:
                    valid = False
                for offset, index in enumerate(components[:3]):
                    if index == "" and offset > 0:
                        continue
                    index_kind = ("v", "vt", "vn")[offset]
                    try:
                        raw_index = int(index)
                        resolved = raw_index - 1 if raw_index > 0 else seen[index_kind] + raw_index
                        if raw_index == 0 or not 0 <= resolved < counts[index_kind]:
                            raise ValueError("index outside defined array")
                        if offset == 0:
                            indices.append(resolved)
                    except ValueError:
                        valid = False
            if not valid or len(indices) != len(fields) - 1:
                errors.append({"line": lineno, "reason": "invalid vertex/UV/normal index"})
                continue
            counts["triangles"] += len(indices) - 2
            a = vertices[indices[0] * 3:indices[0] * 3 + 3]
            for j in range(1, len(indices) - 1):
                b = vertices[indices[j] * 3:indices[j] * 3 + 3]
                c = vertices[indices[j + 1] * 3:indices[j + 1] * 3 + 3]
                ab = [b[i] - a[i] for i in range(3)]
                ac = [c[i] - a[i] for i in range(3)]
                area = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
                if sum(v * v for v in area) < 1e-12:
                    zero_area += 1
                    if len(zero_area_lines) < 8:
                        zero_area_lines.append(lineno)
    if not counts["v"] or not counts["faces"]:
        errors.append({"reason": "empty mesh"})
    return {"counts": dict(counts), "bounds": [bounds_min, bounds_max] if counts["v"] else None,
            "materials": sorted(materials), "zero_area_triangles": zero_area,
            "zero_area_sample_lines": zero_area_lines, "errors_count": len(errors), "errors": errors[:20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: ROOT/Docs/submission-audit.json")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "Docs" / "submission-audit.json"
    findings, warnings, inventory, geometry = [], [], [], {}
    total_bytes = 0
    extension_counts = Counter()
    for path in source_files(root, output):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"kind": "source_symlink_requires_review", "path": relative})
            continue
        lower_parts = {p.lower() for p in path.relative_to(root).parts[:-1]}
        if forbidden := lower_parts.intersection(FORBIDDEN_DIRS):
            findings.append({"kind": "excluded_asset_or_engine_directory", "path": relative, "directories": sorted(forbidden)})
        size = path.stat().st_size
        digest, refs = hash_file(path, path.suffix.lower() in {".uasset", ".umap"})
        inventory.append({"path": relative, "bytes": size, "sha256": digest})
        total_bytes += size
        extension_counts[path.suffix.lower() or "(none)"] += 1
        for ref in refs:
            findings.append({"kind": "restricted_asset_package_reference", "path": relative, "reference": ref})
        if path.suffix.lower() in {".exe", ".dll", ".lib", ".pdb"}:
            findings.append({"kind": "unexpected_binary_in_source_tree", "path": relative})
        if size > 95 * 1024 * 1024:
            warnings.append({"kind": "git_lfs_required", "path": relative, "bytes": size})
        if path.suffix.lower() in CONFIG_SUFFIXES or path.name.startswith(".env"):
            if size <= 4 * 1024 * 1024:
                findings.extend(inspect_config(path, relative))
        elif path.suffix.lower() in CODE_SUFFIXES and path.name != Path(__file__).name and size < 2 * 1024 * 1024:
            for lineno, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
                if KNOWN_SECRET.search(line):
                    findings.append({"kind": "credential_format_detected", "path": relative, "line": lineno})
        if path.suffix.lower() == ".obj":
            result = obj_check(path)
            geometry[relative] = result
            if result["errors_count"]:
                findings.append({"kind": "invalid_obj_geometry", "path": relative, "count": result["errors_count"]})
            if result["zero_area_triangles"]:
                warnings.append({"kind": "degenerate_obj_triangles", "path": relative, "count": result["zero_area_triangles"]})
    indexed = {entry["path"]: entry for entry in inventory}
    provenance = root / "SourceAssets" / "manifest.json"
    provenance_verified = 0
    if provenance.exists():
        try:
            entries = json.loads(provenance.read_text(encoding="utf-8-sig"))
            if not isinstance(entries, list):
                raise ValueError("expected list")
            for entry in entries:
                relative = entry["path"].replace("\\", "/")
                actual = indexed.get(relative)
                if actual is None:
                    findings.append({"kind": "provenance_source_missing", "path": relative})
                elif actual["sha256"] != entry["sha256"] or actual["bytes"] != entry["bytes"]:
                    findings.append({"kind": "provenance_hash_or_size_mismatch", "path": relative})
                else:
                    provenance_verified += 1
        except (ValueError, KeyError, TypeError):
            findings.append({"kind": "invalid_provenance_manifest", "path": "SourceAssets/manifest.json"})
    else:
        findings.append({"kind": "missing_provenance_manifest", "path": "SourceAssets/manifest.json"})
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if not findings else "FAIL",
              "scope": "Source files only; build products, caches and release packages excluded. Technical checks do not determine legal rights.",
              "excluded_directories": sorted(SKIP_DIRS), "file_count": len(inventory), "total_bytes": total_bytes,
              "extensions": dict(sorted(extension_counts.items())), "provenance_entries_verified": provenance_verified,
              "obj_file_count": len(geometry), "obj_total_triangles": sum(item["counts"].get("triangles", 0) for item in geometry.values()),
              "findings": findings, "warnings": warnings, "geometry": geometry, "files": inventory}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output), "files": len(inventory),
                      "bytes": total_bytes, "obj_files": len(geometry), "findings": findings,
                      "warnings": warnings, "provenance_verified": provenance_verified}, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
