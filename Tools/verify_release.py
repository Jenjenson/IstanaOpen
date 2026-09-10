"""Verify complete source inputs or an offline Windows package (Python stdlib only).

python Tools/verify_release.py --source-only
python Tools/verify_release.py --release Releases/Windows --write-manifest
python Tools/verify_release.py --release Releases/Windows --runtime-report Saved/Review/hero.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, minimum: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum:
        raise ValueError(f"Missing or empty required file: {path}")
    with path.open("rb") as stream:
        if stream.read(len(LFS_POINTER)) == LFS_POINTER:
            raise ValueError(f"Git LFS content has not downloaded: {path}. Run git lfs pull.")


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path leaves the source folder: {relative}")
    return path


def verify_source(root: Path) -> dict:
    required = [
        "IstanaOpen.uproject", "Config/DefaultEngine.ini", "Config/DefaultGame.ini",
        "Source/IstanaOpen.Target.cs", "Source/IstanaOpenEditor.Target.cs",
        "Source/IstanaOpen/IstanaOpen.Build.cs", "Source/IstanaOpen/IstanaGameMode.cpp",
        "Tools/build_scene.py", "SourceAssets/manifest.json",
    ]
    for relative in required:
        require_file(root / relative)
    manifest = json.loads((root / "SourceAssets/manifest.json").read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Source asset manifest must be a nonempty list.")
    total = 0
    for item in manifest:
        path = within(root, item["path"])
        require_file(path)
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Source asset differs from its provenance manifest: {path}")
        total += path.stat().st_size
    # Architecture and surroundings are generated original/ODbL geometry outside the scan manifest.
    for folder in ("Architecture", "Environment"):
        meshes = list((root / "SourceAssets" / folder).glob("*.obj"))
        if not meshes:
            raise ValueError(f"Generated geometry is missing: SourceAssets/{folder}")
        for mesh in meshes:
            require_file(mesh, 100)
    project = json.loads((root / "IstanaOpen.uproject").read_text(encoding="utf-8-sig"))
    forbidden = [p["Name"] for p in project.get("Plugins", []) if p.get("Enabled") and "airsim" in p["Name"].lower()]
    if forbidden:
        raise ValueError(f"Unexpected AirSim dependency: {forbidden}")
    return {"source_assets": len(manifest), "source_bytes": total, "source_integrity": "passed"}


def verify_release(release: Path, write_manifest: bool) -> dict:
    if not release.is_dir():
        raise ValueError(f"Release directory does not exist: {release}")
    for relative in (
        "IstanaOpen.exe", "Start_Istana.cmd", "Start_Compatibility.cmd",
        "Install_Prerequisites.cmd", "START_HERE.txt", "source-assets-manifest.json",
        "Engine/Extras/Redist/en-us/UEPrereqSetup_x64.exe",
        "OpenData/osm_context.overpass.json", "OpenData/context_provenance.json",
        "OpenData/generate_environment.py", "Docs/LICENSING.md", "LICENSE", "END_USER_TERMS.md",
    ):
        require_file(release / relative)
    game_bins = list((release / "IstanaOpen/Binaries/Win64").glob("IstanaOpen*.exe"))
    if not game_bins:
        raise ValueError("The actual game executable is missing beneath IstanaOpen/Binaries/Win64.")
    for path in game_bins:
        require_file(path, 1024 * 1024)
    pak_dir = release / "IstanaOpen/Content/Paks"
    containers = []
    for extension in ("*.pak", "*.utoc", "*.ucas"):
        matches = list(pak_dir.glob(extension))
        if not matches:
            raise ValueError(f"Cooked asset container is missing: {pak_dir / extension}")
        for match in matches:
            require_file(match)
        containers.extend(matches)
    notices = [p for p in release.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt") and any(k in p.name.lower() for k in ("licens", "notice", "attribution"))]
    if not notices:
        raise ValueError("Project license/attribution documents are missing from the release.")
    files = sorted(p for p in release.rglob("*") if p.is_file() and p.name != "release-manifest.json")
    unexpected = [p for p in files if "airsim" in p.as_posix().lower() or p.name in ("cesium-request-cache.sqlite",)]
    if unexpected:
        raise ValueError(f"Unexpected legacy/provider file: {unexpected[0]}")
    result = {"release_structure": "passed", "files": len(files), "bytes": sum(p.stat().st_size for p in files), "cooked_containers": len(containers)}
    if write_manifest:
        records = [{"path": p.relative_to(release).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
        (release / "release-manifest.json").write_text(json.dumps({"project": "IstanaOpen", "files": records}, indent=2), encoding="utf-8")
    else:
        manifest_path = release / "release-manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for item in manifest["files"]:
                path = within(release, item["path"])
                # Unreal intentionally emits some empty staged .ini files.
                # Their existence and exact empty-file checksum still matter.
                require_file(path, 0 if item["bytes"] == 0 else 1)
                if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                    raise ValueError(f"Release file failed checksum verification: {path}")
            result["release_checksums"] = "passed"
    return result


def verify_runtime(report_path: Path) -> dict:
    require_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    for key in ("standalone_world", "camera_possessed", "screenshot_saved"):
        if report.get(key) is not True:
            raise ValueError(f"Runtime check failed: {key}")
    if report.get("static_mesh_actors", 0) < 1:
        raise ValueError("Runtime world contains no static mesh actors.")
    if report.get("viewport_width", 0) < 640 or report.get("viewport_height", 0) < 360:
        raise ValueError("Runtime viewport was smaller than the review minimum.")
    if report.get("average_fps_after_warmup", 0) <= 0 or report.get("measured_frames", 0) < 1:
        raise ValueError("Runtime report contains no valid frame measurements.")
    if not report.get("screenshot"):
        raise ValueError("Runtime report does not name a screenshot.")
    require_file(Path(report["screenshot"]), 1000)
    return {"runtime_checks": "passed", "gpu": report.get("gpu"), "average_fps": report["average_fps_after_warmup"], "screenshot": report["screenshot"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--release", type=Path, default=PROJECT / "Releases/Windows")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--runtime-report", type=Path)
    args = parser.parse_args()
    try:
        result = verify_source(args.source_root.resolve()) if args.source_only else verify_release(args.release.resolve(), args.write_manifest)
        if args.runtime_report:
            result.update(verify_runtime(args.runtime_report))
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
