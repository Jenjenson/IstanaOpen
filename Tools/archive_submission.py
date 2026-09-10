"""Archive the complete public source and portable Windows package (stdlib only).

python Tools/archive_submission.py --source-only --dry-run
python Tools/archive_submission.py --dry-run --list
python Tools/archive_submission.py --date 2026-09-10

Run final archives after validation documents and the release are complete.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("Config", "Content", "Source", "SourceAssets", "Tools", "Docs", "ThirdParty")
SKIP = {"__pycache__", ".git", "cache", ".cache", "deriveddatacache", "intermediate", "saved"}
SOURCE_SKIP = SKIP | {"build", "release", "releases", "binaries"}
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
UNREAL_PACKAGE_MAGIC = b"\xc1\x83\x2a\x9e"


def root_file_allowed(path: Path) -> bool:
    name = path.name.lower()
    return name in {".gitignore", ".gitattributes"} or path.suffix.lower() == ".uproject" or any(
        name == prefix or name.startswith(prefix + ".") for prefix in ("readme", "license", "end_user_terms")
    )


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError(f"Unsafe ZIP entry name: {name!r}")
    return path.as_posix()


def check_file(path: Path, base: Path, source: bool) -> None:
    if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Refusing a link or file outside its input root: {path}")
    with path.open("rb") as stream:
        header = stream.read(128)
    if header.lstrip(b"\xef\xbb\xbf\r\n ").startswith(LFS_POINTER):
        raise ValueError(f"Git LFS pointer instead of actual asset: {path}. Run git lfs pull first.")
    if source and path.suffix.lower() in {".uasset", ".umap"} and not header.startswith(UNREAL_PACKAGE_MAGIC):
        raise ValueError(f"Unreal source asset does not have a binary package header: {path}")


def collect(base: Path, source: bool) -> list[tuple[Path, str, int, int]]:
    """Return whitelisted paths with relative ZIP names and a mutation snapshot."""
    if not base.is_dir():
        raise ValueError(f"Input directory does not exist: {base}")
    folders = [base / name for name in SOURCE_DIRS] if source else [base]
    if source:
        missing = [str(p.relative_to(base)) for p in folders if not p.is_dir()]
        if missing:
            raise ValueError(f"Source folders missing: {', '.join(missing)}")
        files = [p for p in base.iterdir() if p.is_file() and root_file_allowed(p)]
        for name in (".gitignore", ".gitattributes", "IstanaOpen.uproject", "README.md", "LICENSE", "END_USER_TERMS.md"):
            if not (base / name).is_file():
                raise ValueError(f"Required source file missing: {name}")
    else:
        files = []
        for name in ("IstanaOpen.exe", "Start_Istana.cmd", "START_HERE.txt", "END_USER_TERMS.md", "source-assets-manifest.json"):
            if not (base / name).is_file():
                raise ValueError(f"Required portable release file missing: {name}")
    excluded = SOURCE_SKIP if source else SKIP
    for folder in folders:
        if folder.is_symlink() or not folder.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"Refusing linked source directory: {folder}")
        for current, dirs, names in os.walk(folder, followlinks=False):
            current = Path(current)
            dirs[:] = sorted(n for n in dirs if n.casefold() not in excluded)
            for name in dirs:
                p = current / name
                if p.is_symlink() or not p.resolve().is_relative_to(base.resolve()):
                    raise ValueError(f"Refusing linked directory: {p}")
            files.extend(current / n for n in names if Path(n).suffix.lower() not in {".pyc", ".pyo"})
    result = []
    for path in sorted(files, key=lambda p: p.relative_to(base).as_posix().casefold()):
        check_file(path, base, source)
        name = safe_name(path.relative_to(base).as_posix())
        stat = path.stat()
        result.append((path, name, stat.st_size, stat.st_mtime_ns))
    if not result:
        raise ValueError("No files selected")
    if source and not any(p.suffix.lower() == ".uasset" for p, _, _, _ in result):
        raise ValueError("Source archive must include real .uasset files")
    return result


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def archive(files, target: Path, prefix: str, overwrite: bool) -> dict:
    if target.exists() and not overwrite:
        raise ValueError(f"Archive already exists: {target}. Use a new --date or --overwrite.")
    temporary = target.with_name(target.name + f".partial-{os.getpid()}")
    # A failed attempt leaves only its own diagnostic partial file; inputs are untouched.
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3,
                         allowZip64=True, strict_timestamps=False) as output:
        for index, (path, name, size, modified) in enumerate(files, 1):
            before = path.stat()
            if (before.st_size, before.st_mtime_ns) != (size, modified):
                raise ValueError(f"Input changed after preflight: {path}")
            output.write(path, safe_name(prefix + "/" + name),
                         compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (size, modified):
                raise ValueError(f"Input changed while archiving: {path}")
            if index % 100 == 0 or index == len(files):
                print(f"{target.name}: {index}/{len(files)} files", flush=True)
    print(f"Verifying CRC for {target.name}", flush=True)
    with zipfile.ZipFile(temporary, "r") as output:
        names = output.namelist()
        if len(names) != len(files) or len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate or missing entries")
        for name in names:
            safe_name(name)
        bad = output.testzip()
        if bad:
            raise ValueError(f"Archive CRC failed: {bad}")
    checksum = digest(temporary)
    temporary.replace(target)
    return dict(archive=target.name, bytes=target.stat().st_size, sha256=checksum, crc="passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source-only", action="store_true")
    mode.add_argument("--release-only", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--release-root", type=Path, help="Defaults to SOURCE_ROOT/Releases/Windows")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD filename version")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report files/size; write nothing")
    parser.add_argument("--list", action="store_true", help="Print every selected relative path and size")
    parser.add_argument("--overwrite", action="store_true", help="Replace only matching completed output archives after verification")
    args = parser.parse_args()
    root = args.source_root.resolve()
    release = (args.release_root or root / "Releases/Windows").resolve()
    destination = root / "Releases"
    jobs = []
    if not args.release_only:
        jobs.append(("Source", root, True))
    if not args.source_only:
        jobs.append(("Windows", release, False))
    prepared = []
    for label, base, source in jobs:
        files = collect(base, source)
        target = destination / f"IstanaOpen-{label}-{args.date.isoformat()}.zip"
        if target.exists() and not args.overwrite and not args.dry_run:
            raise ValueError(f"Archive already exists: {target}; use --overwrite explicitly.")
        if args.list:
            for _, name, size, _ in files:
                print(f"{label}\t{size}\t{name}")
        print(json.dumps(dict(kind=label, files=len(files), uncompressed_bytes=sum(f[2] for f in files),
                              output=str(target), binary_assets_verified=source, dry_run=args.dry_run)), flush=True)
        prepared.append((files, target, "IstanaOpen-" + label))
    if args.dry_run:
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    receipts = [archive(files, target, prefix, args.overwrite) for files, target, prefix in prepared]
    known = {r["archive"]: r["sha256"] for r in receipts}
    # Keep the other completed archive for this same version in the checksum file.
    for label in ("Source", "Windows"):
        target = destination / f"IstanaOpen-{label}-{args.date.isoformat()}.zip"
        if target.is_file() and target.name not in known:
            known[target.name] = digest(target)
    sums = destination / "SHA256SUMS.txt"
    temporary_sums = sums.with_name(sums.name + f".partial-{os.getpid()}")
    temporary_sums.write_text("".join(f"{value}  {name}\n" for name, value in sorted(known.items())), encoding="ascii")
    temporary_sums.replace(sums)
    print(json.dumps(dict(archives=receipts, checksums=str(sums)), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Archive failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
