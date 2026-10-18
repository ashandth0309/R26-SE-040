"""Where the vision model files live and how they are verified.

Model binaries are never committed.  ``vision_models.json`` pins each file
by URL (a fixed upstream commit) and SHA-256 so a download can be checked.
Models are looked up in ``$BUDDY_VISION_MODELS`` or ``~/.buddy/models/vision``.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("vision_models.json")


class ModelMissing(RuntimeError):
    """A required model file is absent or does not match its hash."""


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_dir() -> Path:
    env = os.environ.get("BUDDY_VISION_MODELS")

    return Path(env).expanduser() if env else \
        Path("~/.buddy/models/vision").expanduser()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def model_path(key: str, directory: Path | None = None,
               manifest: dict | None = None, verify: bool = True) -> Path:
    """Path of a model, checked against the manifest hash."""

    manifest = manifest or load_manifest()

    if key not in manifest["models"]:
        raise ModelMissing(f"unknown model key {key!r}")

    entry = manifest["models"][key]
    path = (directory or model_dir()) / entry["file"]

    if not path.is_file():
        raise ModelMissing(
            f"{key}: {path} not found; run scripts/fetch_vision_models.py")

    if verify and sha256_of(path) != entry["sha256"]:
        raise ModelMissing(f"{key}: {path} does not match the pinned hash")

    return path


def available(key: str, directory: Path | None = None) -> bool:
    try:
        model_path(key, directory)

        return True
    except ModelMissing:
        return False


def fetch(keys: list[str] | None = None, directory: Path | None = None,
          manifest: dict | None = None,
          opener=urllib.request.urlopen) -> dict[str, str]:
    """Download missing models and verify them.  Returns key -> status.

    A file whose hash does not match is deleted and reported, never kept.
    """

    manifest = manifest or load_manifest()
    directory = directory or model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    for key in keys or list(manifest["models"]):
        entry = manifest["models"][key]
        dest = directory / entry["file"]

        if dest.is_file() and sha256_of(dest) == entry["sha256"]:
            status[key] = "present"

            continue

        tmp = dest.with_suffix(dest.suffix + ".part")

        try:
            with opener(entry["url"], timeout=120) as resp:
                tmp.write_bytes(resp.read())
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            status[key] = f"download failed: {exc}"

            continue

        if sha256_of(tmp) != entry["sha256"]:
            tmp.unlink(missing_ok=True)
            status[key] = "hash mismatch (discarded)"

            continue

        os.replace(tmp, dest)
        status[key] = "downloaded"

    return status
