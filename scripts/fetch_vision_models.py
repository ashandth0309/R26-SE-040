#!/usr/bin/env python3
"""Download the pinned vision models and verify their hashes.

    python3 scripts/fetch_vision_models.py            # all models
    python3 scripts/fetch_vision_models.py face_detector face_embedder

Files go to $BUDDY_VISION_MODELS or ~/.buddy/models/vision (never into git).
"""

import sys

from buddy_core.vision import models


def main(argv: list[str]) -> int:
    manifest = models.load_manifest()
    keys = argv or None
    status = models.fetch(keys, manifest=manifest)

    for key, text in status.items():
        entry = manifest["models"][key]
        print(f"{key:16s} {text:22s} {entry['licence']:11s} {entry['file']}")

    print(f"directory: {models.model_dir()}")

    return 0 if all(s in ("present", "downloaded")
                    for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
