"""What the vision subsystem keeps, where, and for how long (Phase 19).

``DATA_INVENTORY`` is the single list the docs and tests are checked
against.  If a new feature stores something, it must appear here.
"""

from __future__ import annotations

import re
from pathlib import Path

DATA_INVENTORY = (
    {"data": "face templates (128-number vectors), name, per-person "
             "settings, consent record, audit log",
     "where": "SQLite file, owner-only (0600), default ~/.buddy/vision/"
              "faces.db",
     "created_when": "only after the person confirms out loud",
     "retention": "until the person asks to be forgotten, or the optional "
                  "purge for people not seen for N days",
     "delete": "'forget my face' (secure delete + VACUUM) or the store API",
     "photos": False, "encrypted_at_rest": False},
    {"data": "rejected identity matches (negative examples, vectors only)",
     "where": "same SQLite file", "created_when": "the person corrects a "
     "wrong recognition", "retention": "kept with the profile, newest 20",
     "delete": "deleted with the profile", "photos": False,
     "encrypted_at_rest": False},
    {"data": "check-in cooldowns and counters per person",
     "where": "same SQLite file", "created_when": "after an offer",
     "retention": "with the profile", "delete": "deleted with the profile",
     "photos": False, "encrypted_at_rest": False},
    {"data": "taught objects (feature vectors and a name)",
     "where": "SQLite file, owner-only", "created_when": "only after a "
     "spoken yes", "retention": "until forgotten",
     "delete": "'forget <name>' or forget everything", "photos": False,
     "encrypted_at_rest": False},
    {"data": "where taught or chosen objects were last seen (place, "
             "coarse region, time)",
     "where": "SQLite file, owner-only", "created_when": "only while "
     "object memory is switched on (off by default)",
     "retention": "7 days by default", "delete": "'forget where things "
     "are' wipes it", "photos": False, "encrypted_at_rest": False},
    {"data": "vision settings", "where": "JSON file, owner-only",
     "created_when": "when changed", "retention": "until changed",
     "delete": "delete the file", "photos": False,
     "encrypted_at_rest": False},
    {"data": "camera frames", "where": "memory only", "created_when":
     "while the camera is on", "retention": "dropped within moments "
     "(newest frames only)", "delete": "nothing is written to disk",
     "photos": False, "encrypted_at_rest": False},
    {"data": "expression, posture, gesture and fall states",
     "where": "memory only", "created_when": "while analysed",
     "retention": "a few seconds to minutes", "delete": "nothing is "
     "written to disk", "photos": False, "encrypted_at_rest": False},
)

# Modules that are allowed to write files at all, and why:
#   face_store, object_teach, object_memory, vision_settings: user data above
#   models, object_embedding: downloaded / derived model files
#   camera_service: the camera lock file (no image data)
#   evaluation, expression_eval, object_eval: JSON reports
#   eval_report: Markdown report from those JSON files
_WRITERS_ALLOWED = {"face_store.py", "object_teach.py", "object_memory.py",
                    "vision_settings.py", "models.py", "object_embedding.py",
                    "camera_service.py", "evaluation.py",
                    "expression_eval.py", "object_eval.py", "resources.py",
                    "benchmark.py", "eval_report.py"}
_IMAGE_WRITE = re.compile(
    r"\b(imwrite|imencode|\.save\(|write_image|VideoWriter|tofile\()")


def scan_for_image_writes(package_dir: str | Path) -> list[str]:
    """Return 'file:line' for any code that could write an image to disk.

    The vision package must not save frames, crops or faces.  The only
    tolerated hits are none; evaluation tools read images, they never
    write them.
    """

    hits = []

    for p in sorted(Path(package_dir).glob("*.py")):
        if p.name == "privacy.py":
            continue

        for n, line in enumerate(p.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]

            if _IMAGE_WRITE.search(code):
                hits.append(f"{p.name}:{n}")

    return hits


def files_that_open_for_writing(package_dir: str | Path) -> set[str]:
    """Modules that open files in a write mode (for review)."""

    out = set()
    pat = re.compile(r"open\([^)]*['\"][wax]b?\+?['\"]|write_text|"
                     r"write_bytes|os\.open\(")

    for p in Path(package_dir).glob("*.py"):
        if p.name != "privacy.py" and pat.search(p.read_text()):
            out.add(p.name)

    return out


def allowed_writers() -> set[str]:
    return set(_WRITERS_ALLOWED)
