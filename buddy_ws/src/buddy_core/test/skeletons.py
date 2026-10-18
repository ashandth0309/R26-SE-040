"""Hand-built skeletons for geometry tests (NOT real pose output)."""

import numpy as np

from buddy_core.vision import pose as P


def make(points: dict, vis: float = 0.95, hidden=()) -> P.Pose:
    xy = np.zeros((33, 2), np.float32)
    v = np.zeros(33, np.float32)

    for idx, (x, y) in points.items():
        xy[idx] = (x, y)
        v[idx] = 0.0 if idx in hidden else vis

    ok = v > 0.3
    pts = xy[ok]
    box = (float(pts[:, 0].min()), float(pts[:, 1].min()),
           float(pts[:, 0].max()), float(pts[:, 1].max()))

    return P.Pose(xy, v, 0.9, 0, box)


def standing(x=300.0, top=100.0, scale=1.0):
    s = scale

    return make({
        P.NOSE: (x, top), P.L_SHOULDER: (x - 30 * s, top + 60 * s),
        P.R_SHOULDER: (x + 30 * s, top + 60 * s),
        P.L_WRIST: (x - 40 * s, top + 190 * s),
        P.R_WRIST: (x + 40 * s, top + 190 * s),
        P.L_HIP: (x - 20 * s, top + 200 * s),
        P.R_HIP: (x + 20 * s, top + 200 * s),
        P.L_KNEE: (x - 20 * s, top + 320 * s),
        P.R_KNEE: (x + 20 * s, top + 320 * s),
        P.L_ANKLE: (x - 20 * s, top + 440 * s),
        P.R_ANKLE: (x + 20 * s, top + 440 * s)})


def sitting(x=300.0, top=200.0):
    return make({
        P.NOSE: (x, top), P.L_SHOULDER: (x - 30, top + 60),
        P.R_SHOULDER: (x + 30, top + 60),
        P.L_WRIST: (x - 40, top + 150), P.R_WRIST: (x + 40, top + 150),
        P.L_HIP: (x - 20, top + 200), P.R_HIP: (x + 20, top + 200),
        P.L_KNEE: (x - 25, top + 215), P.R_KNEE: (x + 25, top + 215),
        P.L_ANKLE: (x - 25, top + 330), P.R_ANKLE: (x + 25, top + 330)})


def lying(y=450.0, left=100.0):
    # head at left, feet at right, body horizontal
    return make({
        P.NOSE: (left, y), P.L_SHOULDER: (left + 60, y - 25),
        P.R_SHOULDER: (left + 60, y + 25),
        P.L_WRIST: (left + 190, y - 30), P.R_WRIST: (left + 190, y + 30),
        P.L_HIP: (left + 200, y - 18), P.R_HIP: (left + 200, y + 18),
        P.L_KNEE: (left + 320, y - 18), P.R_KNEE: (left + 320, y + 18),
        P.L_ANKLE: (left + 440, y - 18), P.R_ANKLE: (left + 440, y + 18)})


def bending(x=300.0, top=200.0):
    # torso leaning ~45 degrees forward
    return make({
        P.NOSE: (x + 140, top + 120),
        P.L_SHOULDER: (x + 85, top + 100), P.R_SHOULDER: (x + 105, top + 80),
        P.L_HIP: (x - 5, top + 200), P.R_HIP: (x + 15, top + 200),
        P.L_KNEE: (x - 5, top + 320), P.R_KNEE: (x + 15, top + 320),
        P.L_ANKLE: (x - 5, top + 440), P.R_ANKLE: (x + 15, top + 440)})


def upper_body_only(x=300.0, top=100.0):
    return make({
        P.NOSE: (x, top), P.L_SHOULDER: (x - 30, top + 60),
        P.R_SHOULDER: (x + 30, top + 60),
        P.L_HIP: (x - 20, top + 200), P.R_HIP: (x + 20, top + 200),
        P.L_KNEE: (x - 20, top + 320), P.R_KNEE: (x + 20, top + 320),
        P.L_ANKLE: (x - 20, top + 440), P.R_ANKLE: (x + 20, top + 440)},
        hidden=(P.L_KNEE, P.R_KNEE, P.L_ANKLE, P.R_ANKLE))


def blend(a: P.Pose, b: P.Pose, alpha: float) -> P.Pose:
    xy = (1 - alpha) * a.xy + alpha * b.xy
    vis = np.minimum(a.visibility, b.visibility)
    ok = vis > 0.3
    pts = xy[ok]

    return P.Pose(xy.astype(np.float32), vis, 0.9, 0,
                  (float(pts[:, 0].min()), float(pts[:, 1].min()),
                   float(pts[:, 0].max()), float(pts[:, 1].max())))


def transition(a, b, t0, duration, fps=10):
    """Frames easing from pose a to pose b over `duration` seconds."""

    n = max(1, int(duration * fps))

    return [(t0 + i / fps, blend(a, b, (i + 1) / n)) for i in range(n)]


def hold(pose, t0, seconds, fps=10):
    return [(t0 + i / fps, pose) for i in range(int(seconds * fps))]
