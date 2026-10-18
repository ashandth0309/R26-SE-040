"""Vision companion engine (Sobiya_AdvancedVision).

Perception only.  Nothing in this package publishes motor commands.  The
camera is owned by exactly one service (``camera_service``); every other
module consumes frames from the shared bus or from ``/camera/image_raw``.
"""
