"""BUDDY launch: voice_only (see docs/buddy_deployment_guide.md)."""

from buddy_orchestrator import launch_support as ls
from buddy_orchestrator.launch_util import description


def generate_launch_description():
    return description(ls.voice_only())
