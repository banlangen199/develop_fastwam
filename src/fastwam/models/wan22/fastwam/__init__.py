"""Original FastWAM model family."""

from .model import FastWAM
from .joint import FastWAMJoint
from .idm import FastWAMIDM

__all__ = ["FastWAM", "FastWAMJoint", "FastWAMIDM"]
