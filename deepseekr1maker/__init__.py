"""
DeepSeek R1 Maker - A Python library that automates the process of creating and training models like DeepSeek R1.
"""

from .__version__ import __version__

# Import main components for easier access
from . import model
from . import training
from . import rewards
from . import config
from . import data
from . import utils
from . import cli

__all__ = [
    "model",
    "training",
    "rewards",
    "config",
    "data",
    "utils",
    "cli",
    "__version__",
]
