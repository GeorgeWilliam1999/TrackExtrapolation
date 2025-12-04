# particle.py
"""
Re-export particle_state class from the canonical Classes module.

This module provides backward-compatibility imports. For new code,
import directly from Classes.particle.
"""

import sys
import os

# Add parent directory to path for Classes import
_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from Classes.particle import particle_state

__all__ = ["particle_state"]
