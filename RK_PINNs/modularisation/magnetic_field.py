# magnetic_field.py
"""
Re-export magnetic field classes from the canonical Classes module.

This module provides backward-compatibility imports. For new code,
import directly from Classes.magnetic_field.
"""

import sys
import os

# Add parent directory to path for Classes import
_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from Classes.magnetic_field import (
    MagneticField,
    Quadratic_Field,
    LHCb_Field,
)

__all__ = ["MagneticField", "Quadratic_Field", "LHCb_Field"]