"""Expose the RoboTwin policy API when this repository is cloned under ``RoboTwin/policy/actionunet``."""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_IMPLEMENTATION = _ROOT / "actionunet"
_OPENPI_SRC = _ROOT / "src"
_CLIENT_SRC = _ROOT / "packages" / "openpi-client" / "src"

for _path in (_OPENPI_SRC, _CLIENT_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(_IMPLEMENTATION) not in __path__:
    __path__.append(str(_IMPLEMENTATION))

from .robotwin_client import eval  # noqa: E402,F401
from .robotwin_client import get_model  # noqa: E402,F401
from .robotwin_client import reset_model  # noqa: E402,F401
