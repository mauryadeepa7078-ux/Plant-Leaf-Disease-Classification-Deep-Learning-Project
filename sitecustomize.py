"""Optional local package bootstrap.

On the development machine, pandas lives in an existing Python 3.11
environment and one missing dependency (pytz) was placed in ``vendor/`` to
avoid modifying that shared environment further.  A normal installation via
``pip install -r requirements.txt`` does not need this file.

Python imports ``sitecustomize`` automatically when it is on ``sys.path``
(which it is when commands are run from the project root).  The hook is a
no-op on machines without a ``vendor`` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
