"""Subprocess environment hygiene.

The system install (openclaw-hardening#125) launches this CLI through a
wrapper that exports ``PYTHONHOME``, ``PYTHONPATH``, and ``LD_LIBRARY_PATH``
for a private Python 3.12 runtime under ``/opt``. Children we spawn — the
Node ``openclaw`` CLI — must not inherit them: the immediate child is
unaffected, but any grandchild that resolves ``python3`` would combine the
base image's interpreter with the private 3.12 stdlib, a fatal
``Py_Initialize`` abort rather than a graceful failure (#101).
"""

from __future__ import annotations

import os

_PRIVATE_RUNTIME_VARS = ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH")


def clean_env() -> dict[str, str]:
    """A copy of ``os.environ`` without the launcher's private-runtime vars.

    Pass as ``env=`` to every ``subprocess.run`` that execs a non-Python tool.
    """
    env = dict(os.environ)
    for key in _PRIVATE_RUNTIME_VARS:
        env.pop(key, None)
    return env
