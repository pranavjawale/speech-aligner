"""Top-level config package. `import config` exposes everything in config.py.

Kept at repo top level (not inside src/asrpipe) so scripts and non-package tools
can read the same single source of truth. Requires the repo ROOT on PYTHONPATH
(setup_env.sh adds it).
"""
from config.config import *   # noqa: F401,F403
from config import config as config   # allow `from config import config`
