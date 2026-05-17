from __future__ import annotations

import os
import shutil
import sys


DEFAULT_CONDA_ENV = "yangxinyao"
_REEXEC_FLAG = "MELD_CONDA_ENV_REEXEC"


def ensure_conda_env(required_env: str = DEFAULT_CONDA_ENV) -> None:
    """Re-run script entry points inside the required conda environment."""
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env == required_env:
        return
    if os.environ.get(_REEXEC_FLAG) == "1":
        raise RuntimeError(
            f"Expected conda env '{required_env}', but current env is "
            f"'{current_env or '<none>'}'."
        )

    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            f"Scripts must run in conda env '{required_env}', but conda was not found."
        )

    env = os.environ.copy()
    env[_REEXEC_FLAG] = "1"
    os.execvpe(
        conda,
        [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            required_env,
            "python",
            *sys.argv,
        ],
        env,
    )
