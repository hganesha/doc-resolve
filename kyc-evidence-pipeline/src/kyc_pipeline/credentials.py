"""Secrets from files or the environment.

For a secret NAME, the first of these that is set wins:
1. NAME_FILE: the path of a file holding the value (the Docker and Kubernetes convention);
2. KYC_SECRETS_DIR/NAME: a file named after the secret in a secrets folder;
3. NAME: the environment variable itself (convenient locally, but visible to child
   processes and often captured in crash reports).
Secret files must not be readable by group or others.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path


def _read(path: Path, name: str) -> str:
    try:
        mode = path.stat().st_mode
    except OSError:
        raise ValueError(f"Secret file for {name} is not readable") from None
    if os.name == "posix" and mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError(f"Secret file for {name} must not be accessible by group or others (chmod 600)")
    return path.read_text(encoding="utf-8").strip()


def get_secret(name: str, default: str = "") -> str:
    if os.environ.get(name + "_FILE"):
        return _read(Path(os.environ[name + "_FILE"]).expanduser(), name)
    folder = os.environ.get("KYC_SECRETS_DIR")
    if folder and (Path(folder).expanduser() / name).is_file():
        return _read(Path(folder).expanduser() / name, name)
    return os.environ.get(name, default)
